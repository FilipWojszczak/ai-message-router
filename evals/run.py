"""Run the labeled cases through the real ``route_message`` against a local Ollama.

    uv run python -m evals.run --model qwen3:1.7b --model qwen3:4b-instruct-2507-q4_K_M

Prompt, tool, fallback and deadline are the shipped ones. No email is sent:
``aiosmtplib.send`` is replaced by a capturing fake. The script never pulls a model.
The Markdown report goes to stdout, progress to stderr.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Final, Literal, cast
from unittest import mock

import aiosmtplib
import httpx

from evals.dataset import Case, DatasetError, load_cases
from evals.metrics import (
    FallbackReason,
    ModelMeta,
    ModelSummary,
    RunMeta,
    RunRecord,
    render_markdown,
    summarize,
)
from message_router.agent import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    RouterAgent,
    RoutingContext,
    # Private, but reused on purpose: the eval must report the same fallback reason
    # that route_message logs.
    _fallback_reason,
    _last_ai_message,
    build_agent,
    build_chat_model,
    route_message,
)
from message_router.config import Settings
from message_router.prompts import SYSTEM_PROMPT

REQUESTER: Final = "eval@example.com"
# Ollama's default context; the 2000-char message limit is sized for it.
EXPECTED_CONTEXT: Final = 4096
LIBRARIES: Final = ("langchain", "langchain-ollama", "ollama")

type SplitChoice = Literal["all", "dev", "holdout"]


class EvalError(Exception):
    """The run cannot continue; the message is shown to the user as is."""


# --- Ollama admin calls (sync, outside the event loop) -------------------------------


class OllamaAdmin:
    """The few non-chat endpoints the eval needs."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=30)

    def _request(self, method: str, path: str, body: Any = None) -> httpx.Response:
        try:
            return self._client.request(method, path, json=body)
        except httpx.TransportError as exc:
            raise EvalError(
                f"Ollama is unreachable at {self._base_url} ({type(exc).__name__})."
            ) from exc

    def _json(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        response = self._request(method, path, body)
        if response.is_error:
            raise EvalError(f"{method} {path} returned HTTP {response.status_code}.")
        return response.json()

    def version(self) -> str:
        return self._json("GET", "/api/version")["version"]

    def tags(self) -> dict[str, dict[str, Any]]:
        """Pulled models by name. The only endpoint that reports the digest."""
        return {m["name"]: m for m in self._json("GET", "/api/tags")["models"]}

    def show(self, model: str) -> dict[str, Any]:
        response = self._request("POST", "/api/show", {"model": model})
        if response.status_code == 404:
            raise EvalError(
                f"Model {model} is not pulled. Run: "
                f"docker exec ollama-eval ollama pull {model}"
            )
        if response.is_error:
            raise EvalError(f"POST /api/show returned HTTP {response.status_code}.")
        return response.json()

    def loaded(self) -> dict[str, dict[str, Any]]:
        return {m["name"]: m for m in self._json("GET", "/api/ps")["models"]}

    def unload_all(self) -> None:
        """Free the RAM held by other models, and make the next call a cold one."""
        for name in self.loaded():
            self._json("POST", "/api/generate", {"model": name, "keep_alive": 0})
        for _ in range(100):
            if not self.loaded():
                return
            time.sleep(0.1)
        raise EvalError(f"Models are still loaded: {sorted(self.loaded())}.")


# --- eval-only shims -----------------------------------------------------------------


@dataclass
class CapturingSend:
    """Stands in for ``aiosmtplib.send``: records the envelope recipients only."""

    recipients: list[str] = field(default_factory=list)

    async def __call__(
        self, message: EmailMessage, **kwargs: Any
    ) -> tuple[dict[str, Any], str]:
        self.recipients.extend(kwargs["recipients"])
        return {}, "OK"


class CapturingAgent:
    """Delegates to the compiled agent and keeps the final state of the last run.

    ``route_message`` returns only the routing result; the state's last AIMessage
    carries Ollama's timings and token counts.
    """

    def __init__(self, agent: RouterAgent) -> None:
        self._agent = agent
        self.state: dict[str, Any] | None = None

    async def ainvoke(
        self, graph_input: Any, *, context: RoutingContext
    ) -> dict[str, Any]:
        self.state = None  # a run that raises must not leave the previous state
        state = await self._agent.ainvoke(graph_input, context=context)
        self.state = state
        return state


# --- running -------------------------------------------------------------------------


def _seconds(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else round(nanoseconds / 1e9, 3)


async def _run_case(
    agent: CapturingAgent,
    send: CapturingSend,
    case: Case,
    settings: Settings,
    *,
    rep: int,
    cold: bool = False,
) -> RunRecord:
    already_sent = len(send.recipients)
    start = time.perf_counter()
    try:
        result = await route_message(
            agent,  # pyright: ignore[reportArgumentType]  # duck-typed wrapper
            REQUESTER,
            case.message,
            settings=settings,
        )
    except (LLMTimeoutError, LLMResponseError) as exc:
        return RunRecord(
            case_id=case.id,
            rep=rep,
            cold=cold,
            wall_s=round(time.perf_counter() - start, 3),
            error=type(exc).__name__,
        )
    except LLMUnavailableError as exc:
        raise EvalError(f"Ollama became unreachable at case {case.id}.") from exc
    wall_s = round(time.perf_counter() - start, 3)

    sent = send.recipients[already_sent:]
    if sent != [result.department.value]:
        raise EvalError(
            f"Case {case.id}: expected exactly one send to "
            f"{result.department.value}, captured {sent}."
        )
    last_ai = _last_ai_message(agent.state["messages"]) if agent.state else None
    timings = last_ai.response_metadata if last_ai is not None else {}
    reason = _fallback_reason(last_ai) if result.fallback else None
    return RunRecord(
        case_id=case.id,
        rep=rep,
        cold=cold,
        department=result.department,
        fallback=result.fallback,
        fallback_reason=cast("FallbackReason | None", reason),
        wall_s=wall_s,
        load_duration_s=_seconds(timings.get("load_duration")),
        prompt_eval_duration_s=_seconds(timings.get("prompt_eval_duration")),
        eval_duration_s=_seconds(timings.get("eval_duration")),
        prompt_eval_count=timings.get("prompt_eval_count"),
        eval_count=timings.get("eval_count"),
        done_reason=timings.get("done_reason"),
    )


def _progress(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


async def _run_model(
    settings: Settings, cases: Sequence[Case], *, reps: int
) -> list[RunRecord]:
    """One cold call (the first case), then ``reps`` sequential passes in file order."""
    model = settings.ollama_model
    agent = CapturingAgent(build_agent(build_chat_model(settings)))
    send = CapturingSend()
    with mock.patch.object(aiosmtplib, "send", send):
        cold = await _run_case(agent, send, cases[0], settings, rep=0, cold=True)
        _progress(
            f"{model}: cold call {cold.wall_s:.1f} s (load {cold.load_duration_s} s)"
        )
        records = [cold]
        for rep in range(1, reps + 1):
            start = time.perf_counter()
            batch = [
                await _run_case(agent, send, case, settings, rep=rep) for case in cases
            ]
            tool_calls = sum(r.error is None and not r.fallback for r in batch)
            _progress(
                f"{model}: pass {rep}/{reps} in {time.perf_counter() - start:.0f} s, "
                f"tool calls {tool_calls}/{len(batch)}"
            )
            records += batch
    return records


# --- metadata ------------------------------------------------------------------------


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _cpu_model() -> str | None:
    match = re.search(r"^model name\s*:\s*(.+)$", _read("/proc/cpuinfo"), re.MULTILINE)
    return match.group(1).strip() if match else (platform.processor() or None)


def _ram_gb() -> float | None:
    match = re.search(r"^MemTotal:\s*(\d+) kB", _read("/proc/meminfo"), re.MULTILINE)
    return round(int(match.group(1)) / 1024**2, 1) if match else None


def _run_meta(
    settings: Settings, ollama_version: str, *, split: SplitChoice, reps: int
) -> RunMeta:
    return RunMeta(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        ollama_version=ollama_version,
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count(),
        ram_gb=_ram_gb(),
        libraries={name: importlib.metadata.version(name) for name in LIBRARIES},
        num_predict=settings.ollama_num_predict,
        reasoning=settings.ollama_reasoning,
        timeout_s=settings.ollama_timeout,
        system_prompt_sha=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        split=split,
        reps=reps,
        models={},
    )


def _model_meta(
    model: str, tag: dict[str, Any] | None, show: dict[str, Any], ps: dict[str, Any]
) -> ModelMeta:
    """Describe the model as it ran; fail unless it was CPU-only at the default context.

    Anything else (GPU offload, another context size) makes the latency and the
    truncation figures incomparable with the compose stack.
    """
    size, size_vram = ps.get("size"), ps.get("size_vram")
    context = ps.get("context_length")
    if size_vram != 0 or context != EXPECTED_CONTEXT:
        raise EvalError(
            f"{model} ran with size_vram={size_vram}, context_length={context}; "
            f"expected CPU only (size_vram=0) and context {EXPECTED_CONTEXT}."
        )
    details = show.get("details", {})
    return ModelMeta(
        digest=tag["digest"][:12] if tag else None,
        size_gb=round(tag["size"] / 1e9, 2) if tag else None,
        parameter_size=details.get("parameter_size"),
        quantization=details.get("quantization_level"),
        capabilities=tuple(show.get("capabilities", ())),
        processor="100% CPU",
        context_length=context,
        loaded_size_gb=round(size / 1e9, 2) if size else None,
    )


def _write_result(
    out_dir: Path,
    split: SplitChoice,
    meta: RunMeta,
    summary: ModelSummary,
    records: Sequence[RunRecord],
) -> Path:
    """Full runs own ``<slug>.json``; partial runs get a ``.<split>`` suffix."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", summary.model)
    suffix = "" if split == "all" else f".{split}"
    path = out_dir / f"{slug}{suffix}.json"
    payload = {
        "meta": meta.model_dump(mode="json"),
        "summary": summary.model_dump(mode="json"),
        "records": [record.model_dump(mode="json") for record in records],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


# --- CLI -----------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run the labeled routing cases against a local Ollama.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="Ollama model tag; repeat to compare several (run in this order)",
    )
    parser.add_argument("--reps", type=_positive_int, default=2)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--out", type=Path, default=Path("docs/evals/results"))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> str:
    split: SplitChoice = args.split
    cases = [case for case in load_cases() if split in ("all", case.split)]
    admin = OllamaAdmin(args.ollama_url)
    ollama_version = admin.version()
    tags = admin.tags()
    # Check every model before running any: a missing one fails in seconds.
    shows = {model: admin.show(model) for model in args.models}

    summaries: list[ModelSummary] = []
    meta: RunMeta | None = None
    for model in args.models:
        settings = Settings(ollama_model=model, ollama_url=args.ollama_url)
        if meta is None:
            meta = _run_meta(settings, ollama_version, split=split, reps=args.reps)
        admin.unload_all()
        records = asyncio.run(_run_model(settings, cases, reps=args.reps))
        ps = admin.loaded().get(model)
        if ps is None:
            raise EvalError(f"{model} is not loaded after its run; cannot verify it.")
        model_meta = _model_meta(model, tags.get(model), shows[model], ps)
        summary = summarize(model, cases, records)
        path = _write_result(
            args.out,
            split,
            meta.model_copy(update={"models": {model: model_meta}}),
            summary,
            records,
        )
        _progress(f"{model}: wrote {path}")
        meta = meta.model_copy(update={"models": meta.models | {model: model_meta}})
        summaries.append(summary)
    if meta is None:  # unreachable: --model is required
        raise EvalError("No model given.")
    return render_markdown(summaries, meta)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        report = run(args)
    except (EvalError, DatasetError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(report, end="")


if __name__ == "__main__":
    main()
