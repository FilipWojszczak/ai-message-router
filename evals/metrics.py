"""Metrics over eval run records, and their Markdown report. Pure: no I/O."""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, computed_field

from evals.dataset import Case
from message_router.departments import Department

# Decision rule gates (spec D3), fixed before the full run.
MIN_TOOL_CALL_RATE: Final = 0.95
MIN_ACCURACY: Final = 0.85
MAX_SHORT_P95_S: Final = 5.0
MAX_LONG_WALL_S: Final = 20.0
# Messages below this length count as "short" for the latency gate.
SHORT_MAX_LENGTH: Final = 500

type FallbackReason = Literal["no_tool_call", "invalid_tool_call"]
type Split = Literal["dev", "holdout"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class RunRecord(_Frozen):
    """One `route_message` call. Durations are seconds."""

    case_id: str
    rep: int  # 0 is the cold call; warm passes start at 1
    cold: bool = False
    department: Department | None = None  # None when the call raised
    fallback: bool = False
    fallback_reason: FallbackReason | None = None
    error: str | None = None  # exception class name
    wall_s: float
    load_duration_s: float | None = None
    prompt_eval_duration_s: float | None = None
    eval_duration_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None  # "length" means the output hit num_predict


class Rate(_Frozen):
    hits: int
    total: int

    @computed_field
    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None


class Quality(_Frozen):
    """Rates over one slice of warm records. tool_call + miss + errors = 100%."""

    tool_call: Rate  # the agent's own tool call sent the email
    miss: Rate  # the code fallback sent it: no_tool_call + invalid_tool_call
    no_tool_call: Rate
    invalid_tool_call: Rate
    errors: Rate  # the call raised (timeout, Ollama error); nothing was sent
    accuracy: Rate  # final department is in the case's accepted set


class Outcome(_Frozen):
    case_id: str
    rep: int
    detail: str  # a fallback reason, an error name or the chosen department


class ModelSummary(_Frozen):
    model: str
    cases: int
    reps: int
    splits: tuple[Split, ...]
    overall: Quality
    by_split: dict[Split, Quality]
    accuracy_by_lang: dict[str, Rate]
    accuracy_by_department: dict[str, Rate]  # by the first expected department
    task_examples: Rate
    injection_followed: Rate
    consistency: Rate  # cases whose reps all chose the same department
    inconsistent_case_ids: tuple[str, ...]
    missed: tuple[Outcome, ...]  # fallbacks and errors
    wrong: tuple[Outcome, ...]  # department outside the accepted set
    truncated_case_ids: tuple[str, ...]  # done_reason == "length"
    short_wall_p50_s: float | None
    short_wall_p95_s: float | None
    long_wall_max_s: float | None
    max_wall_s: float | None  # over every call, the cold one included
    cold_load_s: float | None
    cold_wall_s: float | None
    cold_prompt_eval_s: float | None  # the system prompt evaluated uncached
    prompt_eval_count_mean: float | None
    prompt_eval_count_max: int | None
    eval_count_mean: float | None
    eval_count_max: int | None
    prompt_eval_duration_mean_s: float | None
    eval_duration_mean_s: float | None
    gates: dict[str, bool | None]  # None: not measurable in this run
    passes_gates: bool | None  # None unless the run covered both splits


class ModelMeta(_Frozen):
    digest: str | None = None
    size_gb: float | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    capabilities: tuple[str, ...] = ()
    processor: str | None = None
    context_length: int | None = None
    loaded_size_gb: float | None = None


class RunMeta(_Frozen):
    timestamp_utc: str
    ollama_version: str
    cpu_model: str | None
    cpu_count: int | None
    ram_gb: float | None
    libraries: dict[str, str]
    num_predict: int
    reasoning: bool
    timeout_s: float
    system_prompt_sha: str
    split: Literal["all", "dev", "holdout"]
    reps: int
    models: dict[str, ModelMeta]


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q% at or below it."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if not 0 < q <= 100:
        raise ValueError("q must be in (0, 100]")
    return ordered[math.ceil(q * len(ordered) / 100) - 1]


def _is_tool_call(record: RunRecord) -> bool:
    return record.error is None and not record.fallback


def _is_correct(case: Case, record: RunRecord) -> bool:
    return record.error is None and record.department in case.expected


def _rate(records: Sequence[RunRecord], hit: Callable[[RunRecord], bool]) -> Rate:
    return Rate(hits=sum(hit(record) for record in records), total=len(records))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _present[T](values: Iterable[T | None]) -> list[T]:
    return [value for value in values if value is not None]


def _quality(by_id: Mapping[str, Case], records: Sequence[RunRecord]) -> Quality:
    def fell_back(reason: FallbackReason) -> Callable[[RunRecord], bool]:
        return lambda r: r.error is None and r.fallback and r.fallback_reason == reason

    return Quality(
        tool_call=_rate(records, _is_tool_call),
        miss=_rate(records, lambda r: r.error is None and r.fallback),
        no_tool_call=_rate(records, fell_back("no_tool_call")),
        invalid_tool_call=_rate(records, fell_back("invalid_tool_call")),
        errors=_rate(records, lambda r: r.error is not None),
        accuracy=_rate(records, lambda r: _is_correct(by_id[r.case_id], r)),
    )


def _accuracy_by(
    by_id: Mapping[str, Case],
    records: Sequence[RunRecord],
    key: Callable[[Case], str],
) -> dict[str, Rate]:
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        groups.setdefault(key(by_id[record.case_id]), []).append(record)
    return {
        name: _rate(group, lambda r: _is_correct(by_id[r.case_id], r))
        for name, group in sorted(groups.items())
    }


def summarize(
    model: str, cases: Sequence[Case], records: Sequence[RunRecord]
) -> ModelSummary:
    """Aggregate one model's records. Cold records feed only the cold figures."""
    by_id = {case.id: case for case in cases}
    if unknown := {record.case_id for record in records} - set(by_id):
        raise ValueError(f"records for unknown cases: {sorted(unknown)}")
    warm = [record for record in records if not record.cold]
    cold = [record for record in records if record.cold]
    splits: tuple[Split, ...] = tuple(sorted({case.split for case in cases}))

    def of(predicate: Callable[[Case], bool]) -> list[RunRecord]:
        return [record for record in warm if predicate(by_id[record.case_id])]

    overall = _quality(by_id, warm)
    task_examples = _rate(
        of(lambda c: "task_example" in c.tags),
        lambda r: _is_correct(by_id[r.case_id], r),
    )
    injection_followed = _rate(
        of(lambda c: c.injected is not None),
        lambda r: r.department == by_id[r.case_id].injected,
    )

    choices: dict[str, set[Department | None]] = {}
    for record in warm:
        choices.setdefault(record.case_id, set()).add(record.department)
    inconsistent = tuple(sorted(cid for cid, seen in choices.items() if len(seen) > 1))

    short_walls = [r.wall_s for r in of(lambda c: len(c.message) < SHORT_MAX_LENGTH)]
    long_walls = [r.wall_s for r in of(lambda c: "long" in c.tags)]
    short_p95 = percentile(short_walls, 95) if short_walls else None
    long_max = max(long_walls, default=None)
    prompt_counts = _present(r.prompt_eval_count for r in warm)
    eval_counts = _present(r.eval_count for r in warm)

    gates: dict[str, bool | None] = {
        "tool_call_rate": None
        if overall.tool_call.value is None
        else overall.tool_call.value >= MIN_TOOL_CALL_RATE,
        "accuracy": None
        if overall.accuracy.value is None
        else overall.accuracy.value >= MIN_ACCURACY,
        "task_examples": None
        if task_examples.total == 0
        else task_examples.hits == task_examples.total,
        "short_p95": None if short_p95 is None else short_p95 <= MAX_SHORT_P95_S,
        "long_wall": None if long_max is None else long_max <= MAX_LONG_WALL_S,
    }
    full_set = splits == ("dev", "holdout")

    return ModelSummary(
        model=model,
        cases=len(cases),
        reps=len({record.rep for record in warm}),
        splits=splits,
        overall=overall,
        by_split={
            split: _quality(by_id, of(lambda c, split=split: c.split == split))
            for split in splits
        },
        accuracy_by_lang=_accuracy_by(by_id, warm, lambda c: c.lang),
        accuracy_by_department=_accuracy_by(by_id, warm, lambda c: c.expected[0].value),
        task_examples=task_examples,
        injection_followed=injection_followed,
        consistency=Rate(hits=len(choices) - len(inconsistent), total=len(choices)),
        inconsistent_case_ids=inconsistent,
        missed=tuple(
            Outcome(
                case_id=r.case_id,
                rep=r.rep,
                detail=r.error or r.fallback_reason or "fallback",
            )
            for r in warm
            if not _is_tool_call(r)
        ),
        wrong=tuple(
            Outcome(
                case_id=r.case_id,
                rep=r.rep,
                detail=r.error or (r.department.value if r.department else "none"),
            )
            for r in warm
            if not _is_correct(by_id[r.case_id], r)
        ),
        truncated_case_ids=tuple(
            sorted({r.case_id for r in warm if r.done_reason == "length"})
        ),
        short_wall_p50_s=percentile(short_walls, 50) if short_walls else None,
        short_wall_p95_s=short_p95,
        long_wall_max_s=long_max,
        max_wall_s=max((r.wall_s for r in records), default=None),
        cold_load_s=cold[0].load_duration_s if cold else None,
        cold_wall_s=cold[0].wall_s if cold else None,
        cold_prompt_eval_s=cold[0].prompt_eval_duration_s if cold else None,
        prompt_eval_count_mean=_mean(prompt_counts),
        prompt_eval_count_max=max(prompt_counts, default=None),
        eval_count_mean=_mean(eval_counts),
        eval_count_max=max(eval_counts, default=None),
        prompt_eval_duration_mean_s=_mean(
            _present(r.prompt_eval_duration_s for r in warm)
        ),
        eval_duration_mean_s=_mean(_present(r.eval_duration_s for r in warm)),
        gates=gates,
        passes_gates=all(gate is True for gate in gates.values()) if full_set else None,
    )


# --- Markdown report -----------------------------------------------------------------

_TABLE_HEADER: Final = (
    "| Model | Tool calls (≥ 95%) | Accuracy (≥ 85%) | Task examples (100%) "
    "| Warm p95, < 500 chars (≤ 5 s) | Long case, max (≤ 20 s) | Injection followed "
    "| Consistent | Cold load | All gates |"
)
_TABLE_RULE: Final = "|---|---|---|---|---|---|---|---|---|---|"


def _pct(rate: Rate) -> str:
    if rate.value is None:
        return "n/a"
    return f"{rate.value:.1%} ({rate.hits}/{rate.total})"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _mark(gate: bool | None) -> str:
    return "" if gate is None else (" ✓" if gate else " ✗")


def _short(address: str) -> str:
    return address.removesuffix("example.com")


def _miss_line(quality: Quality) -> str:
    line = (
        f"miss {_pct(quality.miss)}: no_tool_call {_pct(quality.no_tool_call)}, "
        f"invalid_tool_call {_pct(quality.invalid_tool_call)}"
    )
    if quality.errors.hits:
        line += f"; errors {_pct(quality.errors)}"
    return line


def _rates(rates: Mapping[str, Rate]) -> str:
    return ", ".join(f"{_short(name)} {_pct(rate)}" for name, rate in rates.items())


def _outcomes(outcomes: Sequence[Outcome]) -> str:
    counts = Counter((o.case_id, _short(o.detail)) for o in outcomes)
    if not counts:
        return "none"
    return ", ".join(
        f"`{case_id}` ({detail} ×{count})"  # noqa: RUF001  # a real times sign
        for (case_id, detail), count in counts.items()
    )


def _ids(case_ids: Sequence[str]) -> str:
    return ", ".join(f"`{case_id}`" for case_id in case_ids) or "none"


def _table_row(summary: ModelSummary) -> str:
    gates = summary.gates
    if summary.passes_gates is None:
        verdict = "n/a (partial run)"
    else:
        verdict = "pass" if summary.passes_gates else "FAIL"
    cells = [
        f"`{summary.model}`",
        _pct(summary.overall.tool_call) + _mark(gates["tool_call_rate"]),
        _pct(summary.overall.accuracy) + _mark(gates["accuracy"]),
        _pct(summary.task_examples) + _mark(gates["task_examples"]),
        _secs(summary.short_wall_p95_s) + _mark(gates["short_p95"]),
        _secs(summary.long_wall_max_s) + _mark(gates["long_wall"]),
        _pct(summary.injection_followed),
        _pct(summary.consistency),
        _secs(summary.cold_load_s),
        verdict,
    ]
    return "| " + " | ".join(cells) + " |"


def _model_line(meta: ModelMeta) -> str:
    return (
        f"- Model: digest `{meta.digest}`, {_num(meta.size_gb, 2)} GB on disk, "
        f"{meta.parameter_size} {meta.quantization}; {meta.processor}, "
        f"context {meta.context_length}, {_num(meta.loaded_size_gb, 2)} GB loaded; "
        f"capabilities: {', '.join(meta.capabilities) or 'n/a'}"
    )


def _detail_lines(summary: ModelSummary, meta: ModelMeta | None) -> list[str]:
    lines = [f"### `{summary.model}`", ""]
    if meta is not None:
        lines.append(_model_line(meta))
    lines.append(f"- Tool-call miss rate: {_miss_line(summary.overall)}")
    if len(summary.by_split) > 1:
        lines += [
            f"  - {split}: {_miss_line(quality)}"
            for split, quality in summary.by_split.items()
        ]
    by_split = {split: quality.accuracy for split, quality in summary.by_split.items()}
    lines += [
        f"- Missed (code fallback or error): {_outcomes(summary.missed)}",
        f"- Accuracy by split: {_rates(by_split)}",
        f"- Accuracy by language: {_rates(summary.accuracy_by_lang)}",
        f"- Accuracy by department: {_rates(summary.accuracy_by_department)}",
        f"- Wrong department: {_outcomes(summary.wrong)}",
        f"- Reps disagree: {_ids(summary.inconsistent_case_ids)}",
        f"- Warm latency: p50 {_secs(summary.short_wall_p50_s)}, "
        f"p95 {_secs(summary.short_wall_p95_s)} (< {SHORT_MAX_LENGTH} chars); "
        f"long case max {_secs(summary.long_wall_max_s)}; "
        f"max over all calls {_secs(summary.max_wall_s)}",
        f"- Cold call (page cache likely warm, so a best case): "
        f"load {_secs(summary.cold_load_s)}, wall {_secs(summary.cold_wall_s)}, "
        f"uncached prompt eval {_secs(summary.cold_prompt_eval_s)}",
        f"- Tokens: prompt mean {_num(summary.prompt_eval_count_mean)}, "
        f"max {_num(summary.prompt_eval_count_max)}; "
        f"output mean {_num(summary.eval_count_mean, 1)}, "
        f"max {_num(summary.eval_count_max)}; "
        f"mean prompt eval {_secs(summary.prompt_eval_duration_mean_s)}, "
        f"mean eval {_secs(summary.eval_duration_mean_s)}",
        f"- Output cut at num_predict: {_ids(summary.truncated_case_ids)}",
    ]
    return lines


def _meta_lines(meta: RunMeta, summaries: Sequence[ModelSummary]) -> list[str]:
    libraries = ", ".join(f"{name} {ver}" for name, ver in meta.libraries.items())
    sets = sorted(
        {
            f"{s.cases} cases × {s.reps} rep{'' if s.reps == 1 else 's'}"  # noqa: RUF001
            for s in summaries
        }
    )
    return [
        f"- Run: {meta.timestamp_utc}, Ollama {meta.ollama_version}, "
        f"split `{meta.split}` ({'; '.join(sets)})",
        f"- Hardware: {meta.cpu_model}, {meta.cpu_count} threads, "
        f"{_num(meta.ram_gb)} GB RAM",
        f"- Settings: num_predict {meta.num_predict}, reasoning {meta.reasoning}, "
        f"timeout {meta.timeout_s:g} s, system prompt `{meta.system_prompt_sha}`",
        f"- Libraries: {libraries}",
    ]


def render_markdown(summaries: Sequence[ModelSummary], meta: RunMeta) -> str:
    """The comparison table plus per-model details, ready for docs/evals/README.md."""
    lines = ["## Routing eval results", "", *_meta_lines(meta, summaries), ""]
    lines += [_TABLE_HEADER, _TABLE_RULE]
    lines += [_table_row(summary) for summary in summaries]
    for summary in summaries:
        lines += ["", *_detail_lines(summary, meta.models.get(summary.model))]
    return "\n".join(lines) + "\n"
