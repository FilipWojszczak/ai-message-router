"""docker/ollama_pull.py against a fake Ollama on 127.0.0.1 (in-process, no Docker)."""

import importlib.util
import json
import socket
import threading
import time
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, override

import pytest

pytestmark = pytest.mark.unit

MODEL = "fake:1b"
NOT_FOUND = (404, '{"error": "model not found"}')


def _load_script() -> ModuleType:
    # Not a package: it is bind-mounted into a plain python container.
    path = Path(__file__).parents[1] / "docker" / "ollama_pull.py"
    spec = importlib.util.spec_from_file_location("ollama_pull", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ollama_pull = _load_script()

# (status, body), or None to close the connection without answering.
type Reply = tuple[int, str] | None


@dataclass
class FakeOllama:
    replies: dict[str, Reply] = field(default_factory=dict)
    generate_delay: float = 0.0
    received: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.received]


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOllama]:
    fake = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:  # the name http.server dispatches POST to
            length = int(self.headers.get("Content-Length", 0))
            fake.received.append((self.path, json.loads(self.rfile.read(length))))
            if self.path == "/api/generate" and fake.generate_delay:
                time.sleep(fake.generate_delay)
            reply = fake.replies.get(self.path, NOT_FOUND)
            if reply is None:
                return
            status, body = reply
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    # Port 0: the OS picks a free port; nothing is published or exposed.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    monkeypatch.setenv("OLLAMA_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("OLLAMA_MODEL", MODEL)
    # The registry preflight connects to the fake too, never to the internet.
    monkeypatch.setattr(ollama_pull, "REGISTRY", ("127.0.0.1", port))
    yield fake
    server.shutdown()
    server.server_close()


def _exit_message() -> Any:
    with pytest.raises(SystemExit) as exc_info:
        ollama_pull.main()
    return exc_info.value.code


def _ndjson(*messages: dict[str, str]) -> str:
    return "".join(json.dumps(message) + "\n" for message in messages)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_present_model_is_only_loaded(
    fake_ollama: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_ollama.replies = {"/api/show": (200, "{}"), "/api/generate": (200, "{}")}

    ollama_pull.main()

    assert fake_ollama.paths == ["/api/show", "/api/generate"]
    # No keep_alive: the server's OLLAMA_KEEP_ALIVE must decide.
    assert fake_ollama.received[1] == ("/api/generate", {"model": MODEL})
    out = capsys.readouterr().out
    assert f"Model {MODEL} already present, skipping pull." in out
    assert f"Model {MODEL} loaded in " in out


def test_missing_model_is_pulled_then_loaded(
    fake_ollama: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_ollama.replies = {
        "/api/pull": (
            200,
            _ndjson(
                {"status": "pulling manifest"},
                {"status": "pulling manifest"},
                {"status": "success"},
            ),
        ),
        "/api/generate": (200, "{}"),
    }

    ollama_pull.main()

    assert fake_ollama.paths == ["/api/show", "/api/pull", "/api/generate"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [f"Pulling {MODEL} ...", "pulling manifest", "success"]
    assert lines[3].startswith(f"Model {MODEL} loaded in ")


def test_unreachable_registry_fails_before_pulling(
    fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama_pull, "REGISTRY", ("127.0.0.1", _closed_port()))

    message = _exit_message()

    assert message.startswith(f"ERROR: model {MODEL} is missing and 127.0.0.1 is")
    assert fake_ollama.paths == ["/api/show"]


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [{"error": "pull model manifest: file does not exist"}],
            "ERROR: pull failed: pull model manifest: file does not exist",
        ),
        (
            [{"status": "pulling manifest"}],
            "ERROR: pull ended without success (last status: pulling manifest).",
        ),
    ],
    ids=["error", "no_success"],
)
def test_failed_pull_stops_before_loading(
    messages: list[dict[str, str]], expected: str, fake_ollama: FakeOllama
) -> None:
    fake_ollama.replies = {"/api/pull": (200, _ndjson(*messages))}

    assert _exit_message() == expected
    assert "/api/generate" not in fake_ollama.paths


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (
            (500, '{"error": "model requires more system memory (5.3 GiB)"}'),
            "model requires more system memory (5.3 GiB)",
        ),
        ((500, "boom"), "boom"),
        ((500, ""), "HTTP 500"),
        (None, "Remote end closed connection without response"),
    ],
    ids=["ollama_error", "non_json_body", "empty_body", "connection_dropped"],
)
def test_load_failure_exits_with_the_reason(
    reply: Reply, reason: str, fake_ollama: FakeOllama
) -> None:
    fake_ollama.replies = {"/api/show": (200, "{}"), "/api/generate": reply}

    assert _exit_message() == f"ERROR: could not load model {MODEL}: {reason}"


def test_load_timeout_exits(
    fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama_pull, "LOAD_TIMEOUT", 0.2)
    fake_ollama.generate_delay = 1.0
    fake_ollama.replies = {"/api/show": (200, "{}"), "/api/generate": (200, "{}")}

    assert _exit_message() == f"ERROR: could not load model {MODEL}: timed out"


def test_unexpected_show_error_is_not_swallowed(fake_ollama: FakeOllama) -> None:
    fake_ollama.replies = {"/api/show": (500, "{}")}

    with pytest.raises(urllib.error.HTTPError):
        ollama_pull.main()
    assert fake_ollama.paths == ["/api/show"]
