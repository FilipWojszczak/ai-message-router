"""One-shot for the `ollama-pull` compose service: pull the model, then load it.

Runs on the plain python image (stdlib only), bind-mounted by docker-compose.yml.
`api` waits for this to exit 0, so the API never starts before the model is in RAM.
Every failure exits non-zero with a readable message, which makes
`docker compose up -d` fail before the API starts.
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from typing import Any

# Checked before a pull: without internet, a pull hangs instead of failing.
REGISTRY = ("registry.ollama.ai", 443)
# Seconds; loading 2.5 GB from a slow disk can take minutes.
LOAD_TIMEOUT = 300.0


def post(
    base: str, path: str, body: dict[str, Any], timeout: float | None
) -> HTTPResponse:
    # S310: the URL is OLLAMA_URL from compose wiring (http://ollama:11434), not
    # user input, so no file: or custom scheme can reach urllib.
    req = urllib.request.Request(  # noqa: S310
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def is_present(base: str, model: str) -> bool:
    """Whether the model is already in Ollama's volume. Local, works offline."""
    try:
        post(base, "/api/show", {"model": model}, timeout=10).close()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        return False
    return True


def pull(base: str, model: str) -> None:
    """Pull through the server, printing each new status once."""
    try:
        socket.create_connection(REGISTRY, timeout=5).close()
    except OSError as e:
        sys.exit(
            f"ERROR: model {model} is missing and {REGISTRY[0]} is unreachable ({e})."
        )

    print(f"Pulling {model} ...", flush=True)
    last: str | None = None
    # No timeout: large models on slow links legitimately take long.
    with post(base, "/api/pull", {"model": model}, timeout=None) as resp:
        for line in resp:
            msg = json.loads(line)
            if "error" in msg:
                sys.exit(f"ERROR: pull failed: {msg['error']}")
            if msg.get("status") != last:
                last = msg.get("status")
                print(last, flush=True)
    if last != "success":
        sys.exit(f"ERROR: pull ended without success (last status: {last}).")


def warm_up(base: str, model: str) -> None:
    """Load the model into RAM.

    A generate call without a prompt only loads the model. It sends no keep_alive, so
    the server's OLLAMA_KEEP_ALIVE decides how long the model stays.
    """
    start = time.monotonic()
    try:
        post(base, "/api/generate", {"model": model}, timeout=LOAD_TIMEOUT).close()
    except urllib.error.HTTPError as e:
        # Ollama explains itself in the body, e.g. not enough system memory.
        body = e.read().decode(errors="replace")
        try:
            reason = json.loads(body)["error"]
        except ValueError, KeyError, TypeError:
            reason = body.strip() or f"HTTP {e.code}"
        sys.exit(f"ERROR: could not load model {model}: {reason}")
    except OSError as e:
        # A timeout, or the connection dropped (e.g. Ollama killed for memory).
        sys.exit(f"ERROR: could not load model {model}: {e}")
    print(f"Model {model} loaded in {time.monotonic() - start:.1f} s.", flush=True)


def main() -> None:
    base = os.environ["OLLAMA_URL"]
    model = os.environ["OLLAMA_MODEL"]
    if is_present(base, model):
        print(f"Model {model} already present, skipping pull.", flush=True)
    else:
        pull(base, model)
    warm_up(base, model)


if __name__ == "__main__":
    main()
