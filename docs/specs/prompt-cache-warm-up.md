# Prompt-cache warm-up at API startup

Status: draft, 2026-11-17. Implements stage 4's optional item **O5** and closes the first
"known gap" of D5 (`docs/specs/stage-4-model-selection-and-routing-quality.md`).
Suggested branch: `feat/prompt-cache-warm-up`.

## Goal

The first routed request after the stack starts should be about as fast as later ones
(~1.5 s instead of ~4.2–5.6 s on the dev machine).

The `ollama-pull` one-shot already loads the model weights into RAM (D5), but not the KV
cache of the prompt. About 95% of every prompt is identical: the system prompt and the
`send_email` tool schema (~440 tokens in total). The first request evaluates all of it
uncached; later requests reuse Ollama's prompt prefix cache and only evaluate the ~20-token
user message. Measured on the running stack (`alpine/ollama:0.40.15`, CPU, model
`qwen3:4b-instruct-2507-q4_K_M`), calling `build_chat_model(...).bind_tools([send_email])`
with `SystemMessage(SYSTEM_PROMPT)` and a `format_user_message(...)` message:

| Case | Prompt tokens | Prefill (`prompt_eval_duration`) | Total |
|---|---|---|---|
| Forced cache miss (unique first token in the system prompt) | 469 | 3249 ms | 4711 ms |
| Cache hit | 439 | 220 ms | 1741 ms |
| Cache hit | 440 | 183 ms | 1514 ms |

API logs of the same stack: first request `latency_ms=4230`, then 1379 and 1462 (438–449
input tokens). The direct `bind_tools` call above hit the cache entry that the real agent
requests had filled, which shows the two paths produce the same prompt prefix.

The fix: at API startup, make one best-effort model call with the real system prompt,
the bound tool and a fixed dummy message. That fills Ollama's prefix cache before the API
accepts traffic.

## Non-goals

- Re-priming after an Ollama-only restart (`docker compose restart ollama`). The cache is
  lost with the model; the first request then pays the load and the uncached prompt,
  as today (D5 gap 2, stays documented). `docker compose up -d` or restarting `api`
  warms it again.
- An e2e test that asserts first-request latency (timing-based, flaky across machines).
  Verified manually instead (see End-to-end verification).
- Updating the project `README.md`. Only `docs/evals/README.md` changes.
- Inspecting or logging the warm-up's answer beyond the elapsed time (no department,
  no tool-call check).
- Shortening the warm-up's generation (`num_predict=1` or similar): the warm-up uses
  exactly the real request's options, so nothing can differ or trigger a reload.
- Any change to `SYSTEM_PROMPT`, `DEPARTMENT_SCOPES`, the API contract,
  `docker-compose.yml`, `docker/ollama_pull.py` or the eval runner.
- Changing `OLLAMA_NUM_PARALLEL` (D7).

## Design

### Decisions

- **D1. Blocking, bounded.** The lifespan awaits the warm-up before `yield`. uvicorn
  accepts connections only after the lifespan starts up, so `healthy` (and
  `docker compose up -d --wait`) now means "model loaded **and** prompt cached",
  in line with D5's "API starts only when ready". Expected startup cost on the dev machine:
  ~4.5 s (prefill ~3.2 s + generation ~1.3 s).
- **D2. Deadline = `settings.ollama_timeout`** (60 s, the same as a real request). No new
  knob. A worst-case hang delays startup by 60 s. That is still under the API
  healthcheck's unhealthy threshold of ~80 s (30 s `start_period`, during which failures
  don't count, plus 5 retries × 10 s).
- **D3. Best-effort.** Any `Exception` from the warm-up (Ollama unreachable, model
  missing, timeout, a parsing error) is logged as a WARNING with the exception type only,
  and startup continues. Real requests then behave exactly as today (503/502/504 or a
  slow first request). `asyncio.CancelledError` (a `BaseException`) is not caught, so a
  shutdown during startup is not delayed.
- **D4. Same path as a real request.** The warm-up calls the same `ChatOllama` instance
  the agent uses (same `num_predict`, `temperature`, `reasoning`, client timeout), with
  `.bind_tools([send_email])`, `[SystemMessage(SYSTEM_PROMPT),
  HumanMessage(format_user_message(WARMUP_MESSAGE))]`. The tool never runs: this is a
  model call, not an agent run, so no email can be sent. The answer is discarded.
- **D5. Toggle: `ollama_warmup: bool = True`** in `Settings` (env `OLLAMA_WARMUP`). Unit
  tests that run the lifespan set it to false. Not added to `docker-compose.yml`: the
  default is already true (single source of truth).
- **D6. Logging.** INFO `Prompt cache warmed in <ms> ms` on success; WARNING
  `Prompt cache warm-up failed: error=<ExceptionType> latency_ms=<ms>` on failure;
  INFO `Prompt cache warm-up disabled` when the setting is off. No message text, no
  Ollama error text (consistent with `route_message`'s logging).

### Files

| File | Change |
|---|---|
| `src/message_router/config.py` | Add `ollama_warmup: bool = True` with a comment (what it does, why it blocks startup). |
| `src/message_router/agent.py` | Add `WARMUP_MESSAGE: Final[str]` (fixed neutral text, e.g. `"Startup warm-up request."`) and `async def warm_up_prompt_cache(model: BaseChatModel, *, timeout: float) -> None`. It builds the messages as in D4, runs `model.bind_tools([send_email]).ainvoke(...)` under `asyncio.timeout(timeout)` and logs per D6. It never raises `Exception`. |
| `src/message_router/main.py` | In `lifespan`, build the chat model once, pass it to `build_agent`, then `if settings.ollama_warmup: await warm_up_prompt_cache(model, timeout=settings.ollama_timeout)`, else log "disabled". Update the "no I/O: the model loads on the first request" comment. |
| `tests/fakes.py` | `ToolCallingFakeModel.bind_tools` records the bound tools (e.g. `bound_tools: list[Sequence[Any]]`), still returns `self`. |
| `tests/test_agent.py` | Unit tests for `warm_up_prompt_cache` (see Test plan). |
| `tests/test_app.py` | `client` fixture sets `OLLAMA_WARMUP=false`. New lifespan tests with a monkeypatched `warm_up_prompt_cache`. |
| `tests/test_messages_api.py` | `app` fixture sets `OLLAMA_WARMUP=false` (its `client` runs the lifespan with the real `ChatOllama`). |
| `.github/workflows/ci.yml` | Comment-only: the "Start the stack" step name/comment no longer says the model is "not exercised". |
| `docs/evals/README.md` | Replace the first "known gap" (warm-up doesn't fill the prompt cache) with the new behavior and the measured before/after first-request latency. Keep the Ollama-only restart gap. |

Interface sketch (not final code):

```python
WARMUP_MESSAGE: Final[str] = "Startup warm-up request."


async def warm_up_prompt_cache(model: BaseChatModel, *, timeout: float) -> None:
    """Fill Ollama's prompt prefix cache with the system prompt and tool schema.

    One model call, the same prefix as a real request; the answer is discarded and the
    tool never runs. Best-effort: logs and returns on any error.
    """
```

## Data & migrations

None. No database. One new environment setting (`OLLAMA_WARMUP`, default true). The
project has no `.env.example` and the stack starts without a `.env` file, so nothing else
changes.

## External calls & cost

- One extra Ollama `/api/chat` call per API process start: ~440 prompt tokens, ~24
  generated tokens, ~4.5 s on the dev machine (local model, no money cost). Holds
  Ollama's single slot for that time. No requests can be waiting yet, because the API
  doesn't accept connections until the warm-up finishes.
- Every later request saves ~3 s of prefill while the cache entry survives (until Ollama
  restarts or the entry is evicted).
- CI: the "Start the stack" step (`docker compose up -d --build --wait --wait-timeout 600
  api`) now exercises the model once. Expect it to take a few seconds to ~20 s longer
  on a GitHub runner, well within 600 s and the healthcheck threshold. Its step comment
  "(model loaded, not exercised)" becomes inaccurate: update it to say the warm-up runs
  one model call.

## Edge cases

1. **Ollama unreachable** (host run without Ollama): `httpx.ConnectError` right away →
   WARNING, the API starts. Same as today for requests (503).
2. **Model not pulled** (host run with the wrong `OLLAMA_MODEL`): `ollama.ResponseError`
   404 → WARNING, the API starts; requests get 502 as today.
3. **Ollama hangs or a very slow machine**: cut off at `ollama_timeout` (60 s) → WARNING,
   the API starts. The cache may stay cold, so the first request is slow but succeeds.
4. **Model not in RAM** (e.g. after an Ollama-only restart followed by an api restart):
   the warm-up also pays the weight load (~2.5 s) and then fills the cache. That is a
   bonus, not a goal.
5. **API restart with a warm Ollama**: the warm-up hits the existing cache entry (~1.5 s).
6. **Shutdown (SIGTERM) during the warm-up**: `CancelledError` propagates and the process
   exits promptly.
7. **Connection refused for a few seconds after the container starts**: already true
   today while uvicorn starts. It is now ~4.5 s longer. `--wait` and the healthcheck
   already cover this.
8. **Prefix drift**: if a future change makes the agent send a different prompt prefix
   (e.g. extra system text from middleware, a different tool list), the warm-up silently
   stops helping without breaking anything. The parity unit test (Test plan, 3) guards
   this.
9. **Multiple uvicorn workers** (not used today): each worker warms once; later ones hit
   the cache. Harmless.
10. **Privacy**: the warm-up sends only fixed text, never user data.

## Test plan

All unit tests (`@pytest.mark.unit`), offline, using `tests/fakes.py`. No e2e test
changes; there are no `llm` tests in this project.

`tests/test_agent.py`:

1. **Sends the right messages and tool**: with `fake_model(tool_call("other@example.com"))`,
   `warm_up_prompt_cache` makes exactly one call. `received[0]` is
   `[SystemMessage(SYSTEM_PROMPT), HumanMessage(format_user_message(WARMUP_MESSAGE))]`,
   and the bound tools are `[send_email]`.
2. **The tool never runs**: with `fake_send` and a scripted tool call in the answer,
   `fake_send.calls` stays empty.
3. **Prefix parity with the agent**: run `route_message` on a `build_agent(fake)`
   (with `fake_send`) and `warm_up_prompt_cache` on a second fake. Assert the same
   `SystemMessage` content in first position and the same bound tool names. This pins D4
   and catches edge case 8.
4. **Best-effort on errors**: `raising_model(httpx.ConnectError(...))`,
   `raising_model(ollama.ResponseError("not found", 404))` and
   `raising_model(RuntimeError())` → no exception, one WARNING record with the
   exception type (`caplog`), no message text in the log.
5. **Timeout**: `raising_model(delay=1.0)` with `timeout=0.05` → returns within ~0.5 s,
   WARNING with `TimeoutError`.
6. **Success log**: INFO record containing `Prompt cache warmed in`.

`tests/test_app.py`:

7. Existing `client` fixture sets `OLLAMA_WARMUP=false`, so existing tests make no
   network calls. Optionally assert the "disabled" INFO record. The same applies to the
   `app` fixture in `tests/test_messages_api.py`, the only other place that runs the
   lifespan.
8. **Lifespan calls the warm-up when enabled**: monkeypatch
   `message_router.main.warm_up_prompt_cache` with a recording async fake. With
   `OLLAMA_WARMUP` unset (default true) and `OLLAMA_TIMEOUT=12`, entering `TestClient`
   calls it once with `timeout=12.0` and the same model instance the agent was built with.
9. **Lifespan skips it when disabled**: `OLLAMA_WARMUP=false` → the fake is not called.
10. Settings: `Settings(ollama_model="m").ollama_warmup is True`, and
    `OLLAMA_WARMUP=false` parses to `False`.

### Done criteria (run and show output)

```bash
uv run ruff format . && uv run ruff check --fix .
uv run pyright
uv run pytest -m "not e2e" -x -q
uv run pre-commit run --all-files
```

## End-to-end verification steps

```bash
# 1. Rebuild and start; --wait returns only after the warm-up (healthy = warm)
docker compose up -d --build --wait api
docker compose logs api | grep -E "Prompt cache"
#    expect: "Prompt cache warmed in ~4500 ms" (dev machine)

# 2. First real request: fast, routed, email in MailHog
curl -s -X POST http://localhost:8000/api/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"email": "jan.nowak@example.com", "message": "I forgot my password and cannot log in"}'
docker compose logs api | grep "Routed message" | tail -1
#    expect: department=help-desk@example.com fallback=False latency_ms ~1400-1900
#    (before: 4230). Email visible at http://localhost:8025 with Reply-To set.

# 3. No warm-up email was sent
#    expect: MailHog shows only the email from step 2 (none to other@ at startup).

# 4. api-only restart: warm-up hits the existing cache
docker compose restart api && docker compose logs api | grep "Prompt cache" | tail -1
#    expect: ~1500 ms

# 5. Failure path, host run without Ollama: the API still starts
OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M OLLAMA_URL=http://localhost:1 \
  uv run uvicorn message_router.main:app
#    expect: WARNING "Prompt cache warm-up failed: error=ConnectError", then
#    "Application startup complete"; Ctrl+C to stop.

# 6. Disabled
OLLAMA_WARMUP=false OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M \
  uv run uvicorn message_router.main:app
#    expect: INFO "Prompt cache warm-up disabled".

# 7. Regression: e2e suite (full stack)
uv run pytest -m e2e -q
```

## Open questions

None blocking. To confirm during implementation:

1. That `create_agent`'s model node binds the tools with no extra kwargs that change the
   rendered prompt (the probe's cache hit says so for the current langchain version;
   check with `uv pip show langchain langchain-ollama`). Test 3 pins the parts that are
   visible offline.
2. The exact `WARMUP_MESSAGE` wording is free: only the shared prefix before it matters.
