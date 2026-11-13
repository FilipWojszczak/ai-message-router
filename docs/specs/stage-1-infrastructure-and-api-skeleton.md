# Stage 1: infrastructure and API skeleton

Status: draft, 2026-11-13. Covers acceptance criteria 1 and 2 from `CLAUDE.md`.
Suggested branch: `feat/stage-1-infrastructure`.

Measurements quoted below were taken on the dev machine (x86_64, 16 cores, Docker Engine
29.8.2, Compose v5.6.0) with throwaway containers during the spec investigation.

## Goal

On a fresh clone with no `.env` file, `docker compose up -d`:

1. starts `ollama` (healthy) and pulls the configured model through a one-shot `ollama-pull`
   service;
2. starts `mailhog` (healthy);
3. starts `api` (healthy) only after `ollama` and `mailhog` are healthy and `ollama-pull`
   exited 0;
4. serves Swagger UI at `http://localhost:8000/api/v1/docs`.

## Non-goals

- LLM/agent logic, LangChain dependencies, prompts, tool calling.
- The message endpoint, email sending, SMTP settings. `SMTP_HOST`/`SMTP_PORT` are added to
  Settings and compose together with email sending.
- `README.md` (later stage).
- A readiness endpoint or any dependency check from inside the API (decided: liveness only).
- Model warm-up, `OLLAMA_KEEP_ALIVE`, LLM request timeouts, final model choice: stage 4.
- GPU support. `alpine/ollama` is CPU-only; GPU would mean switching back to
  `ollama/ollama` (open question 7).
- Publishing SMTP 1025 or Ollama 11434 to the host.
- Restart policies, resource limits, digest pinning, production hardening.
- Replacing MailHog. Mailpit would run natively on arm64, but `mailhog/mailhog` is a hard
  constraint.

## Design

### Decisions and evidence

**D1. Model pull: one-shot `ollama-pull` service running Python** (not an entrypoint
wrapper, not the API).

- The `ollama` server does the download into its volume. `ollama-pull` only drives it over
  Ollama's HTTP API with an inline, stdlib-only Python script. It runs on
  `python:3.14-slim`, the API's base image, so it costs no extra download.
- Why Python and not the `ollama` CLI: `alpine/ollama` has no bash, and its BusyBox has no
  `nc`, `wget` or `nslookup` (only `ping` and `timeout`). The bash `/dev/tcp` preflight is
  therefore impossible there, and a `ping`-based check would misfire wherever ICMP is
  blocked.
- Script steps:
  1. `POST /api/show`: 200 means the model is present, so exit 0; 404 means continue.
     Works offline.
  2. Preflight: `socket.create_connection(("registry.ollama.ai", 443), timeout=5)`. On
     failure, print a clear error and exit 1.
  3. `POST /api/pull`, streamed. Print each new status once. Exit 1 on an `error` message
     or if the stream ends without `success`.
- Why the preflight: with no internet, `ollama pull` **hangs** silently. Measured >150 s on
  `ollama/ollama`, both with the model cached and missing, on a disconnected container and
  on an `--internal` network. Not re-measured on `alpine/ollama` (same upstream code); the
  preflight makes it moot.
- Verified on `alpine/ollama:0.40.15`, with the script inlined exactly as in the compose
  sketch below:

  | Case | Result |
  |---|---|
  | Online, model missing | Pulled in 37 s, exit 0 |
  | Online, model cached | "already present, skipping pull", 1 s, exit 0 |
  | Unknown model name | `ERROR: pull failed: pull model manifest: file does not exist`, exit 1 |
  | No internet, model cached | "already present, skipping pull", under 1 s, exit 0 |
  | No internet, model missing | `ERROR: model qwen3:1.7b is missing and registry.ollama.ai is unreachable ([Errno -3] Temporary failure in name resolution).`, 1 s, exit 1 |

- Why step 1 matters: Compose re-runs an exited one-shot container on every `up` (verified).
  The guard makes repeat `up` take about a second and need no network.
- Why not an entrypoint wrapper: `ollama` stays the vanilla image (default `serve`
  entrypoint, correct SIGTERM handling). A failure surfaces as
  `service "ollama-pull" didn't complete successfully: exit 1` instead of an unhealthy
  container after healthcheck retries, and pull logs are separate.
- Why inline instead of a mounted script file: no bind mounts on a fresh clone, so no SELinux
  `:z` labels, no CRLF breakage on Windows checkouts and no Docker Desktop file-sharing
  issues. The cost: the script must not contain `$`, because Compose would interpolate it
  from the host shell (a literal `$` must be written `$$`). The current script has none.

**D2. The API's readiness gate is "model present on disk".**

- `api.depends_on`: `ollama: service_healthy`, `ollama-pull: service_completed_successfully`,
  `mailhog: service_healthy`. Compose waits for these conditions inside `up` itself (also with
  `-d`). So `docker compose up -d` blocks during the first model download and exits non-zero
  if the pull fails.
- Loading the model into RAM is not gated. Cold load of `qwen3:0.6b` measured 0.78 s (on
  `ollama/ollama`; same Ollama release line). Ollama
  unloads it after 5 min idle (`keep_alive` default, seen in `/api/ps`). Stage 4 decides
  warm-up and keep-alive for the final model.
- First start on CPU is dominated by downloads (see "External calls & cost"). Progress is
  visible via `docker compose logs -f ollama-pull`.

**D3. The model-name default lives only in compose.**

```yaml
x-ollama-model: &ollama-model ${OLLAMA_MODEL:-qwen3:0.6b}
```

- Used as `OLLAMA_MODEL: *ollama-model` by `ollama-pull` and `api`. Verified with
  `docker compose config`: both aliases are interpolated, a shell override applies to both, and
  an empty `OLLAMA_MODEL=` falls back to the default (`:-` semantics).
- In Settings, `ollama_model` is **required** (no default), so nothing is duplicated. Host
  runs must set `OLLAMA_MODEL`.
- Placeholder `qwen3:0.6b`: the smallest tool-capable Qwen tag. It is a 523 MB download
  (499 MB on disk). `ollama show` reports `completion`, `tools` and `thinking` (default on).
  The thinking default matters only from stage 4.

**D4. Host ports: only API 8000 and MailHog UI 8025, both bound to `127.0.0.1`.**

- Why `127.0.0.1`: Docker implements published ports with iptables rules that bypass `ufw`.
  MailHog and the API have no auth.
- SMTP 1025 and Ollama 11434 stay on the compose network. The dev machine already runs a
  native Ollama 0.15.4 (systemd) on `:11434`, so publishing the container's port would make
  `up` fail there.
- Consequence for host runs (`uv run uvicorn …`): the `localhost:11434` default reaches the
  **native** Ollama, not the container. There is no SMTP on the host (relevant from the email
  stage).

**D5. Healthchecks use only tools that exist in each image (checked inside the images).**

| Service | Image facts | Healthcheck |
|---|---|---|
| `ollama` | `alpine/ollama`: Wolfi base; BusyBox `sh`; no bash/curl/wget/nc; has `ollama` CLI; image env `OLLAMA_HOST=0.0.0.0` | `["CMD", "ollama", "list"]`: rc=0 about 0.3 s after start |
| `mailhog` | Alpine, BusyBox 1.31.1 `wget` and `nc`; no curl | `["CMD-SHELL", "nc -z localhost 1025 && wget -q --spider http://localhost:8025/api/v2/messages"]`: healthy in ~2 s; checks SMTP (what the API needs) and the UI/API (what you and e2e need) |
| `api` | `python:3.14-slim`; no curl/wget | `["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=2)"]`: any exception means non-zero |
| `ollama-pull` | n/a (one-shot) | none |

- Shared timing via a YAML anchor merged into each healthcheck:
  `interval: 10s, timeout: 5s, retries: 5, start_period: 30s, start_interval: 1s`.
- Why a generous `start_period` costs nothing: during it, probes run every `start_interval`
  (1 s) and the first success marks the service healthy at once. Failures only count after
  the period, which absorbs slow emulated starts on Apple Silicon.
- `start_interval` needs Docker Engine 25 or newer.

**D6. MailHog on Apple Silicon.**

- `mailhog/mailhog:v1.0.1` (same digest as `latest`, built 2020-08-11) is a single
  `linux/amd64` manifest; there is no arm64 variant.
- Compose sets `platform: linux/amd64` explicitly. On arm64 Docker Desktop this runs the
  image under emulation (Rosetta or QEMU) instead of emitting a platform-mismatch warning,
  and it documents the constraint. It is a no-op on amd64.
- Not verified on a Mac (see open questions).
- `alpine/ollama` is multi-arch (amd64, arm64), so Ollama runs natively on Apple Silicon.
  MailHog is the only emulated service.

**D7. Pinned images.**

- `alpine/ollama:0.40.15`: a CPU-only Ollama build, 44 MB compressed on amd64 and 33 MB on
  arm64 (199 MB locally). Built 2026-11-08, same digest as `latest` on 2026-11-13, one
  patch release behind upstream 0.40.16. Entrypoint (`ollama serve`), port and model path
  (`/root/.ollama`) match the official image.
- Why not `ollama/ollama`: 8.52 GB locally, of which 4.65 GB is `/usr/lib/ollama`, almost
  all CUDA libraries (`mlx_cuda_v13` 2.0 GB, `cuda_v12` 1.3 GB, `cuda_v13` 1.1 GB) that a
  CPU-only stack never loads. Ollama publishes no CPU-only tag.
- Trade-offs accepted: `alpine/ollama` comes from a third-party community namespace on
  Docker Hub. It is published neither by Ollama nor by Alpine Linux (whose official image is
  plain `alpine`). It can lag upstream or stop being updated. The pinned version tag avoids
  silent changes (open question 8).
- `mailhog/mailhog:v1.0.1`.
- API base `python:3.14-slim`, as specified.
- uv binary from `ghcr.io/astral-sh/uv:0.13.8`: same as the local uv; multi-arch verified.

**D8. No top-level `name:` in compose.** The project name is the clone's directory name, so
a verification clone in another directory gets its own containers and volume.

**D9. Multi-stage Dockerfile.**

- The runtime image contains only the venv: no uv, no build cache, no source tree.
- Details are below.

### Files

New:

- `Dockerfile`
- `docker-compose.yml`
- `src/message_router/config.py`: `Settings`, `get_settings()`
- `src/message_router/main.py`: `create_app()`, module-level `app`
- `src/message_router/api/__init__.py`, `src/message_router/api/health.py`: health router
- `tests/test_app.py` (unit)
- `tests/e2e/__init__.py`, `tests/e2e/test_stack.py` (e2e)

Changed:

- `pyproject.toml` and `uv.lock` via `uv add fastapi uvicorn pydantic-settings` (runtime).
  Check the resolved versions and current docs at implementation time. `httpx` stays a dev
  dependency (TestClient, e2e).
- `.github/workflows/ci.yml`: replace the TODO (see CI).
- `CLAUDE.md`: Docker commands in "Commands" (see below).

Unchanged:

- `.dockerignore` already excludes `.venv`, `.env*`, `tests/` and `docs/`. It keeps
  `README.md`, which `uv_build` needs because `pyproject.toml` declares
  `readme = "README.md"`.

Implementation order:

1. Dependencies, config and app, plus unit tests.
2. Dockerfile.
3. Compose.
4. CI and CLAUDE.md.
5. e2e tests.
6. End-to-end verification.

### Settings: `src/message_router/config.py`

```python
class Settings(BaseSettings):
    # Env vars only; no env_file.
    model_config = SettingsConfigDict(frozen=True)

    # Host-run default; compose sets the container URL.
    ollama_url: str = "http://localhost:11434"
    # Required; the default lives only in compose (D3).
    ollama_model: str = Field(min_length=1)


@functools.cache
def get_settings() -> Settings: ...
```

- Env names `OLLAMA_URL` and `OLLAMA_MODEL` (case-insensitive, no prefix) match the
  `CLAUDE.md` wiring.
- `ollama_url` is a `str`, not `HttpUrl`, because pydantic's `HttpUrl` appends a trailing `/`
  to bare hosts, which bites when building URLs in stage 4.
- There is deliberately no `.env` support, per the project rule that the stack starts with
  no `.env`.

### App: `src/message_router/main.py` and `api/health.py`

```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Message Router",
        version=importlib.metadata.version("message-router"),
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/api/v1")
    return app


app = create_app()
```

- `lifespan` calls `get_settings()` and logs the model and URL. A missing or empty
  `OLLAMA_MODEL` therefore fails at **startup**: uvicorn exits and the container exits
  non-zero. It does not fail on the first request.
- `create_app()` itself reads no settings, so importing `message_router.main` without env
  vars is safe (tests, tooling).
- `GET /api/v1/health` returns `200 {"status": "ok"}` (`HealthResponse` with
  `status: Literal["ok"]`). It does no I/O and is tagged `health`.
- The version comes from package metadata, keeping `pyproject.toml` the single source of
  truth.

### Dockerfile (multi-stage), indicative

```dockerfile
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Layer 1: third-party deps only; rebuilt only when pyproject.toml/uv.lock change
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
# Layer 2: the project itself, installed as a regular (non-editable) package
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.14-slim
RUN useradd --system --no-log-init --uid 10001 --no-create-home app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "message_router.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

What each choice buys:

- `UV_COMPILE_BYTECODE=1`: `.pyc` files are precompiled, so the container starts faster.
- `UV_LINK_MODE=copy`: the cache mount is a separate filesystem, so hardlinks are
  impossible; this silences the warning.
- `UV_PYTHON_DOWNLOADS=0`: the venv must use the image's `/usr/local/bin/python3.14`, which
  also exists in the runtime stage. The same base tag in both stages is what makes copying
  the venv valid.
- `--locked`: the build fails if `uv.lock` is out of date with `pyproject.toml`. CI
  catches this.
- `--no-editable`: the venv doesn't reference `/app/src`, so the runtime stage needs only
  the venv.
- The venv is root-owned in the runtime stage, so it is read-only for the `app` user.
- `useradd -l` (`--no-log-init`) satisfies hadolint DL3046.
- Exec-form `CMD`: uvicorn is PID 1 and receives SIGTERM from `docker compose down`
  directly, giving a graceful stop instead of a 10 s kill.
- `--host 0.0.0.0`: required inside a container, because the container's loopback is not
  reachable through a published port.
- No `# syntax=` line is needed: the built-in BuildKit frontend supports `RUN --mount`.

### `docker-compose.yml`, indicative

```yaml
x-ollama-url: &ollama-url http://ollama:11434
x-ollama-model: &ollama-model ${OLLAMA_MODEL:-qwen3:0.6b}
x-healthcheck-timing: &healthcheck-timing
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 30s
  start_interval: 1s

services:
  api:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      OLLAMA_URL: *ollama-url
      OLLAMA_MODEL: *ollama-model
    depends_on:
      ollama:
        condition: service_healthy
      ollama-pull:
        condition: service_completed_successfully
      mailhog:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=2)"]
      <<: *healthcheck-timing

  mailhog:
    image: mailhog/mailhog:v1.0.1
    platform: linux/amd64          # amd64-only image; emulated on Apple Silicon (D6)
    ports:
      - "127.0.0.1:8025:8025"      # UI + HTTP API; SMTP 1025 stays internal
    healthcheck:
      test: ["CMD-SHELL", "nc -z localhost 1025 && wget -q --spider http://localhost:8025/api/v2/messages"]
      <<: *healthcheck-timing

  ollama:
    image: alpine/ollama:0.40.15     # CPU-only build (D7)
    volumes:
      - ollama-models:/root/.ollama
    healthcheck:
      test: ["CMD", "ollama", "list"]
      <<: *healthcheck-timing

  ollama-pull:
    image: python:3.14-slim          # same base as the API image: no extra download (D1)
    entrypoint: ["python", "-c"]
    command:
      - |
        import json, os, socket, sys, urllib.error, urllib.request

        base = os.environ["OLLAMA_URL"]
        model = os.environ["OLLAMA_MODEL"]


        def post(path, body, timeout):
            req = urllib.request.Request(
                base + path,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(req, timeout=timeout)


        # 1. Already present? /api/show is local, works offline.
        try:
            post("/api/show", {"model": model}, timeout=10).close()
            print(f"Model {model} already present, skipping pull.")
            sys.exit(0)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

        # 2. Preflight: a pull hangs without internet, so fail fast instead.
        try:
            socket.create_connection(("registry.ollama.ai", 443), timeout=5).close()
        except OSError as e:
            sys.exit(f"ERROR: model {model} is missing and registry.ollama.ai is unreachable ({e}).")

        # 3. Pull via the server; stream progress, print each new status once.
        #    No timeout: large models on slow links legitimately take long.
        print(f"Pulling {model} ...", flush=True)
        last = None
        with post("/api/pull", {"model": model}, timeout=None) as resp:
            for line in resp:
                msg = json.loads(line)
                if "error" in msg:
                    sys.exit(f"ERROR: pull failed: {msg['error']}")
                if msg.get("status") != last:
                    last = msg.get("status")
                    print(last, flush=True)
        if last != "success":
            sys.exit(f"ERROR: pull ended without success (last status: {last}).")
    environment:
      OLLAMA_URL: *ollama-url
      OLLAMA_MODEL: *ollama-model
    depends_on:
      ollama:
        condition: service_healthy
    restart: "no"

volumes:
  ollama-models:
```

How this follows the configuration rules:

- The API gets only literal container wiring (`OLLAMA_URL`) plus the shared model name. Both
  are YAML anchors because `api` and `ollama-pull` both need them.
- No Settings default is repeated in compose.
- There is no `env_file:` and no valueless variable.
- The inline script must stay free of `$`: Compose interpolates `$` in `command` from the
  **host** shell at parse time (a literal `$` would need `$$`).

### CI: `.github/workflows/ci.yml`

Replace the TODO in `build-and-test` with:

```yaml
      - name: Validate docker-compose.yml
        run: docker compose config -q  # runs without .env, like a fresh clone

      - name: Build API image
        run: docker compose build api  # catches Dockerfile breakage and uv.lock drift (--locked)
```

e2e tests stay out of CI: they need the full stack and the downloaded model.

### `CLAUDE.md`: additions to "Commands"

- Start the stack: `docker compose up -d`. The first run builds the API image and pulls the
  model (minutes).
- Start and wait until healthy: `docker compose up -d --wait`
- Rebuild the API after code changes: `docker compose up -d --build api`
- Status, including the one-shot pull: `docker compose ps -a`
- Logs: `docker compose logs -f api`, `docker compose logs ollama-pull`
- Validate compose: `docker compose config -q`
- Stop, keeping the model: `docker compose down`
- Stop and **delete the downloaded model**: `docker compose down -v`
- e2e tests (need the running stack): `uv run pytest -m e2e -q`
- Host run (uses the native Ollama on `localhost:11434`; no SMTP):
  `OLLAMA_MODEL=qwen3:0.6b uv run uvicorn message_router.main:app --reload`

## Data & migrations

- No database and no migrations.
- The only persistent state is the named volume `ollama-models` at `/root/.ollama`
  (~0.5 GB for the placeholder). `docker compose down -v` deletes it.
- MailHog stores messages in memory, so they are lost when the container restarts.
  Acceptable for a PoC.

## External calls & cost

- No LLM inference, no tokens, no paid APIs in this stage.
- One-time downloads on a fresh machine (local sizes per `docker image ls`):
  - `alpine/ollama:0.40.15`: 199 MB (44 MB compressed download on amd64).
  - `mailhog/mailhog:v1.0.1`: 572 MB.
  - `python:3.14-slim`: 181 MB, shared by the API build and `ollama-pull`.
  - The uv image (builder stage only).
  - PyPI wheels during the build.
  - The model `qwen3:0.6b`: 523 MB (pulled in 24 s here).
- Repeat `up`: the model check is local (`/api/show`) with no network; images are cached.
- CI: `docker compose build api` pulls `python:3.14-slim`, the uv image and PyPI deps. The
  expected time is ~30–60 s, not measured. CI pulls no Ollama or MailHog images.

## Edge cases

1. **No internet, model cached.** `ollama-pull` exits 0 via `/api/show`, with no network
   access. The stack starts.
2. **No internet, model missing.** The preflight fails in about a second and prints an
   explicit error; `ollama-pull` exits 1. The API is not started and `up -d` exits
   non-zero. Recovery: restore the network and run `docker compose up -d` again.
3. **Registry reachable but the blob CDN blocked or stalled.** The pull can hang, and
   `up` blocks. There is deliberately no overall timeout, because a large model on a slow
   link legitimately takes long. Diagnose with `docker compose logs -f ollama-pull`; Ctrl+C
   is safe. Known limitation.
4. **Interrupted pull.** Ollama keeps partial blobs and resumes on the next pull. Verified
   on `alpine/ollama:0.40.15`: `ollama pull qwen3:1.7b` killed at 128 MB of 1.4 GB, and the
   second pull started at 132 MB. Tested with the CLI client killed, not the `ollama-pull`
   script, which uses the same server endpoint.
5. **`OLLAMA_MODEL` changed.** The API is recreated (its env changed) and `ollama-pull`
   pulls the new model. The old one stays in the volume until
   `docker compose exec ollama ollama rm <old>`.
6. **`OLLAMA_MODEL=` (empty) on the host.** `:-` falls back to the default (verified).
7. **`OLLAMA_MODEL` exported in the shell** (e.g. for native Ollama work). Compose uses it;
   this is intended, but can surprise. Check with `docker compose config | grep OLLAMA_MODEL`.
8. **`OLLAMA_MODELS` (plural)** is Ollama's models-directory variable. Do not confuse it with
   `OLLAMA_MODEL`.
9. **Host port 8000 or 8025 busy.** `up` fails with "address already in use". Ports are
   literal by design (no extra variables). Port 11434 is not published, so a native Ollama
   on the host never conflicts.
10. **Host run without `OLLAMA_MODEL`.** Startup fails with a pydantic `ValidationError`, by
    design. A host run talks to the native Ollama, which needs the model pulled natively.
11. **`docker compose up -d` returns before the API is healthy** (no `--wait`). A request in
    the first ~1–2 s may get connection refused. Scripts and verification use `--wait`.
    Verified that `--wait` returns 0 with an exited one-shot dependency on Compose v5.6.0.
12. **Ollama crashes after startup.** The API keeps reporting healthy (liveness only).
    Handling LLM call failures belongs to stage 4.
13. **Swagger UI assets are loaded from a CDN** (FastAPI default, jsdelivr). With an offline
    browser, `/api/v1/docs` returns 200 but renders blank. Accepted for the PoC.
14. **Stale API image after code changes.** Use `docker compose up -d --build`.
15. **Apple Silicon.** MailHog runs emulated (D6); `alpine/ollama` runs natively (arm64).
    Docker Desktop's VM memory limit caps the model size. Both matter in stage 4.
16. **Old Docker.** `start_interval` needs Engine 25 or newer; `depends_on` conditions need
    Compose v2. Tested on Engine 29.8.2 and Compose v5.6.0.
17. **Ports bound to `127.0.0.1`** are unreachable from other machines or a VM host.
    Intended.
18. **Unknown or misspelled model name.** `ollama-pull` exits 1 with Ollama's own error
    (`pull model manifest: file does not exist`, verified), so the API is not started.

## Test plan

Tests cover only the hard constraints that stage 1 touches: Swagger at `/api/v1/docs`, and
the containerised stack. Everything else is covered by the verification steps below.

Unit (`uv run pytest -m "not e2e" -x -q`), in `tests/test_app.py`:

- `test_swagger_served`: with `OLLAMA_MODEL` set via `monkeypatch`,
  `TestClient(create_app())` returns 200 `text/html` for `GET /api/v1/docs`.

e2e (`uv run pytest -m e2e -q`; needs `docker compose up -d --wait`; not in CI), in
`tests/e2e/test_stack.py`:

- `test_swagger_served`: `GET http://localhost:8000/api/v1/docs` returns 200 (API container).
- `test_mailhog_up`: `GET http://localhost:8025/api/v2/messages` returns 200 (MailHog
  container).

The Ollama container is not tested from the host: its port isn't published, and on the dev
machine `localhost:11434` is the native Ollama. Model presence is checked in verification
step 5.

`tests/test_smoke.py` is unchanged.

### Done criteria (run and show output)

```bash
uv run ruff format . && uv run ruff check --fix .
uv run pyright
uv run pytest -m "not e2e" -x -q
uv run pre-commit run --all-files     # hadolint (Dockerfile), yamllint/check-yaml (compose, CI)
docker compose config                  # required by CLAUDE.md for compose/settings changes
```

## End-to-end verification steps

Run these after the implementation is committed on the feature branch (a true fresh clone
needs a commit). During implementation, run steps 2–14 in the working tree instead.

The clone directory is named `mr-fresh` so its compose project, containers and volume don't
collide with the dev checkout (D8).

```bash
# 0. Free ports 8000/8025: stop the dev stack (keeps its model volume)
cd ~/Desktop/ai-message-router && docker compose down

# 1. Fresh clone, no .env
FRESH="$(mktemp -d)/mr-fresh"
git clone --branch feat/stage-1-infrastructure ~/Desktop/ai-message-router "$FRESH"
cd "$FRESH" && ls -A && test ! -e .env && echo "OK: no .env"

# 2. Compose valid without .env; show the effective wiring
docker compose config -q && echo "OK: compose config valid"
docker compose config | grep -nE 'OLLAMA_(MODEL|URL|HOST)|host_ip|published|platform'

# 3. Acceptance criterion 1: exactly the acceptance command, cold, timed
time docker compose up -d
#    expect: exit 0; API image built; ollama-pull pulled qwen3:0.6b

# 4. All healthy
docker compose up -d --wait && docker compose ps -a
#    expect: api, mailhog, ollama "Up ... (healthy)"; ollama-pull "Exited (0)"

# 5. Model pulled and tool-capable
docker compose logs ollama-pull | tail -3
docker compose exec ollama ollama list
docker compose exec ollama ollama show qwen3:0.6b | sed -n '/Capabilities/,/^$/p'   # expect "tools"

# 6. Acceptance criterion 2: health and Swagger
curl -s http://localhost:8000/api/v1/health; echo                                    # {"status":"ok"}
curl -s -o /dev/null -w '%{http_code} %{content_type}\n' http://localhost:8000/api/v1/docs   # 200 text/html; charset=utf-8
curl -s http://localhost:8000/api/v1/openapi.json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["info"]["version"], list(d["paths"]))'
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/docs                  # 404

# 7. MailHog UI and API
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8025/                      # 200
curl -s http://localhost:8025/api/v2/messages; echo                                  # {"total":0,...}

# 8. Port exposure: only 8000 and 8025, loopback only
docker compose ps --format '{{.Service}}: {{.Ports}}'
#    expect: api 127.0.0.1:8000->8000/tcp; mailhog 1025/tcp, 127.0.0.1:8025->8025/tcp;
#            ollama 11434/tcp (no host mapping)

# 9. Wiring and image hygiene inside the API container
docker compose exec api env | grep '^OLLAMA_'          # OLLAMA_URL=http://ollama:11434, OLLAMA_MODEL=qwen3:0.6b
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://ollama:11434/api/version').read().decode())"
docker compose exec api id                             # uid=10001(app), not root
docker compose exec api sh -c 'command -v uv || echo "no uv in runtime image"'

# 10. e2e tests (from the dev checkout; they only hit localhost ports)
(cd ~/Desktop/ai-message-router && uv run pytest -m e2e -q)

# 11. Idempotent repeat `up`: fast, no download
time docker compose up -d && docker compose logs ollama-pull | tail -1   # "... already present, skipping pull."

# 12. Model survives down/up (volume kept)
docker compose down && time docker compose up -d --wait && docker compose logs ollama-pull | tail -1

# 13. Fail fast on bad settings (empty model name): API exits with a ValidationError
docker compose run --rm --no-deps -e OLLAMA_MODEL= api; echo "exit=$?"   # non-zero

# 14. Offline behaviour (optional; cuts ALL host networking, so restore it afterwards)
nmcli networking off
docker compose up -d --force-recreate ollama-pull && docker compose logs ollama-pull | tail -1   # cached: exit 0
OLLAMA_MODEL=qwen3:1.7b docker compose up -d; echo "exit=$?"     # missing: fails fast, non-zero
docker compose logs ollama-pull | tail -1                          # "ERROR: model ... unreachable."
docker compose ps -a                                               # api not running
nmcli networking on
docker compose up -d --wait                                        # back to the default model

# 15. Clean slate: WARNING, deletes THIS clone's model volume (~0.5 GB re-download)
docker compose down -v && docker volume ls --filter name=mr-fresh   # volume gone
time docker compose up -d --wait && docker compose ps -a            # re-pulls, all healthy

# 16. Cleanup: removes only mr-fresh_* resources, not the dev checkout's volume
docker compose down -v && cd ~ && rm -rf "$(dirname "$FRESH")"
```

On an Apple Silicon Mac, if available, run steps 1–7 plus
`docker compose exec mailhog uname -m` (expect `x86_64`, i.e. emulated). MailHog must reach
healthy.

## Open questions

1. **Apple Silicon:** MailHog under `platform: linux/amd64` emulation is unverified (no Mac
   here). The hard constraint leaves no fallback image, so a failure there would need a
   decision on the constraint itself.
2. **Interrupted pulls:** resolved. Verified on `alpine/ollama:0.40.15` that a killed pull
   leaves partial blobs and the next pull resumes (see edge case 4).
3. **Pull hang with a partially blocked network** (registry reachable, CDN not): revisit
   only if it shows up in practice.
4. **Stage 4:** model warm-up, `OLLAMA_KEEP_ALIVE`, LLM timeouts, and whether to disable
   qwen3's default thinking mode for tool calling.
5. **README (later stage):** document minimum versions (Docker Engine 25 or newer, Compose
   v2), the first-start download (~1.5 GB of images and model, by local sizes), the macOS
   CPU-only/memory caveats, and why `alpine/ollama` replaced `ollama/ollama`.
6. **Swagger UI offline:** self-host the Swagger static assets, or accept the CDN
   dependency (current assumption: accept).
7. **`alpine/ollama` upkeep:** it is a third-party build. Bump the pinned tag deliberately,
   and if it falls far behind upstream or stops updating, revisit D7.
