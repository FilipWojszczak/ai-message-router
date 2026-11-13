# ai-message-router

## Project goal

PoC of an AI message router: an HTTP API
receives `email` + free-text `message`, a LangChain agent backed by a local Ollama model
picks the target department and sends the message by calling an email tool (tool calling
is required, not optional). Emails are captured by MailHog.

### Hard constraints

- Containers only: API on an official `python` image, `mailhog/mailhog`, and Ollama on
  `alpine/ollama` (CPU-only build, ~44 MB download) or `ollama/ollama` (8.5 GB locally).
- Departments: human-resources@, help-desk@, it@, kadry@ (all @example.com);
  `other@example.com` is the fallback.
- The sent email must have `Reply-To` set to the requester's email.
- Swagger/OpenAPI at `/api/v1/docs`.

### Acceptance criteria (all must pass on a fresh clone)

1. `docker compose up -d` starts a working API, MailHog and Ollama (model pulled automatically).
2. Swagger available at `/api/v1/docs`.
3. `README.md` with run instructions, architecture decisions and a sample cURL.
4. A request to the endpoint results in a new email visible in the MailHog web UI.
5. The email is addressed to the correct department.
6. The email has the correct `Reply-To` header.

## Environment

- This project does NOT use Dev Containers. This overrides the global rule.
  Python runs on the host via uv (`uv run ...`); the full stack runs via `docker compose`.

## Configuration

- This project has no secrets (local Ollama, MailHog without auth). This overrides the
  global `.env` rule: the stack must start with no `.env` file present.
- `docker compose up -d` on a fresh clone is the main acceptance criterion. Never add
  `env_file:` or a variable without a value to `docker-compose.yml`.
- Each setting has exactly one source of truth:
  - app defaults live in pydantic-settings and target host runs (`localhost`);
  - `docker-compose.yml` sets only container wiring as literal values
    (`OLLAMA_URL: http://ollama:11434`, `SMTP_HOST: mailhog`);
  - `${VAR:-default}` only for values compose itself needs in more than one service
    (e.g. the Ollama model name, shared by the model-pull step and the API).
  Never repeat a pydantic-settings default in `docker-compose.yml`.
- When changing `docker-compose.yml` or settings, verify with `docker compose config`
  and show the output.

## Commands

- Sync dependencies: `uv sync`
- Format & lint: `uv run ruff format . && uv run ruff check --fix .`
- Type check: `uv run pyright`
- Fast tests (offline, no Docker): `uv run pytest -m "not e2e" -x -q`
- Single test: `uv run pytest tests/path/test_file.py::test_name -x -q`
- All pre-commit hooks: `uv run pre-commit run --all-files`

There are no `llm` tests in this project: the only model is local Ollama, used by `e2e` tests.

## Compact instructions

When compacting, keep: the current branch, the list of modified files, the verification
commands used, and any failing tests with their error messages.
