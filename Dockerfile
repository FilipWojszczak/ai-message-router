FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv
# Precompile .pyc (faster start); copy instead of hardlink (cache mount is another FS);
# use the image's Python, which the runtime stage has at the same path.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Layer 1: third-party deps only; rebuilt only when pyproject.toml/uv.lock change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
# Layer 2: the project itself, as a regular (non-editable) package, so the venv
# does not reference /app/src.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.14-slim
RUN useradd --system --no-log-init --uid 10001 --no-create-home app
# Root-owned venv: read-only for the app user. No uv, cache or sources in this stage.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# Numeric uid (the "app" user): resolvable without /etc/passwd (hadolint DL3066).
USER 10001
EXPOSE 8000
# Exec form: uvicorn is PID 1 and gets SIGTERM directly (graceful stop).
CMD ["uvicorn", "message_router.main:app", "--host", "0.0.0.0", "--port", "8000"]
