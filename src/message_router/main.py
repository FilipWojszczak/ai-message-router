import importlib.metadata
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from message_router.api import health
from message_router.config import get_settings

# uvicorn configures this logger, so messages show up without extra logging setup.
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Validate settings at startup: a missing OLLAMA_MODEL stops the server here,
    # not on the first request.
    settings = get_settings()
    logger.info("Ollama model %s at %s", settings.ollama_model, settings.ollama_url)
    yield


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
