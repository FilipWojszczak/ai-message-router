import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel

from message_router import main
from message_router.agent import RouterAgent, build_agent
from message_router.config import Settings, get_settings
from message_router.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    # No model call at startup: these tests must stay offline.
    monkeypatch.setenv("OLLAMA_WARMUP", "false")
    get_settings.cache_clear()
    # The context manager runs the lifespan, which validates the settings.
    with TestClient(create_app()) as client:
        yield client
    get_settings.cache_clear()


@pytest.mark.unit
def test_swagger_served(client: TestClient) -> None:
    response = client.get("/api/v1/docs")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


# --- prompt-cache warm-up in the lifespan ---------------------------------------------


class _Recorder:
    """Records the models passed to build_agent and the warm-up calls."""

    def __init__(self) -> None:
        self.agent_models: list[BaseChatModel] = []
        self.warmups: list[tuple[BaseChatModel, dict[str, Any]]] = []

    def build_agent(self, model: BaseChatModel) -> RouterAgent:
        self.agent_models.append(model)
        return build_agent(model)

    async def warm_up(self, model: BaseChatModel, **kwargs: Any) -> None:
        self.warmups.append((model, kwargs))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Recorder]:
    recorder = _Recorder()
    monkeypatch.setattr(main, "build_agent", recorder.build_agent)
    monkeypatch.setattr(main, "warm_up_prompt_cache", recorder.warm_up)
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.delenv("OLLAMA_WARMUP", raising=False)
    get_settings.cache_clear()
    yield recorder
    get_settings.cache_clear()


@pytest.mark.unit
def test_lifespan_warms_up_the_agents_model(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_TIMEOUT", "12")

    with TestClient(create_app()):
        pass

    [agent_model] = recorder.agent_models
    [(warmup_model, kwargs)] = recorder.warmups
    assert warmup_model is agent_model
    assert kwargs == {"timeout": 12.0}


@pytest.mark.unit
def test_lifespan_skips_warm_up_when_disabled(
    recorder: _Recorder,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    monkeypatch.setenv("OLLAMA_WARMUP", "false")

    with TestClient(create_app()):
        pass

    assert recorder.warmups == []
    assert "Prompt cache warm-up disabled" in caplog.text


@pytest.mark.unit
def test_warm_up_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_WARMUP", raising=False)
    assert Settings(ollama_model="m").ollama_warmup is True

    monkeypatch.setenv("OLLAMA_WARMUP", "false")
    assert Settings(ollama_model="m").ollama_warmup is False
