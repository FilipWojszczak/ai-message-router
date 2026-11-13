from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from message_router.config import get_settings
from message_router.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
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
