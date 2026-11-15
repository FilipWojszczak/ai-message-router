from collections.abc import Callable, Iterator

import aiosmtplib
import httpx
import ollama
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from message_router.agent import build_agent
from message_router.api.messages import (
    DELIVERY_FAILED,
    LLM_ERROR,
    LLM_TIMEOUT,
    LLM_UNAVAILABLE,
    get_agent,
)
from message_router.config import get_settings
from message_router.departments import Department
from message_router.main import create_app
from tests.fakes import FakeSend, ToolCallingFakeModel, fake_model, tool_call

pytestmark = pytest.mark.unit

URL = "/api/v1/messages"
REQUESTER = "jan.nowak@example.com"
MESSAGE = "I forgot my password and cannot log in"

type UseModel = Callable[[ToolCallingFakeModel], ToolCallingFakeModel]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    get_settings.cache_clear()
    yield create_app()
    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # The lifespan still builds the real ChatOllama (no I/O); requests use the fake.
    with TestClient(app) as client:
        yield client


@pytest.fixture
def use_model(app: FastAPI) -> UseModel:
    """Serve requests with an agent built on the given fake model."""

    def use(model: ToolCallingFakeModel) -> ToolCallingFakeModel:
        agent = build_agent(model)
        app.dependency_overrides[get_agent] = lambda: agent
        return model

    return use


@pytest.mark.parametrize(
    ("response", "department", "fallback"),
    [
        (tool_call("help-desk@example.com"), "help-desk@example.com", False),
        (AIMessage(content="No idea."), "other@example.com", True),
    ],
    ids=["agent", "fallback"],
)
def test_routes_message(
    response: AIMessage,
    department: str,
    fallback: bool,
    client: TestClient,
    use_model: UseModel,
    fake_send: FakeSend,
) -> None:
    use_model(fake_model(response))

    result = client.post(URL, json={"email": REQUESTER, "message": MESSAGE})

    assert result.status_code == 200
    [(msg, _)] = fake_send.calls
    assert result.json() == {
        "department": department,
        "fallback": fallback,
        "message_id": msg["Message-ID"],
    }
    assert msg["To"] == department
    assert msg["Reply-To"] == REQUESTER


@pytest.mark.parametrize(
    "body",
    [
        {"message": MESSAGE},
        {"email": REQUESTER},
        {"email": REQUESTER, "message": MESSAGE, "mesage": "typo"},
        {"email": "Jan <jan@example.com>", "message": MESSAGE},
        {"email": "jan@localhost", "message": MESSAGE},
        {"email": REQUESTER, "message": ""},
        {"email": REQUESTER, "message": " \n\t "},
        {"email": REQUESTER, "message": "x" * 2001},
    ],
    ids=[
        "missing_email",
        "missing_message",
        "extra_field",
        "display_name",
        "special_use_domain",
        "empty_message",
        "blank_message",
        "too_long",
    ],
)
def test_invalid_input_is_422(
    body: dict[str, str],
    client: TestClient,
    use_model: UseModel,
    fake_send: FakeSend,
) -> None:
    model = use_model(fake_model(tool_call("it@example.com")))

    result = client.post(URL, json=body)

    assert result.status_code == 422
    assert model.received == []
    assert fake_send.calls == []


def test_max_length_message_is_accepted(
    client: TestClient, use_model: UseModel, fake_send: FakeSend
) -> None:
    use_model(fake_model(tool_call("it@example.com")))

    result = client.post(URL, json={"email": REQUESTER, "message": "ż" * 2000})

    assert result.status_code == 200


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (httpx.ConnectError("ollama:11434 refused"), 503, LLM_UNAVAILABLE),
        (httpx.ReadTimeout("read timed out"), 504, LLM_TIMEOUT),
        (ollama.ResponseError("model 'x' not found", 404), 502, LLM_ERROR),
    ],
    ids=["unavailable", "timeout", "model_error"],
)
def test_model_failures(
    error: Exception,
    status: int,
    detail: str,
    client: TestClient,
    use_model: UseModel,
    fake_send: FakeSend,
) -> None:
    use_model(ToolCallingFakeModel(messages=iter([]), error=error))

    result = client.post(URL, json={"email": REQUESTER, "message": MESSAGE})

    assert result.status_code == status
    assert result.json() == {"detail": detail}
    assert fake_send.calls == []


def test_delivery_failure_is_502(
    client: TestClient, use_model: UseModel, fake_send: FakeSend
) -> None:
    fake_send.error = aiosmtplib.SMTPConnectError("mailhog:1025 refused")
    use_model(fake_model(tool_call("it@example.com")))

    result = client.post(URL, json={"email": REQUESTER, "message": MESSAGE})

    assert result.status_code == 502
    assert result.json() == {"detail": DELIVERY_FAILED}
    assert len(fake_send.calls) == 1  # no fallback attempt


def test_openapi_documents_the_endpoint(client: TestClient) -> None:
    spec = client.get("/api/v1/openapi.json").json()

    operation = spec["paths"]["/api/v1/messages"]["post"]
    assert set(operation["responses"]) == {"200", "422", "502", "503", "504"}
    schemas = spec["components"]["schemas"]
    assert schemas["MessageRequest"]["properties"]["message"]["maxLength"] == 2000
    assert schemas["MessageRequest"]["additionalProperties"] is False
    assert set(schemas["Department"]["enum"]) == {d.value for d in Department}
