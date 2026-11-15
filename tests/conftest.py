import aiosmtplib
import pytest

from message_router.config import Settings
from tests.fakes import FakeSend


@pytest.fixture
def fake_send(monkeypatch: pytest.MonkeyPatch) -> FakeSend:
    fake = FakeSend()
    monkeypatch.setattr(aiosmtplib, "send", fake)
    return fake


@pytest.fixture
def settings() -> Settings:
    return Settings(ollama_model="test-model")
