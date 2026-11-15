from email import message_from_bytes, policy
from email.message import EmailMessage

import aiosmtplib
import pytest

from message_router.config import Settings
from message_router.departments import Department
from message_router.mailer import (
    EmailDeliveryError,
    build_message,
    send_department_email,
)
from tests.fakes import FakeSend

pytestmark = pytest.mark.unit

REQUESTER = "jan.nowak@example.com"
SENDER = "message-router@example.com"
MESSAGE = "Nie działa mi komputer"
TRICKY_MESSAGE = (
    "Zażółć gęślą jaźń\n"
    ".\n"
    "Bcc: x@y.com\n"
    "  leading spaces\n"
    "trailing spaces   \n"
    "ostatnia linia"
)


def _normalize_body(text: str) -> str:
    """CRLF on the wire and the final newline added by set_content don't count."""
    return text.replace("\r\n", "\n").removesuffix("\n")


def _reparse(msg: EmailMessage) -> EmailMessage:
    parsed = message_from_bytes(msg.as_bytes(), policy=policy.default)
    assert isinstance(parsed, EmailMessage)
    return parsed


def test_departments_are_exactly_the_allowed_addresses() -> None:
    assert {d.value for d in Department} == {
        "human-resources@example.com",
        "help-desk@example.com",
        "it@example.com",
        "kadry@example.com",
        "other@example.com",
    }
    with pytest.raises(ValueError):
        Department("ceo@example.com")


def test_build_message_headers() -> None:
    msg = build_message(Department.IT, REQUESTER, MESSAGE, sender=SENDER)

    assert msg["To"] == "it@example.com"
    assert msg["From"] == SENDER
    assert msg["Reply-To"] == REQUESTER
    assert msg["Subject"] == f"[message-router] Message from {REQUESTER}"
    assert msg["Date"]
    assert msg["Message-ID"].endswith("@example.com>")
    assert "Cc" not in msg
    assert "Bcc" not in msg
    assert msg["Content-Type"] == 'text/plain; charset="utf-8"'


def test_body_round_trips_unchanged() -> None:
    msg = build_message(Department.IT, REQUESTER, TRICKY_MESSAGE, sender=SENDER)

    content = _reparse(msg).get_content()

    assert _normalize_body(content) == _normalize_body(TRICKY_MESSAGE)


def test_header_injection_in_body_stays_in_body() -> None:
    injected = "Hello\r\n\r\nBcc: x@y.com\r\nTo: evil@x.com"

    parsed = _reparse(build_message(Department.IT, REQUESTER, injected, sender=SENDER))

    assert "Bcc" not in parsed
    assert parsed["To"] == "it@example.com"
    assert "Bcc: x@y.com" in parsed.get_content()


@pytest.mark.parametrize(
    "requester",
    [
        "jan@example.com\r\nBcc: x@y.com",
        "jan@example.com\nBcc: x@y.com",
        "a@b.com, evil@x.com",
        "Jan <jan@example.com>",
        "żółw@example.com",
        "jan.example.com",
        "jan@localhost",
        "",
        # RFC 2047 encoded words that EmailMessage would decode into other addresses
        "=?utf-8?q?attacker=40evil.com=2C_b?=@example.com",
        "=?utf-8?b?SGFja2Vk?=@example.com",
    ],
)
async def test_invalid_requester_rejected_before_io(
    requester: str, fake_send: FakeSend, settings: Settings
) -> None:
    with pytest.raises(ValueError):
        await send_department_email(
            Department.IT, requester, MESSAGE, settings=settings
        )

    assert fake_send.calls == []


async def test_invalid_department_rejected_before_io(
    fake_send: FakeSend, settings: Settings
) -> None:
    with pytest.raises(ValueError):
        await send_department_email(
            "ceo@example.com",  # pyright: ignore[reportArgumentType]  # runtime check
            REQUESTER,
            MESSAGE,
            settings=settings,
        )

    assert fake_send.calls == []


@pytest.mark.parametrize("message", ["", "   ", "\n\t \r\n"])
async def test_empty_message_rejected_before_io(
    message: str, fake_send: FakeSend, settings: Settings
) -> None:
    with pytest.raises(ValueError):
        await send_department_email(
            Department.IT, REQUESTER, message, settings=settings
        )

    assert fake_send.calls == []


@pytest.mark.parametrize(
    ("requester", "expected"),
    [
        ("Jan.Nowak@EXAMPLE.com", "Jan.Nowak@example.com"),
        ("jan@bücher.de", "jan@xn--bcher-kva.de"),
    ],
)
def test_requester_is_normalized(requester: str, expected: str) -> None:
    msg = build_message(Department.IT, requester, MESSAGE, sender=SENDER)

    assert msg["Reply-To"] == expected
    assert msg["Subject"] == f"[message-router] Message from {expected}"


async def test_send_uses_explicit_envelope_and_settings(fake_send: FakeSend) -> None:
    settings = Settings(
        ollama_model="test-model",
        smtp_host="smtp.test",
        smtp_port=2525,
        smtp_timeout=3.5,
        smtp_from="router@corp.example.com",
    )

    message_id = await send_department_email(
        Department.KADRY, REQUESTER, MESSAGE, settings=settings
    )

    [(msg, kwargs)] = fake_send.calls
    assert kwargs == {
        "sender": "router@corp.example.com",
        "recipients": ["kadry@example.com"],
        "hostname": "smtp.test",
        "port": 2525,
        "timeout": 3.5,
    }
    assert message_id == msg["Message-ID"]
    assert message_id.endswith("@corp.example.com>")


@pytest.mark.parametrize(
    "error",
    [
        aiosmtplib.SMTPConnectError("Error connecting to localhost on port 1025"),
        aiosmtplib.SMTPConnectTimeoutError("Timed out connecting to localhost"),
        aiosmtplib.SMTPServerDisconnected("Unexpected EOF received"),
        aiosmtplib.SMTPRecipientsRefused(
            [aiosmtplib.SMTPRecipientRefused(550, "No such user", "it@example.com")]
        ),
        aiosmtplib.SMTPDataError(554, "Transaction failed"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_smtp_failures_become_delivery_error(
    error: aiosmtplib.SMTPException, fake_send: FakeSend, settings: Settings
) -> None:
    fake_send.error = error

    with pytest.raises(EmailDeliveryError) as exc_info:
        await send_department_email(
            Department.IT, REQUESTER, TRICKY_MESSAGE, settings=settings
        )

    assert exc_info.value.__cause__ is error
    text = str(exc_info.value)
    assert "it@example.com" in text
    assert type(error).__name__ in text
    assert "Zażółć" not in text
    assert REQUESTER not in text


def test_smtp_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_TIMEOUT", "SMTP_FROM"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(ollama_model="test-model")

    assert settings.smtp_host == "localhost"
    assert settings.smtp_port == 1025
    assert settings.smtp_timeout == 10.0
    assert settings.smtp_from == "message-router@example.com"
