"""Real SMTP delivery to the compose MailHog.

Needs only MailHog: `docker compose up -d --wait mailhog`.
"""

import asyncio
import socket
from email import message_from_string, policy
from email.message import EmailMessage
from uuid import uuid4

import aiosmtplib
import pytest

from message_router.config import Settings
from message_router.departments import Department
from message_router.mailer import EmailDeliveryError, send_department_email
from tests.e2e.conftest import MailHog, normalize_body

pytestmark = pytest.mark.e2e

REQUESTER = "jan.nowak@example.com"


async def test_email_delivered_to_mailhog(mailhog: MailHog) -> None:
    token = uuid4().hex
    message = (
        f"Zażółć gęślą jaźń\n.\nBcc: x@y.com\n  token {token}  \nNie działa mi komputer"
    )
    # Explicit kwargs override only the model; SMTP_* come from defaults or env.
    settings = Settings(ollama_model="unused")

    message_id = await send_department_email(
        Department.IT, REQUESTER, message, settings=settings
    )
    items = await mailhog.find(message_id)

    assert len(items) == 1
    [item] = items
    # Content.Headers hold raw RFC 2047 text, so parse the raw message instead.
    parsed = message_from_string(item["Raw"]["Data"], policy=policy.default)
    assert isinstance(parsed, EmailMessage)
    assert parsed["To"] == "it@example.com"
    assert parsed["Reply-To"] == REQUESTER
    assert parsed["Subject"] == f"[message-router] Message from {REQUESTER}"
    assert parsed["Message-ID"] == message_id
    assert "Bcc" not in parsed
    assert normalize_body(parsed.get_content()) == normalize_body(message)
    assert item["Raw"]["To"] == ["it@example.com"]
    assert item["Raw"]["From"] == settings.smtp_from


def _closed_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_unreachable_smtp_raises_delivery_error() -> None:
    settings = Settings(
        ollama_model="unused",
        smtp_host="127.0.0.1",
        smtp_port=_closed_loopback_port(),
        smtp_timeout=2.0,
    )

    async with asyncio.timeout(5):  # connection refused must fail fast
        with pytest.raises(EmailDeliveryError) as exc_info:
            await send_department_email(
                Department.IT, REQUESTER, "Nie działa mi komputer", settings=settings
            )

    assert isinstance(exc_info.value.__cause__, aiosmtplib.SMTPConnectError)
