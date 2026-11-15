"""Real SMTP delivery to the compose MailHog.

Needs only MailHog: `docker compose up -d --wait mailhog`.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from email import message_from_string, policy
from email.message import EmailMessage
from typing import Any
from uuid import uuid4

import aiosmtplib
import httpx2
import pytest

from message_router.config import Settings
from message_router.departments import Department
from message_router.mailer import EmailDeliveryError, send_department_email

pytestmark = pytest.mark.e2e

MAILHOG_URL = "http://localhost:8025"
REQUESTER = "jan.nowak@example.com"


@dataclass
class MailHog:
    """MailHog HTTP API client that remembers which messages the test created."""

    client: httpx2.AsyncClient
    created: list[str] = field(default_factory=list)

    async def find(self, message_id: str) -> list[dict[str, Any]]:
        """Poll for messages whose raw text contains ``message_id`` (up to ~5 s).

        Searches by the Message-ID header, not by a body token: the body is base64
        (MailHog has no 8BITMIME) and MailHog's search doesn't decode it.
        """
        params = {"kind": "containing", "query": message_id.strip("<>")}
        for _ in range(50):
            response = await self.client.get("/api/v2/search", params=params)
            response.raise_for_status()
            if items := response.json()["items"]:
                self.created.extend(item["ID"] for item in items)
                return items
            await asyncio.sleep(0.1)
        return []


@pytest.fixture
async def mailhog() -> AsyncIterator[MailHog]:
    async with httpx2.AsyncClient(base_url=MAILHOG_URL) as client:
        mailhog = MailHog(client)
        yield mailhog
        # Delete only this test's messages, never `DELETE /api/v1/messages`, which
        # would wipe the inbox someone may be looking at.
        for message_id in mailhog.created:
            await client.delete(f"/api/v1/messages/{message_id}")


def _normalize_body(text: str) -> str:
    """CRLF on the wire and the final newline added by set_content don't count."""
    return text.replace("\r\n", "\n").removesuffix("\n")


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
    assert _normalize_body(parsed.get_content()) == _normalize_body(message)
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
