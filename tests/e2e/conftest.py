import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest

MAILHOG_URL = "http://localhost:8025"


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


def normalize_body(text: str) -> str:
    """CRLF on the wire and the final newline added by set_content don't count."""
    return text.replace("\r\n", "\n").removesuffix("\n")
