"""Full-stack routing: API -> agent -> Ollama -> tool -> MailHog.

Needs the whole stack (`docker compose up -d --wait`) and the model. Not run in CI.
"""

from email import message_from_string, policy
from email.message import EmailMessage

import httpx2
import pytest

from tests.e2e.conftest import MailHog, normalize_body

pytestmark = pytest.mark.e2e

API_URL = "http://localhost:8000"
REQUESTER = "jan.nowak@example.com"


@pytest.mark.xfail(
    reason="qwen3:0.6b rarely emits a real tool call with this prompt, so the request "
    "falls back to other@; stage 4 picks the model and tunes the prompt",
    strict=False,
)
async def test_password_reset_routed_to_help_desk(mailhog: MailHog) -> None:
    message = "I forgot my password and cannot log in"

    # Above the API's 60 s deadline, so a slow model shows up as 504, not a client
    # timeout.
    async with httpx2.AsyncClient(base_url=API_URL, timeout=90) as client:
        response = await client.post(
            "/api/v1/messages", json={"email": REQUESTER, "message": message}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    # Find the email first, so the fixture deletes it even if routing was wrong.
    items = await mailhog.find(body["message_id"])
    assert body["department"] == "help-desk@example.com"
    assert body["fallback"] is False
    assert len(items) == 1
    [item] = items
    parsed = message_from_string(item["Raw"]["Data"], policy=policy.default)
    assert isinstance(parsed, EmailMessage)
    assert parsed["To"] == "help-desk@example.com"
    assert parsed["Reply-To"] == REQUESTER
    assert parsed["Message-ID"] == body["message_id"]
    assert normalize_body(parsed.get_content()) == normalize_body(message)
    assert item["Raw"]["To"] == ["help-desk@example.com"]
