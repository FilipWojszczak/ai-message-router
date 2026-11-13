"""Smoke tests against the running stack (`docker compose up -d --wait`)."""

import httpx2
import pytest

pytestmark = pytest.mark.e2e

API_URL = "http://localhost:8000"
MAILHOG_URL = "http://localhost:8025"


def test_swagger_served() -> None:
    response = httpx2.get(f"{API_URL}/api/v1/docs")

    assert response.status_code == 200


def test_mailhog_up() -> None:
    response = httpx2.get(f"{MAILHOG_URL}/api/v2/messages")

    assert response.status_code == 200
