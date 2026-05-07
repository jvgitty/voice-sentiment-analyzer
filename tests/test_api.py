"""Integration tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from vsa.api import app


VALID_BODY = {
    "audio_url": "https://example.test/audio.wav",
    "callback_url": "https://example.test/callback",
    "callback_secret": "shared-secret-1234567890abcdef",
    "metadata": {"note_id": "abc"},
    "request_id": "req-1",
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("API_KEY", "test-key")
    return TestClient(app)


class TestAnalyzeEndpoint:
    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_missing_authorization_returns_401(self, client: TestClient) -> None:
        response = client.post("/analyze", json=VALID_BODY)
        assert response.status_code == 401
