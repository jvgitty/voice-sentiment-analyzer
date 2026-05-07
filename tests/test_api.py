"""Integration tests for the FastAPI application."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from vsa.api import app
from vsa.auth import CallbackSigner


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
        # Valid auth so we exercise body validation specifically, not the auth gate.
        response = client.post(
            "/analyze",
            json={},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 422

    def test_missing_authorization_returns_401(self, client: TestClient) -> None:
        response = client.post("/analyze", json=VALID_BODY)
        assert response.status_code == 401

    def test_full_happy_path_returns_result_and_fires_signed_callback(
        self, client: TestClient, fixture_wav_bytes: bytes
    ) -> None:
        audio_url = "https://example.test/audio.wav"
        callback_url = "https://example.test/callback"
        secret = "shared-secret-1234567890abcdef"
        body = {
            "audio_url": audio_url,
            "callback_url": callback_url,
            "callback_secret": secret,
            "metadata": {"note_id": "abc"},
            "request_id": "req-1",
        }

        with respx.mock(assert_all_called=True) as mocker:
            mocker.get(audio_url).mock(
                return_value=httpx.Response(
                    200,
                    headers={
                        "Content-Type": "audio/wav",
                        "Content-Length": str(len(fixture_wav_bytes)),
                    },
                    content=fixture_wav_bytes,
                )
            )
            callback_route = mocker.post(callback_url).mock(
                return_value=httpx.Response(204)
            )

            response = client.post(
                "/analyze",
                json=body,
                headers={"Authorization": "Bearer test-key"},
            )

            assert response.status_code == 200
            payload = response.json()
            assert payload["schema_version"] == "1.0"
            assert payload["audio"]["sample_rate"] == 16000

            assert callback_route.call_count == 1
            sent = callback_route.calls.last.request
            sent_body = sent.content
            sent_sig = sent.headers.get("x-signature-256")
            assert sent_sig is not None
            assert CallbackSigner.verify(sent_body, secret, sent_sig)

            # Verify the callback body echoes request_id + metadata and contains the result
            import json

            decoded = json.loads(sent_body)
            assert decoded["request_id"] == "req-1"
            assert decoded["status"] == "completed"
            assert decoded["metadata"] == {"note_id": "abc"}
            assert decoded["result"]["schema_version"] == "1.0"
