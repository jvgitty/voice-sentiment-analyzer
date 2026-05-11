"""Integration tests for the FastAPI application."""

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from vsa.api import _pipeline, app
from vsa.auth import CallbackSigner
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo


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


class TestProductionAudioAllowlist:
    """The production allowlist embedded in the FastAPI module is the
    single source of truth for which audio formats real callers can
    submit. These tests pin that decision so a refactor that drops a
    format fails loudly."""

    def test_includes_m4a_aac_mp4_webm_for_pixel_voice_notes(self) -> None:
        from vsa.api import _ALLOWED_AUDIO_TYPES

        # Pixel voice notes arrive as audio/mp4 or audio/m4a; webm is
        # the natural format for browser-based recorders we may add
        # later. aac is the codec inside both mp4 and m4a containers
        # and shows up as a top-level type from some uploaders.
        assert "audio/mp4" in _ALLOWED_AUDIO_TYPES
        assert "audio/m4a" in _ALLOWED_AUDIO_TYPES
        assert "audio/aac" in _ALLOWED_AUDIO_TYPES
        assert "audio/webm" in _ALLOWED_AUDIO_TYPES

    def test_still_includes_legacy_formats(self) -> None:
        from vsa.api import _ALLOWED_AUDIO_TYPES

        # Regression guard: the original set must keep working so
        # existing callers (curl tests, the CLI, the wav fixture in
        # this very test suite) keep flowing.
        assert "audio/wav" in _ALLOWED_AUDIO_TYPES
        assert "audio/x-wav" in _ALLOWED_AUDIO_TYPES
        assert "audio/mpeg" in _ALLOWED_AUDIO_TYPES
        assert "audio/mp3" in _ALLOWED_AUDIO_TYPES
        assert "audio/ogg" in _ALLOWED_AUDIO_TYPES
        assert "audio/flac" in _ALLOWED_AUDIO_TYPES


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
            assert payload["schema_version"] == "2.0"
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
            assert decoded["result"]["schema_version"] == "2.0"

    def test_null_transcription_fires_failed_status_callback(
        self, client: TestClient, fixture_wav_bytes: bytes
    ) -> None:
        """When the analyzer pipeline produces a result with no transcript,
        the outbound callback must report status='failed' rather than
        'completed' with an empty result. The supabase callback handler
        relies on this to take a single failure code path on hard
        transcription failures (model crash, language filter rejection,
        non-speech audio, etc.)."""
        audio_url = "https://example.test/audio.wav"
        callback_url = "https://example.test/callback"
        secret = "shared-secret-1234567890abcdef"

        class _NullTranscriptPipeline:
            async def analyze(self, audio_path: Path) -> AnalyzeResult:
                now = datetime.now(timezone.utc)
                return AnalyzeResult(
                    audio=AudioInfo(
                        duration_seconds=1.0,
                        sample_rate=16000,
                        channels=1,
                    ),
                    transcription=None,
                    processing=ProcessingInfo(
                        started_at=now,
                        completed_at=now,
                        errors=["transcription failed: simulated"],
                    ),
                )

        app.dependency_overrides[_pipeline] = lambda: _NullTranscriptPipeline()
        try:
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
                    json={
                        "audio_url": audio_url,
                        "callback_url": callback_url,
                        "callback_secret": secret,
                        "metadata": {"note_id": "abc"},
                        "request_id": "req-fail",
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert response.status_code == 200

                assert callback_route.call_count == 1
                sent = callback_route.calls.last.request
                sent_body = sent.content
                sent_sig = sent.headers.get("x-signature-256")
                assert sent_sig is not None
                assert CallbackSigner.verify(sent_body, secret, sent_sig)

                import json

                decoded = json.loads(sent_body)
                assert decoded["request_id"] == "req-fail"
                assert decoded["status"] == "failed"
                assert decoded["metadata"] == {"note_id": "abc"}
                assert decoded["error"] is not None
        finally:
            app.dependency_overrides.pop(_pipeline, None)
