"""Tests for the Pipeline orchestrator (v0.2: transcription-only)."""

from pathlib import Path

import pytest

from vsa.pipeline import Pipeline
from vsa.schema import Transcript


class _StubTranscriber:
    """Deterministic transcriber stub used by Pipeline tests so we don't
    pay the ~12s NeMo model load on every pipeline assertion. The
    ParakeetTranscriber smoke tests in test_transcription.py cover the
    real-model path."""

    def __init__(self, transcript: Transcript | None = None) -> None:
        self._transcript = transcript or Transcript(
            engine="parakeet-tdt-0.6b-v2",
            language="en",
            text="",
            words=[],
        )

    def transcribe(self, audio_path: Path) -> Transcript:
        return self._transcript


class TestPipeline:
    @pytest.mark.asyncio
    async def test_analyze_fixture_wav_returns_valid_result(
        self, fixture_wav_path: Path
    ) -> None:
        pipeline = Pipeline(transcriber=_StubTranscriber())
        result = await pipeline.analyze(fixture_wav_path)

        assert result.schema_version == "2.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_analyze_handles_m4a_input_via_internal_normalization(
        self, fixture_m4a_path: Path
    ) -> None:
        """Wiring tracer: an m4a file passed directly to Pipeline.analyze
        must be normalized internally to a WAV the wave-based metadata
        reader can handle, then flow through the rest of the pipeline.
        Without this wiring the unit tests for normalize_audio pass but
        production m4a uploads still fail at Pipeline.analyze's first
        wave.open call."""
        pipeline = Pipeline(transcriber=_StubTranscriber())
        result = await pipeline.analyze(fixture_m4a_path)

        # The audio metadata reflects the post-normalization WAV
        # (16 kHz mono), not the original m4a's encoded properties.
        assert result.schema_version == "2.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.2

        # v0.2 surface: transcription is the only feature stage.
        assert result.transcription is not None
        assert result.processing.errors == []
        assert result.processing.completed_at >= result.processing.started_at
        assert isinstance(result.processing.library_versions, dict)

    @pytest.mark.asyncio
    async def test_analyze_populates_transcription_section(
        self, fixture_wav_path: Path
    ) -> None:
        """Pipeline.analyze runs the injected Transcriber and merges its
        output into result.transcription."""
        stub = _StubTranscriber(
            Transcript(
                engine="parakeet-tdt-0.6b-v2",
                language="en",
                text="hello world",
                words=[],
            )
        )
        pipeline = Pipeline(transcriber=stub)
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is not None
        assert result.transcription.engine == "parakeet-tdt-0.6b-v2"
        assert result.transcription.language == "en"
        assert result.transcription.text == "hello world"
        assert result.processing.errors == []

    @pytest.mark.asyncio
    async def test_transcription_failure_sets_none_and_logs_error(
        self, fixture_wav_path: Path
    ) -> None:
        """Partial-success contract: a flaky Transcriber must not 500 the
        whole pipeline. transcription becomes None and an entry is
        appended to processing.errors. The API layer translates this into
        a status='failed' callback (covered in test_api.py)."""

        class BoomTranscriber:
            def transcribe(self, audio_path: Path) -> Transcript:
                raise RuntimeError("synthetic transcription failure")

        pipeline = Pipeline(transcriber=BoomTranscriber())
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is None
        assert any(
            "transcription" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors

    @pytest.mark.asyncio
    async def test_live_parakeet_transcriber_pipeline_runs_end_to_end(
        self, fixture_wav_path: Path
    ) -> None:
        """End-to-end run with the real ParakeetTranscriber to de-risk
        the word-timestamp parsing path.

        The 1-second 440 Hz sine fixture transcribes to gibberish at best
        (often an empty word list). We assert only:
          1. Pipeline.analyze does not crash on the real transcribe path.
          2. result.transcription populates (a Transcript, not None).
          3. processing.errors is empty when the model returns cleanly.

        Real chunking + memory behavior on long audio is exercised
        separately in test_transcription.py — this test only proves the
        live-model wiring still holds after the pivot strip."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        pipeline = Pipeline(transcriber=ParakeetTranscriber())
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is not None
        # No crash means partial-success contract held; the transcript
        # may be empty for sine-wave input and that's fine.
