"""Tests for the Pipeline orchestrator (v0.2: transcription + extraction)."""

from pathlib import Path

import pytest

from vsa.extraction.schema import ExtractionResult, Task
from vsa.extraction.types import VoiceNoteType
from vsa.pipeline import Pipeline
from vsa.schema import Transcript, Word


class _StubTranscriber:
    """Deterministic transcriber stub used by Pipeline tests so we don't
    pay the ~12s NeMo model load on every pipeline assertion. The real
    Parakeet path is covered by tests/test_transcription.py."""

    def __init__(self, transcript: Transcript | None = None) -> None:
        self._transcript = transcript or Transcript(
            engine="parakeet-tdt-0.6b-v2",
            language="en",
            text="hello world",
            words=[Word(w="hello", start=0.0, end=0.3, conf=0.9)],
        )

    def transcribe(self, audio_path: Path) -> Transcript:
        return self._transcript


class _StubExtractor:
    """Deterministic LLM-extractor stub. Records the last extract()
    call's args so tests can assert that the per-request voice-note
    type override actually flowed from the API layer through the
    Pipeline into the extractor."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self._result = result or ExtractionResult(
            title="A stub title",
            summary="A stub summary.",
            type="idea",
            tags=["stub-tag"],
        )
        # Recorded args from the most recent call. None until called.
        self.last_transcript: str | None = None
        self.last_voice_note_types: list[VoiceNoteType] | None = None
        self.last_fallback_type: str | None = None

    def extract(
        self,
        transcript: str,
        voice_note_types: list[VoiceNoteType] | None = None,
        fallback_type: str = "other",
        summary_max_words: int = 50,
    ) -> ExtractionResult:
        self.last_transcript = transcript
        self.last_voice_note_types = voice_note_types
        self.last_fallback_type = fallback_type
        return self._result


class TestPipeline:
    @pytest.mark.asyncio
    async def test_analyze_fixture_wav_returns_valid_result(
        self, fixture_wav_path: Path
    ) -> None:
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=_StubExtractor(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.schema_version == "2.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_analyze_populates_transcription_and_extraction(
        self, fixture_wav_path: Path
    ) -> None:
        """Both feature stages run when transcription succeeds with
        non-empty text. ``extraction`` is populated, ``processing.errors``
        is empty."""
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=_StubExtractor(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is not None
        assert result.transcription.text == "hello world"
        assert result.extraction is not None
        assert result.extraction.title == "A stub title"
        assert result.processing.errors == []

    @pytest.mark.asyncio
    async def test_analyze_handles_m4a_input_via_internal_normalization(
        self, fixture_m4a_path: Path
    ) -> None:
        """Wiring tracer: an m4a file passed directly to Pipeline.analyze
        must be normalized internally to a WAV before the metadata read."""
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=_StubExtractor(),
        )
        result = await pipeline.analyze(fixture_m4a_path)

        assert result.schema_version == "2.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.2
        assert result.transcription is not None
        assert result.extraction is not None
        assert result.processing.errors == []

    @pytest.mark.asyncio
    async def test_transcription_failure_skips_extraction(
        self, fixture_wav_path: Path
    ) -> None:
        """When transcription raises, the pipeline must not call the
        extractor at all — there's nothing to extract from. The
        partial-success contract leaves both fields ``None`` and
        records the transcription error."""

        class BoomTranscriber:
            def transcribe(self, audio_path: Path) -> Transcript:
                raise RuntimeError("synthetic transcription failure")

        extractor = _StubExtractor()
        pipeline = Pipeline(
            transcriber=BoomTranscriber(),
            extractor=extractor,
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is None
        assert result.extraction is None
        # Critical: the extractor was never invoked.
        assert extractor.last_transcript is None
        assert any(
            "transcription" in err.lower() for err in result.processing.errors
        ), result.processing.errors

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_extraction(
        self, fixture_wav_path: Path
    ) -> None:
        """A transcript with empty/whitespace-only text is functionally
        a transcription failure for the LLM — skip extraction rather
        than burning a 30s LLM call to extract from nothing. The
        pipeline must not raise; ``extraction`` is just ``None``."""
        silent = Transcript(
            engine="parakeet-tdt-0.6b-v2",
            language="en",
            text="   \n  ",
            words=[],
        )
        extractor = _StubExtractor()
        pipeline = Pipeline(
            transcriber=_StubTranscriber(silent),
            extractor=extractor,
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is not None  # transcript exists
        assert result.extraction is None  # but nothing to extract from
        assert extractor.last_transcript is None  # extractor not called

    @pytest.mark.asyncio
    async def test_extraction_failure_records_error_without_failing_request(
        self, fixture_wav_path: Path
    ) -> None:
        """Partial-success contract: when the LLM fails on a valid
        transcript, the request still ``completes`` — the transcript
        itself is a useful deliverable. ``extraction`` becomes ``None``,
        an entry is appended to ``processing.errors``, and the API
        layer (covered by test_api.py) emits status=completed because
        ``transcription is not None``."""

        class BoomExtractor:
            def extract(
                self,
                transcript: str,
                voice_note_types=None,
                fallback_type: str = "other",
                summary_max_words: int = 50,
            ) -> ExtractionResult:
                raise RuntimeError("synthetic extraction failure")

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=BoomExtractor(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is not None
        assert result.extraction is None
        assert any(
            "extraction" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors

    @pytest.mark.asyncio
    async def test_voice_note_types_override_flows_to_extractor(
        self, fixture_wav_path: Path
    ) -> None:
        """Per-request override of the voice-note type catalog must
        reach the extractor verbatim. This is the integration point for
        clients with their own taxonomies (the v2 multi-tenant story
        builds on this primitive)."""
        custom = [
            VoiceNoteType(
                name="legal-call",
                description="Summary of a call with a legal client.",
            ),
        ]
        extractor = _StubExtractor()
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=extractor,
        )
        await pipeline.analyze(
            fixture_wav_path, voice_note_types=custom
        )

        assert extractor.last_voice_note_types is not None
        assert len(extractor.last_voice_note_types) == 1
        assert extractor.last_voice_note_types[0].name == "legal-call"

    @pytest.mark.asyncio
    async def test_no_voice_note_types_override_means_extractor_uses_defaults(
        self, fixture_wav_path: Path
    ) -> None:
        """When the caller omits ``voice_note_types``, the pipeline
        passes ``None`` to the extractor. The extractor (not the
        pipeline) is responsible for falling back to the default
        catalog — this keeps the default-catalog source-of-truth in
        one place."""
        extractor = _StubExtractor()
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            extractor=extractor,
        )
        await pipeline.analyze(fixture_wav_path)

        assert extractor.last_voice_note_types is None

    @pytest.mark.asyncio
    async def test_transcript_text_is_what_reaches_the_extractor(
        self, fixture_wav_path: Path
    ) -> None:
        """The extractor receives the transcript's ``.text`` field, not
        the word-level list. A future schema change that moves text out
        of that field should fail this test loudly."""
        transcript = Transcript(
            engine="parakeet-tdt-0.6b-v2",
            language="en",
            text="something specific to assert on",
            words=[],
        )
        extractor = _StubExtractor()
        pipeline = Pipeline(
            transcriber=_StubTranscriber(transcript),
            extractor=extractor,
        )
        await pipeline.analyze(fixture_wav_path)

        assert extractor.last_transcript == "something specific to assert on"


class TestPipelineLiveTranscriber:
    """End-to-end run with the real ParakeetTranscriber + a stub
    extractor. De-risks the word-timestamp parsing path after the
    pivot strip — proves the live-model wiring still holds."""

    @pytest.mark.asyncio
    async def test_live_parakeet_pipeline_runs_end_to_end(
        self, fixture_wav_path: Path
    ) -> None:
        from vsa.transcription.parakeet import ParakeetTranscriber

        pipeline = Pipeline(
            transcriber=ParakeetTranscriber(),
            extractor=_StubExtractor(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        # Transcription must populate. The text may be empty (the
        # 1-second sine fixture transcribes to gibberish at best);
        # whether extraction runs depends on whether the model
        # produced non-empty text. Both outcomes are valid here.
        assert result.transcription is not None
