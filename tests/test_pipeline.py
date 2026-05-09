"""Tests for the Pipeline orchestrator."""

from pathlib import Path

import pytest

from vsa.pipeline import Pipeline
from vsa.schema import (
    CategoricalEmotion,
    DimensionalEmotion,
    EmotionResult,
    Transcript,
)


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


class _StubEmotionAnalyzer:
    """Deterministic EmotionAnalyzer stub for Pipeline tests so we don't
    load both wav2vec2 backbones on every pipeline assertion. The real-model
    path is covered by the smoke tests in test_emotion.py."""

    def __init__(self, result: EmotionResult | None = None) -> None:
        self._result = result or EmotionResult(
            dimensional=DimensionalEmotion(
                model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                arousal=0.5,
                valence=0.5,
                dominance=0.5,
            ),
            categorical=CategoricalEmotion(
                model="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                label="neutral",
                scores={
                    "neutral": 1.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "angry": 0.0,
                },
            ),
        )

    def analyze(self, audio_path: Path) -> EmotionResult:
        return self._result


class TestPipeline:
    @pytest.mark.asyncio
    async def test_analyze_fixture_wav_returns_valid_result(
        self, fixture_wav_path: Path
    ) -> None:
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.schema_version == "1.0"
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
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_m4a_path)

        # The audio metadata reflects the post-normalization WAV
        # (16 kHz mono), not the original m4a's encoded properties.
        assert result.schema_version == "1.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.2

        # Slices 2+3+4+5+6+7 wire up every section; windows is now a
        # non-empty list of WindowMetrics covering the full audio.
        from vsa.schema import WindowMetrics

        assert result.transcription is not None
        assert result.acoustic is not None
        assert result.emotion is not None
        assert result.prosody is not None
        assert result.composite is not None
        assert isinstance(result.windows, list)
        assert len(result.windows) >= 1
        assert all(isinstance(w, WindowMetrics) for w in result.windows)

        # processing block is populated and errors[] is empty
        assert result.processing.errors == []
        assert result.processing.completed_at >= result.processing.started_at
        assert isinstance(result.processing.library_versions, dict)


    @pytest.mark.asyncio
    async def test_analyze_populates_acoustic_section(
        self, fixture_wav_path: Path
    ) -> None:
        """Pipeline.analyze runs AcousticAnalyzer and merges its output."""
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.acoustic is not None
        assert result.processing.errors == []
        # Sanity: the wired-in AcousticFeatures should reflect the 440 Hz
        # fixture, not stub zeros.
        assert 380.0 <= result.acoustic.pitch.mean_hz <= 500.0


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
        pipeline = Pipeline(
            transcriber=stub,
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
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
        whole pipeline. transcription becomes None, an entry is appended to
        processing.errors, and other sections (acoustic) still populate."""

        class BoomTranscriber:
            def transcribe(self, audio_path: Path) -> Transcript:
                raise RuntimeError("synthetic transcription failure")

        pipeline = Pipeline(
            transcriber=BoomTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is None
        assert any(
            "transcription" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors
        # Other sections continue running.
        assert result.acoustic is not None


    @pytest.mark.asyncio
    async def test_transcription_failure_skips_prosody_and_logs_error(
        self, fixture_wav_path: Path
    ) -> None:
        """When transcription returns None, prosody has no input and must be
        None. A dedicated 'prosody skipped' error is appended so the caller
        can tell prosody-was-skipped apart from transcription-failed."""

        class BoomTranscriber:
            def transcribe(self, audio_path: Path) -> Transcript:
                raise RuntimeError("synthetic transcription failure")

        pipeline = Pipeline(
            transcriber=BoomTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.transcription is None
        assert result.prosody is None
        assert any(
            "prosody" in err.lower() and "transcription" in err.lower()
            for err in result.processing.errors
        ), result.processing.errors
        # Other sections continue running.
        assert result.acoustic is not None
        assert result.emotion is not None


    @pytest.mark.asyncio
    async def test_acoustic_failure_sets_none_and_logs_error(
        self, fixture_wav_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-success contract: a flaky AcousticAnalyzer must not 500
        the whole pipeline. The section becomes None and an entry is appended
        to processing.errors."""
        from vsa.features.acoustic import AcousticAnalyzer

        def boom(self, audio_path: Path) -> None:
            raise RuntimeError("synthetic acoustic failure")

        monkeypatch.setattr(AcousticAnalyzer, "analyze", boom)

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.acoustic is None
        assert len(result.processing.errors) == 1
        assert "acoustic" in result.processing.errors[0].lower()
        assert "synthetic" in result.processing.errors[0]


    @pytest.mark.asyncio
    async def test_analyze_populates_emotion_section(
        self, fixture_wav_path: Path
    ) -> None:
        """Pipeline.analyze runs EmotionAnalyzer and merges its output into
        result.emotion. Stubbed to keep the test fast — the real-model path
        is covered by the EmotionAnalyzer smoke tests in test_emotion.py."""
        stub = _StubEmotionAnalyzer(
            EmotionResult(
                dimensional=DimensionalEmotion(
                    model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                    arousal=0.42,
                    valence=0.58,
                    dominance=0.61,
                ),
                categorical=CategoricalEmotion(
                    model="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    label="happy",
                    scores={
                        "neutral": 0.1,
                        "happy": 0.7,
                        "sad": 0.1,
                        "angry": 0.1,
                    },
                ),
            )
        )
        pipeline = Pipeline(
            transcriber=_StubTranscriber(), emotion_analyzer=stub
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.emotion is not None
        assert result.emotion.dimensional is not None
        assert result.emotion.dimensional.arousal == pytest.approx(0.42)
        assert result.emotion.categorical is not None
        assert result.emotion.categorical.label == "happy"
        assert result.processing.errors == []


    @pytest.mark.asyncio
    async def test_analyze_populates_prosody_section(
        self, fixture_wav_path: Path
    ) -> None:
        """Pipeline.analyze runs ProsodyAnalyzer when transcription succeeds.

        Even with a stub transcript that has no words (the shape Slice 3's
        sine-wave fixture produces today), prosody should populate with
        zero-valued fields rather than being None."""
        from vsa.schema import ProsodyFeatures

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert isinstance(result.prosody, ProsodyFeatures)
        assert result.prosody.speaking_rate_wpm == 0.0
        assert result.prosody.pause_count == 0
        assert result.prosody.filler_rate == 0.0
        assert result.processing.errors == []


    @pytest.mark.asyncio
    async def test_live_parakeet_transcriber_pipeline_populates_prosody(
        self, fixture_wav_path: Path
    ) -> None:
        """End-to-end run with the real ParakeetTranscriber to de-risk
        Slice 3's word-timestamp parsing.

        The 1-second 440 Hz sine fixture transcribes to gibberish at best
        (often an empty word list). We assert only:
          1. Pipeline.analyze does not crash on the real transcribe path.
          2. result.prosody populates (i.e. ProsodyFeatures, not None) —
             the transcript shape produced by Slice 3 is consumable by
             ProsodyAnalyzer without raising.
          3. Numeric fields are sane (non-negative, no NaN).

        If the fixture ever produces *non-empty* words and the prosody
        section gets weird (e.g. negative speaking rate, NaN pauses), that
        signals Slice 3's word-timestamp key-handling needs review."""
        from vsa.transcription.parakeet import ParakeetTranscriber
        from vsa.schema import ProsodyFeatures

        pipeline = Pipeline(
            transcriber=ParakeetTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        # No crash, prosody populated.
        assert isinstance(result.prosody, ProsodyFeatures)
        assert result.transcription is not None

        # Sanity ranges. None of these should ever go negative regardless
        # of what the real model emitted.
        assert result.prosody.speaking_rate_wpm >= 0.0
        assert result.prosody.speaking_rate_sps >= 0.0
        assert result.prosody.pause_count >= 0
        assert result.prosody.pause_total_seconds >= 0.0
        assert result.prosody.pause_mean_seconds >= 0.0
        assert 0.0 <= result.prosody.filler_rate <= 1.0


    @pytest.mark.asyncio
    async def test_composite_scorer_failure_sets_none_and_logs_error(
        self,
        fixture_wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Partial-success contract: when CompositeScorer.score raises (e.g.
        a config bug or a registry mismatch), Pipeline.analyze sets
        result.composite to None and appends an entry to
        processing.errors. Other sections still populate."""
        from vsa.composites import CompositeScorer

        def boom(self: CompositeScorer, inputs, *args, **kwargs) -> None:
            # *args/**kwargs swallow Slice 7's overrides= kwarg without
            # this stub having to know about it.
            raise RuntimeError("synthetic composite failure")

        monkeypatch.setattr(CompositeScorer, "score", boom)

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.composite is None
        assert any(
            "composite" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors
        # Other sections continue running.
        assert result.acoustic is not None
        assert result.transcription is not None
        assert result.emotion is not None
        assert result.prosody is not None


    @pytest.mark.asyncio
    async def test_analyze_populates_composite_section(
        self, fixture_wav_path: Path
    ) -> None:
        """Pipeline.analyze runs CompositeScorer last and merges its
        output. With acoustic+prosody+emotion all populated, every
        composite should score successfully and the section should be a
        CompositeScores."""
        from vsa.composites import CompositeScores

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert isinstance(result.composite, CompositeScores)
        assert result.composite.confidence is not None
        assert result.composite.engagement is not None
        assert result.composite.calmness is not None
        assert 0.0 <= result.composite.confidence <= 1.0
        assert 0.0 <= result.composite.engagement <= 1.0
        assert 0.0 <= result.composite.calmness <= 1.0
        assert result.processing.errors == []


    @pytest.mark.asyncio
    async def test_windowed_failure_sets_windows_none_and_logs_error(
        self, fixture_wav_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-success contract: when WindowedAnalyzer.analyze raises,
        result.windows is None, an entry is appended to processing.errors,
        and every other section (acoustic / transcription / emotion /
        prosody / composite) still populates."""
        from vsa.windowed import WindowedAnalyzer

        def boom(self, **kwargs) -> None:
            raise RuntimeError("synthetic windowed failure")

        monkeypatch.setattr(WindowedAnalyzer, "analyze", boom)

        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)

        assert result.windows is None
        assert any(
            "window" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors
        # Other sections continue running.
        assert result.acoustic is not None
        assert result.transcription is not None
        assert result.emotion is not None
        assert result.prosody is not None
        assert result.composite is not None


    @pytest.mark.asyncio
    async def test_emotion_failure_sets_none_and_logs_error(
        self, fixture_wav_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-success contract: when the entire EmotionAnalyzer.analyze
        raises (not just one inner model — that's handled inside the
        analyzer), Pipeline.analyze sets result.emotion = None and appends
        an entry to processing.errors. Other sections still populate."""
        from vsa.features.emotion import EmotionAnalyzer

        def boom(self: EmotionAnalyzer, audio_path: Path) -> None:
            raise RuntimeError("synthetic emotion failure")

        monkeypatch.setattr(EmotionAnalyzer, "analyze", boom)

        pipeline = Pipeline(transcriber=_StubTranscriber())
        result = await pipeline.analyze(fixture_wav_path)

        assert result.emotion is None
        assert any(
            "emotion" in err.lower() and "synthetic" in err
            for err in result.processing.errors
        ), result.processing.errors
        # Other sections continue running.
        assert result.acoustic is not None
        assert result.transcription is not None
