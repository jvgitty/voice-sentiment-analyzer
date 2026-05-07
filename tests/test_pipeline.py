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

        # Slices 2+3+5 wire up acoustic, transcription, and emotion;
        # remaining sections are still None.
        assert result.transcription is not None
        assert result.acoustic is not None
        assert result.emotion is not None
        assert result.prosody is None
        assert result.composite is None
        assert result.windows is None

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
