"""Tests for the EmotionAnalyzer (dimensional + categorical)."""

from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Cycle 1: Pydantic schema models for emotion outputs.
# ---------------------------------------------------------------------------


class TestEmotionSchemaModels:
    def test_dimensional_emotion_model_constructs(self) -> None:
        from vsa.schema import DimensionalEmotion

        de = DimensionalEmotion(
            model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            arousal=0.5,
            valence=0.4,
            dominance=0.6,
        )
        assert de.model.endswith("emotion-msp-dim")
        assert 0.0 <= de.arousal <= 1.0
        assert 0.0 <= de.valence <= 1.0
        assert 0.0 <= de.dominance <= 1.0

    def test_categorical_emotion_model_constructs(self) -> None:
        from vsa.schema import CategoricalEmotion

        ce = CategoricalEmotion(
            model="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            label="neutral",
            scores={"neutral": 0.4, "happy": 0.3, "sad": 0.2, "angry": 0.1},
        )
        assert ce.label == "neutral"
        assert sum(ce.scores.values()) == pytest.approx(1.0)

    def test_emotion_result_allows_partial_population(self) -> None:
        from vsa.schema import (
            CategoricalEmotion,
            DimensionalEmotion,
            EmotionResult,
        )

        # Both populated
        full = EmotionResult(
            dimensional=DimensionalEmotion(
                model="m", arousal=0.5, valence=0.5, dominance=0.5
            ),
            categorical=CategoricalEmotion(
                model="m2", label="neutral", scores={"neutral": 1.0}
            ),
        )
        assert full.dimensional is not None
        assert full.categorical is not None

        # Only dimensional populated
        only_dim = EmotionResult(
            dimensional=DimensionalEmotion(
                model="m", arousal=0.5, valence=0.5, dominance=0.5
            ),
            categorical=None,
        )
        assert only_dim.categorical is None

        # Both None (analyzer-level total failure)
        empty = EmotionResult(dimensional=None, categorical=None)
        assert empty.dimensional is None
        assert empty.categorical is None


# ---------------------------------------------------------------------------
# Cycle 2: EmotionAnalyzer construction is lazy — no model load.
# ---------------------------------------------------------------------------


class TestEmotionAnalyzerConstruction:
    def test_class_importable(self) -> None:
        from vsa.features.emotion import EmotionAnalyzer

        assert EmotionAnalyzer is not None

    def test_constructor_does_not_load_models(self) -> None:
        """Lazy-load contract: instantiating EmotionAnalyzer must not pull
        either underlying model into memory. Both wav2vec2 backbones are
        ~1GB each, so we defer loading to the first .analyze() call."""
        from vsa.features.emotion import EmotionAnalyzer

        analyzer = EmotionAnalyzer()
        # Internal model handles must be unset until analyze() is called.
        assert analyzer._dimensional_model is None
        assert analyzer._dimensional_processor is None
        assert analyzer._categorical_classifier is None


# ---------------------------------------------------------------------------
# Cycle 3+: smoke tests against the real models on the sine-wave fixture.
# Models are loaded once for the whole module via a session-scoped fixture.
# ---------------------------------------------------------------------------

_IEMOCAP_LABELS = {"neutral", "happy", "sad", "angry"}


@pytest.fixture(scope="session")
def emotion_analyzer():
    """Session-scoped EmotionAnalyzer. Loading both wav2vec2 backbones is
    several seconds + several GB; we only pay it once across all smoke
    tests in this module."""
    from vsa.features.emotion import EmotionAnalyzer

    return EmotionAnalyzer()


class TestEmotionAnalyzerSmoke:
    """Smoke tests against the real models. The fixture audio is a 1s 440 Hz
    sine — both models will produce garbage outputs, so we assert only on
    output shape, never on content (per Slice 5 brief & PRD user story 35)."""

    def test_analyze_returns_emotion_result(
        self, emotion_analyzer, fixture_wav_path: Path
    ) -> None:
        from vsa.schema import EmotionResult

        result = emotion_analyzer.analyze(fixture_wav_path)
        assert isinstance(result, EmotionResult)

    def test_dimensional_section_shape(
        self, emotion_analyzer, fixture_wav_path: Path
    ) -> None:
        result = emotion_analyzer.analyze(fixture_wav_path)
        if result.dimensional is None:
            pytest.skip("dimensional model failed; covered by failure-isolation tests")
        d = result.dimensional
        assert d.model.endswith("emotion-msp-dim")
        for v in (d.arousal, d.valence, d.dominance):
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_categorical_section_shape(
        self, emotion_analyzer, fixture_wav_path: Path
    ) -> None:
        result = emotion_analyzer.analyze(fixture_wav_path)
        if result.categorical is None:
            pytest.skip("categorical model failed; covered by failure-isolation tests")
        c = result.categorical
        assert "IEMOCAP" in c.model
        assert c.label in _IEMOCAP_LABELS
        # Probability scores: one per class, non-negative, summing to ~1.
        assert set(c.scores.keys()) == _IEMOCAP_LABELS
        for v in c.scores.values():
            assert isinstance(v, float)
            assert v >= 0.0
        assert sum(c.scores.values()) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Cycles 4+5: independent failure isolation between the two models.
# ---------------------------------------------------------------------------


class TestEmotionAnalyzerFailureIsolation:
    """If one underlying emotion model crashes, the other section must
    still populate. The analyzer-level partial-success contract: per the
    Slice 5 brief and PRD acceptance criteria, an EmotionAnalyzer must
    handle each model's failure independently."""

    def test_dimensional_failure_leaves_categorical_populated(
        self,
        emotion_analyzer,
        fixture_wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force the dimensional inference path to raise. Categorical
        should still produce a CategoricalEmotion section."""
        from vsa.features.emotion import EmotionAnalyzer

        def boom(self: EmotionAnalyzer, audio_path: Path) -> Any:
            raise RuntimeError("synthetic dimensional failure")

        monkeypatch.setattr(EmotionAnalyzer, "_run_dimensional", boom)

        result = emotion_analyzer.analyze(fixture_wav_path)
        assert result.dimensional is None
        assert result.categorical is not None
        assert result.categorical.label in _IEMOCAP_LABELS

    def test_categorical_failure_leaves_dimensional_populated(
        self,
        emotion_analyzer,
        fixture_wav_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mirror of the previous test: categorical raises, dimensional
        section is still produced."""
        from vsa.features.emotion import EmotionAnalyzer

        def boom(self: EmotionAnalyzer, audio_path: Path) -> Any:
            raise RuntimeError("synthetic categorical failure")

        monkeypatch.setattr(EmotionAnalyzer, "_run_categorical", boom)

        result = emotion_analyzer.analyze(fixture_wav_path)
        assert result.categorical is None
        assert result.dimensional is not None
        for v in (
            result.dimensional.arousal,
            result.dimensional.valence,
            result.dimensional.dominance,
        ):
            assert 0.0 <= v <= 1.0
