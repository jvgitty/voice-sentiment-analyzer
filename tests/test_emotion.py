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
