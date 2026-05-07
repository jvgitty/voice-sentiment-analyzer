"""Tests for the CompositeScorer (slice 6).

The composites.yaml at the repo root is the source of truth — these tests
read from it where possible rather than re-asserting hardcoded names so
that tuning the YAML doesn't break the tests for the wrong reason.
"""

from pathlib import Path

import pytest

from vsa.composites import CompositeScorer, ScoreInputs
from vsa.schema import (
    AcousticFeatures,
    DimensionalEmotion,
    EmotionResult,
    LoudnessFeatures,
    PitchFeatures,
    ProsodyFeatures,
    SpectralFeatures,
    VoiceQualityFeatures,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSITES_YAML = REPO_ROOT / "composites.yaml"


def _make_acoustic(
    *,
    jitter: float = 0.005,
    shimmer: float = 0.02,
    pitch_std: float = 10.0,
    pitch_range: float = 200.0,
    loudness_std: float = 8.0,
) -> AcousticFeatures:
    return AcousticFeatures(
        pitch=PitchFeatures(
            mean_hz=200.0,
            median_hz=200.0,
            std_hz=pitch_std,
            min_hz=100.0,
            max_hz=100.0 + pitch_range,
            range_hz=pitch_range,
        ),
        loudness=LoudnessFeatures(
            mean_db=-20.0, std_db=loudness_std, rms_mean=0.1
        ),
        voice_quality=VoiceQualityFeatures(
            jitter_local=jitter,
            shimmer_local=shimmer,
            hnr_db=20.0,
            voiced_unvoiced_ratio=0.7,
        ),
        spectral=SpectralFeatures(
            centroid_mean=1500.0,
            rolloff_mean=4000.0,
            bandwidth_mean=2000.0,
            mfcc_means=[0.0] * 13,
        ),
    )


def _make_prosody(
    *,
    speaking_rate_wpm: float = 140.0,
    filler_rate: float = 0.0,
    pause_total_seconds: float = 0.0,
) -> ProsodyFeatures:
    return ProsodyFeatures(
        speaking_rate_wpm=speaking_rate_wpm,
        speaking_rate_sps=speaking_rate_wpm / 60.0 * 1.5,
        pause_count=0,
        pause_total_seconds=pause_total_seconds,
        pause_mean_seconds=0.0,
        filler_rate=filler_rate,
    )


def _make_emotion(
    *, arousal: float = 0.5, valence: float = 0.5, dominance: float = 0.5
) -> EmotionResult:
    return EmotionResult(
        dimensional=DimensionalEmotion(
            model="test",
            arousal=arousal,
            valence=valence,
            dominance=dominance,
        ),
        categorical=None,
    )


class TestCompositesYamlLoad:
    def test_composites_yaml_exists_at_repo_root(self) -> None:
        assert COMPOSITES_YAML.exists(), (
            f"composites.yaml is the spec file for slice 6 and must live at the "
            f"repo root: {COMPOSITES_YAML}"
        )

    def test_from_yaml_loads_without_error(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        assert scorer is not None


class TestCompositesYamlInvariants:
    def test_each_composite_weights_sum_to_one(self) -> None:
        """The shipped composites.yaml is the spec; for the formula to be
        a clean weighted average producing values in [0, 1], each
        composite's weights MUST sum to 1.0 within float tolerance.

        Iterates over composites discovered in the YAML rather than naming
        them, so adding a new composite (or renaming one) only requires a
        YAML edit.
        """
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        for name, components in scorer.weights_per_composite().items():
            total = sum(components.values())
            assert total == pytest.approx(1.0, abs=1e-9), (
                f"composite {name!r} weights sum to {total}, not 1.0"
            )


class TestConfidenceFormula:
    """Hand-computed confidence values against the YAML formula.

    Note on the placeholders: energy_steadiness and pace_steadiness are
    hardcoded to 1.0 in slice 6 (they need windowed CoV data that arrives
    in slice 7). Their weights are 0.20 and 0.10 respectively, so the
    "all minimum" case can never go below 0.20*1.0 + 0.10*1.0 = 0.30.
    The expected values below take that into account.
    """

    def test_all_maximum_inputs_score_one(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.005, shimmer=0.02),
            prosody=_make_prosody(filler_rate=0.0),
            emotion=_make_emotion(dominance=1.0),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.confidence == pytest.approx(1.0, abs=1e-9)

    def test_all_minimum_inputs_only_get_placeholder_weight(self) -> None:
        """jitter at the high end, shimmer high, filler_rate high,
        dominance=0 — every input-driven subscore is 0. The two slice-7
        placeholders (energy_steadiness, pace_steadiness) still contribute
        their hardcoded 1.0 each, so confidence = 0.20 + 0.10 = 0.30."""
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.025, shimmer=0.10),
            prosody=_make_prosody(filler_rate=0.08),
            emotion=_make_emotion(dominance=0.0),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.confidence == pytest.approx(0.30, abs=1e-9)

    def test_midpoint_inputs_score_midpoint_plus_placeholders(self) -> None:
        """Each input-driven subscore at 0.5; placeholders at 1.0.
        Weighted sum: 0.20*0.5 + 0.15*0.5 + 0.20*1.0 + 0.15*0.5 + 0.10*1.0
        + 0.20*0.5 = 0.10 + 0.075 + 0.20 + 0.075 + 0.10 + 0.10 = 0.65."""
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.015, shimmer=0.06),
            prosody=_make_prosody(filler_rate=0.04),
            emotion=_make_emotion(dominance=0.5),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.confidence == pytest.approx(0.65, abs=1e-9)


class TestEngagementFormula:
    """Hand-computed engagement values. No slice-7 placeholders here, so
    the all-min case truly hits 0.0 and all-max truly hits 1.0."""

    def test_all_maximum_inputs_score_one(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(pitch_range=200.0, loudness_std=8.0),
            prosody=_make_prosody(
                speaking_rate_wpm=180.0, pause_total_seconds=0.0
            ),
            emotion=_make_emotion(arousal=1.0),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.engagement == pytest.approx(1.0, abs=1e-9)

    def test_all_minimum_inputs_score_zero(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(pitch_range=50.0, loudness_std=1.0),
            prosody=_make_prosody(
                speaking_rate_wpm=100.0, pause_total_seconds=4.0
            ),
            emotion=_make_emotion(arousal=0.0),
            audio_duration_seconds=10.0,  # 4s/10s = 0.4 ratio = clipped max
        )
        out = scorer.score(inputs)
        assert out.engagement == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_inputs_score_midpoint(self) -> None:
        """range=125 (mid 50..200), loud_std=4.5 (mid 1..8), arousal=0.5,
        wpm=140 (mid 100..180), pause_ratio=0.2 (mid 0..0.4) -> all
        subscores 0.5 -> engagement 0.5."""
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(pitch_range=125.0, loudness_std=4.5),
            prosody=_make_prosody(
                speaking_rate_wpm=140.0, pause_total_seconds=2.0
            ),
            emotion=_make_emotion(arousal=0.5),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.engagement == pytest.approx(0.5, abs=1e-9)


class TestCalmnessFormula:
    """Hand-computed calmness values. No slice-7 placeholders here.

    All-max means: low pitch_std, low jitter, low arousal, high valence,
    low/unhurried wpm. The labels carry the inversions.
    """

    def test_all_maximum_inputs_score_one(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.005, pitch_std=10.0),
            prosody=_make_prosody(speaking_rate_wpm=120.0),
            emotion=_make_emotion(arousal=0.0, valence=1.0),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.calmness == pytest.approx(1.0, abs=1e-9)

    def test_all_minimum_inputs_score_zero(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.025, pitch_std=50.0),
            prosody=_make_prosody(speaking_rate_wpm=200.0),
            emotion=_make_emotion(arousal=1.0, valence=0.0),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.calmness == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_inputs_score_midpoint(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(jitter=0.015, pitch_std=30.0),
            prosody=_make_prosody(speaking_rate_wpm=160.0),
            emotion=_make_emotion(arousal=0.5, valence=0.5),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        assert out.calmness == pytest.approx(0.5, abs=1e-9)


class TestComponentsBreakdown:
    """For any non-degenerate input, the contributions in
    ``_components[composite]`` should sum to the composite score within
    float tolerance. This is the invariant that lets a journaling user
    see WHY a score moved (not just THAT it did)."""

    def test_components_sum_equals_composite_for_each_composite(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        inputs = ScoreInputs(
            acoustic=_make_acoustic(
                jitter=0.012,
                shimmer=0.05,
                pitch_std=22.0,
                pitch_range=140.0,
                loudness_std=5.0,
            ),
            prosody=_make_prosody(
                speaking_rate_wpm=150.0,
                filler_rate=0.03,
                pause_total_seconds=1.5,
            ),
            emotion=_make_emotion(arousal=0.6, valence=0.7, dominance=0.55),
            audio_duration_seconds=10.0,
        )
        out = scorer.score(inputs)
        score_map = {
            "confidence": out.confidence,
            "engagement": out.engagement,
            "calmness": out.calmness,
        }
        for composite_name, composite_score in score_map.items():
            assert composite_score is not None
            contributions = out.components[composite_name]
            # Every contribution must be a number (no None) for this
            # all-inputs-present case, and they should sum to the score.
            total = sum(v for v in contributions.values() if v is not None)
            assert total == pytest.approx(composite_score, abs=1e-9), (
                f"composite {composite_name}: sum of contributions {total} "
                f"!= composite score {composite_score}"
            )
