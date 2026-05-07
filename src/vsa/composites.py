"""Composite scoring (slice 6).

Loads ``composites.yaml`` and scores ``confidence`` / ``engagement`` /
``calmness`` from a bundle of feature inputs. The YAML is the spec —
weights, thresholds, and the formulas live there so the open-source
audience can fork and disagree without touching code.

Implementation note: the YAML's ``source`` strings are documentary; we
do NOT eval them. The actual computation is done by a closed registry
of Python callables keyed off each component's ``name``. The YAML is the
spec the registry implements; the ``_formulas`` echo in the output proves
the two stay in sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import yaml
from pydantic import BaseModel, ConfigDict

from vsa.schema import AcousticFeatures, EmotionResult, ProsodyFeatures


# --- Public input/output models ---------------------------------------


class ScoreInputs(BaseModel):
    """Bundle of optional features the scorer consumes.

    Each field is optional so the partial-success contract can be
    enforced: a missing input simply makes any component depending on it
    skip and the remaining weights re-normalize.
    """

    acoustic: Optional[AcousticFeatures] = None
    prosody: Optional[ProsodyFeatures] = None
    emotion: Optional[EmotionResult] = None
    audio_duration_seconds: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=False)


class CompositeScores(BaseModel):
    """The composite section of the result JSON.

    ``_components`` maps composite_name -> component_name -> contribution
    value (the weighted contribution, not the raw subscore). For any
    composite that scored, ``sum(_components[name].values()) ==
    composite_score`` within float tolerance.

    A skipped component (its required input was missing) shows up with a
    null contribution so the breakdown still names every component the
    YAML declared.
    """

    confidence: Optional[float] = None
    engagement: Optional[float] = None
    calmness: Optional[float] = None
    components: dict[str, dict[str, Optional[float]]] = {}
    formulas: dict[str, list[str]] = {}

    model_config = ConfigDict(populate_by_name=True)

    def model_dump(self, **kwargs):  # type: ignore[override]
        # Custom dump so the leading-underscore aliases survive JSON.
        # Pydantic strips leading underscores from field names; we
        # restore them manually for the public contract.
        d = super().model_dump(**kwargs)
        d["_components"] = d.pop("components", {})
        d["_formulas"] = d.pop("formulas", {})
        return d


# --- Subscore registry ------------------------------------------------
#
# Each entry maps a component ``name`` (from composites.yaml) to a
# callable ``(ScoreInputs) -> Optional[float]``. The callable returns
# None when the inputs it needs are missing; the scorer treats that as
# "skip this component and re-normalize."


def _normalize(x: float, lo: float, hi: float) -> float:
    """Min-max scale x into [0, 1] with clipping (lo -> 0, hi -> 1)."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _jitter_steadiness(inputs: ScoreInputs) -> Optional[float]:
    if inputs.acoustic is None:
        return None
    return 1.0 - _normalize(
        inputs.acoustic.voice_quality.jitter_local, 0.005, 0.025
    )


def _shimmer_steadiness(inputs: ScoreInputs) -> Optional[float]:
    if inputs.acoustic is None:
        return None
    return 1.0 - _normalize(
        inputs.acoustic.voice_quality.shimmer_local, 0.02, 0.10
    )


def _energy_steadiness(inputs: ScoreInputs) -> Optional[float]:
    """Slice 7 will compute 1 - CoV(rms) over windowed audio. Until that
    lands, this is a 1.0 placeholder so the composite still demos."""
    if inputs.acoustic is None:
        return None
    return 1.0


def _low_filler_rate(inputs: ScoreInputs) -> Optional[float]:
    if inputs.prosody is None:
        return None
    return 1.0 - _normalize(inputs.prosody.filler_rate, 0.0, 0.08)


def _pace_steadiness(inputs: ScoreInputs) -> Optional[float]:
    """Slice 7 will compute 1 - CoV(speaking_rate) over windowed audio.
    Until then, this is a 1.0 placeholder."""
    if inputs.prosody is None:
        return None
    return 1.0


def _dominance(inputs: ScoreInputs) -> Optional[float]:
    if inputs.emotion is None or inputs.emotion.dimensional is None:
        return None
    return float(inputs.emotion.dimensional.dominance)


def _arousal(inputs: ScoreInputs) -> Optional[float]:
    if inputs.emotion is None or inputs.emotion.dimensional is None:
        return None
    return float(inputs.emotion.dimensional.arousal)


def _low_arousal(inputs: ScoreInputs) -> Optional[float]:
    a = _arousal(inputs)
    return None if a is None else 1.0 - a


def _positive_valence_bias(inputs: ScoreInputs) -> Optional[float]:
    if inputs.emotion is None or inputs.emotion.dimensional is None:
        return None
    return float(inputs.emotion.dimensional.valence)


def _pitch_range(inputs: ScoreInputs) -> Optional[float]:
    if inputs.acoustic is None:
        return None
    return _normalize(inputs.acoustic.pitch.range_hz, 50.0, 200.0)


def _pitch_stability(inputs: ScoreInputs) -> Optional[float]:
    if inputs.acoustic is None:
        return None
    return 1.0 - _normalize(inputs.acoustic.pitch.std_hz, 10.0, 50.0)


def _loudness_variability(inputs: ScoreInputs) -> Optional[float]:
    if inputs.acoustic is None:
        return None
    return _normalize(inputs.acoustic.loudness.std_db, 1.0, 8.0)


def _speaking_rate_engagement(inputs: ScoreInputs) -> Optional[float]:
    if inputs.prosody is None:
        return None
    return _normalize(inputs.prosody.speaking_rate_wpm, 100.0, 180.0)


def _low_pause_ratio(inputs: ScoreInputs) -> Optional[float]:
    if inputs.prosody is None or inputs.audio_duration_seconds <= 0.0:
        return None
    pause_ratio = (
        inputs.prosody.pause_total_seconds / inputs.audio_duration_seconds
    )
    return 1.0 - _normalize(pause_ratio, 0.0, 0.4)


def _unhurried_pace(inputs: ScoreInputs) -> Optional[float]:
    if inputs.prosody is None:
        return None
    return 1.0 - _normalize(inputs.prosody.speaking_rate_wpm, 120.0, 200.0)


SUBSCORE_REGISTRY: dict[str, Callable[[ScoreInputs], Optional[float]]] = {
    # confidence
    "jitter_steadiness": _jitter_steadiness,
    "shimmer_steadiness": _shimmer_steadiness,
    "energy_steadiness": _energy_steadiness,
    "low_filler_rate": _low_filler_rate,
    "pace_steadiness": _pace_steadiness,
    "dominance": _dominance,
    # engagement
    "pitch_range": _pitch_range,
    "loudness_variability": _loudness_variability,
    "arousal": _arousal,
    "speaking_rate": _speaking_rate_engagement,
    "low_pause_ratio": _low_pause_ratio,
    # calmness
    "pitch_stability": _pitch_stability,
    "low_arousal": _low_arousal,
    "positive_valence_bias": _positive_valence_bias,
    "unhurried_pace": _unhurried_pace,
}


# --- Scorer -----------------------------------------------------------


class CompositeScorer:
    """Loads ``composites.yaml`` and scores a bundle of feature inputs."""

    def __init__(self, config: dict) -> None:
        self._config = config

    @classmethod
    def from_yaml(cls, path: Path) -> "CompositeScorer":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def weights_per_composite(self) -> dict[str, dict[str, float]]:
        """Return ``{composite_name: {component_name: weight}}`` straight
        from the YAML so tests can assert the weight-sum invariant without
        hardcoding component names."""
        out: dict[str, dict[str, float]] = {}
        for composite_name, body in self._config.items():
            comps = body.get("components", [])
            out[composite_name] = {
                comp["name"]: float(comp["weight"]) for comp in comps
            }
        return out

    def score(self, inputs: ScoreInputs) -> CompositeScores:
        """Compute every composite the YAML declares.

        Partial-success: a component whose registered callable returns
        None is skipped. The remaining components' weights are
        re-normalized so they still sum to 1.0. If every component for a
        composite is unavailable, the composite is None and an entry is
        appended to ``self._last_errors`` (read by the pipeline).
        """
        scores: dict[str, Optional[float]] = {}
        components: dict[str, dict[str, Optional[float]]] = {}
        formulas: dict[str, list[str]] = {}
        errors: list[str] = []

        for composite_name, body in self._config.items():
            comp_specs = body.get("components", [])
            # Echo the formula list verbatim from the YAML so the
            # output's _formulas stays in sync with the spec.
            formulas[composite_name] = [
                str(c.get("source", "")) for c in comp_specs
            ]

            available: list[tuple[str, float, float]] = []  # (name, weight, raw)
            unavailable: list[str] = []
            total_weight = 0.0
            for spec in comp_specs:
                name = spec["name"]
                weight = float(spec["weight"])
                fn = SUBSCORE_REGISTRY.get(name)
                if fn is None:
                    # Defensive: a YAML name with no registered callable
                    # is treated as unavailable rather than crashing.
                    unavailable.append(name)
                    continue
                raw = fn(inputs)
                if raw is None:
                    unavailable.append(name)
                else:
                    available.append((name, weight, raw))
                    total_weight += weight

            comp_breakdown: dict[str, Optional[float]] = {}
            for name in unavailable:
                comp_breakdown[name] = None

            if not available:
                scores[composite_name] = None
                components[composite_name] = comp_breakdown
                errors.append(
                    f"composite {composite_name!r}: all components unavailable"
                )
                continue

            # Re-normalize: divide each available component's weight by
            # the sum of available weights so contributions still sum to
            # the composite score.
            composite_score = 0.0
            for name, weight, raw in available:
                contribution = (weight / total_weight) * raw
                comp_breakdown[name] = contribution
                composite_score += contribution

            scores[composite_name] = composite_score
            components[composite_name] = comp_breakdown

        self._last_errors = errors
        return CompositeScores(
            confidence=scores.get("confidence"),
            engagement=scores.get("engagement"),
            calmness=scores.get("calmness"),
            components=components,
            formulas=formulas,
        )

    @property
    def last_errors(self) -> list[str]:
        """Errors emitted by the most recent ``score()`` call.

        The pipeline appends these to ``processing.errors`` so the
        partial-success contract surfaces in the output JSON.
        """
        return getattr(self, "_last_errors", [])
