"""Orchestrator: read audio metadata, run feature analyzers, assemble result."""

import os
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from vsa import __version__ as _vsa_version
from vsa.composites import CompositeScorer, ScoreInputs
from vsa.features.acoustic import AcousticAnalyzer
from vsa.features.emotion import EmotionAnalyzer
from vsa.features.prosody import ProsodyAnalyzer
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo
from vsa.transcription import make_transcriber
from vsa.windowed import WindowedAnalyzer

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


# composites.yaml lives at the repo root; it's the editable spec for the
# three composite formulas. Resolved relative to this file so the test
# suite (which runs with cwd=tests/...) finds it without env tricks.
_COMPOSITES_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent / "composites.yaml"
)


def _energy_steadiness_from_windows(windows) -> float | None:
    """Compute ``1 - coefficient_of_variation(loudness_mean_db_per_window)``
    clamped to [0, 1] from a list of ``WindowMetrics``. Returns None when
    fewer than two windows have a non-None ``loudness_mean_db`` (CoV is
    undefined for n<2) so the whole-audio scoring falls back to the
    YAML's 1.0 placeholder behavior.

    Loudness is measured in dB FS (negative values), so we shift to
    absolute values before taking the CoV — otherwise the mean (a
    negative number) flips the sign and produces nonsense.
    """
    if windows is None:
        return None
    loudnesses = [
        w.loudness_mean_db
        for w in windows
        if w.loudness_mean_db is not None
    ]
    if len(loudnesses) < 2:
        return None
    abs_vals = [abs(x) for x in loudnesses]
    mean = sum(abs_vals) / len(abs_vals)
    if mean == 0.0:
        return 1.0
    var = sum((x - mean) ** 2 for x in abs_vals) / len(abs_vals)
    std = var ** 0.5
    cov = std / mean
    return max(0.0, min(1.0, 1.0 - cov))


def _default_window_seconds() -> float:
    """Read WINDOW_SECONDS from env (default 30s). Invalid values fall
    back to the default so a typo can't take down the whole pipeline."""
    raw = os.environ.get("WINDOW_SECONDS")
    if raw is None:
        return 30.0
    try:
        v = float(raw)
        return v if v > 0.0 else 30.0
    except ValueError:
        return 30.0


class Pipeline:
    def __init__(
        self,
        acoustic_analyzer: AcousticAnalyzer | None = None,
        transcriber: "Transcriber | None" = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
        prosody_analyzer: ProsodyAnalyzer | None = None,
        composite_scorer: CompositeScorer | None = None,
        windowed_analyzer: WindowedAnalyzer | None = None,
    ) -> None:
        self._acoustic = acoustic_analyzer or AcousticAnalyzer()
        # Default transcriber is whichever engine ``TRANSCRIBER_ENGINE``
        # selects (Parakeet by default, faster-whisper when the env var
        # is "whisper"). Both implementations are lazy — model load only
        # happens on the first transcribe() call — so constructing the
        # Pipeline at FastAPI startup stays cheap.
        if transcriber is None:
            transcriber = make_transcriber()
        self._transcriber: "Transcriber" = transcriber
        # Default EmotionAnalyzer is also lazy: it only loads the two
        # wav2vec2 backbones on the first .analyze() call. Tests inject a
        # stub via emotion_analyzer= to skip that ~GBs cold-start.
        self._emotion = emotion_analyzer or EmotionAnalyzer()
        # ProsodyAnalyzer is pure-function and trivially cheap; no lazy
        # loading needed. It runs after transcription (because it
        # consumes the transcript) and after acoustic+emotion (so its
        # failure can't take them down).
        self._prosody = prosody_analyzer or ProsodyAnalyzer()
        # CompositeScorer loads composites.yaml (the editable spec) and
        # runs at the very end, after every feature analyzer. The YAML
        # lives at the repo root; tests can inject a stub via
        # composite_scorer= to skip the file load.
        self._composite_scorer = composite_scorer or CompositeScorer.from_yaml(
            _COMPOSITES_YAML_PATH
        )
        # WindowedAnalyzer reads WINDOW_SECONDS at Pipeline construction
        # time so the env var is the single switch (FastAPI handler /
        # CLI / tests all share it). Tests can inject a custom analyzer
        # via windowed_analyzer= when they need explicit window sizes.
        self._windowed = windowed_analyzer or WindowedAnalyzer(
            window_seconds=_default_window_seconds()
        )

    async def analyze(self, audio_path: Path) -> AnalyzeResult:
        started_at = datetime.now(timezone.utc)
        errors: list[str] = []

        with wave.open(str(audio_path), "rb") as f:
            n_frames = f.getnframes()
            sample_rate = f.getframerate()
            channels = f.getnchannels()
            duration_seconds = n_frames / sample_rate

        # Transcription runs first — downstream analyzers (prosody, in a
        # later slice) consume the transcript. A failure here must not
        # short-circuit the rest of the pipeline.
        try:
            transcription = self._transcriber.transcribe(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            transcription = None
            errors.append(f"transcription failed: {e}")

        try:
            acoustic = self._acoustic.analyze(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            acoustic = None
            errors.append(f"acoustic analysis failed: {e}")

        # EmotionAnalyzer.analyze itself contains independent per-model
        # try/except, so a single inner-model crash returns a partially
        # populated EmotionResult. The wrapper here only fires when the
        # whole analyzer raises (e.g. a load-time error before either
        # model runs) — that's the analyzer-level partial-success
        # contract, distinct from the per-model one.
        try:
            emotion = self._emotion.analyze(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            emotion = None
            errors.append(f"emotion analysis failed: {e}")

        # Prosody runs last among the feature analyzers so its failure
        # cannot take down acoustic/emotion. It also depends on the
        # Transcript: when transcription failed earlier we have no input,
        # so prosody is set to None and a dedicated error is logged.
        if transcription is None:
            prosody = None
            errors.append(
                "prosody skipped: transcription unavailable"
            )
        else:
            try:
                prosody = self._prosody.analyze(transcription, duration_seconds)
            except Exception as e:  # noqa: BLE001 -- partial-success contract
                prosody = None
                errors.append(f"prosody analysis failed: {e}")

        # Windows pass runs after every per-audio analyzer so the heavy
        # backbones are loaded at most once and we can reuse the loaded
        # emotion/acoustic analyzers per window. A whole-pass failure
        # nulls result.windows and surfaces an error string but cannot
        # block composite scoring.
        try:
            windows = self._windowed.analyze(
                audio_path=audio_path,
                transcript=transcription,
                composite_scorer=self._composite_scorer,
                emotion_analyzer=self._emotion,
                acoustic_analyzer=self._acoustic,
                prosody=prosody,
            )
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            windows = None
            errors.append(f"windowed analysis failed: {e}")

        # Composite scoring runs last because it consumes everything
        # above. A whole-scorer crash (config error, registry mismatch)
        # nulls the composite section but does not take down the rest of
        # the pipeline — same partial-success contract as every other
        # analyzer. Per-composite failures (a missing input nulling out
        # one composite while others succeed) are the scorer's own job
        # and surface as scorer.last_errors.
        #
        # When a windows pass succeeded with at least two valid loudness
        # values we override the all-1.0 ``energy_steadiness`` placeholder
        # with the real coefficient-of-variation across windows. Single-
        # window audio leaves the placeholder alone (CoV is undefined for
        # n<2). ``pace_steadiness`` remains a placeholder; it requires
        # word-level transcript windowing, out of scope for Slice 7.
        energy_steadiness_override = _energy_steadiness_from_windows(windows)
        try:
            composite = self._composite_scorer.score(
                ScoreInputs(
                    acoustic=acoustic,
                    prosody=prosody,
                    emotion=emotion,
                    audio_duration_seconds=duration_seconds,
                ),
                overrides={"energy_steadiness": energy_steadiness_override}
                if energy_steadiness_override is not None
                else None,
            )
            errors.extend(self._composite_scorer.last_errors)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            composite = None
            errors.append(f"composite scoring failed: {e}")

        completed_at = datetime.now(timezone.utc)

        return AnalyzeResult(
            audio=AudioInfo(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            ),
            transcription=transcription,
            acoustic=acoustic,
            emotion=emotion,
            prosody=prosody,
            composite=composite,
            windows=windows,
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
