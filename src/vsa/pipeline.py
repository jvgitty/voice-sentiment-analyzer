"""Orchestrator: read audio metadata, run feature analyzers, assemble result."""

import gc
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
from vsa.preprocess import normalize_audio
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo
from vsa.transcription import make_transcriber
from vsa.windowed import WindowedAnalyzer

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


# composites.yaml is the editable spec for the three composite formulas.
# Its location depends on how vsa is installed:
#   - Dev checkout:  /repo/composites.yaml  — sibling of /repo/src/vsa/.
#   - Docker / Fly:  /app/composites.yaml   — copied in by the Dockerfile.
#   - pip install:   tries /app/, falls back to the dev-checkout layout
#                    (which doesn't apply but is tried last for safety).
# An override env var lets advanced operators point at a custom location.
def _resolve_composites_yaml_path() -> Path:
    override = os.environ.get("COMPOSITES_YAML_PATH")
    if override:
        return Path(override)
    candidates = [
        Path("/app/composites.yaml"),  # Docker / Fly convention.
        Path(__file__).resolve().parent.parent.parent / "composites.yaml",  # dev checkout.
        Path.cwd() / "composites.yaml",  # arbitrary cwd fallback.
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to the first candidate; CompositeScorer.from_yaml will
    # raise a clear FileNotFoundError if it really isn't there.
    return candidates[0]


_COMPOSITES_YAML_PATH = _resolve_composites_yaml_path()


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


def _evict_between_phases() -> bool:
    """Whether Pipeline should drop heavy models the moment they stop
    being needed for the rest of the request. Default: enabled.

    Set ``VSA_EVICT_BETWEEN_PHASES=0`` (or "false"/"no") to keep the
    lazy caches resident across the full request — useful when
    profiling or benchmarking many requests in a single warm process,
    where the per-request reload cost dominates the memory savings.
    On Fly.io with auto-suspend, the next request after a cold-start
    reloads from scratch anyway, so eviction is essentially free in
    production.
    """
    raw = os.environ.get("VSA_EVICT_BETWEEN_PHASES", "1")
    return raw.strip().lower() not in ("0", "false", "no", "")


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

        # Normalize whatever format we were handed (m4a, mp4, aac, webm,
        # mp3, ogg, flac, or wav itself) to 16 kHz mono PCM WAV before
        # the wave-based metadata read below. Caller still owns the
        # original ``audio_path``; we own the normalized tmp file and
        # clean it up at the end of this method.
        normalized_path = normalize_audio(audio_path)
        try:
            return await self._analyze_normalized(
                normalized_path, started_at, errors
            )
        finally:
            if normalized_path != audio_path and normalized_path.exists():
                normalized_path.unlink(missing_ok=True)

    async def _analyze_normalized(
        self,
        audio_path: Path,
        started_at: datetime,
        errors: list[str],
    ) -> AnalyzeResult:
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

        # The transcriber model (Parakeet ~2 GB, Whisper variable) is not
        # used by any downstream phase. Drop it now so its weights are
        # collectable before the emotion backbones load — this is the
        # single biggest win against the OOMs observed during the v0.1.1
        # smoke test. Stub transcribers without a release() are skipped.
        if _evict_between_phases():
            release = getattr(self._transcriber, "release", None)
            if callable(release):
                release()
                gc.collect()

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

        # The categorical (SpeechBrain IEMOCAP) backbone is dead weight
        # from this point on: WindowedAnalyzer skips categorical inference
        # per window (see vsa.windowed module docstring), and nothing else
        # downstream consumes it. Drop it so the per-window dimensional
        # forward passes have ~1.3 GB more headroom. Stub analyzers
        # without a release_categorical() are skipped.
        if _evict_between_phases():
            release_cat = getattr(self._emotion, "release_categorical", None)
            if callable(release_cat):
                release_cat()
                gc.collect()

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
