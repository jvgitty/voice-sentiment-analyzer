"""Orchestrator: read audio metadata, run feature analyzers, assemble result."""

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

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


# composites.yaml lives at the repo root; it's the editable spec for the
# three composite formulas. Resolved relative to this file so the test
# suite (which runs with cwd=tests/...) finds it without env tricks.
_COMPOSITES_YAML_PATH = (
    Path(__file__).resolve().parent.parent.parent / "composites.yaml"
)


class Pipeline:
    def __init__(
        self,
        acoustic_analyzer: AcousticAnalyzer | None = None,
        transcriber: "Transcriber | None" = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
        prosody_analyzer: ProsodyAnalyzer | None = None,
        composite_scorer: CompositeScorer | None = None,
    ) -> None:
        self._acoustic = acoustic_analyzer or AcousticAnalyzer()
        # Default: NeMo-backed Parakeet TDT. Constructed eagerly because
        # ParakeetTranscriber is itself lazy — it does not load the model
        # until the first transcribe() call.
        if transcriber is None:
            from vsa.transcription.parakeet import ParakeetTranscriber

            transcriber = ParakeetTranscriber()
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

        # Composite scoring runs last because it consumes everything
        # above. A whole-scorer crash (config error, registry mismatch)
        # nulls the composite section but does not take down the rest of
        # the pipeline — same partial-success contract as every other
        # analyzer. Per-composite failures (a missing input nulling out
        # one composite while others succeed) are the scorer's own job
        # and surface as scorer.last_errors.
        try:
            composite = self._composite_scorer.score(
                ScoreInputs(
                    acoustic=acoustic,
                    prosody=prosody,
                    emotion=emotion,
                    audio_duration_seconds=duration_seconds,
                )
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
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
