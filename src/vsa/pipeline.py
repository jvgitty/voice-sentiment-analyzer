"""Orchestrator: read audio metadata, run feature analyzers, assemble result."""

import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from vsa import __version__ as _vsa_version
from vsa.features.acoustic import AcousticAnalyzer
from vsa.features.emotion import EmotionAnalyzer
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


class Pipeline:
    def __init__(
        self,
        acoustic_analyzer: AcousticAnalyzer | None = None,
        transcriber: "Transcriber | None" = None,
        emotion_analyzer: EmotionAnalyzer | None = None,
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
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
