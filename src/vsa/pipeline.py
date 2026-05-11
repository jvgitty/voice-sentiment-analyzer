"""Orchestrator: normalize audio, transcribe, assemble result.

v0.2 scope (post-pivot): transcription-only. The pipeline previously also
ran acoustic, emotion, prosody, windowed, and composite-scoring stages;
those are archived under the v0.1.1-archived-sentiment tag. The LLM
extraction stage that replaces them lands in a follow-up PR — keeping
this module's surface stable now so wiring it in becomes a small
diff later.
"""

import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from vsa import __version__ as _vsa_version
from vsa.preprocess import normalize_audio
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo
from vsa.transcription import make_transcriber

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


class Pipeline:
    def __init__(self, transcriber: "Transcriber | None" = None) -> None:
        # Default transcriber is whichever engine ``TRANSCRIBER_ENGINE``
        # selects (Parakeet by default, faster-whisper when the env var
        # is "whisper"). Both implementations are lazy — model load only
        # happens on the first transcribe() call — so constructing the
        # Pipeline at FastAPI startup stays cheap.
        if transcriber is None:
            transcriber = make_transcriber()
        self._transcriber: "Transcriber" = transcriber

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

        # Transcription is the only feature stage in v0.2. A failure here
        # must not 500 the request — the partial-success contract from
        # the v0.1.1 pipeline carries forward: ``transcription = None`` +
        # an entry in ``errors`` lets the API layer emit a
        # ``status: "failed"`` callback rather than a half-empty
        # ``"completed"`` one.
        try:
            transcription = self._transcriber.transcribe(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            transcription = None
            errors.append(f"transcription failed: {e}")

        completed_at = datetime.now(timezone.utc)

        return AnalyzeResult(
            audio=AudioInfo(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            ),
            transcription=transcription,
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
