"""Orchestrator: normalize audio, transcribe, extract, assemble result.

v0.2 pipeline shape:

    audio_file
      → ffmpeg-normalize to 16 kHz mono PCM WAV
      → Parakeet transcription (chunked for long audio)
      → LLM extraction (Qwen3.5-9B-Instruct + JSON-constrained sampling)
      → assemble AnalyzeResult

Partial-success contract:

* Transcription failure ⇒ ``transcription = None``, the API layer emits
  a ``status: "failed"`` callback. No point running extraction on
  nothing.
* Empty-transcript (silent / non-speech audio) ⇒ same: extraction is
  skipped, status: failed.
* Extraction failure on a valid transcript ⇒ ``extraction = None`` and
  an entry in ``processing.errors``, but ``status: "completed"``
  because the transcript itself is a useful deliverable. Downstream
  consumers can branch on ``result.extraction is None``.
"""

import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vsa import __version__ as _vsa_version
from vsa.extraction.llm import LlmExtractor
from vsa.extraction.types import (
    DEFAULT_FALLBACK_TYPE,
    VoiceNoteType,
)
from vsa.preprocess import normalize_audio
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo
from vsa.transcription import make_transcriber

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


class Pipeline:
    def __init__(
        self,
        transcriber: "Transcriber | None" = None,
        extractor: LlmExtractor | None = None,
    ) -> None:
        # Default transcriber is whichever engine ``TRANSCRIBER_ENGINE``
        # selects (Parakeet by default, faster-whisper when the env var
        # is "whisper"). Both implementations are lazy — model load only
        # happens on the first transcribe() call — so constructing the
        # Pipeline at FastAPI startup stays cheap.
        if transcriber is None:
            transcriber = make_transcriber()
        self._transcriber: "Transcriber" = transcriber
        # Same lazy contract for the LLM extractor: construction is
        # cheap, the ~5 GB GGUF doesn't load until the first extract()
        # call. Tests inject a stub via ``extractor=`` to skip that.
        self._extractor = extractor or LlmExtractor()

    async def analyze(
        self,
        audio_path: Path,
        voice_note_types: Optional[list[VoiceNoteType]] = None,
    ) -> AnalyzeResult:
        """Run the full pipeline on ``audio_path``.

        Args:
            audio_path: Path to the input audio file. Any format
                ffmpeg can decode is accepted (m4a, mp4, aac, webm,
                mp3, ogg, flac, wav).
            voice_note_types: Optional per-request override for the
                voice-note type catalog passed into the LLM extractor.
                When ``None``, the default catalog from
                :mod:`vsa.extraction.types` is used.
        """
        started_at = datetime.now(timezone.utc)
        errors: list[str] = []

        # Normalize whatever format we were handed to 16 kHz mono PCM
        # WAV before the wave-based metadata read below. Caller still
        # owns the original ``audio_path``; we own the normalized tmp
        # file and clean it up at the end of this method.
        normalized_path = normalize_audio(audio_path)
        try:
            return await self._analyze_normalized(
                normalized_path,
                started_at,
                errors,
                voice_note_types,
            )
        finally:
            if normalized_path != audio_path and normalized_path.exists():
                normalized_path.unlink(missing_ok=True)

    async def _analyze_normalized(
        self,
        audio_path: Path,
        started_at: datetime,
        errors: list[str],
        voice_note_types: Optional[list[VoiceNoteType]],
    ) -> AnalyzeResult:
        with wave.open(str(audio_path), "rb") as f:
            n_frames = f.getnframes()
            sample_rate = f.getframerate()
            channels = f.getnchannels()
            duration_seconds = n_frames / sample_rate

        # Transcription is the gate. A failure here means we have
        # nothing useful for extraction, so we record the error and
        # return early — the API layer will emit a status="failed"
        # callback rather than running extraction on nothing.
        try:
            transcription = self._transcriber.transcribe(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            transcription = None
            errors.append(f"transcription failed: {e}")

        # Extraction runs only when we have a non-empty transcript.
        # An empty transcript (silent audio, no-speech filter, etc.)
        # is functionally the same as a transcription failure — skip
        # the LLM call rather than asking it to extract from nothing.
        extraction = None
        if (
            transcription is not None
            and transcription.text.strip()
        ):
            try:
                extraction = self._extractor.extract(
                    transcript=transcription.text,
                    voice_note_types=voice_note_types,
                    fallback_type=DEFAULT_FALLBACK_TYPE,
                )
            except Exception as e:  # noqa: BLE001 -- partial-success contract
                # Extraction failure on a valid transcript does NOT
                # fail the whole request: the transcript itself is a
                # useful deliverable. The error is recorded so the
                # caller can see it; status stays "completed" at the
                # API layer because ``transcription is not None``.
                extraction = None
                errors.append(f"extraction failed: {e}")

        completed_at = datetime.now(timezone.utc)

        return AnalyzeResult(
            audio=AudioInfo(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            ),
            transcription=transcription,
            extraction=extraction,
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
