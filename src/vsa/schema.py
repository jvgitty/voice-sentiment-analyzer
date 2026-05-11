"""Pydantic models for the analyzer's request, result, and callback shapes.

v0.2 scope: transcription + LLM-based structured extraction. The
extraction layer's input/output types live under :mod:`vsa.extraction`;
this module re-exposes the result type on :class:`AnalyzeResult` and
accepts a per-request voice-note-type override on :class:`AnalyzeRequest`.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from vsa.extraction.schema import ExtractionResult
from vsa.extraction.types import VoiceNoteType


class AudioInfo(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int


class ProcessingInfo(BaseModel):
    started_at: datetime
    completed_at: datetime
    library_versions: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class Word(BaseModel):
    w: str
    start: float
    end: float
    conf: float


class Transcript(BaseModel):
    engine: str
    language: str
    text: str
    words: list[Word] = Field(default_factory=list)


class AnalyzeResult(BaseModel):
    schema_version: str = "2.0"
    audio: AudioInfo
    transcription: Optional[Transcript] = None
    extraction: Optional[ExtractionResult] = None
    processing: ProcessingInfo


class AnalyzeRequest(BaseModel):
    audio_url: HttpUrl
    callback_url: HttpUrl
    callback_secret: str = Field(min_length=16)
    metadata: dict[str, Any]
    request_id: str
    # Optional per-request override for the voice-note type catalog.
    # When omitted, the default catalog from vsa.extraction.types
    # applies. The multi-tenant per-client persistent overrides
    # described in docs/ROADMAP.md build on top of this.
    voice_note_types: Optional[list[VoiceNoteType]] = None


class CallbackBody(BaseModel):
    request_id: str
    status: Literal["completed", "failed"]
    metadata: dict[str, Any]
    result: Optional[AnalyzeResult] = None
    error: Optional[dict[str, Any]] = None
