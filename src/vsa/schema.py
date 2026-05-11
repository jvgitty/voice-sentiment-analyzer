"""Pydantic models for the analyzer's request, result, and callback shapes.

v0.2 scope (post-pivot): transcription-only. The structured-extraction
layer (LLM-derived tags / entities / tasks) lands in a follow-up PR; for
now ``AnalyzeResult`` carries the Parakeet transcript and the audio /
processing metadata that came with it.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


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
    processing: ProcessingInfo


class AnalyzeRequest(BaseModel):
    audio_url: HttpUrl
    callback_url: HttpUrl
    callback_secret: str = Field(min_length=16)
    metadata: dict[str, Any]
    request_id: str


class CallbackBody(BaseModel):
    request_id: str
    status: Literal["completed", "failed"]
    metadata: dict[str, Any]
    result: Optional[AnalyzeResult] = None
    error: Optional[dict[str, Any]] = None
