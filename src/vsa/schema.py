"""Pydantic models for the analyzer's request, result, and callback shapes."""

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


class AnalyzeResult(BaseModel):
    schema_version: str = "1.0"
    audio: AudioInfo
    transcription: Optional[Any] = None
    acoustic: Optional[Any] = None
    prosody: Optional[Any] = None
    emotion: Optional[Any] = None
    composite: Optional[Any] = None
    windows: Optional[Any] = None
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
