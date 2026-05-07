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


class PitchFeatures(BaseModel):
    mean_hz: float
    median_hz: float
    std_hz: float
    min_hz: float
    max_hz: float
    range_hz: float


class LoudnessFeatures(BaseModel):
    mean_db: float
    std_db: float
    rms_mean: float


class VoiceQualityFeatures(BaseModel):
    jitter_local: float
    shimmer_local: float
    hnr_db: float
    voiced_unvoiced_ratio: float


class SpectralFeatures(BaseModel):
    centroid_mean: float
    rolloff_mean: float
    bandwidth_mean: float
    mfcc_means: list[float]


class AcousticFeatures(BaseModel):
    pitch: PitchFeatures
    loudness: LoudnessFeatures
    voice_quality: VoiceQualityFeatures
    spectral: SpectralFeatures


class AnalyzeResult(BaseModel):
    schema_version: str = "1.0"
    audio: AudioInfo
    transcription: Optional[Any] = None
    acoustic: Optional[AcousticFeatures] = None
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
