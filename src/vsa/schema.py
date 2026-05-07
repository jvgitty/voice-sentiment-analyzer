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


class DimensionalEmotion(BaseModel):
    """Continuous arousal / valence / dominance from a regression-head model
    (e.g. ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``). Each
    dimension is a probability-like scalar in [0, 1]."""

    model: str
    arousal: float
    valence: float
    dominance: float


class CategoricalEmotion(BaseModel):
    """Discrete emotion label + per-class probability scores from a
    classifier (e.g. SpeechBrain's IEMOCAP wav2vec2 model). ``scores`` keys
    are the class names; values sum to ~1.0."""

    model: str
    label: str
    scores: dict[str, float]


class EmotionResult(BaseModel):
    """Combined emotion output. Either component may be ``None`` if its
    underlying model failed independently — the analyzer treats them as
    independent so one model crashing doesn't wipe out the other."""

    dimensional: Optional[DimensionalEmotion] = None
    categorical: Optional[CategoricalEmotion] = None


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


class ProsodyFeatures(BaseModel):
    """Prosody-level features derived from a Transcript + audio duration.

    All fields are derived purely from text/timestamps — no model load.
    See ``vsa.features.prosody.ProsodyAnalyzer`` for the formulas."""

    speaking_rate_wpm: float
    speaking_rate_sps: float
    pause_count: int
    pause_total_seconds: float
    pause_mean_seconds: float
    filler_rate: float


class WindowMetrics(BaseModel):
    """Headline metrics over a single time-tiled audio window (Slice 7).

    A ``WindowedAnalyzer`` produces a list of these tiling
    ``[0, audio_duration)`` with no gaps and no overlaps. The last window
    may be shorter when the duration does not divide evenly. Every metric
    field is ``Optional`` so a window where one analyzer failed (or was
    skipped to keep cost bounded) still serializes."""

    start_sec: float
    end_sec: float
    pitch_mean_hz: Optional[float] = None
    loudness_mean_db: Optional[float] = None
    arousal: Optional[float] = None
    valence: Optional[float] = None
    confidence: Optional[float] = None
    engagement: Optional[float] = None
    calmness: Optional[float] = None


class AnalyzeResult(BaseModel):
    schema_version: str = "1.0"
    audio: AudioInfo
    transcription: Optional[Transcript] = None
    acoustic: Optional[AcousticFeatures] = None
    prosody: Optional[ProsodyFeatures] = None
    emotion: Optional[EmotionResult] = None
    # ``composite`` is a vsa.composites.CompositeScores when scoring
    # succeeds, ``None`` when the scorer raised. The annotation is a
    # forward-string-style ``Any`` to avoid a circular import: schema is
    # imported by composites for ScoreInputs.
    composite: Optional[Any] = None
    # Per-window headline metrics from the WindowedAnalyzer (Slice 7);
    # None when windowing failed outright. Empty list is a valid value
    # but unusual: even a sub-window-length audio yields one entry.
    windows: Optional[list[WindowMetrics]] = None
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
