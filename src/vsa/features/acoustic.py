"""AcousticAnalyzer: wraps parselmouth (Praat) and librosa to extract
pitch, voice quality, spectral, and loudness features from audio."""

from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from vsa.schema import (
    AcousticFeatures,
    LoudnessFeatures,
    PitchFeatures,
    SpectralFeatures,
    VoiceQualityFeatures,
)


class AcousticAnalyzer:
    def analyze(self, audio_path: Path) -> AcousticFeatures:
        sound = parselmouth.Sound(str(audio_path))

        pitch_obj = sound.to_pitch()
        f0 = pitch_obj.selected_array["frequency"]
        voiced = f0[f0 > 0.0]
        mean_hz = float(voiced.mean()) if voiced.size else 0.0
        voiced_unvoiced_ratio = (
            float(voiced.size) / float(f0.size) if f0.size else 0.0
        )

        # Voice quality via Praat's PointProcess (jitter, shimmer) and
        # Harmonicity (HNR).
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = float(
            call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        )
        shimmer_local = float(
            call(
                [sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        )
        harmonicity = sound.to_harmonicity()
        hnr_db = float(call(harmonicity, "Get mean", 0, 0))

        # Spectral features via librosa. Load audio mono so MFCCs are
        # deterministic shape (13, n_frames).
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [float(v) for v in mfcc.mean(axis=1)]

        return AcousticFeatures(
            pitch=PitchFeatures(
                mean_hz=mean_hz,
                median_hz=0.0,
                std_hz=0.0,
                min_hz=0.0,
                max_hz=0.0,
                range_hz=0.0,
            ),
            loudness=LoudnessFeatures(mean_db=0.0, std_db=0.0, rms_mean=0.0),
            voice_quality=VoiceQualityFeatures(
                jitter_local=jitter_local,
                shimmer_local=shimmer_local,
                hnr_db=hnr_db,
                voiced_unvoiced_ratio=voiced_unvoiced_ratio,
            ),
            spectral=SpectralFeatures(
                centroid_mean=0.0,
                rolloff_mean=0.0,
                bandwidth_mean=0.0,
                mfcc_means=mfcc_means,
            ),
        )
