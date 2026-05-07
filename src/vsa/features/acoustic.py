"""AcousticAnalyzer: wraps parselmouth (Praat) and librosa to extract
pitch, voice quality, spectral, and loudness features from audio."""

from pathlib import Path

import parselmouth

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
        # Praat returns 0 for unvoiced frames; filter those out for stats.
        f0 = pitch_obj.selected_array["frequency"]
        voiced = f0[f0 > 0.0]
        mean_hz = float(voiced.mean()) if voiced.size else 0.0

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
                jitter_local=0.0,
                shimmer_local=0.0,
                hnr_db=0.0,
                voiced_unvoiced_ratio=0.0,
            ),
            spectral=SpectralFeatures(
                centroid_mean=0.0,
                rolloff_mean=0.0,
                bandwidth_mean=0.0,
                mfcc_means=[0.0] * 13,
            ),
        )
