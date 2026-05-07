"""Tests for AcousticAnalyzer and the AcousticFeatures schema."""

from pathlib import Path


class TestAcousticFeaturesSchema:
    def test_can_construct_with_stub_values(self) -> None:
        """Smoke test: AcousticFeatures imports cleanly with full nested shape."""
        from vsa.schema import (
            AcousticFeatures,
            LoudnessFeatures,
            PitchFeatures,
            SpectralFeatures,
            VoiceQualityFeatures,
        )

        features = AcousticFeatures(
            pitch=PitchFeatures(
                mean_hz=440.0,
                median_hz=440.0,
                std_hz=0.0,
                min_hz=440.0,
                max_hz=440.0,
                range_hz=0.0,
            ),
            loudness=LoudnessFeatures(
                mean_db=-20.0,
                std_db=0.0,
                rms_mean=0.5,
            ),
            voice_quality=VoiceQualityFeatures(
                jitter_local=0.01,
                shimmer_local=0.05,
                hnr_db=15.0,
                voiced_unvoiced_ratio=0.9,
            ),
            spectral=SpectralFeatures(
                centroid_mean=1500.0,
                rolloff_mean=3000.0,
                bandwidth_mean=1000.0,
                mfcc_means=[0.0] * 13,
            ),
        )

        assert features.pitch.mean_hz == 440.0
        assert features.loudness.rms_mean == 0.5
        assert features.voice_quality.hnr_db == 15.0
        assert len(features.spectral.mfcc_means) == 13
