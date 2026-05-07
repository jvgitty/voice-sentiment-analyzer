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


class TestAcousticAnalyzer:
    def test_pitch_mean_hz_in_tolerance_band_for_440hz_fixture(
        self, fixture_wav_path: Path
    ) -> None:
        """440 Hz pure sine fixture; parselmouth typically lands within +/- 20Hz.

        Tolerance band [380, 500] is intentionally wide to absorb library
        version drift.
        """
        from vsa.features.acoustic import AcousticAnalyzer

        analyzer = AcousticAnalyzer()
        features = analyzer.analyze(fixture_wav_path)

        assert 380.0 <= features.pitch.mean_hz <= 500.0

    def test_voice_quality_features_are_non_negative_floats(
        self, fixture_wav_path: Path
    ) -> None:
        """jitter, shimmer, hnr_db, voiced_unvoiced_ratio are >= 0.

        Praat returns these as ratios / dB / fractions; for any non-pathological
        signal they should be finite and non-negative. The 440 Hz fixture is
        fully voiced and tonal, so HNR should be large and most frames should
        be detected as voiced -- forces a real implementation, not stubs.
        """
        from vsa.features.acoustic import AcousticAnalyzer

        analyzer = AcousticAnalyzer()
        features = analyzer.analyze(fixture_wav_path)
        vq = features.voice_quality

        assert isinstance(vq.jitter_local, float)
        assert isinstance(vq.shimmer_local, float)
        assert isinstance(vq.hnr_db, float)
        assert isinstance(vq.voiced_unvoiced_ratio, float)
        assert vq.jitter_local >= 0.0
        assert vq.shimmer_local >= 0.0
        assert vq.hnr_db >= 0.0
        assert vq.voiced_unvoiced_ratio >= 0.0
        # Tonal pure sine: HNR should be substantially > 1 dB and most frames
        # voiced. Force real implementation.
        assert vq.hnr_db > 1.0
        assert vq.voiced_unvoiced_ratio > 0.5

    def test_spectral_mfcc_means_is_13_element_list_of_floats(
        self, fixture_wav_path: Path
    ) -> None:
        """librosa returns 13 MFCCs by default; we mean-pool over time."""
        from vsa.features.acoustic import AcousticAnalyzer

        analyzer = AcousticAnalyzer()
        features = analyzer.analyze(fixture_wav_path)
        mfcc = features.spectral.mfcc_means

        assert isinstance(mfcc, list)
        assert len(mfcc) == 13
        for v in mfcc:
            assert isinstance(v, float)
        # Default-stub list is all zeros; force a real implementation by
        # asserting at least one MFCC coefficient is non-zero.
        assert any(v != 0.0 for v in mfcc)

    def test_loudness_rms_mean_non_negative_and_mean_db_finite(
        self, fixture_wav_path: Path
    ) -> None:
        """RMS amplitude is bounded below by 0 by definition.

        mean_db is in dB FS so negative values are normal and expected (a
        non-clipped sine sits below 0 dB FS). The contract is just 'finite'.
        """
        import math

        from vsa.features.acoustic import AcousticAnalyzer

        analyzer = AcousticAnalyzer()
        features = analyzer.analyze(fixture_wav_path)
        loudness = features.loudness

        assert isinstance(loudness.rms_mean, float)
        assert loudness.rms_mean >= 0.0
        # Force real implementation: a non-silent fixture should have RMS > 0.
        assert loudness.rms_mean > 0.0
        assert isinstance(loudness.mean_db, float)
        assert math.isfinite(loudness.mean_db)
