"""Tests for ProsodyAnalyzer and the ProsodyFeatures schema model."""

from __future__ import annotations


class TestProsodyFeaturesSchema:
    def test_prosody_features_constructs_with_all_six_fields(self) -> None:
        """ProsodyFeatures imports cleanly from vsa.schema and accepts the
        six Slice 4 fields."""
        from vsa.schema import ProsodyFeatures

        features = ProsodyFeatures(
            speaking_rate_wpm=120.0,
            speaking_rate_sps=2.5,
            pause_count=3,
            pause_total_seconds=1.7,
            pause_mean_seconds=0.5667,
            filler_rate=0.05,
        )
        assert features.speaking_rate_wpm == 120.0
        assert features.speaking_rate_sps == 2.5
        assert features.pause_count == 3
        assert features.pause_total_seconds == 1.7
        assert features.pause_mean_seconds == 0.5667
        assert features.filler_rate == 0.05
