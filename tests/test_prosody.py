"""Tests for ProsodyAnalyzer and the ProsodyFeatures schema model."""

from __future__ import annotations

from vsa.schema import Transcript, Word


def _make_transcript(words: list[tuple[str, float, float]]) -> Transcript:
    """Helper: build a Transcript from (text, start, end) tuples."""
    return Transcript(
        engine="parakeet-tdt-0.6b-v2",
        language="en",
        text=" ".join(w for w, _, _ in words),
        words=[Word(w=w, start=s, end=e, conf=1.0) for w, s, e in words],
    )


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


class TestSpeakingRateWpm:
    def test_wpm_is_words_over_minutes_of_audio_duration(self) -> None:
        """10 words spoken across a 5-second audio clip = 120 WPM.

        Per the spec, the denominator is the full audio duration (not just
        the spoken interval), so silence at either end still drags the rate
        down — that's intentional, since slow/halting delivery should look
        slow."""
        from vsa.features.prosody import ProsodyAnalyzer

        # 10 evenly spaced words over 5 seconds of audio. Each word lasts
        # 0.4s, gap of 0.1s. Words land at [0.0,0.4], [0.5,0.9], ...,
        # [4.5,4.9]. None of these gaps exceed the 0.3s pause threshold.
        words: list[tuple[str, float, float]] = [
            (f"word{i}", i * 0.5, i * 0.5 + 0.4) for i in range(10)
        ]
        transcript = _make_transcript(words)

        features = ProsodyAnalyzer().analyze(transcript, audio_duration_sec=5.0)
        assert features.speaking_rate_wpm == 120.0
