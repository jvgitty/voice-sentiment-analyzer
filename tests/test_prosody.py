"""Tests for ProsodyAnalyzer and the ProsodyFeatures schema model."""

from __future__ import annotations

import pytest

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


class TestFillerRate:
    def test_filler_rate_counts_unigrams_and_bigrams_case_insensitive(self) -> None:
        """20 words, 4 fillers (3 unigrams + 1 bigram) = filler_rate 0.20.

        Verifies (a) case-insensitive unigram match, (b) bigram detection
        for "you know", (c) the bigram counts as one filler regardless of
        which word is examined."""
        from vsa.features.prosody import ProsodyAnalyzer

        # Build a 20-word transcript. Fillers placed at fixed positions:
        #   - "Um"  at index 0  (case-insensitive unigram)
        #   - "uh"  at index 5  (unigram)
        #   - "Like" at index 10 (case-insensitive unigram)
        #   - "you know" bigram at indices 15-16 (counts as 1 filler)
        # The bigram "you know" should be counted once, not twice — i.e.
        # 4 filler slots in 20 words = 0.20.
        raw = [
            "Um",     "the",   "rain",   "in",     "Spain",
            "uh",     "falls", "mainly", "on",     "the",
            "Like",   "the",   "plain",  "and",    "also",
            "you",    "know",  "the",    "hills",  "elsewhere",
        ]
        words = [(w, float(i), float(i) + 0.4) for i, w in enumerate(raw)]
        transcript = _make_transcript(words)

        features = ProsodyAnalyzer().analyze(transcript, audio_duration_sec=20.0)
        assert features.filler_rate == 4 / 20


class TestPauseStatistics:
    def test_pauses_above_threshold_are_counted_summed_and_averaged(self) -> None:
        """A "pause" is a gap between consecutive words exceeding 0.3s.

        Synthesized 5-word transcript with two qualifying gaps (0.5s
        and 1.2s) plus two sub-threshold gaps (0.1s and 0.2s). Expect
        pause_count == 2, total ≈ 1.7, mean ≈ 0.85."""
        from vsa.features.prosody import ProsodyAnalyzer

        # Word layout. Each word is 0.4s long.
        # w0: 0.0 -> 0.4
        # gap = 0.1s (sub-threshold)
        # w1: 0.5 -> 0.9
        # gap = 0.5s (PAUSE)
        # w2: 1.4 -> 1.8
        # gap = 0.2s (sub-threshold)
        # w3: 2.0 -> 2.4
        # gap = 1.2s (PAUSE)
        # w4: 3.6 -> 4.0
        words = [
            ("a", 0.0, 0.4),
            ("b", 0.5, 0.9),
            ("c", 1.4, 1.8),
            ("d", 2.0, 2.4),
            ("e", 3.6, 4.0),
        ]
        transcript = _make_transcript(words)

        features = ProsodyAnalyzer().analyze(transcript, audio_duration_sec=4.0)
        assert features.pause_count == 2
        assert features.pause_total_seconds == pytest.approx(1.7)
        assert features.pause_mean_seconds == pytest.approx(0.85)
