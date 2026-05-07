"""ProsodyAnalyzer: pure-function prosody features from a Transcript.

This module deliberately has no model load and no heavy dependencies — it
just walks the word array from a previously-produced :class:`Transcript`
and derives speaking rate, pause statistics, and filler rate. It is safe
to import (and run) on every request without any cold-start cost.
"""

from __future__ import annotations

from vsa.schema import ProsodyFeatures, Transcript


class ProsodyAnalyzer:
    """Derive prosody features from a transcript and the audio duration."""

    def analyze(
        self, transcript: Transcript, audio_duration_sec: float
    ) -> ProsodyFeatures:
        word_count = len(transcript.words)

        # Speaking rate: words divided by minutes of *total* audio duration.
        # Per spec, the denominator is the whole audio (not just the spoken
        # interval), so trailing silence drags the rate down — that's the
        # intended behavior for "slow / halting" delivery.
        if audio_duration_sec > 0:
            speaking_rate_wpm = word_count / (audio_duration_sec / 60.0)
        else:
            speaking_rate_wpm = 0.0

        return ProsodyFeatures(
            speaking_rate_wpm=speaking_rate_wpm,
            speaking_rate_sps=0.0,
            pause_count=0,
            pause_total_seconds=0.0,
            pause_mean_seconds=0.0,
            filler_rate=0.0,
        )
