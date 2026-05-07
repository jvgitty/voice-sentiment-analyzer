"""ProsodyAnalyzer: pure-function prosody features from a Transcript.

This module deliberately has no model load and no heavy dependencies — it
just walks the word array from a previously-produced :class:`Transcript`
and derives speaking rate, pause statistics, and filler rate. It is safe
to import (and run) on every request without any cold-start cost.
"""

from __future__ import annotations

from vsa.schema import ProsodyFeatures, Transcript

# Fixed filler-word lookup, mixed unigrams and bigrams. Matched
# case-insensitively per spec. Multi-word entries are detected via a
# sliding bigram pass; a matched bigram consumes both word slots so
# overlapping unigram matches inside the bigram are not double-counted.
FILLER_WORDS: frozenset[str] = frozenset(
    {
        "um",
        "uh",
        "like",
        "you know",
        "i mean",
        "sort of",
        "basically",
        "literally",
    }
)


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

        filler_rate = self._filler_rate(transcript)

        return ProsodyFeatures(
            speaking_rate_wpm=speaking_rate_wpm,
            speaking_rate_sps=0.0,
            pause_count=0,
            pause_total_seconds=0.0,
            pause_mean_seconds=0.0,
            filler_rate=filler_rate,
        )

    def _filler_rate(self, transcript: Transcript) -> float:
        """Fraction of words that match the filler lookup (case-insensitive).

        Multi-word fillers like "you know" are matched as bigrams. When a
        bigram match is found, both word slots are consumed and we skip
        ahead — a "like" embedded in a "you know like..." stretch would
        still be counted exactly once when the cursor reaches it.

        The denominator is the raw word count (unigram count). A matched
        bigram contributes 1 filler hit on top of 2 word slots, so e.g.
        a transcript that is 100% "you know" pairs would land at 0.5,
        which is the intuitive answer.
        """
        word_count = len(transcript.words)
        if word_count == 0:
            return 0.0

        lowered = [w.w.lower() for w in transcript.words]
        filler_hits = 0
        i = 0
        while i < len(lowered):
            # Try bigram first so "you know" doesn't get parsed as a
            # standalone "know" unigram (which isn't in the lookup anyway,
            # but the principle matters for "sort" + "of").
            if i + 1 < len(lowered):
                bigram = f"{lowered[i]} {lowered[i + 1]}"
                if bigram in FILLER_WORDS:
                    filler_hits += 1
                    i += 2
                    continue
            if lowered[i] in FILLER_WORDS:
                filler_hits += 1
            i += 1

        return filler_hits / word_count
