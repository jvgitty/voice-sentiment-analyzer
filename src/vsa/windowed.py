"""Time-windowed analyzer (Slice 7).

Tiles ``[0, audio_duration)`` into fixed-size windows, re-runs the
headline metrics per window, and emits a list of ``WindowMetrics`` for
the time-series view in the result JSON.

The windows tile the audio with no gaps and no overlaps. The last
window may be shorter when ``audio_duration % window_seconds != 0``.
For audio shorter than ``window_seconds``, a single window covering
``[0, audio_duration)`` is produced.
"""

from __future__ import annotations


class WindowedAnalyzer:
    """Tile audio into fixed-size windows and (in later cycles) score
    headline metrics per window."""

    def __init__(self, window_seconds: float = 30.0) -> None:
        self._window_seconds = float(window_seconds)

    def _tile(self, audio_duration_sec: float) -> list[tuple[float, float]]:
        """Return ``[(start_sec, end_sec), ...]`` tiling
        ``[0, audio_duration_sec)`` with no gaps and no overlaps. The last
        tuple may have ``end_sec - start_sec < window_seconds`` when the
        duration does not divide evenly. For ``audio_duration_sec <=
        window_seconds`` a single window covering the full audio is
        returned."""
        if audio_duration_sec <= 0.0:
            return []
        # Single-window short-audio case: avoid spinning a degenerate
        # zero-length tail when duration <= window.
        if audio_duration_sec <= self._window_seconds:
            return [(0.0, float(audio_duration_sec))]
        # Full-length back-to-back windows for the bulk of the audio.
        tiles: list[tuple[float, float]] = []
        start = 0.0
        while start + self._window_seconds <= audio_duration_sec:
            tiles.append((start, start + self._window_seconds))
            start += self._window_seconds
        # Partial trailing window so coverage extends to the audio end.
        # ``start < audio_duration_sec`` is the only safe condition: float
        # rounding means the equality case may register either as a clean
        # multiple (already handled above) or a sub-microsecond tail; the
        # latter we accept rather than introducing an epsilon.
        if start < audio_duration_sec:
            tiles.append((start, float(audio_duration_sec)))
        return tiles
