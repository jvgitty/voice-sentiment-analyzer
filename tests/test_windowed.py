"""Tests for the WindowedAnalyzer (Slice 7)."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestWindowMetricsSchema:
    def test_window_metrics_imports_and_constructs(self) -> None:
        """The WindowMetrics Pydantic model is importable from vsa.schema and
        constructs cleanly with just start_sec/end_sec; metric fields default
        to None so a window with all-failed inference still serializes."""
        from vsa.schema import WindowMetrics

        wm = WindowMetrics(start_sec=0.0, end_sec=30.0)
        assert wm.start_sec == 0.0
        assert wm.end_sec == 30.0
        assert wm.pitch_mean_hz is None
        assert wm.loudness_mean_db is None
        assert wm.arousal is None
        assert wm.valence is None
        assert wm.confidence is None
        assert wm.engagement is None
        assert wm.calmness is None


class TestWindowedAnalyzerTiling:
    def test_audio_shorter_than_window_yields_single_window(self) -> None:
        """For audio shorter than the window size, _tile must produce a
        single window covering the whole audio, not zero windows or one
        with end_sec past the audio duration."""
        from vsa.windowed import WindowedAnalyzer

        analyzer = WindowedAnalyzer(window_seconds=30.0)
        tiles = analyzer._tile(audio_duration_sec=10.0)
        assert tiles == [(0.0, 10.0)]

    def test_even_multiple_yields_back_to_back_full_windows(self) -> None:
        """When duration is an exact multiple of window_seconds, _tile
        produces back-to-back full-length windows with no partial tail."""
        from vsa.windowed import WindowedAnalyzer

        analyzer = WindowedAnalyzer(window_seconds=30.0)
        tiles = analyzer._tile(audio_duration_sec=60.0)
        assert tiles == [(0.0, 30.0), (30.0, 60.0)]

    def test_partial_last_window_is_included_with_full_coverage(self) -> None:
        """Audio not divisible by window_seconds produces a shorter final
        window covering the remainder. Tiles must have no gaps, no
        overlaps, and cover [0, duration) exactly."""
        from vsa.windowed import WindowedAnalyzer

        analyzer = WindowedAnalyzer(window_seconds=30.0)
        tiles = analyzer._tile(audio_duration_sec=70.0)
        assert tiles == [(0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]
        # Coverage invariants: first window starts at 0, last ends at
        # duration, and each tile starts where the previous ended.
        assert tiles[0][0] == 0.0
        assert tiles[-1][1] == 70.0
        for prev, curr in zip(tiles, tiles[1:]):
            assert curr[0] == prev[1], (
                f"gap or overlap between {prev} and {curr}"
            )
