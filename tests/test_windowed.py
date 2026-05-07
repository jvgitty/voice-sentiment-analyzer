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
