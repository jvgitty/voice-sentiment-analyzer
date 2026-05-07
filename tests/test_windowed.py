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

    def test_window_seconds_is_configurable(self) -> None:
        """The constructor's window_seconds argument actually drives the
        tiling — this guards against accidentally hardcoding 30s in _tile."""
        from vsa.windowed import WindowedAnalyzer

        analyzer = WindowedAnalyzer(window_seconds=15.0)
        tiles = analyzer._tile(audio_duration_sec=45.0)
        assert tiles == [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0)]

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


class TestWindowedAnalyzerEndToEnd:
    def test_analyze_returns_single_window_with_pitch_for_short_fixture(
        self, fixture_wav_path: Path
    ) -> None:
        """End-to-end smoke: WindowedAnalyzer.analyze on the 1s 440 Hz
        sine fixture returns exactly one WindowMetrics covering the full
        audio. With the real AcousticAnalyzer the window's pitch_mean_hz
        should reflect the 440 Hz tone (broad tolerance). EmotionAnalyzer
        is stubbed to None so we don't pay the wav2vec2 load cost in this
        wiring test — the dimensional fields are allowed to be None."""
        from vsa.composites import CompositeScorer
        from vsa.features.acoustic import AcousticAnalyzer
        from vsa.pipeline import _COMPOSITES_YAML_PATH
        from vsa.windowed import WindowedAnalyzer

        analyzer = WindowedAnalyzer(window_seconds=30.0)
        scorer = CompositeScorer.from_yaml(_COMPOSITES_YAML_PATH)
        windows = analyzer.analyze(
            audio_path=fixture_wav_path,
            transcript=None,
            composite_scorer=scorer,
            emotion_analyzer=None,  # skip dimensional inference
            acoustic_analyzer=AcousticAnalyzer(),
        )
        assert len(windows) == 1
        window = windows[0]
        assert window.start_sec == 0.0
        assert abs(window.end_sec - 1.0) < 0.05
        assert window.pitch_mean_hz is not None
        assert 380.0 <= window.pitch_mean_hz <= 500.0
