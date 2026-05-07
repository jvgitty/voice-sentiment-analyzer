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


class TestEnergySteadinessOverride:
    def test_score_overrides_replace_registry_value_for_energy_steadiness(
        self,
    ) -> None:
        """CompositeScorer.score(overrides={...}) replaces the registry's
        placeholder with the supplied value. Confidence with override=0.0
        must differ from confidence with override=1.0 (the placeholder)
        because energy_steadiness has 0.20 weight in the YAML."""
        from vsa.composites import CompositeScorer, ScoreInputs
        from vsa.pipeline import _COMPOSITES_YAML_PATH
        from vsa.schema import (
            AcousticFeatures,
            DimensionalEmotion,
            EmotionResult,
            LoudnessFeatures,
            PitchFeatures,
            ProsodyFeatures,
            SpectralFeatures,
            VoiceQualityFeatures,
        )

        acoustic = AcousticFeatures(
            pitch=PitchFeatures(
                mean_hz=200.0,
                median_hz=200.0,
                std_hz=10.0,
                min_hz=100.0,
                max_hz=300.0,
                range_hz=200.0,
            ),
            loudness=LoudnessFeatures(
                mean_db=-20.0, std_db=4.0, rms_mean=0.1
            ),
            voice_quality=VoiceQualityFeatures(
                jitter_local=0.005,
                shimmer_local=0.02,
                hnr_db=20.0,
                voiced_unvoiced_ratio=0.7,
            ),
            spectral=SpectralFeatures(
                centroid_mean=1500.0,
                rolloff_mean=4000.0,
                bandwidth_mean=2000.0,
                mfcc_means=[0.0] * 13,
            ),
        )
        prosody = ProsodyFeatures(
            speaking_rate_wpm=140.0,
            speaking_rate_sps=4.0,
            pause_count=0,
            pause_total_seconds=0.0,
            pause_mean_seconds=0.0,
            filler_rate=0.0,
        )
        emotion = EmotionResult(
            dimensional=DimensionalEmotion(
                model="m", arousal=0.5, valence=0.5, dominance=0.5
            )
        )
        inputs = ScoreInputs(
            acoustic=acoustic,
            prosody=prosody,
            emotion=emotion,
            audio_duration_seconds=60.0,
        )

        scorer = CompositeScorer.from_yaml(_COMPOSITES_YAML_PATH)
        baseline = scorer.score(inputs)
        overridden = scorer.score(
            inputs, overrides={"energy_steadiness": 0.0}
        )
        assert baseline.confidence is not None
        assert overridden.confidence is not None
        # energy_steadiness has 0.20 weight; baseline uses placeholder=1.0,
        # override forces 0.0, so confidence drops by ~0.20.
        assert overridden.confidence < baseline.confidence
        assert (baseline.confidence - overridden.confidence) == pytest.approx(
            0.20, abs=1e-6
        )


class TestWindowSecondsEnvVar:
    @pytest.mark.asyncio
    async def test_pipeline_constructs_windowed_analyzer_from_env_var(
        self, fixture_wav_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting WINDOW_SECONDS in the env causes Pipeline construction
        to use that value for the WindowedAnalyzer. We verify by setting
        it to a sub-fixture-duration value (0.4s) so the 1s fixture tiles
        into multiple windows; with the default 30s it would be one."""
        from tests.test_pipeline import (
            _StubEmotionAnalyzer,
            _StubTranscriber,
        )
        from vsa.pipeline import Pipeline

        monkeypatch.setenv("WINDOW_SECONDS", "0.4")
        pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
        )
        result = await pipeline.analyze(fixture_wav_path)
        assert result.windows is not None
        # 1.0s / 0.4s -> tiles at [0.0, 0.4), [0.4, 0.8), [0.8, 1.0).
        assert len(result.windows) == 3
        assert result.windows[0].start_sec == 0.0
        assert result.windows[0].end_sec == pytest.approx(0.4)
        assert result.windows[-1].end_sec == pytest.approx(1.0)


class TestPipelineEnergySteadinessFromWindows:
    @pytest.mark.asyncio
    async def test_pipeline_injects_energy_steadiness_from_window_loudness_cov(
        self, fixture_wav_path: Path
    ) -> None:
        """End-to-end: a stubbed WindowedAnalyzer that returns windows with
        varying loudness causes whole-audio confidence to differ from the
        all-1.0-placeholder baseline produced by a single-window stub.

        This pins the contract that Pipeline.analyze derives the
        energy_steadiness override from the windows pass and threads it
        through CompositeScorer.score."""
        from vsa.pipeline import Pipeline
        from vsa.schema import WindowMetrics
        from tests.test_pipeline import _StubEmotionAnalyzer, _StubTranscriber

        class _StubWindowedSingle:
            """Single window — CoV undefined for n<2 → no override → the
            YAML's placeholder (1.0) controls confidence."""

            def analyze(self, **kwargs):
                return [
                    WindowMetrics(
                        start_sec=0.0, end_sec=1.0, loudness_mean_db=-20.0
                    )
                ]

        class _StubWindowedVarying:
            """Windows with strongly varying loudness — high CoV → low
            energy_steadiness → confidence drops below the baseline."""

            def analyze(self, **kwargs):
                return [
                    WindowMetrics(
                        start_sec=0.0, end_sec=0.5, loudness_mean_db=-10.0
                    ),
                    WindowMetrics(
                        start_sec=0.5, end_sec=1.0, loudness_mean_db=-50.0
                    ),
                ]

        baseline_pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
            windowed_analyzer=_StubWindowedSingle(),
        )
        baseline_result = await baseline_pipeline.analyze(fixture_wav_path)

        varying_pipeline = Pipeline(
            transcriber=_StubTranscriber(),
            emotion_analyzer=_StubEmotionAnalyzer(),
            windowed_analyzer=_StubWindowedVarying(),
        )
        varying_result = await varying_pipeline.analyze(fixture_wav_path)

        assert baseline_result.composite is not None
        assert varying_result.composite is not None
        assert baseline_result.composite.confidence is not None
        assert varying_result.composite.confidence is not None
        # Varying loudness across windows → CoV>0 → energy_steadiness<1
        # → confidence strictly below the placeholder baseline.
        assert (
            varying_result.composite.confidence
            < baseline_result.composite.confidence
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
