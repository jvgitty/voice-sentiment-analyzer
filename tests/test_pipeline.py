"""Tests for the Pipeline orchestrator."""

from pathlib import Path

import pytest

from vsa.pipeline import Pipeline


class TestPipeline:
    @pytest.mark.asyncio
    async def test_analyze_fixture_wav_returns_valid_result(
        self, fixture_wav_path: Path
    ) -> None:
        pipeline = Pipeline()
        result = await pipeline.analyze(fixture_wav_path)

        assert result.schema_version == "1.0"
        assert result.audio.sample_rate == 16000
        assert result.audio.channels == 1
        assert abs(result.audio.duration_seconds - 1.0) < 0.1

        # Slice 1: all feature sections are None
        assert result.transcription is None
        assert result.acoustic is None
        assert result.prosody is None
        assert result.emotion is None
        assert result.composite is None
        assert result.windows is None

        # processing block is populated and errors[] is empty
        assert result.processing.errors == []
        assert result.processing.completed_at >= result.processing.started_at
        assert isinstance(result.processing.library_versions, dict)
