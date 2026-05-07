"""Orchestrator: read audio metadata and assemble the AnalyzeResult."""

import wave
from datetime import datetime, timezone
from pathlib import Path

from vsa import __version__ as _vsa_version
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo


class Pipeline:
    async def analyze(self, audio_path: Path) -> AnalyzeResult:
        started_at = datetime.now(timezone.utc)

        with wave.open(str(audio_path), "rb") as f:
            n_frames = f.getnframes()
            sample_rate = f.getframerate()
            channels = f.getnchannels()
            duration_seconds = n_frames / sample_rate

        completed_at = datetime.now(timezone.utc)

        return AnalyzeResult(
            audio=AudioInfo(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            ),
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=[],
            ),
        )
