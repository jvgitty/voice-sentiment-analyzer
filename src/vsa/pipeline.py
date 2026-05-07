"""Orchestrator: read audio metadata, run feature analyzers, assemble result."""

import wave
from datetime import datetime, timezone
from pathlib import Path

from vsa import __version__ as _vsa_version
from vsa.features.acoustic import AcousticAnalyzer
from vsa.schema import AnalyzeResult, AudioInfo, ProcessingInfo


class Pipeline:
    def __init__(self, acoustic_analyzer: AcousticAnalyzer | None = None) -> None:
        self._acoustic = acoustic_analyzer or AcousticAnalyzer()

    async def analyze(self, audio_path: Path) -> AnalyzeResult:
        started_at = datetime.now(timezone.utc)
        errors: list[str] = []

        with wave.open(str(audio_path), "rb") as f:
            n_frames = f.getnframes()
            sample_rate = f.getframerate()
            channels = f.getnchannels()
            duration_seconds = n_frames / sample_rate

        try:
            acoustic = self._acoustic.analyze(audio_path)
        except Exception as e:  # noqa: BLE001 -- partial-success contract
            acoustic = None
            errors.append(f"acoustic analysis failed: {e}")

        completed_at = datetime.now(timezone.utc)

        return AnalyzeResult(
            audio=AudioInfo(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
            ),
            acoustic=acoustic,
            processing=ProcessingInfo(
                started_at=started_at,
                completed_at=completed_at,
                library_versions={"vsa": _vsa_version},
                errors=errors,
            ),
        )
