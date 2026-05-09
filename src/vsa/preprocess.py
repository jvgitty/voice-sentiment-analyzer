"""Normalize arbitrary audio inputs to 16 kHz mono PCM WAV.

The analyzer's downstream pipeline (``Pipeline.analyze``) reads audio
via Python's stdlib ``wave`` module, which only handles RIFF/WAVE
containers with PCM payloads. Real-world voice notes arrive as m4a
(Pixel default), aac, mp4, webm, mp3, ogg, or flac. This module
bridges that gap by shelling out to ffmpeg and producing a normalized
WAV that the rest of the pipeline can consume unchanged.

Single public entry point: :func:`normalize_audio`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


_TARGET_SAMPLE_RATE = 16000
_TARGET_CHANNELS = 1

# EBU R128 loudness normalization targets, ported from
# local-transcription-service/src/transcribe/preprocess.py defaults
# (a podcast-ish target). Quiet outdoor voice notes and loud indoor
# voice notes end up at comparable perceived loudness so downstream
# acoustic feature analyzers (loudness mean/std, jitter, shimmer)
# operate on a consistent baseline across recordings.
_LOUDNORM_TARGET_LUFS = -16.0
_LOUDNORM_TRUE_PEAK = -1.5
_LOUDNORM_LRA = 11.0


class AudioNormalizationError(Exception):
    """Raised when ffmpeg fails to convert an input audio file.

    Wraps the underlying ffmpeg stderr so callers can surface a
    meaningful error to the API client without leaking subprocess
    plumbing.
    """


def normalize_audio(input_path: Path) -> Path:
    """Convert any audio file to 16 kHz mono PCM WAV via ffmpeg.

    Returns a path to a freshly-created temporary WAV file. The caller
    owns the file and is responsible for deleting it when done.

    Always re-encodes — even when the input is already WAV — so that
    downstream readers see a consistent sample rate, channel layout,
    and codec regardless of source. The cost is sub-second for typical
    voice-note-length audio.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioNormalizationError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it is "
            "available to the analyzer process."
        )

    fd, dst_str = tempfile.mkstemp(prefix="vsa-normalized-", suffix=".wav")
    dst = Path(dst_str)
    # Close the descriptor immediately; ffmpeg writes to the path,
    # not to an open file handle.
    os.close(fd)

    loudnorm_filter = (
        f"loudnorm=I={_LOUDNORM_TARGET_LUFS}"
        f":TP={_LOUDNORM_TRUE_PEAK}"
        f":LRA={_LOUDNORM_LRA}"
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",                 # overwrite the tmp file we just made
            "-loglevel", "error",
            "-i", str(input_path),
            "-af", loudnorm_filter,
            "-ac", str(_TARGET_CHANNELS),
            "-ar", str(_TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le",  # 16-bit signed little-endian PCM
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Clean up the failed-write temp file before raising.
        if dst.exists():
            dst.unlink(missing_ok=True)
        raise AudioNormalizationError(
            f"ffmpeg returned exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return dst
