"""Shared pytest fixtures."""

from pathlib import Path
import wave

# Pre-import the heavyweight ML stack (pandas/pyarrow via NeMo) BEFORE
# librosa / audioread are loaded by other test modules. On Windows + Python
# 3.13, loading pyarrow after librosa has been imported triggers a fatal
# access violation during DLL initialisation. Importing pandas first warms
# up pyarrow against a clean process state and sidesteps the collision.
import pandas  # noqa: F401  -- imported for side effects (pyarrow init).

import numpy as np
import pytest


@pytest.fixture
def fixture_wav_path(tmp_path: Path) -> Path:
    """Synthesize 1s of 440 Hz sine at 16 kHz mono, return its path."""
    sample_rate = 16000
    duration_seconds = 1.0
    frequency = 440.0

    n_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)
    samples_int16 = (samples * 32767).astype(np.int16)

    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples_int16.tobytes())

    return path


@pytest.fixture
def fixture_wav_bytes(fixture_wav_path: Path) -> bytes:
    """Raw bytes of the synthesized fixture WAV."""
    return fixture_wav_path.read_bytes()
