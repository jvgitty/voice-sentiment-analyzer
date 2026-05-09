"""Tests for the audio-format normalization step.

The analyzer's downstream pipeline reads via Python's stdlib ``wave``
module which only handles WAV. Real-world voice notes arrive as m4a /
aac / mp4 / webm. The ``normalize_audio`` function bridges that gap by
converting any input format to 16 kHz mono PCM WAV via ffmpeg.
"""

import wave
from pathlib import Path

import pytest

from vsa.preprocess import AudioNormalizationError, normalize_audio


class TestNormalizeAudio:
    def test_m4a_input_returns_path_readable_by_wave_module(
        self, fixture_m4a_path: Path
    ) -> None:
        """The load-bearing tracer: an m4a file in, a WAV path out that
        the stdlib wave module can read. If this passes, the central
        problem this slice exists to solve (analyzer rejects Pixel voice
        notes) is fixed."""
        wav_path = normalize_audio(fixture_m4a_path)

        with wave.open(str(wav_path), "rb") as f:
            n_frames = f.getnframes()
            sample_rate = f.getframerate()
            channels = f.getnchannels()

        assert sample_rate == 16000
        assert channels == 1
        assert n_frames > 0

    def test_wav_input_passes_through_to_normalized_wav(
        self, fixture_wav_path: Path
    ) -> None:
        """Regression guard: the existing happy path (WAV in) must keep
        working. The output is still 16 kHz mono PCM and the returned
        path differs from the input (always a freshly-created tmp file)
        so the caller never accidentally clobbers the source."""
        wav_path = normalize_audio(fixture_wav_path)

        assert wav_path != fixture_wav_path
        with wave.open(str(wav_path), "rb") as f:
            assert f.getframerate() == 16000
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2  # 16-bit
            assert f.getnframes() > 0

    def test_garbage_input_raises_audio_normalization_error(
        self, tmp_path: Path
    ) -> None:
        """A non-audio file (e.g. someone uploaded a text doc with the
        wrong content-type, or the file is corrupt) must raise a typed
        AudioNormalizationError that callers can convert to a meaningful
        HTTP error. The error message should include something derived
        from ffmpeg's stderr so debugging upstream isn't a black box."""
        bogus = tmp_path / "not-actually-audio.bin"
        bogus.write_bytes(b"this is plain text, not an audio file at all")

        with pytest.raises(AudioNormalizationError) as exc_info:
            normalize_audio(bogus)

        # The error string should at least mention ffmpeg's exit code so
        # the operator can correlate against ffmpeg logs.
        assert "ffmpeg" in str(exc_info.value).lower()
