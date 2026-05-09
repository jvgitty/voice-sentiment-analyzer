"""Tests for the Transcriber interface and ParakeetTranscriber."""

import wave
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pytest


# parakeet_transcriber is a session-scoped fixture in conftest.py so the
# Slice 9 schema-parity tests can share the same warmed instance.


class TestTranscriberInterface:
    def test_word_model_has_required_fields(self) -> None:
        from vsa.schema import Word

        word = Word(w="hello", start=0.0, end=0.5, conf=0.9)
        assert word.w == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.conf == 0.9

    def test_transcript_model_has_required_fields(self) -> None:
        from vsa.schema import Transcript, Word

        words = [Word(w="hi", start=0.0, end=0.2, conf=0.95)]
        transcript = Transcript(
            engine="parakeet-tdt-0.6b-v2",
            language="en",
            text="hi",
            words=words,
        )
        assert transcript.engine == "parakeet-tdt-0.6b-v2"
        assert transcript.language == "en"
        assert transcript.text == "hi"
        assert transcript.words == words

    def test_transcript_default_words_empty_list(self) -> None:
        from vsa.schema import Transcript

        transcript = Transcript(
            engine="parakeet-tdt-0.6b-v2", language="en", text=""
        )
        assert transcript.words == []

    def test_transcriber_protocol_importable(self) -> None:
        """Transcriber interface must be importable from vsa.transcription.base."""
        from vsa.transcription.base import Transcriber

        # Protocol/ABC: anything with .transcribe(audio_path) -> Transcript
        # qualifies. Just verify the symbol exists.
        assert Transcriber is not None


class TestParakeetTranscriberConstruction:
    def test_class_importable(self) -> None:
        from vsa.transcription.parakeet import ParakeetTranscriber

        assert ParakeetTranscriber is not None

    def test_constructor_does_not_load_model(self) -> None:
        """Lazy-load contract: constructing the transcriber must not pull
        the model into memory. The ~2GB model is only loaded the first time
        ``transcribe`` is called."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        transcriber = ParakeetTranscriber()
        # The internal model handle must be unset until transcribe() runs.
        assert transcriber._model is None

    def test_engine_name_advertised(self) -> None:
        from vsa.transcription.parakeet import ParakeetTranscriber

        transcriber = ParakeetTranscriber()
        assert transcriber.engine == "parakeet-tdt-0.6b-v2"


def _write_silence_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> Path:
    """Write a silent 16-bit mono WAV of ``duration_sec`` to ``path``.

    Used by the chunking tests so we can drive a fake transcribe() with
    real WAV input without paying any model-load or model-inference
    cost. Silence keeps the file tiny on disk regardless of duration.
    """
    n_samples = int(round(sample_rate * duration_sec))
    samples = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return path


class _FakeHypothesis:
    """Stand-in for a NeMo TDT hypothesis. Mirrors the duck-typed shape
    that ``_hypotheses_to_text_and_words`` reads: ``.text`` (str) and
    ``.timestamp`` (dict with a ``"word"`` list)."""

    def __init__(self, text: str, words: list[dict[str, Any]]) -> None:
        self.text = text
        self.timestamp = {"word": words}


class _FakeAsrModel:
    """Fake NeMo ASR model for chunking tests. Records every transcribe()
    call's input audio path + the duration of that file. By default
    returns a single fake hypothesis with one word so we can verify
    timestamp-offset math without loading a real model."""

    def __init__(
        self,
        per_chunk_text: str = "x",
        per_chunk_words: list[dict[str, Any]] | None = None,
    ) -> None:
        self.calls: list[tuple[str, float]] = []
        self._per_chunk_text = per_chunk_text
        # Default per-chunk words: a single token at 0.0–0.1s so tests
        # can assert timestamp offsets are added correctly.
        self._per_chunk_words = (
            per_chunk_words
            if per_chunk_words is not None
            else [{"word": "x", "start": 0.0, "end": 0.1, "confidence": 0.9}]
        )

    def transcribe(self, paths: list[str], timestamps: bool = False) -> list[_FakeHypothesis]:
        assert timestamps is True, "ParakeetTranscriber must request word timestamps"
        assert len(paths) == 1, "Chunked path passes one chunk per call"
        with wave.open(paths[0], "rb") as f:
            duration = f.getnframes() / float(f.getframerate())
        self.calls.append((paths[0], duration))
        return [
            _FakeHypothesis(
                text=self._per_chunk_text,
                words=list(self._per_chunk_words),
            )
        ]


class TestParakeetTranscriberChunking:
    """Chunking lets the transcriber handle voice notes up to ~20 minutes
    without OOMing the 8 GB Fly machine. These tests inject a fake NeMo
    model so we can assert chunking behavior without loading the real
    ~2 GB Parakeet checkpoint."""

    def test_short_audio_uses_single_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audio shorter than the chunk threshold takes the original
        single-call code path with zero overhead. The fake model should
        see exactly one transcribe() call covering the full duration."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        wav = _write_silence_wav(tmp_path / "short.wav", duration_sec=5.0)
        monkeypatch.setenv("PARAKEET_CHUNK_SECONDS", "60")

        fake = _FakeAsrModel(per_chunk_text="hello")
        t = ParakeetTranscriber()
        t._model = fake  # type: ignore[assignment] -- bypass lazy load

        result = t.transcribe(wav)

        assert len(fake.calls) == 1
        # The single call sees the original file (full 5s duration).
        assert abs(fake.calls[0][1] - 5.0) < 0.01
        assert result.text == "hello"
        assert len(result.words) == 1
        # Single-pass: word timestamps are NOT offset.
        assert result.words[0].start == pytest.approx(0.0)

    def test_long_audio_is_chunked_into_sequential_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audio longer than the chunk threshold is split into ceil(duration
        / chunk_seconds) sequential transcribe() calls, each on a chunk
        whose duration is at most ``chunk_seconds``."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        # 7 chunks at 5s each = 35s of audio.
        wav = _write_silence_wav(tmp_path / "long.wav", duration_sec=35.0)
        monkeypatch.setenv("PARAKEET_CHUNK_SECONDS", "5")

        fake = _FakeAsrModel(per_chunk_text="seg")
        t = ParakeetTranscriber()
        t._model = fake  # type: ignore[assignment]

        result = t.transcribe(wav)

        assert len(fake.calls) == 7
        # Every chunk file is at most chunk_seconds long. The last chunk
        # may be shorter when duration doesn't divide evenly; here it's
        # exactly 5s because 35 / 5 = 7.
        for _, chunk_duration in fake.calls:
            assert chunk_duration <= 5.0 + 0.01
            assert chunk_duration > 0.0
        # Merged text concatenates per-chunk texts with single spaces.
        assert result.text == "seg seg seg seg seg seg seg"

    def test_long_audio_handles_partial_trailing_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When duration doesn't divide evenly the last chunk is shorter,
        not dropped. 12s audio with 5s chunks → [5s, 5s, 2s]."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        wav = _write_silence_wav(tmp_path / "uneven.wav", duration_sec=12.0)
        monkeypatch.setenv("PARAKEET_CHUNK_SECONDS", "5")

        fake = _FakeAsrModel()
        t = ParakeetTranscriber()
        t._model = fake  # type: ignore[assignment]

        t.transcribe(wav)

        assert len(fake.calls) == 3
        durations = sorted(d for _, d in fake.calls)
        # Two full chunks at 5s + one tail at ~2s.
        assert abs(durations[0] - 2.0) < 0.01
        assert abs(durations[1] - 5.0) < 0.01
        assert abs(durations[2] - 5.0) < 0.01

    def test_chunk_word_timestamps_are_offset_to_global_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Word-level timestamps from each chunk must be shifted by the
        chunk's start_sec so the merged transcript is in global audio
        time. Without this, prosody analysis (which consumes word
        timings) would see every chunk's words starting at 0.0."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        wav = _write_silence_wav(tmp_path / "long.wav", duration_sec=15.0)
        monkeypatch.setenv("PARAKEET_CHUNK_SECONDS", "5")

        # Each chunk reports a word at 0.5–0.8s in chunk-local time.
        # After offset, expected word starts are: 0.5, 5.5, 10.5.
        fake = _FakeAsrModel(
            per_chunk_words=[
                {"word": "w", "start": 0.5, "end": 0.8, "confidence": 0.9}
            ]
        )
        t = ParakeetTranscriber()
        t._model = fake  # type: ignore[assignment]

        result = t.transcribe(wav)

        assert len(result.words) == 3
        starts = [w.start for w in result.words]
        ends = [w.end for w in result.words]
        assert starts == pytest.approx([0.5, 5.5, 10.5])
        assert ends == pytest.approx([0.8, 5.8, 10.8])

    def test_invalid_chunk_seconds_env_falls_back_to_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bogus PARAKEET_CHUNK_SECONDS value must not crash the
        transcriber. It falls back to the 60s default — so a 30s file
        takes the single-pass path."""
        from vsa.transcription.parakeet import ParakeetTranscriber

        wav = _write_silence_wav(tmp_path / "med.wav", duration_sec=30.0)
        monkeypatch.setenv("PARAKEET_CHUNK_SECONDS", "not-a-number")

        fake = _FakeAsrModel()
        t = ParakeetTranscriber()
        t._model = fake  # type: ignore[assignment]

        t.transcribe(wav)

        # 30s ≤ 60s default → single call, not chunked.
        assert len(fake.calls) == 1


class TestParakeetTranscriberSmoke:
    """Smoke tests that exercise the real NeMo model. The fixture audio is
    a 1s 440 Hz sine — the model will produce a garbage / empty transcript,
    so we assert only on output shape, never on content."""

    def test_transcribe_returns_transcript_with_correct_engine_and_language(
        self, parakeet_transcriber, fixture_wav_path: Path
    ) -> None:
        result = parakeet_transcriber.transcribe(fixture_wav_path)

        assert result.engine == "parakeet-tdt-0.6b-v2"
        assert result.language == "en"

    def test_transcribe_returns_string_text_and_word_list(
        self, parakeet_transcriber, fixture_wav_path: Path
    ) -> None:
        result = parakeet_transcriber.transcribe(fixture_wav_path)

        assert isinstance(result.text, str)
        assert isinstance(result.words, list)
        # Every word entry, if any, conforms to the Word schema.
        for word in result.words:
            assert isinstance(word.w, str)
            assert isinstance(word.start, float)
            assert isinstance(word.end, float)
            assert isinstance(word.conf, float)
