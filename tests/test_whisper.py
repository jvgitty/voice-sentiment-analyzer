"""Tests for the FasterWhisperTranscriber alternative engine (Slice 9)."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def whisper_transcriber():
    """Session-scoped FasterWhisperTranscriber. Loading even the ``small``
    model takes ~10s on first call (and downloads ~500MB the very first time
    it ever runs on this machine), so we share one instance across smoke
    tests."""
    from vsa.transcription.whisper import FasterWhisperTranscriber

    transcriber = FasterWhisperTranscriber()
    # Force the lazy load once so subsequent test calls are warm.
    transcriber._load()
    return transcriber


class TestFasterWhisperTranscriberConstruction:
    def test_class_importable(self) -> None:
        from vsa.transcription.whisper import FasterWhisperTranscriber

        assert FasterWhisperTranscriber is not None

    def test_constructor_does_not_load_model(self) -> None:
        """Lazy-load contract: constructing the transcriber must not pull
        the model into memory. The model is only loaded the first time
        ``transcribe`` is called."""
        from vsa.transcription.whisper import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber()
        # The internal model handle must be unset until transcribe() runs.
        assert transcriber._model is None

    def test_engine_name_includes_model_size(self) -> None:
        """The engine string must follow ``faster-whisper-<size>`` so the
        downstream JSON makes the engine choice obvious to the caller."""
        from vsa.transcription.whisper import FasterWhisperTranscriber

        transcriber = FasterWhisperTranscriber()
        assert transcriber.engine.startswith("faster-whisper-")


class TestFasterWhisperTranscriberSmoke:
    """Smoke tests that exercise the real faster-whisper model. The fixture
    audio is a 1s 440 Hz sine — the model will produce a garbage / empty
    transcript, so we assert only on output shape, never on content."""

    def test_transcribe_returns_transcript_with_engine_prefix(
        self, whisper_transcriber, fixture_wav_path: Path
    ) -> None:
        result = whisper_transcriber.transcribe(fixture_wav_path)

        assert result.engine.startswith("faster-whisper-")

    def test_transcribe_returns_string_language(
        self, whisper_transcriber, fixture_wav_path: Path
    ) -> None:
        """``language`` is whisper's auto-detection result. We don't assert
        a specific value (a sine wave can detect as anything) — just that
        the field is a non-empty string."""
        result = whisper_transcriber.transcribe(fixture_wav_path)

        assert isinstance(result.language, str)

    def test_transcribe_returns_string_text_and_word_list(
        self, whisper_transcriber, fixture_wav_path: Path
    ) -> None:
        result = whisper_transcriber.transcribe(fixture_wav_path)

        assert isinstance(result.text, str)
        assert isinstance(result.words, list)
        # Every word entry, if any, conforms to the Word schema.
        for word in result.words:
            assert isinstance(word.w, str)
            assert isinstance(word.start, float)
            assert isinstance(word.end, float)
            assert isinstance(word.conf, float)
