"""Tests for the Transcriber interface and ParakeetTranscriber."""

from pathlib import Path
from typing import Protocol, runtime_checkable

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
