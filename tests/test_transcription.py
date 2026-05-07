"""Tests for the Transcriber interface and ParakeetTranscriber."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest


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
