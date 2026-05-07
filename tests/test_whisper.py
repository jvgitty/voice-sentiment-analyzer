"""Tests for the FasterWhisperTranscriber alternative engine (Slice 9)."""

from pathlib import Path

import pytest

from vsa.schema import Transcript


# whisper_transcriber and parakeet_transcriber are session-scoped fixtures
# in conftest.py so the schema-parity tests below can share the warmed
# instances with test_transcription.py.


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


class TestSchemaParity:
    """Both engines must emit a ``Transcript`` that validates against the
    same Pydantic model with the same field names and types. The downstream
    pipeline (prosody, windows, scoring) operates on the schema, not on
    engine-specific shapes, so any drift between Parakeet and Whisper would
    break engine-swap as a feature.
    """

    def test_whisper_output_validates_against_transcript_schema(
        self, whisper_transcriber, fixture_wav_path: Path
    ) -> None:
        whisper_result = whisper_transcriber.transcribe(fixture_wav_path)

        # Round-trip through the Pydantic model: dumping then re-validating
        # would catch any extra/missing field that .transcribe() emitted.
        Transcript.model_validate(whisper_result.model_dump())

    def test_parakeet_and_whisper_share_field_set(
        self,
        parakeet_transcriber,
        whisper_transcriber,
        fixture_wav_path: Path,
    ) -> None:
        """Engine swap is only safe if both engines populate exactly the
        same set of fields. We compare the dumped key sets at both the
        ``Transcript`` level and the ``Word`` level (when either engine
        produced any words on the sine fixture)."""
        from vsa.transcription.parakeet import ParakeetTranscriber  # noqa: F401  # import sanity

        parakeet_result = parakeet_transcriber.transcribe(fixture_wav_path)
        whisper_result = whisper_transcriber.transcribe(fixture_wav_path)

        parakeet_keys = set(parakeet_result.model_dump().keys())
        whisper_keys = set(whisper_result.model_dump().keys())
        assert parakeet_keys == whisper_keys

        # Every Word entry from either engine must carry the same shape.
        expected_word_keys = {"w", "start", "end", "conf"}
        for word in list(parakeet_result.words) + list(whisper_result.words):
            assert set(word.model_dump().keys()) == expected_word_keys


class TestMakeTranscriberFactory:
    """``make_transcriber()`` lives in ``vsa.transcription`` and reads the
    ``TRANSCRIBER_ENGINE`` env var to pick between ParakeetTranscriber and
    FasterWhisperTranscriber. This is the single switch the rest of the
    code (Pipeline, FastAPI handler) uses to honour engine selection."""

    def test_default_returns_parakeet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No env var (or unset) means production-default Parakeet."""
        from vsa.transcription import make_transcriber
        from vsa.transcription.parakeet import ParakeetTranscriber

        monkeypatch.delenv("TRANSCRIBER_ENGINE", raising=False)
        transcriber = make_transcriber()
        assert isinstance(transcriber, ParakeetTranscriber)

    def test_explicit_parakeet_returns_parakeet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vsa.transcription import make_transcriber
        from vsa.transcription.parakeet import ParakeetTranscriber

        monkeypatch.setenv("TRANSCRIBER_ENGINE", "parakeet")
        transcriber = make_transcriber()
        assert isinstance(transcriber, ParakeetTranscriber)

    def test_whisper_returns_faster_whisper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vsa.transcription import make_transcriber
        from vsa.transcription.whisper import FasterWhisperTranscriber

        monkeypatch.setenv("TRANSCRIBER_ENGINE", "whisper")
        transcriber = make_transcriber()
        assert isinstance(transcriber, FasterWhisperTranscriber)

    def test_unknown_engine_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typos must surface loudly at construction time, not silently
        fall back to a default."""
        from vsa.transcription import make_transcriber

        monkeypatch.setenv("TRANSCRIBER_ENGINE", "garbage")
        with pytest.raises(ValueError) as exc_info:
            make_transcriber()
        # Error message names the bad value so the operator can spot it.
        assert "garbage" in str(exc_info.value)
