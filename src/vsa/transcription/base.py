"""Transcriber interface: any engine that can map an audio file to a
``Transcript`` (text + word-level timestamps + language) qualifies."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from vsa.schema import Transcript


@runtime_checkable
class Transcriber(Protocol):
    """A speech-to-text engine.

    Implementations must produce a ``Transcript`` with the engine name,
    language code, full text, and a (possibly empty) list of word-level
    timestamps. Failures should propagate as exceptions; the pipeline is
    responsible for translating them into the partial-success contract.
    """

    def transcribe(self, audio_path: Path) -> Transcript:
        ...
