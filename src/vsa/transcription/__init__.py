"""Transcription engines (Parakeet, faster-whisper).

The ``make_transcriber()`` factory is the single switch operators flip
to choose an engine — read at Pipeline construction time from the
``TRANSCRIBER_ENGINE`` env var (default: ``parakeet``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vsa.transcription.base import Transcriber


_VALID_ENGINES = ("parakeet", "whisper")


def make_transcriber() -> "Transcriber":
    """Construct a transcriber based on the ``TRANSCRIBER_ENGINE`` env var.

    - ``parakeet`` (default, unset): :class:`ParakeetTranscriber` — the
      production-default NeMo-backed engine baked into the Docker image.
    - ``whisper``: :class:`FasterWhisperTranscriber` — the alternative
      faster-whisper engine. The model is **not** baked into the image;
      it lazy-downloads on first use.
    - Anything else: ``ValueError`` so a typo can't silently fall back
      to an engine the operator did not intend.
    """
    engine = os.environ.get("TRANSCRIBER_ENGINE", "parakeet").strip().lower()
    if engine == "parakeet":
        # Imported lazily so a Whisper-only deployment doesn't pay for
        # NeMo importing at module-load time, and vice versa.
        from vsa.transcription.parakeet import ParakeetTranscriber

        return ParakeetTranscriber()
    if engine == "whisper":
        from vsa.transcription.whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber()
    raise ValueError(
        f"Unknown TRANSCRIBER_ENGINE={engine!r}; "
        f"expected one of {_VALID_ENGINES}."
    )


__all__ = ["make_transcriber"]
