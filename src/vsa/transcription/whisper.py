"""FasterWhisperTranscriber: CTranslate2-backed Whisper alternative.

Selected via the ``TRANSCRIBER_ENGINE=whisper`` env var. The model size is
controlled by ``WHISPER_MODEL`` (default ``small``) and is read at
construction time so the engine string can be set without paying the
model-load cost.

Construction is cheap and side-effect free — the model is pulled into
memory only on the first call to :meth:`transcribe`. This mirrors
``ParakeetTranscriber`` and lets the pipeline be constructed (e.g. for
FastAPI app startup) without paying the cold-start cost when no audio is
being analyzed.

Unlike Parakeet, the Whisper model is **not** baked into the Docker
image — it lazy-downloads on first use. See PR #11 for the deviation
rationale (production-default is Parakeet; bloating the image with a
secondary engine isn't worth it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vsa.schema import Transcript, Word

DEFAULT_MODEL = "small"


class FasterWhisperTranscriber:
    """Alternative transcription engine backed by faster-whisper.

    Outputs a :class:`Transcript` with word-level timestamps. The model
    is lazy-loaded on the first :meth:`transcribe` call and cached for
    the lifetime of the instance. The model size is read from the
    ``WHISPER_MODEL`` env var at construction time (default: ``small``).
    """

    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = model_size or os.environ.get(
            "WHISPER_MODEL", DEFAULT_MODEL
        )
        self.engine: str = f"faster-whisper-{self._model_size}"
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            # Imported lazily so that simply constructing the transcriber
            # (or importing this module) does not pull faster-whisper /
            # ctranslate2 into memory.
            from faster_whisper import WhisperModel

            # ``device="cpu"`` keeps deployments portable: the CTranslate2
            # GPU path requires cuBLAS/cuDNN at runtime, which is not in
            # the base Docker image. Self-hosters with GPUs can subclass
            # or pass an env override later.
            self._model = WhisperModel(self._model_size, device="cpu")
        return self._model

    def release(self) -> None:
        """Drop the loaded Whisper model so its weights become eligible
        for GC. Mirrors :meth:`ParakeetTranscriber.release`; the next
        :meth:`transcribe` call reloads lazily.
        """
        self._model = None

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._load()
        # ``word_timestamps=True`` makes faster-whisper attach per-segment
        # word objects with .word/.start/.end/.probability — we map
        # probability -> conf to keep parity with Parakeet's word schema.
        segments, info = model.transcribe(
            str(audio_path), word_timestamps=True
        )

        text_parts: list[str] = []
        words: list[Word] = []
        # ``segments`` is a generator — iterating consumes it and triggers
        # the actual transcription work.
        for seg in segments:
            seg_text = getattr(seg, "text", "") or ""
            text_parts.append(seg_text)
            for w in getattr(seg, "words", None) or []:
                words.append(
                    Word(
                        w=getattr(w, "word", "") or "",
                        start=float(getattr(w, "start", 0.0) or 0.0),
                        end=float(getattr(w, "end", 0.0) or 0.0),
                        conf=float(getattr(w, "probability", 0.0) or 0.0),
                    )
                )

        # ``info.language`` is whisper's auto-detected language code.
        language = getattr(info, "language", "") or ""

        return Transcript(
            engine=self.engine,
            language=language,
            text="".join(text_parts).strip(),
            words=words,
        )
