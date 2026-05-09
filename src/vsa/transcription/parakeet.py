"""ParakeetTranscriber: NeMo-backed wrapper around
``nvidia/parakeet-tdt-0.6b-v2``.

The 0.6B model weighs roughly 2GB on disk and several seconds to load.
Construction is therefore cheap and side-effect free — the model is pulled
into memory only on the first call to :meth:`transcribe`. This matches the
lazy pattern used by ``AcousticAnalyzer`` and lets the pipeline be
constructed (e.g. for FastAPI app startup) without paying the load cost
when no audio is being analyzed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vsa.schema import Transcript, Word

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
ENGINE_ID = "parakeet-tdt-0.6b-v2"
LANGUAGE = "en"  # Parakeet TDT 0.6B v2 is English-only.


class ParakeetTranscriber:
    """Default transcription engine backed by NVIDIA NeMo's Parakeet TDT.

    Outputs a :class:`Transcript` with word-level timestamps. The model is
    lazy-loaded on the first :meth:`transcribe` call and cached for the
    lifetime of the instance.
    """

    engine: str = ENGINE_ID

    def __init__(self) -> None:
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            # Imported lazily so that simply constructing the transcriber
            # (or importing this module) does not pull NeMo into memory.
            import nemo.collections.asr as nemo_asr

            self._model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        return self._model

    def release(self) -> None:
        """Drop the loaded NeMo model so its weights become eligible for
        GC. Used by Pipeline to reclaim ~2 GB of resident memory after the
        transcription phase, since downstream analyzers do not call back
        into the transcriber. The next :meth:`transcribe` call reloads
        lazily.
        """
        self._model = None

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._load()
        hypotheses = model.transcribe([str(audio_path)], timestamps=True)
        if not hypotheses:
            return Transcript(
                engine=ENGINE_ID, language=LANGUAGE, text="", words=[]
            )

        hyp = hypotheses[0]
        text = getattr(hyp, "text", "") or ""

        words: list[Word] = []
        timestamp = getattr(hyp, "timestamp", None) or {}
        # NeMo's RNNT/TDT hypotheses expose timestamps as a dict whose
        # ``word`` entry is a list of ``{word, start, end, ...}`` dicts.
        # Confidence is not always present at word granularity; default
        # to 0.0 when missing rather than fabricating a value.
        word_entries = timestamp.get("word", []) if isinstance(timestamp, dict) else []
        for entry in word_entries:
            w = entry.get("word") or entry.get("char") or ""
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
            conf = float(entry.get("confidence", entry.get("conf", 0.0)) or 0.0)
            words.append(Word(w=w, start=start, end=end, conf=conf))

        return Transcript(
            engine=ENGINE_ID,
            language=LANGUAGE,
            text=text,
            words=words,
        )
