"""ParakeetTranscriber: NeMo-backed wrapper around
``nvidia/parakeet-tdt-0.6b-v2``.

The 0.6B model weighs roughly 2GB on disk and several seconds to load.
Construction is therefore cheap and side-effect free — the model is pulled
into memory only on the first call to :meth:`transcribe`. This matches the
lazy pattern used by ``AcousticAnalyzer`` and lets the pipeline be
constructed (e.g. for FastAPI app startup) without paying the load cost
when no audio is being analyzed.

Chunked inference for long audio
--------------------------------
Parakeet TDT's Conformer encoder uses self-attention whose memory is
quadratic in sequence length. On CPU with 8 GB of RAM, single-pass
transcription is bounded to roughly 60–90 seconds of audio: a 13-minute
voice note OOM'd a Fly machine at 7.89 GB resident even though only the
~2 GB model itself was loaded. To handle the long voice notes the
service is built for (up to ~20 minutes), :meth:`transcribe`
auto-chunks audio longer than ``PARAKEET_CHUNK_SECONDS`` (default 60s)
into sequential calls to ``model.transcribe()`` and merges the results,
offsetting word timestamps by each chunk's start time.

Short audio (≤ chunk size) takes the original single-pass code path
with zero overhead.
"""

from __future__ import annotations

import gc
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

from vsa.schema import Transcript, Word

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
ENGINE_ID = "parakeet-tdt-0.6b-v2"
LANGUAGE = "en"  # Parakeet TDT 0.6B v2 is English-only.

CHUNK_SECONDS_ENV = "PARAKEET_CHUNK_SECONDS"
DEFAULT_CHUNK_SECONDS = 60.0


def _chunk_seconds() -> float:
    """Read the per-chunk duration in seconds from the environment.

    Default 60s sits well below the empirical OOM ceiling on shared-cpu-4x
    with 8 GB. Operators can tune higher on bigger machines (or lower if
    a future model bump pushes attention memory up). Invalid values fall
    back to the default rather than crashing the transcriber.
    """
    raw = os.environ.get(CHUNK_SECONDS_ENV)
    if raw is None:
        return DEFAULT_CHUNK_SECONDS
    try:
        v = float(raw)
        return v if v > 0.0 else DEFAULT_CHUNK_SECONDS
    except ValueError:
        return DEFAULT_CHUNK_SECONDS


def _audio_duration_seconds(audio_path: Path) -> float:
    """Return ``audio_path`` duration in seconds via stdlib ``wave``.

    Cheap (header-only read) and dependency-free. The Pipeline normalizes
    every input to 16 kHz mono PCM WAV before reaching the transcriber,
    so ``wave`` is always sufficient here.
    """
    with wave.open(str(audio_path), "rb") as f:
        return f.getnframes() / float(f.getframerate())


def _slice_to_wav(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    dst_path: Path,
) -> Path:
    """Write the audio slice ``[start_sec, end_sec)`` from ``audio_path``
    to a fresh WAV at ``dst_path``. Uses stdlib ``wave`` so we don't pay
    a librosa decode per chunk for the segmentation step itself."""
    with wave.open(str(audio_path), "rb") as src:
        sr = src.getframerate()
        sw = src.getsampwidth()
        nc = src.getnchannels()
        start_frame = max(0, int(round(start_sec * sr)))
        end_frame = min(src.getnframes(), int(round(end_sec * sr)))
        src.setpos(start_frame)
        frames = src.readframes(max(0, end_frame - start_frame))
    with wave.open(str(dst_path), "wb") as dst:
        dst.setnchannels(nc)
        dst.setsampwidth(sw)
        dst.setframerate(sr)
        dst.writeframes(frames)
    return dst_path


def _trim_malloc() -> None:
    """Force glibc's malloc to return free pages to the OS.

    Without this, RSS grows monotonically across ``model.transcribe()``
    calls even when Python correctly drops every reference: glibc keeps
    the freed pages on its own free list rather than ``munmap``-ing
    them, and ``gc.collect()`` only releases to glibc's allocator, not
    to the kernel. After ~10 sequential 60s-chunk transcribe calls
    that's enough to OOM an 8 GB Fly machine even though no individual
    chunk peaks anywhere near the limit.

    ``malloc_trim(0)`` is a glibc-specific syscall; it's a best-effort
    no-op on musl (Alpine), Windows, and macOS. We swallow the load
    error there so dev environments without glibc still work.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # No glibc on this platform (musl, Windows, macOS dev) — the
        # accumulation pattern this guards against is glibc-specific
        # anyway, so a no-op here is correct.
        pass


def _hypotheses_to_text_and_words(
    hypotheses: Any, offset_sec: float
) -> tuple[str, list[Word]]:
    """Parse NeMo's first-hypothesis text + word-level timestamps,
    shifting every ``start``/``end`` by ``offset_sec``. Returns empty
    text and word list when the model produced no hypothesis (e.g.
    silent chunk). Confidence defaults to 0.0 when absent."""
    if not hypotheses:
        return "", []
    hyp = hypotheses[0]
    text = getattr(hyp, "text", "") or ""

    timestamp = getattr(hyp, "timestamp", None) or {}
    word_entries = (
        timestamp.get("word", []) if isinstance(timestamp, dict) else []
    )
    words: list[Word] = []
    for entry in word_entries:
        w = entry.get("word") or entry.get("char") or ""
        start = float(entry.get("start", 0.0)) + offset_sec
        end = float(entry.get("end", entry.get("start", 0.0))) + offset_sec
        conf = float(entry.get("confidence", entry.get("conf", 0.0)) or 0.0)
        words.append(Word(w=w, start=start, end=end, conf=conf))
    return text, words


class ParakeetTranscriber:
    """Default transcription engine backed by NVIDIA NeMo's Parakeet TDT.

    Outputs a :class:`Transcript` with word-level timestamps. The model is
    lazy-loaded on the first :meth:`transcribe` call and cached for the
    lifetime of the instance.

    Long audio is auto-chunked into sequential ``model.transcribe()`` calls
    of at most ``PARAKEET_CHUNK_SECONDS`` (default 60s) each, so peak
    transcription memory stays roughly constant regardless of input
    duration. See module docstring for the rationale.
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
        chunk_sec = _chunk_seconds()
        duration_sec = _audio_duration_seconds(audio_path)

        # Short audio: original single-shot path. No chunking overhead,
        # bit-for-bit identical behavior to pre-chunking releases.
        if duration_sec <= chunk_sec:
            hypotheses = model.transcribe([str(audio_path)], timestamps=True)
            text, words = _hypotheses_to_text_and_words(
                hypotheses, offset_sec=0.0
            )
            return Transcript(
                engine=ENGINE_ID, language=LANGUAGE, text=text, words=words
            )

        # Long audio: tile [0, duration_sec) with non-overlapping chunks
        # of at most ``chunk_sec`` each, transcribe each chunk in a
        # separate ``model.transcribe()`` call, then merge. Sequential
        # (not batched) so peak memory stays at one chunk's worth of
        # encoder activations.
        text_parts: list[str] = []
        all_words: list[Word] = []
        with tempfile.TemporaryDirectory(prefix="vsa-parakeet-") as tmp:
            tmp_dir = Path(tmp)
            start_sec = 0.0
            chunk_idx = 0
            while start_sec < duration_sec:
                end_sec = min(start_sec + chunk_sec, duration_sec)
                chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.wav"
                _slice_to_wav(audio_path, start_sec, end_sec, chunk_path)

                hypotheses = model.transcribe(
                    [str(chunk_path)], timestamps=True
                )
                chunk_text, chunk_words = _hypotheses_to_text_and_words(
                    hypotheses, offset_sec=start_sec
                )
                if chunk_text:
                    text_parts.append(chunk_text)
                all_words.extend(chunk_words)

                # Drop the chunk file as soon as it's been consumed and
                # nudge the GC so encoder activations from this call
                # don't accumulate into the next chunk's peak. Then
                # tell glibc to give the freed pages back to the kernel
                # — without that, RSS grows ~600 MB per chunk even with
                # gc.collect() and OOMs around chunk 11 on 8 GB Fly.
                chunk_path.unlink(missing_ok=True)
                gc.collect()
                _trim_malloc()

                start_sec += chunk_sec
                chunk_idx += 1

        return Transcript(
            engine=ENGINE_ID,
            language=LANGUAGE,
            text=" ".join(text_parts).strip(),
            words=all_words,
        )
