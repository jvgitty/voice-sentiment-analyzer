"""LlmExtractor: llama-cpp-python wrapper around the Qwen3.5-9B-Instruct
GGUF model used for transcript → structured-JSON extraction.

The model is lazy-loaded on the first :meth:`extract` call and cached
for the lifetime of the instance, matching the lazy pattern used by
the transcribers. Pipeline construction at FastAPI startup stays cheap.

JSON output is constrained two ways:

1. **At the sampler** via llama-cpp-python's
   ``response_format={"type": "json_object"}``, which tells the model
   to emit valid JSON only. The model card for Qwen3.5-9B-Instruct
   reports IFEval 91.5, so the model is already strong at structured
   output; this is belt to its suspenders.
2. **At parse time** via Pydantic's :class:`ExtractionResult.model_validate`.
   If the LLM somehow emits valid JSON that nonetheless violates our
   schema (wrong types, missing required fields), this raises and the
   Pipeline records a partial-success error.

Configuration via env vars
--------------------------
``LLM_MODEL_PATH``       — absolute path to the GGUF file.
                           Default: ``/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf``.
``LLM_CONTEXT_SIZE``     — context window in tokens. Default: ``8192``.
                           A 20-min transcript runs ~4-5K tokens, and the
                           system prompt is ~1K, so 8K leaves room for
                           the JSON output.
``LLM_THREADS``          — n_threads for CPU inference.
                           Default: ``0`` (let llama.cpp auto-pick).
``LLM_TEMPERATURE``      — sampling temperature. Default: ``0.2``.
                           Low because we want consistent extraction,
                           not creative writing.
``LLM_MAX_OUTPUT_TOKENS`` — cap on JSON output length. Default: ``2048``.
                           Plenty for our schema even with many tasks.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any

from vsa.extraction.prompt import build_system_prompt, build_user_prompt
from vsa.extraction.schema import ExtractionResult
from vsa.extraction.types import DEFAULT_FALLBACK_TYPE, VoiceNoteType


DEFAULT_MODEL_PATH = "/opt/models/qwen3.5-9b-instruct-q4_k_m.gguf"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_THREADS = 0  # 0 = let llama.cpp auto-detect
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing or
    invalid values rather than crashing the extractor."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class LlmExtractor:
    """Local-LLM extractor returning a strict :class:`ExtractionResult`.

    Construction is cheap and side-effect free. The ~5 GB GGUF model is
    loaded on the first :meth:`extract` call and cached. Tests inject a
    stub extractor via the Pipeline's constructor to skip that load.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_path = os.environ.get(
            "LLM_MODEL_PATH", DEFAULT_MODEL_PATH
        )
        self._context_size = _env_int(
            "LLM_CONTEXT_SIZE", DEFAULT_CONTEXT_SIZE
        )
        self._threads = _env_int("LLM_THREADS", DEFAULT_THREADS)
        self._temperature = _env_float(
            "LLM_TEMPERATURE", DEFAULT_TEMPERATURE
        )
        self._max_output_tokens = _env_int(
            "LLM_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS
        )

    def _load(self) -> Any:
        if self._model is None:
            # Imported lazily so that simply constructing the extractor
            # (or importing this module) does not pull llama-cpp-python
            # — and through it, several hundred MB of native libs —
            # into the process at app startup.
            from llama_cpp import Llama

            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._context_size,
                n_threads=self._threads,
                # Force CPU. Production deployment is shared-cpu Fly; if
                # someone runs this on a GPU box they can override with
                # n_gpu_layers via a future env var.
                n_gpu_layers=0,
                # Quiet llama.cpp's startup banner from polluting logs.
                # Errors still surface via exceptions.
                verbose=False,
            )
        return self._model

    def release(self) -> None:
        """Drop the loaded LLM so its weights are eligible for GC.

        Mirrors :meth:`ParakeetTranscriber.release`. Useful if a future
        pipeline phase needs the memory back, or when running the
        extractor as a one-shot CLI command. The next :meth:`extract`
        call lazily reloads.
        """
        self._model = None
        gc.collect()

    def extract(
        self,
        transcript: str,
        voice_note_types: list[VoiceNoteType] | None = None,
        fallback_type: str = DEFAULT_FALLBACK_TYPE,
        summary_max_words: int = 50,
    ) -> ExtractionResult:
        """Run extraction on a transcript and return a validated result.

        Args:
            transcript: The plain-text transcript of the voice note.
            voice_note_types: Per-request override for the type catalog.
                When ``None``, the default catalog is used.
            fallback_type: Type name to use when nothing in the catalog
                fits. Defaults to ``other``.
            summary_max_words: Hard cap on the ``summary`` field's
                word count. Communicated to the LLM in the prompt.

        Raises:
            RuntimeError: When the LLM returns no content.
            json.JSONDecodeError: When the LLM emits non-JSON despite
                the response-format constraint.
            pydantic.ValidationError: When the JSON parses but doesn't
                match :class:`ExtractionResult`.
        """
        model = self._load()
        system = build_system_prompt(
            voice_note_types=voice_note_types,
            fallback_type=fallback_type,
            summary_max_words=summary_max_words,
        )
        user = build_user_prompt(transcript)

        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            # JSON-object mode: llama.cpp constrains the sampler so the
            # output is guaranteed to parse as JSON. Pydantic validates
            # the parsed object against our schema below.
            response_format={"type": "json_object"},
        )

        # llama-cpp-python's create_chat_completion returns an OpenAI-
        # shaped dict: ``{choices: [{message: {content: "..."}}]}``.
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(
                "LLM returned no choices; cannot extract"
            )
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(
                "LLM returned empty content; cannot extract"
            )

        parsed = json.loads(content)
        return ExtractionResult.model_validate(parsed)
