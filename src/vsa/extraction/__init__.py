"""LLM-based structured extraction from voice-note transcripts.

The :class:`vsa.extraction.llm.LlmExtractor` runs a local LLM (Qwen3.5-9B-Instruct
by default, llama-cpp-python runtime) against a transcript and returns
an :class:`ExtractionResult` with title / summary / type / tags /
people / places / projects / businesses / tech_stack / tasks / etc.

The extractor's contract:

* Lazy model load on the first :meth:`extract` call (the GGUF file is
  several GB; we don't pay that cost at app startup).
* JSON output is grammar-constrained at the sampler so the model
  cannot emit malformed JSON. Pydantic validation is a belt-and-
  suspenders catch on top of that.
* Per-request override of the voice-note type catalog is supported via
  the ``voice_note_types`` argument; the default catalog (idea /
  journal / task / meditation + an ``other`` fallback) applies when no
  override is provided.

See :mod:`vsa.extraction.types` for the default catalog,
:mod:`vsa.extraction.schema` for the output types, and
:mod:`vsa.extraction.llm` for the runtime wrapper.
"""

from vsa.extraction.schema import ExtractionResult, Task
from vsa.extraction.types import (
    DEFAULT_FALLBACK_TYPE,
    DEFAULT_VOICE_NOTE_TYPES,
    VoiceNoteType,
)

__all__ = [
    "DEFAULT_FALLBACK_TYPE",
    "DEFAULT_VOICE_NOTE_TYPES",
    "ExtractionResult",
    "Task",
    "VoiceNoteType",
]
