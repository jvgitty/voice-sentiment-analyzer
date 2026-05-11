"""Pydantic models for the LLM extraction output.

The schema mirrors the JSON the LLM is asked to emit in the system
prompt. Every field is either required-string, optional-string-or-null,
or a list whose default is empty — matching the prompt's "all keys
required; use empty string, empty array, or null where nothing applies"
contract.

Pydantic validation is the second line of defense behind grammar-
constrained sampling at the LLM level. If the LLM's output deviates
from this schema, the extractor raises and the Pipeline records a
partial-success error rather than 500ing the whole request.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single extracted action item.

    ``due`` is a strict YYYY-MM-DD when the transcript names a date,
    otherwise ``None``. ``priority`` is one of low/medium/high when the
    speaker explicitly signals urgency, otherwise ``None`` — we do not
    let the LLM infer priority from tone.
    """

    text: str = Field(min_length=1)
    due: Optional[str] = None  # YYYY-MM-DD or None
    priority: Optional[Literal["low", "medium", "high"]] = None


class ExtractionResult(BaseModel):
    """Structured extraction over a single voice-note transcript.

    Field semantics match the system-prompt contract:

    * ``title``: 5-10 words, specific, no colons or semicolons.
    * ``summary``: a 1-2 sentence summary capped at ``summary_max_words``
      tokens (passed into the prompt at extraction time).
    * ``type``: the name of one entry from the active voice-note type
      catalog (default or per-request override), or the fallback type
      when nothing fits.
    * ``mood``: short phrase describing the speaker's mood. Optional.
    * ``voice_note_location``: where the speaker is physically located,
      *only* if explicitly stated. Distinct from ``locations`` below.
    * ``tags``: 3-8 Obsidian-style hashtags (lowercase-hyphens, no
      leading ``#``).
    * ``themes``: 1-5 longer conceptual descriptors.
    * ``locations``: place names *mentioned* in the transcript (not the
      speaker's own location).
    * ``people``, ``projects``, ``businesses``, ``tech_stack``: named
      entity buckets.
    * ``tasks``: explicit to-dos. See :class:`Task`.

    The schema version is exposed so downstream consumers can branch on
    shape changes without inferring from field presence.
    """

    schema_version: str = "1.0"
    title: str
    summary: str
    type: str
    mood: Optional[str] = None
    voice_note_location: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    businesses: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
