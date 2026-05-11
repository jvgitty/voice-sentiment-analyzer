"""Voice-note type catalog: defaults plus the per-request override type.

A voice-note type is a short categorical label (e.g. ``idea``,
``journal``, ``task``) paired with a one-line description that anchors
the LLM's classification. Callers can override the default catalog on
a per-request basis by including a ``voice_note_types`` array in the
:class:`vsa.schema.AnalyzeRequest` body.

This file is deliberately tiny — it's the public extension point for
clients that want their own taxonomy without forking the prompt itself.
The multi-tenant per-client persistent overrides described in
``docs/ROADMAP.md`` build on top of this primitive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceNoteType(BaseModel):
    """A single user-defined voice-note type entry.

    ``name`` is what the LLM is asked to emit verbatim in the
    extraction's ``type`` field. ``description`` is the disambiguating
    sentence shown to the LLM so it knows when to pick this type over
    a sibling. Keep descriptions concrete and mutually-exclusive — that
    is what drives classification accuracy more than any prompt trick.
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


# Default catalog used when AnalyzeRequest.voice_note_types is omitted.
# These four were chosen as the smallest set that covers the typical
# personal-voice-note use cases (capture, reflect, act, contemplate).
# Clients with different taxonomies override per-request.
DEFAULT_VOICE_NOTE_TYPES: list[VoiceNoteType] = [
    VoiceNoteType(
        name="idea",
        description=(
            "A captured spark of inspiration, observation, or insight. "
            "The speaker is thinking out loud about something new."
        ),
    ),
    VoiceNoteType(
        name="journal",
        description=(
            "Personal reflection — processing events, feelings, or "
            "thoughts about the day. Inward-facing rather than action-"
            "oriented."
        ),
    ),
    VoiceNoteType(
        name="task",
        description=(
            "An actionable item, to-do, or reminder. The speaker is "
            "noting something they need to do or follow up on."
        ),
    ),
    VoiceNoteType(
        name="meditation",
        description=(
            "A contemplative or mindful reflection — broader than a "
            "journal entry, often about values, principles, or larger "
            "patterns rather than specific events."
        ),
    ),
]

# Fallback type the LLM picks when no entry in the catalog fits. Kept
# separate from the catalog itself so per-request overrides don't have
# to remember to include a fallback themselves.
DEFAULT_FALLBACK_TYPE: str = "other"
