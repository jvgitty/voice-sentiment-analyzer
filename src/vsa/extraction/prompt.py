"""System and user prompt construction for the LLM extractor.

Kept separate from the runtime wrapper so the prompt is reviewable in
isolation. Edits to the prompt are far more frequent than edits to the
llama-cpp-python plumbing; splitting the files keeps prompt diffs
small and focused.

The system prompt is rendered at request time, not at module load,
because the voice-note type catalog and the summary length cap are
per-request overrides.
"""

from __future__ import annotations

from vsa.extraction.types import (
    DEFAULT_FALLBACK_TYPE,
    DEFAULT_VOICE_NOTE_TYPES,
    VoiceNoteType,
)


def _render_type_list(types: list[VoiceNoteType]) -> str:
    """Format the voice-note type catalog into a human-readable bullet
    list for the system prompt. Each line is ``- name: description``
    so the LLM can read both the canonical token to emit and the
    semantic anchor that distinguishes it from siblings.
    """
    return "\n".join(f'  - "{t.name}": {t.description}' for t in types)


def build_system_prompt(
    voice_note_types: list[VoiceNoteType] | None = None,
    fallback_type: str = DEFAULT_FALLBACK_TYPE,
    summary_max_words: int = 50,
) -> str:
    """Render the system prompt for a single extraction call.

    Args:
        voice_note_types: Per-request override for the type catalog.
            When ``None``, the default catalog from
            :mod:`vsa.extraction.types` is used.
        fallback_type: The type name the LLM MUST emit when nothing in
            the catalog fits. Defaults to ``other``.
        summary_max_words: Hard cap on the ``summary`` field. Defaults
            to 50 — long enough to be useful, short enough to not eat
            half the response budget.

    Returns:
        The fully-rendered system prompt as a single string.
    """
    types = voice_note_types or DEFAULT_VOICE_NOTE_TYPES
    type_list = _render_type_list(types)

    return f"""You are an enrichment assistant for a personal voice-note pipeline.
You receive the transcript of one voice note (a single English-speaking
person rambling, no multiple speakers) and return STRICT JSON matching the
schema below. You do NOT add commentary, prose, markdown, or code fences.

Principles (non-negotiable):
1. Extraction is the job. Prefer richer extraction over sparse. When a
   field could reasonably be populated from the transcript, populate it.
2. Atomic notes. Treat the transcript as a self-contained unit. Do not
   speculate about other notes or the speaker's broader context.
3. Stay faithful to the transcript. Do not invent people, places,
   projects, tasks, or technologies that the speaker did not mention.

Output JSON schema (all keys required; use empty string, empty array, or
null where nothing applies):

{{
  "title":                string,           // 5-10 words, specific, no colons or semicolons
  "summary":              string,           // <= {summary_max_words} words, 1-2 sentences of key points
  "type":                 string,           // one of the allowed types listed below
  "mood":                 string | null,    // short phrase describing the speaker's mood
  "voice_note_location":  string | null,    // set only if the speaker states where they physically are
  "tags":                 string[],         // 3-8 Obsidian-style hashtags; lowercase-hyphens, no "#", no spaces
  "themes":               string[],         // 1-5 longer conceptual descriptors (what the note is trying to get at)
  "locations":            string[],         // locations MENTIONED in the transcript (not the speaker's physical location)
  "people":               string[],         // people mentioned
  "projects":             string[],         // projects mentioned
  "businesses":           string[],         // businesses / organizations mentioned
  "tech_stack":           string[],         // technologies, tools, frameworks, languages mentioned
  "tasks":                Task[]            // explicit to-dos, action items, or reminders
}}

Task = {{
  "text":     string,                      // the task itself
  "due":      string | null,               // YYYY-MM-DD if a date is stated, else null
  "priority": "low" | "medium" | "high" | null
}}

Allowed values for "type":
{type_list}

If none of the types fits, use "{fallback_type}". You MUST pick exactly one.

Formatting rules for string values:
- Title: no ":" or ";" characters. If needed, rewrite.
- Tags: lowercase, words joined with "-", no "#" prefix, no spaces.
- people/locations/projects/businesses: the entity's natural name
  ("Mom", "New York", "Project Aeryn"). Do not add "[[" or "]]".

Return ONLY the JSON object. No preamble. No trailing text."""


def build_user_prompt(transcript: str) -> str:
    """Render the user-turn message wrapping the raw transcript.

    The triple-quote fence is so the model can see exactly where the
    transcript boundaries are even if the transcript itself contains
    quote marks or other markup. The trailing "Return the JSON object
    now." is a small but measurable anchor against the model adding
    preamble.
    """
    return f'Transcript:\n"""\n{transcript}\n"""\n\nReturn the JSON object now.'
