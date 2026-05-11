"""Unit tests for system/user prompt construction.

These run without the LLM loaded — we just verify the prompt-rendering
logic that flows the voice-note type catalog, fallback type, and
summary length cap into the system prompt.
"""

from vsa.extraction.prompt import build_system_prompt, build_user_prompt
from vsa.extraction.types import (
    DEFAULT_FALLBACK_TYPE,
    DEFAULT_VOICE_NOTE_TYPES,
    VoiceNoteType,
)


class TestBuildSystemPrompt:
    def test_default_catalog_includes_all_four_types(self) -> None:
        """The four default types (idea/journal/task/meditation) must
        all appear in the rendered prompt when no override is passed."""
        prompt = build_system_prompt()

        for t in DEFAULT_VOICE_NOTE_TYPES:
            assert f'"{t.name}"' in prompt, t.name
            # The description must also render — otherwise the LLM has
            # nothing to disambiguate sibling types from.
            assert t.description in prompt, t.name

    def test_default_fallback_type_is_referenced(self) -> None:
        prompt = build_system_prompt()
        assert f'use "{DEFAULT_FALLBACK_TYPE}"' in prompt

    def test_custom_voice_note_types_override_defaults(self) -> None:
        """A per-request override replaces the entire default catalog —
        the defaults must NOT appear in the rendered prompt when an
        override is provided."""
        custom = [
            VoiceNoteType(
                name="legal-call",
                description="A summary of a call with a legal client.",
            ),
            VoiceNoteType(
                name="case-note",
                description="An observation about an active case.",
            ),
        ]
        prompt = build_system_prompt(voice_note_types=custom)

        assert '"legal-call"' in prompt
        assert '"case-note"' in prompt
        # Defaults must be gone — caller asked for their own taxonomy.
        for t in DEFAULT_VOICE_NOTE_TYPES:
            assert f'"{t.name}"' not in prompt, t.name

    def test_custom_fallback_type_propagates(self) -> None:
        prompt = build_system_prompt(fallback_type="uncategorized")
        assert 'use "uncategorized"' in prompt
        # The default fallback must not appear as a fallback when one
        # is explicitly overridden. (It MAY appear in catalog entries
        # if a future test adds it there, so we check the specific
        # "use \"x\"" phrasing rather than substring "other".)
        assert f'use "{DEFAULT_FALLBACK_TYPE}"' not in prompt

    def test_summary_max_words_renders_into_schema_comment(self) -> None:
        prompt = build_system_prompt(summary_max_words=30)
        assert "<= 30 words" in prompt

    def test_strict_json_only_instruction_present(self) -> None:
        """The prompt must keep the 'STRICT JSON, no commentary or
        code fences' rule visible regardless of catalog overrides —
        without it the model occasionally wraps JSON in ```json fences."""
        prompt = build_system_prompt()
        assert "STRICT JSON" in prompt
        assert "no preamble" in prompt.lower() or "no trailing text" in prompt.lower()


class TestBuildUserPrompt:
    def test_transcript_is_wrapped_in_triple_quote_fence(self) -> None:
        """The triple-quote fence is what lets the model see exactly
        where the transcript ends, especially when the transcript
        itself contains quotes or markup."""
        prompt = build_user_prompt("hello world")
        assert '"""\nhello world\n"""' in prompt

    def test_return_json_now_anchor_present(self) -> None:
        """The closing 'Return the JSON object now.' line is a small
        but measurable anchor against the model adding preamble to its
        response."""
        prompt = build_user_prompt("any transcript")
        assert "Return the JSON object now." in prompt
