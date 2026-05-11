"""Unit tests for extraction schema validation.

These exercise the Pydantic constraints on ExtractionResult and Task
without touching the LLM. The point is to fail fast at the second
validation layer (after grammar-constrained sampling) when an LLM ever
emits valid JSON that nonetheless violates our schema.
"""

import pytest
from pydantic import ValidationError

from vsa.extraction.schema import ExtractionResult, Task
from vsa.extraction.types import (
    DEFAULT_FALLBACK_TYPE,
    DEFAULT_VOICE_NOTE_TYPES,
    VoiceNoteType,
)


class TestExtractionResultDefaults:
    def test_minimum_required_fields_validate(self) -> None:
        """Only ``title``, ``summary``, and ``type`` are required;
        every other field has an empty-default. This is the shape the
        LLM emits for a transcript with nothing to extract."""
        result = ExtractionResult(
            title="A short title",
            summary="The summary.",
            type="idea",
        )
        assert result.schema_version == "1.0"
        assert result.mood is None
        assert result.voice_note_location is None
        assert result.tags == []
        assert result.themes == []
        assert result.locations == []
        assert result.people == []
        assert result.projects == []
        assert result.businesses == []
        assert result.tech_stack == []
        assert result.tasks == []


class TestTaskValidation:
    def test_task_with_only_text_validates(self) -> None:
        t = Task(text="buy milk")
        assert t.due is None
        assert t.priority is None

    def test_task_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Task(text="")

    def test_task_priority_enum_enforced(self) -> None:
        Task(text="a", priority="low")
        Task(text="a", priority="medium")
        Task(text="a", priority="high")
        Task(text="a", priority=None)
        with pytest.raises(ValidationError):
            Task(text="a", priority="urgent")  # type: ignore[arg-type]


class TestExtractionResultRoundtrip:
    def test_dict_roundtrip_preserves_all_fields(self) -> None:
        """Dump + parse round-trip matches the original — relevant
        because the callback body serializes via model_dump_json and
        downstream consumers parse it back."""
        original = ExtractionResult(
            title="Meeting with Sarah about API design",
            summary="Discussed REST vs GraphQL for the analyzer service.",
            type="journal",
            mood="energized",
            voice_note_location="home office",
            tags=["api-design", "graphql", "rest"],
            themes=["architecture decisions"],
            locations=["San Francisco"],
            people=["Sarah"],
            projects=["analyzer service"],
            businesses=[],
            tech_stack=["REST", "GraphQL"],
            tasks=[
                Task(text="Send Sarah the RFC link", priority="medium"),
                Task(text="Sketch GraphQL schema", due="2026-05-15"),
            ],
        )
        parsed = ExtractionResult.model_validate(
            original.model_dump(mode="python")
        )
        assert parsed == original


class TestVoiceNoteTypeOverride:
    def test_default_catalog_has_four_entries(self) -> None:
        names = {t.name for t in DEFAULT_VOICE_NOTE_TYPES}
        assert names == {"idea", "journal", "task", "meditation"}

    def test_default_fallback_is_other(self) -> None:
        assert DEFAULT_FALLBACK_TYPE == "other"

    def test_voice_note_type_rejects_empty_strings(self) -> None:
        """Empty name or description would render to garbage in the
        prompt — catch at request validation rather than at LLM call."""
        with pytest.raises(ValidationError):
            VoiceNoteType(name="", description="a valid description")
        with pytest.raises(ValidationError):
            VoiceNoteType(name="idea", description="")
