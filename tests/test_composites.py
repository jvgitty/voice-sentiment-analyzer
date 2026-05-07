"""Tests for the CompositeScorer (slice 6).

The composites.yaml at the repo root is the source of truth — these tests
read from it where possible rather than re-asserting hardcoded names so
that tuning the YAML doesn't break the tests for the wrong reason.
"""

from pathlib import Path

import pytest

from vsa.composites import CompositeScorer


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSITES_YAML = REPO_ROOT / "composites.yaml"


class TestCompositesYamlLoad:
    def test_composites_yaml_exists_at_repo_root(self) -> None:
        assert COMPOSITES_YAML.exists(), (
            f"composites.yaml is the spec file for slice 6 and must live at the "
            f"repo root: {COMPOSITES_YAML}"
        )

    def test_from_yaml_loads_without_error(self) -> None:
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        assert scorer is not None


class TestCompositesYamlInvariants:
    def test_each_composite_weights_sum_to_one(self) -> None:
        """The shipped composites.yaml is the spec; for the formula to be
        a clean weighted average producing values in [0, 1], each
        composite's weights MUST sum to 1.0 within float tolerance.

        Iterates over composites discovered in the YAML rather than naming
        them, so adding a new composite (or renaming one) only requires a
        YAML edit.
        """
        scorer = CompositeScorer.from_yaml(COMPOSITES_YAML)
        for name, components in scorer.weights_per_composite().items():
            total = sum(components.values())
            assert total == pytest.approx(1.0, abs=1e-9), (
                f"composite {name!r} weights sum to {total}, not 1.0"
            )
