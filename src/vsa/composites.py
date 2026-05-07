"""Composite scoring (slice 6).

Loads ``composites.yaml`` and scores ``confidence`` / ``engagement`` /
``calmness`` from a bundle of feature inputs. The YAML is the spec —
weights, thresholds, and the formulas live there so the open-source
audience can fork and disagree without touching code.

Implementation note: the YAML's ``source`` strings are documentary; we
do NOT eval them. The actual computation is done by a closed registry
of Python callables keyed off each component's ``name``. The YAML is the
spec the registry implements; the ``_formulas`` echo in the output proves
the two stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class CompositeScorer:
    """Loads ``composites.yaml`` and scores a bundle of feature inputs.

    Slice 6 only implements the from_yaml loader; the .score() method
    arrives later in the cycle. The class exists now so the smoke test
    can prove the YAML parses.
    """

    def __init__(self, config: dict) -> None:
        # config is the parsed YAML root. Stored verbatim for now;
        # later cycles will validate it via a Pydantic model.
        self._config = config

    @classmethod
    def from_yaml(cls, path: Path) -> "CompositeScorer":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def weights_per_composite(self) -> dict[str, dict[str, float]]:
        """Return ``{composite_name: {component_name: weight}}`` straight
        from the YAML so tests can assert the weight-sum invariant without
        hardcoding component names."""
        out: dict[str, dict[str, float]] = {}
        for composite_name, body in self._config.items():
            comps = body.get("components", [])
            out[composite_name] = {
                comp["name"]: float(comp["weight"]) for comp in comps
            }
        return out
