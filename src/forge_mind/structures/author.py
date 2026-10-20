"""C2: the Author contract -- every way of making geometry emits the same artifact shape.

An author takes a ``RequirementSpec`` and a work dir and returns an ``AuthorResult``: the artifact
pair (STL + STEP -- STEP required of every author so the uniform validator can run), provenance
(WHICH method produced this claim -- load-bearing for both the honesty ledger and the M4 flywheel,
which learns method/topology SELECTION, not millimeters), and the author's self-reported record.
The self-report is advisory: acceptance authority stays with the uniform gate, never the author.

Three implementations exist or are planned: the agentic LLM path (``agentic.loop``), the
closed-form section author (``structures.sections``), and the generative topology path
(``structures.topopt`` + extrusion). Routing between them is a deterministic table over the spec
(never an LLM judgment call) -- see ADR-0097.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.structures.spec import RequirementSpec


@dataclass(frozen=True)
class AuthorResult:
    """What every author hands the uniform gate."""

    provenance: str  # "agentic" | "analytic_isection" | "topopt2d_extrude" | ...
    stl: Path | None  # None = the author failed to produce geometry (fail-soft)
    step: Path | None  # required for the full validator; None degrades B-rep checks to explicit N/A
    self_reported: dict[str, Any] = field(default_factory=dict)  # advisory, never the verdict


# The author contract: (spec, work_dir) -> AuthorResult. Implementations are free functions so the
# router stays a data table (name -> callable), not a class hierarchy.
Author = Callable[[RequirementSpec, Path], AuthorResult]

__all__ = ["Author", "AuthorResult"]
