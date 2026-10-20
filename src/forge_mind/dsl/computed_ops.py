"""Computed-geometry operation MENU (ADR-0019) — the CATEGORICAL discriminated union the
LLM selects from. The LLM picks ``kind`` + bounded categoricals ONLY; deterministic
interpreters/builders own every dimension (the ADR-0009 thesis, extended to computed
geometry). Parallel to the LayoutSpec ``pattern`` union and the component registry.

The menu grows ONE earned op at a time. Today: ``faceted_strut`` (a straight solid prism) and
``lattice`` (a weight-saving vertical-walled cellular grid). Both are STRAIGHT-walled with
obtuse/right dihedrals, so the conservative min_wall reads them exactly. Taper and curved
(loft/sweep) ops remain DEFERRED until the SDF/medial-axis min_wall (ADR-0019 Q5), because it
over-rejects their acute/curved walls.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class FacetedStrutOp(BaseModel):
    """A straight structural prism — the first computed op. The cross-section is
    OBTUSE-dihedral only (hexagon = 120 deg, rectangle = 90 deg), which is why the
    conservative min_wall reads its true thickness (no acute-edge sliver) and it is
    validator-ready today. Categorical only — NO numbers (dims derive from coarse intent).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["faceted_strut"] = "faceted_strut"
    cross_section: Literal["rectangular", "hexagonal"] = Field(
        ..., description="Strut cross-section; both have obtuse dihedrals (no acute slivers)."
    )


class LatticeOp(BaseModel):
    """A weight-saving LATTICE: a grid of straight vertical wall-members unioned into one body,
    leaving vertical through-cells. Validator-ready today because every member is vertical (no
    overhang) and meets at right angles (no acute slivers); the interpreter DERIVES the wall
    thickness (>= min_wall) and the cell counts. Categorical only — NO numbers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["lattice"] = "lattice"
    cell: Literal["grid"] = Field(
        "grid", description="Cell shape. 'grid' = square cells (vertical walls along X and Y)."
    )
    density: Literal["light", "medium", "dense"] = Field(
        ...,
        description=(
            "How much material the lattice keeps. 'light' = big cells/least mass; 'dense' = "
            "small cells/most stiffness. Walls always meet min_wall regardless of this choice."
        ),
    )


# Discriminated union: Pydantic picks the concrete op from the ``kind`` literal.
ComputedOp = Annotated[FacetedStrutOp | LatticeOp, Field(discriminator="kind")]


__all__ = ["ComputedOp", "FacetedStrutOp", "LatticeOp"]
