"""Buildable-kit outputs: turn a generated part into what-to-buy + how-to-make-it.

Two deterministic, ADDITIVE artifacts derived from a finished ``PartProgram`` (never an LLM,
never a gate, no geometry change):

  - ``BillOfMaterials`` (``kit.bom``)          -- what to buy: every fastener / bearing / insert /
    magnet / dowel, with size, quantity, and the cited standard, pulled ONLY from the standards
    registry (a feature with no buyable component is FLAGGED, never guessed).
  - ``MakeInstructions`` (``kit.instructions``) -- how to fabricate it: print orientation,
    suggested material, wall/layer guidance, and (when inserts are present) an install note.
    Everything heuristic is marked advisory; it NEVER gates the part.

``build_kit`` bundles both. The pipeline attaches a ``Kit`` to the result and the CLI surfaces
it, so the same structured object renders in the UI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.program import PartProgram
from forge_mind.kit.bom import BillOfMaterials, BomLine, build_bom
from forge_mind.kit.instructions import MakeInstructions, build_instructions
from forge_mind.validator.metrics import DEFAULT_MATERIAL
from forge_mind.validator.report import ValidationReport


class Kit(BaseModel):
    """A complete buildable kit for one part: what to buy + how to make it. REPORTED, never gated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part_name: str = Field(..., description="The part's name (from its metadata).")
    bom: BillOfMaterials = Field(..., description="What to buy (registry-backed).")
    instructions: MakeInstructions = Field(..., description="How to fabricate it (advisory).")


def build_kit(
    program: PartProgram,
    *,
    material: str = DEFAULT_MATERIAL,
    report: ValidationReport | None = None,
) -> Kit:
    """Build the kit (BOM + make-instructions) for a finished part.

    ``material`` is the source blueprint's material (defaults to PLA when only a bare
    ``PartProgram`` is on hand). ``report`` is optional: when present, the printability verdict
    sharpens the orientation note; the kit is otherwise independent of it. Purely additive --
    reads the program, builds nothing, mutates nothing.
    """
    bom = build_bom(program)
    instructions = build_instructions(program, bom, material=material, report=report)
    return Kit(part_name=program.metadata.name, bom=bom, instructions=instructions)


__all__ = [
    "BillOfMaterials",
    "BomLine",
    "Kit",
    "MakeInstructions",
    "build_bom",
    "build_instructions",
    "build_kit",
]
