"""DSLDraft — a constrained generation schema for the LLM.

The full PartProgram lets any operation be add or subtract. In practice the LLM
kept adding cylinders where it should subtract them (solid posts instead of
holes). So the LLM emits a DSLDraft instead: a list of SOLIDS (always added) and
a list of HOLES (always subtracted). The wrong-mode failure becomes structurally
impossible. We then assemble a PartProgram deterministically.

This is the planned "reduce the DSL surface" move: constrain what the model can
express to exactly what the part needs, so a smaller model succeeds.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.blueprint.schema import Blueprint
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Rotation, Vec3

_OP_ID = r"^[a-zA-Z_][a-zA-Z0-9_]*$"


class DraftSolid(BaseModel):
    """A box that is ADDED to the body."""

    model_config = ConfigDict(extra="forbid")

    op_id: str = Field(..., pattern=_OP_ID)
    position: Vec3
    rotation: Rotation
    width: float = Field(..., gt=0)
    depth: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class DraftHole(BaseModel):
    """A cylinder that is SUBTRACTED from the body (a hole / bore)."""

    model_config = ConfigDict(extra="forbid")

    op_id: str = Field(..., pattern=_OP_ID)
    position: Vec3
    rotation: Rotation
    radius: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class DraftInterface(BaseModel):
    """An interface the part presents (no free-form params, for clean schemas)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    interface_type: InterfaceType
    location: Vec3
    normal: Vec3


class DSLDraft(BaseModel):
    """What the LLM emits: solids, holes, and declared interfaces."""

    model_config = ConfigDict(extra="forbid")

    solids: list[DraftSolid] = Field(..., min_length=1)
    holes: list[DraftHole] = Field(default_factory=list)
    interfaces: list[DraftInterface] = Field(default_factory=list)


def draft_to_program(draft: DSLDraft, blueprint: Blueprint) -> PartProgram:
    """Assemble a PartProgram: solids added first, then holes subtracted.

    Metadata + constraints come from the blueprint (not the model), so the
    validator checks the geometry against the blueprint's promises.
    """
    operations: list[Operation] = []
    for s in draft.solids:
        operations.append(
            Box(
                op_id=s.op_id,
                mode=OpMode.ADD,
                position=s.position,
                rotation=s.rotation,
                width=s.width,
                depth=s.depth,
                height=s.height,
            )
        )
    for h in draft.holes:
        operations.append(
            Cylinder(
                op_id=h.op_id,
                mode=OpMode.SUBTRACT,
                position=h.position,
                rotation=h.rotation,
                radius=h.radius,
                height=h.height,
            )
        )

    interfaces = [
        InterfaceDeclaration(
            name=i.name,
            interface_type=i.interface_type,
            location=i.location,
            normal=i.normal,
        )
        for i in draft.interfaces
    ]

    metadata = Metadata(
        name=blueprint.name,
        units="mm",
        print_orientation=Vec3(
            x=blueprint.print_orientation.x,
            y=blueprint.print_orientation.y,
            z=blueprint.print_orientation.z,
        ),
    )
    constraints = Constraints(
        min_wall_thickness=blueprint.min_wall_mm,
        max_overhang_deg=blueprint.max_overhang_deg,
        max_bbox=Vec3(
            x=blueprint.envelope_mm.x, y=blueprint.envelope_mm.y, z=blueprint.envelope_mm.z
        ),
    )
    return PartProgram(
        metadata=metadata,
        operations=operations,
        interfaces=interfaces,
        constraints=constraints,
    )


__all__ = [
    "DSLDraft",
    "DraftHole",
    "DraftInterface",
    "DraftSolid",
    "draft_to_program",
]
