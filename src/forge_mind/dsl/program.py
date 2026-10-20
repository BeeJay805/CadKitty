"""PartProgram — the top-level DSL document.

A PartProgram is what the DSLGenerator emits, what the Compiler consumes, and
what the validator reasons about. It is JSON-serializable round-trip-safe.

Structure:
    {
      "version": "0.1",
      "metadata": { name, units, print_orientation, description? },
      "operations": [ { op_id, op, mode, position, rotation, ...geometry }, ... ]
    }

Interfaces, anchors, and richer metadata land in later phases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration
from forge_mind.dsl.primitives import Operation, OpMode
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature


class Metadata(BaseModel):
    """Part-level metadata. Drives validator behavior (print orientation) and
    interpretation (units).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    units: Literal["mm"] = Field(
        "mm",
        description="Unit system. Only mm is supported in v0.",
    )
    print_orientation: Vec3 = Field(
        default=Vec3(x=0.0, y=0.0, z=1.0),
        description=(
            "Direction in the part's coordinate frame that points up during 3D "
            "printing. Validator's printability tier uses this to compute "
            "overhang angles."
        ),
    )
    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Free-text description; ignored by the compiler but useful for review.",
    )


class Constraints(BaseModel):
    """Declared design constraints the validator checks the geometry against.

    These come from user intent (flowing intent -> blueprint -> DSL). They are
    the reference the feature/printability tiers compare measured geometry to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_wall_thickness: float | None = Field(
        default=None,
        gt=0,
        description="Minimum wall thickness anywhere (mm). None = not checked.",
    )
    max_overhang_deg: float = Field(
        default=45.0,
        gt=0,
        le=90,
        description="Max overhang angle from vertical that prints without support.",
    )
    max_bbox: Vec3 | None = Field(
        default=None,
        description="Maximum bounding-box size (mm). None = not checked.",
    )


class PartProgram(BaseModel):
    """A complete part design ready for compilation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal["0.1"] = "0.1"
    metadata: Metadata
    operations: list[Operation] = Field(
        ...,
        min_length=1,
        description="Ordered operations. The compiler applies them sequentially.",
    )
    interfaces: list[InterfaceDeclaration] = Field(
        default_factory=list,
        description="Declared mating interfaces, verified by the validator's interface tier.",
    )
    hardware_features: list[HardwareFeature] = Field(
        default_factory=list,
        description=(
            "Declared registry-backed hardware features (bearing seats, clearance holes, heat-set "
            "bosses, ...), verified by the validator's A4 conformance check. Empty for parts with "
            "no standards hardware, which keeps them byte-identical to before this field existed."
        ),
    )
    constraints: Constraints = Field(
        default_factory=Constraints,
        description="Declared design constraints checked by the validator.",
    )

    @field_validator("operations")
    @classmethod
    def _unique_op_ids(cls, ops: list[Operation]) -> list[Operation]:
        seen: set[str] = set()
        for op in ops:
            if op.op_id in seen:
                raise ValueError(f"duplicate op_id: {op.op_id!r}")
            seen.add(op.op_id)
        return ops

    @field_validator("operations")
    @classmethod
    def _first_op_must_add(cls, ops: list[Operation]) -> list[Operation]:
        # A program that starts with a subtract has nothing to subtract from.
        if ops and ops[0].mode != OpMode.ADD:
            raise ValueError("first operation must have mode='add' (no body to subtract from yet)")
        return ops


__all__ = ["Constraints", "Metadata", "PartProgram"]
