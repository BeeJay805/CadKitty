"""DSL primitives — typed operation variants applied sequentially by the compiler.

Each Operation:
- has a unique ``op_id`` (used by topology tags and the validator to reference
  the resulting faces/edges back from the B-rep)
- has a discriminator field ``op`` (Literal) so Pydantic can deserialize the
  correct concrete class from JSON
- declares a ``mode`` (ADD or SUBTRACT) that the compiler honors when combining
  the primitive with the running body
- declares an absolute ``position`` (and optional ``rotation``) in the part's
  coordinate system. Anchors / coordinate frames will be added in a later phase.

Phase 1 ships only ``Box`` and ``Cylinder`` — the minimum needed to express the
NEMA 17 mount fixture. New primitives (HolePattern, Slot, Fillet, ...) get
added in later phases as benchmarks demand them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge_mind.types import ZERO_ROTATION, ZERO_VEC3, Rotation, Vec3


class OpMode(StrEnum):
    """How the primitive combines with the running body."""

    ADD = "add"
    SUBTRACT = "subtract"


class _BaseOp(BaseModel):
    """Fields common to every operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description=(
            "Unique identifier within a PartProgram. Used by the topology index "
            "and the validator to refer back to this operation's resulting faces."
        ),
    )
    mode: OpMode = Field(
        OpMode.ADD,
        description="Whether this primitive is added to or subtracted from the running body.",
    )
    position: Vec3 = Field(
        default=ZERO_VEC3,
        description="Center position in the part's coordinate frame (mm).",
    )
    rotation: Rotation = Field(
        default=ZERO_ROTATION,
        description="Euler rotation (deg) applied before translation.",
    )


class Box(_BaseOp):
    """Axis-aligned rectangular solid centered at ``position``.

    Dimensions are sizes (not half-sizes) along the X / Y / Z axes of the local
    frame, then rotated by ``rotation``, then translated to ``position``.
    """

    op: Literal["box"] = "box"
    width: float = Field(..., gt=0, description="Size along local X axis (mm).")
    depth: float = Field(..., gt=0, description="Size along local Y axis (mm).")
    height: float = Field(..., gt=0, description="Size along local Z axis (mm).")


class Cylinder(_BaseOp):
    """Right circular cylinder centered at ``position``.

    Axis is along local Z by default; use ``rotation`` to orient it elsewhere
    (e.g. ``Rotation(x=90)`` rotates the axis from +Z to +Y).
    """

    op: Literal["cylinder"] = "cylinder"
    radius: float = Field(..., gt=0, description="Cylinder radius (mm).")
    height: float = Field(..., gt=0, description="Cylinder length along local Z (mm).")

    @field_validator("radius")
    @classmethod
    def _radius_sane(cls, v: float) -> float:
        # Above 1000mm is almost certainly a unit mistake (m -> mm).
        if v > 1000.0:
            raise ValueError(f"radius {v}mm looks like a unit error; max 1000mm.")
        return v


class ComputedSolid(_BaseOp):
    """A computed-geometry body whose shape a deterministic builder produces from
    RESOLVED parameters (ADR-0019). The LLM never emits these numbers — they are
    DERIVED by the interpreter from coarse intent + the categorical menu pick
    (`dsl.computed_ops`). The compiler dispatches on ``kind`` to the builder
    registry (`compiler.build_ops.computed`), parallel to how Box/Cylinder map to
    build123d primitives. Additive: prismatic parts emit no ComputedSolid.

    First op (`faceted_strut`): a straight structural prism with an obtuse-dihedral
    cross-section (rectangular/hexagonal), validator-ready today (taper deferred to
    the SDF/medial-axis min_wall — ADR-0019 Q3/Q5).
    """

    op: Literal["computed_solid"] = "computed_solid"
    kind: Literal["faceted_strut"] = Field(
        ..., description="Selects the deterministic builder (the computed-op discriminant)."
    )
    section: Literal["rectangular", "hexagonal"] = Field(
        ..., description="Cross-section of the faceted strut (both obtuse-dihedral)."
    )
    size: float = Field(..., gt=0, description="Cross-section characteristic size (mm), DERIVED.")
    length: float = Field(..., gt=0, description="Member length along its axis (mm), DERIVED.")


class GearSolid(_BaseOp):
    """An external involute spur gear, built by the catalog's deterministic gear
    generator (``agentic.gears.spur_gear``) -- never authored tooth-by-tooth.

    The disc sits with z in [0, thickness], axis +Z at the local origin, one tooth
    centered on +X at ``clock_deg=0`` (the catalog's phase convention, so
    ``mesh_clock_deg`` clocking and the kinematic mesh gate apply unchanged). The
    deterministic assembly builders emit these; the LLM grammar never does (the gear's
    defining parameters are registry/derivation facts, ADR-0009).
    """

    op: Literal["gear_solid"] = "gear_solid"
    module: float = Field(..., gt=0, description="Gear module (mm pitch dia per tooth).")
    teeth: int = Field(..., ge=12, le=120, description="Tooth count (catalog bounds).")
    thickness: float = Field(..., gt=0, description="Face width along local Z (mm).")
    pressure_angle: float = Field(default=20.0, gt=0, lt=45.0, description="Degrees.")
    bore_d: float = Field(default=0.0, ge=0, description="Central bore DIAMETER (mm); 0 = solid.")
    clock_deg: float = Field(default=0.0, description="Tooth-phase rotation about the axis (deg).")


# Discriminated union: Pydantic picks the right class from the ``op`` literal.
Operation = Annotated[
    Box | Cylinder | ComputedSolid | GearSolid,
    Field(discriminator="op"),
]


__all__ = [
    "Box",
    "ComputedSolid",
    "Cylinder",
    "GearSolid",
    "OpMode",
    "Operation",
]
