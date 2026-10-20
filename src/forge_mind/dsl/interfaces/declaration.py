"""InterfaceDeclaration — a part's claim that it presents a known mating
interface at a location/orientation.

This is pure declared intent (data). The geometry that actually verifies the
claim lives in the validator's interface checker library
(``forge_mind.validator.interfaces``). Keeping the declaration in the DSL and
the checker in the validator preserves the layering: DSL = data, validator =
logic.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.types import Vec3


class InterfaceType(StrEnum):
    """The catalog of interfaces the library can check. Extend only when a
    benchmark needs a new mating interface.
    """

    NEMA17_FACE = "nema17_face"
    EXTRUSION_2020_SLOT = "extrusion_2020_slot"
    M5_TNUT = "m5_tnut"
    PI4_MOUNT = "pi4_mount"
    PI4_PORTS = "pi4_ports"
    FAN_FACE = "fan_face"


class InterfaceDeclaration(BaseModel):
    """A declared mating interface on a part."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    interface_type: InterfaceType
    location: Vec3 = Field(
        ...,
        description="Approximate center of the interface face, part coordinates (mm).",
    )
    normal: Vec3 = Field(
        ...,
        description="Outward direction the interface faces (mating direction).",
    )
    params: dict[str, float | int | str | bool] = Field(
        default_factory=dict,
        description="Interface-specific overrides (rarely needed; defaults live in the checker).",
    )


__all__ = ["InterfaceDeclaration", "InterfaceType"]
