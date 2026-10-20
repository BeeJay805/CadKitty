"""IntentSpec — structured intent extracted from free text.

The input to the BlueprintGenerator. In Phase 4 these are hand-authored; the
IntentParser (Phase 6) will produce them from a natural-language prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.interfaces.declaration import InterfaceType


class MatingInterfaceRequest(BaseModel):
    """A thing this part must mate with, named by interface type + target."""

    model_config = ConfigDict(extra="forbid")

    interface_type: InterfaceType
    target: str = Field(..., description="What it mates with, e.g. 'NEMA 17 stepper motor'.")


class IntentConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_wall_mm: float | None = Field(default=None, gt=0)
    material: str | None = None
    max_envelope_mm: list[float] | None = Field(
        default=None,
        description="Optional [x, y, z] max bounding box if the user stated one.",
    )


class IntentSpec(BaseModel):
    """Structured user intent for a single part."""

    model_config = ConfigDict(extra="forbid")

    part_family: str = Field(..., description="e.g. 'motor_mount', 'bracket', 'enclosure'.")
    summary: str = Field(..., description="One-line restatement of what the user wants.")
    mating_interfaces: list[MatingInterfaceRequest] = Field(default_factory=list)
    constraints: IntentConstraints = Field(default_factory=IntentConstraints)
    orientation_notes: str | None = Field(
        default=None,
        description="Free-text orientation hints, e.g. 'shaft faces away from extrusion'.",
    )


__all__ = ["IntentConstraints", "IntentSpec", "MatingInterfaceRequest"]
