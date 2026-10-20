"""Extrusion component models — minimal extrusion-registry data (ADR-0017).

The fourth component CLASS, a flat sibling of ``FanComponent`` / ``MotorComponent``.
A part that bolts to a T-slot extrusion drills a pair of T-nut clearance holes; the
per-profile facts are the slot width, the T-nut bolt clearance (M5 on 2020/2040, M6 on
3030), and the default pair spacing. Frozen Pydantic, pure data, resolved by id.

Honest scope note (ADR-0017 Q7): the multiplier here is thinner than the motor one --
a T-nut mount only drills the hole pair, so 2020 and 2040 produce the same drilled
geometry (both M5); the genuinely-new geometric axis is the bolt size (M6 on 3030). The
``ext_2020`` entry imports today's M5 + 20 mm constants verbatim.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtrusionSlotFacts(BaseModel):
    """A T-slot extrusion's T-nut pair facts (nominal)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot_width_mm: float = Field(
        ..., gt=0, description="Channel slot opening (T-nut fit; provenance)."
    )
    bolt_clearance_radius_mm: float = Field(
        ..., gt=0, description="T-nut bolt clearance radius (M5 r2.6 / M6 r3.3)."
    )
    default_tnut_spacing_mm: float = Field(
        ..., gt=0, description="Default spacing between the 2 T-nut holes (mm)."
    )


class ExtrusionComponent(BaseModel):
    """A real T-slot extrusion a part can bolt to. Pure data, resolved by id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    profile_mm: float = Field(
        ..., gt=0, description="Nominal profile module (20 / 30 mm); provenance/envelope."
    )
    slot: ExtrusionSlotFacts
    source: str = Field(..., min_length=1, description="Provenance / source note.")


__all__ = ["ExtrusionComponent", "ExtrusionSlotFacts"]
