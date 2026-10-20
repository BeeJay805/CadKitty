"""Motor component models — minimal motor-registry data (ADR-0017).

The third component CLASS in the registry, a flat sibling of ``FanComponent``
(``dsl/components/fan.py``). Like a fan, a motor face carries a 4-bolt SQUARE + a
central bore; the differences are exactly the per-size nominals -- the bolt pitch,
the corner-bolt clearance (M3 on NEMA 14/17, M5 on NEMA 23), and the central
pilot-boss (register) clearance. Frozen Pydantic, pure data, resolved by id --
exactly the board/fan pattern.

Naming note: the central bore is the **pilot-boss / register** clearance (the raised
centring ring), NOT the much smaller shaft. The existing ``dsl/interfaces/nema17.py``
calls its 22 mm-dia hole a "shaft bore"; ``pilot_bore_radius_mm`` is the same value,
named for what it actually clears. The NEMA 17 entry imports that constant verbatim so
nothing drifts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MotorMountFacts(BaseModel):
    """A motor face's 4-bolt square + central pilot-boss bore (nominal facts)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bolt_pitch_mm: float = Field(
        ..., gt=0, description="Centre-to-centre of the 4 corner bolts (square)."
    )
    bolt_clearance_radius_mm: float = Field(
        ..., gt=0, description="Corner-bolt clearance radius (M3 r1.6 / M5 r2.6)."
    )
    pilot_bore_radius_mm: float = Field(
        ...,
        gt=0,
        description="Central register/boss clearance radius the face must leave clear (mm).",
    )


class MotorComponent(BaseModel):
    """A real stepper motor a mount can hold. Pure data, resolved by id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    body_mm: float = Field(..., gt=0, description="Square faceplate footprint, X == Y (mm).")
    shaft_diameter_mm: float = Field(
        ..., gt=0, description="Output shaft dia (mm); envelope/provenance only."
    )
    mount: MotorMountFacts
    source: str = Field(..., min_length=1, description="Provenance / source note.")


__all__ = ["MotorComponent", "MotorMountFacts"]
