"""Fan component models — minimal fan-registry data (bench-006, ADR-0011).

The second component CLASS in the registry, a flat sibling of ``BoardComponent``
(``dsl/components/board.py``). Where a board carries a 4-hole rectangle + ports, a
fan carries a 4-bolt SQUARE + a central airflow bore. Frozen Pydantic, pure data,
no behaviour, resolved by id -- exactly the board pattern.

A ``FanComponent`` is everything the adapter_plate interpreter's fan path needs to
build a mount for a real axial fan: the body footprint, the bolt-square mount, and
the airflow bore the part must leave clear. The screw clearance reuses the shared
M3 fact (wrap-don't-delete, ADR-0007); a 40/60 mm fan's corner holes are M3.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FanMountFacts(BaseModel):
    """A fan's 4-bolt square + central airflow bore (nominal facts)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bolt_square_mm: float = Field(..., gt=0, description="Bolt-hole spacing, both axes (mm).")
    screw_clearance_radius_mm: float = Field(
        ..., gt=0, description="Corner-bolt clearance radius (mm)."
    )
    airflow_bore_radius_mm: float = Field(
        ..., gt=0, description="Designed open airflow radius the mount must leave clear (mm)."
    )


class FanComponent(BaseModel):
    """A real axial fan a mount can hold. Pure data, resolved by id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    body_mm: float = Field(..., gt=0, description="Square fan-frame footprint, X == Y (mm).")
    thickness_mm: float = Field(..., gt=0, description="Fan depth (mm); envelope headroom only.")
    mount: FanMountFacts
    source: str = Field(..., min_length=1, description="Provenance / source note.")


__all__ = ["FanComponent", "FanMountFacts"]
