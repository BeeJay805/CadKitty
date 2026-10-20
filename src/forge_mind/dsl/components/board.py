"""Board component models — minimal board-registry data (ADR-0007 Stage D).

Frozen Pydantic models, pure data (no behaviour, no geometry). A ``BoardComponent``
is everything the electronics tray interpreter needs to build a tray for a real PCB:
size, the 4-hole rectangular mount, the screw clearance, board-specific build
defaults, and the connector-bank ports (empty for a board with no ports).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PortEdge(StrEnum):
    """Which tray wall a connector bank opens through, in the tray's frame.

    Only the two edges the Raspberry Pi 4 uses are defined; more are added when a
    board needs them (the tray prints floor-down, so any wall edge is overhang-safe
    as a top-open notch).
    """

    LONG_PLUS_Y = "long_plus_y"  # the +Y long wall; the opening spans X
    SHORT_MINUS_X = "short_minus_x"  # the -X short wall; the opening spans Y


class BoardPortFact(BaseModel):
    """One connector-bank opening the board needs cut in a tray wall."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Bank name, e.g. usb_ethernet.")
    edge: PortEdge
    width_mm: float = Field(..., gt=0, description="Opening width along the wall (mm).")


class BoardMountFacts(BaseModel):
    """The board's 4-hole rectangular mounting pattern (nominal facts)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hole_dx_mm: float = Field(..., gt=0, description="Hole rectangle span along X (mm).")
    hole_dy_mm: float = Field(..., gt=0, description="Hole rectangle span along Y (mm).")
    screw_clearance_radius_mm: float = Field(..., gt=0, description="Screw clearance radius (mm).")
    edge_inset_mm: float = Field(..., gt=0, description="Hole inset from the board edge (mm).")


class BoardDefaults(BaseModel):
    """Tray build defaults specific to this board (not per-design policy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    standoff_height_mm: float = Field(default=4.0, gt=0, description="Standoff height above floor.")
    standoff_outer_diameter_mm: float = Field(
        ...,
        gt=0,
        description="Standoff post OD (mm), sized so OD/2 - screw_r >= a typical min_wall.",
    )


class BoardComponent(BaseModel):
    """A real PCB the electronics tray can hold. Pure data, resolved by id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    length_mm: float = Field(..., gt=0)
    width_mm: float = Field(..., gt=0)
    thickness_mm: float = Field(..., gt=0)
    mount: BoardMountFacts
    defaults: BoardDefaults
    ports: tuple[BoardPortFact, ...] = Field(
        default=(), description="Connector banks; empty = a closed tray (no port cutouts)."
    )
    source: str = Field(..., min_length=1, description="Provenance / source note.")


__all__ = [
    "BoardComponent",
    "BoardDefaults",
    "BoardMountFacts",
    "BoardPortFact",
    "PortEdge",
]
