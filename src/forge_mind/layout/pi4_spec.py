"""ElectronicsTraySpec — the board-generic open-tray LayoutSpec (ADR-0005, ADR-0007).

The second LayoutSpec pattern: an open tray that holds a PCB on standoffs. First
specialised for the Raspberry Pi 4 case (bench-002), then shown to handle a generic
rectangular PCB through the same interpreter (ADR-0007 Stage C). ``Pi4CaseSpec`` is
kept as a backward-compatible alias of ``ElectronicsTraySpec`` (ADR-0007 Stage E).

The deterministic interpreter (``layout/pi4_interpreter.py``,
``interpret_electronics_tray``) turns this spec into a PartProgram. Kept in its own
module so bench-001 stays untouched; the two patterns form a discriminated union on
``pattern`` in ``layout/layoutspec.py``.

Mechanical facts (Raspberry Pi 4 Model B), used as field defaults:
- board 85 x 56 mm, ~1.5 mm PCB
- 4 mounting holes on a 58 x 49 mm rectangle, 3.5 mm in from the board edges
- M2.5 screws (2.7 mm clearance)
- connectors on one long edge (2x USB-A, 2x USB 3.0, Gigabit Ethernet) and one
  short edge (USB-C power, 2x micro-HDMI, 3.5 mm AV); microSD on the opposite
  short edge.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from forge_mind.dsl.components.registry import known_board_component_ids
from forge_mind.dsl.interfaces.pi4 import (
    HOLE_RECT_DX_MM,
    HOLE_RECT_DY_MM,
    M25_CLEARANCE_DIAMETER_MM,
)
from forge_mind.layout.spec import Pattern
from forge_mind.types import Vec3


class PortPolicy(StrEnum):
    """Which connector openings to cut. ``standard`` = the fixed Pi 4 port layout
    (the interpreter owns the exact rectangles); ``none`` = a closed shell.
    """

    NONE = "none"
    STANDARD = "standard"


class VentilationPolicy(StrEnum):
    """How to ventilate the case floor/walls."""

    NONE = "none"
    SLOTS = "slots"
    GRID = "grid"


class LidPolicy(StrEnum):
    """How the case closes. In v1 this shapes the tray's lid-interface features;
    the lid itself is a separate emit (deferred — see ADR-0005).
    """

    NONE = "none"
    SNAP = "snap"
    SCREW = "screw"
    FRICTION = "friction"


class BoardDims(BaseModel):
    """The PCB the case holds. Defaults to a Raspberry Pi 4 Model B."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    length_mm: float = Field(default=85.0, gt=0, description="Board length, X (mm).")
    width_mm: float = Field(default=56.0, gt=0, description="Board width, Y (mm).")
    thickness_mm: float = Field(default=1.5, gt=0, description="PCB thickness (mm).")


class StandoffLayout(BaseModel):
    """The 4 mounting standoffs the board screws onto. Defaults are the Pi 4
    58 x 49 mm hole rectangle, 3.5 mm inset, M2.5 screws.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hole_dx_mm: float = Field(
        default=HOLE_RECT_DX_MM, gt=0, description="Hole spacing along X (mm)."
    )
    hole_dy_mm: float = Field(
        default=HOLE_RECT_DY_MM, gt=0, description="Hole spacing along Y (mm)."
    )
    edge_inset_mm: float = Field(default=3.5, gt=0, description="Hole inset from board edges (mm).")
    height_mm: float = Field(default=4.0, gt=0, description="Standoff height above the floor (mm).")
    outer_diameter_mm: float = Field(
        default=7.0,
        gt=0,
        description="Standoff post diameter (mm). 7.0 (not 6.0) so the M2.5 hole "
        "keeps a >= 2mm material land (resolves the ADR-0005 standoff-land tension).",
    )
    screw_clearance_mm: float = Field(
        default=M25_CLEARANCE_DIAMETER_MM,
        gt=0,
        description="Screw clearance hole diameter, M2.5 = 2.7mm (derived from the canonical radius).",
    )


class ElectronicsTraySpec(BaseModel):
    """Board-generic open-tray spec (the Pi 4 case is its first specialisation).

    Defaults are Raspberry Pi 4 Model B values; override ``board`` + ``standoffs`` for
    any other rectangular board (ADR-0007 Stage C). ``pattern`` serializes to
    ``"pi4_case"`` for backward compatibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.ELECTRONICS_TRAY] = Pattern.ELECTRONICS_TRAY
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 = Field(..., description="Maximum case bounding box (mm).")
    # The USER-stated hard maximum (plumbed deterministically from intent by the router, never
    # LLM-picked -- mirrors ComposablePlateSpec). The envelope_mm above is an LLM ESTIMATE the
    # interpreter GROWS to the derived board footprint (grow-don't-reject, ADR-0035 A2); this is
    # the one real limit growth must honour -- exceeding it is an honest refusal.
    max_envelope_mm: Vec3 | None = Field(
        default=None,
        description="User-stated hard maximum bounding box (mm); growth never exceeds it.",
    )
    material: str = Field(..., min_length=1)
    min_wall_mm: float = Field(..., gt=0, description="Minimum wall/floor thickness (mm).")
    clearance_mm: float = Field(
        ..., ge=0, description="Gap between the board edge and the inner wall (mm)."
    )

    # Canonical board source (ADR-0007 Stage D): when set, the interpreter resolves
    # this id to a BoardComponent and takes the board size, mount rectangle, screw
    # radius, standoff defaults, and ports from it -- the `board`/`standoffs`/`ports`
    # fields below are then IGNORED. When None, the legacy explicit fields are used
    # (so old fixtures without an id keep working unchanged).
    board_component_id: str | None = Field(
        default=None, description="Known board id, e.g. raspberry_pi_4 or generic_rect_pcb."
    )
    board: BoardDims = Field(default_factory=BoardDims)
    standoffs: StandoffLayout = Field(default_factory=StandoffLayout)
    ports: PortPolicy = Field(default=PortPolicy.STANDARD)
    ventilation: VentilationPolicy = Field(default=VentilationPolicy.SLOTS)
    lid: LidPolicy = Field(default=LidPolicy.NONE)
    # print_orientation is no longer a spec field: the electronics tray prints
    # floor-down (+Z up) by construction; the interpreter assumes it. Dropping
    # it from the spec shrinks the structured-output grammar.

    @field_validator("board_component_id")
    @classmethod
    def _known_component(cls, v: str | None) -> str | None:
        if v is not None and v not in known_board_component_ids():
            raise ValueError(
                f"unknown board_component_id {v!r}; known: {sorted(known_board_component_ids())}"
            )
        return v


# Backward-compatible alias (ADR-0007 Stage E): old code, tests, and fixtures refer
# to Pi4CaseSpec. It is literally the same class as ElectronicsTraySpec.
Pi4CaseSpec = ElectronicsTraySpec


__all__ = [
    "BoardDims",
    "ElectronicsTraySpec",
    "LidPolicy",
    "Pi4CaseSpec",
    "PortPolicy",
    "StandoffLayout",
    "VentilationPolicy",
]
