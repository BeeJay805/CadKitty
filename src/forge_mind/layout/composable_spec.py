"""ComposablePlateSpec — the unified plate-family LayoutSpec (ADR-0014).

Collapses ``l_bracket_motor_mount`` + ``adapter_plate`` + ``fan_mount`` into ONE spec:
a base archetype + a list of registry face-interface placements. The LLM picks a MENU
(``base`` + which interfaces on which faces); the composable interpreter
(``composable_interpreter.py``) derives every coordinate, reusing the existing
``place_*`` / ``rules.py`` / legacy ``derive_*`` machinery. No LLM geometry (ADR-0009).

The electronics tray stays a SEPARATE pattern (walled cavity ≠ drilled plate; ADR-0014
Q1). This is data only; the interpreter is the behaviour.

The spec is deliberately a small tuple of bounded categoricals so a future
combination-search loop can ENUMERATE configs (ADR-0014 Q6): ``base`` ∈ a 2-member enum,
each placement is ``(ref ∈ menu, face ∈ menu)`` with optional derived-by-default numbers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.layout.placement import LayoutPattern
from forge_mind.layout.spec import Pattern
from forge_mind.types import Vec3


class BaseArchetype(StrEnum):
    """The parametric base bodies the plate family uses (ADR-0014 Q1)."""

    FLAT_PLATE = "flat_plate"  # one slab (adapter_plate, fan_mount)
    L_BRACKET_BEND = "l_bracket_bend"  # base slab + upright wall at +Y, optional gusset (l_bracket)
    OPEN_BOX = "open_box"  # a flat_plate FLOOR + a perimeter wall ring (a housing / tray)
    ANGLE_BRACKET = (
        "angle_bracket"  # a flat_plate BASE + a rear upright flange (an L / shelf bracket)
    )
    # The walled-base grammar makes new prismatic bodies pure config (a wall set on the floor):
    U_CHANNEL = "u_channel"  # floor + two opposite side walls (a channel / rail)
    TRAY = "tray"  # a deeper 4-wall ring than open_box (a tall tray / bin)
    PARTIAL_ENCLOSURE = "partial_enclosure"  # floor + three walls, open on one side
    ROUND_PLATE = "round_plate"  # a CIRCULAR floor (a round plate / disc mount / flange / lid)


class InterfaceRef(StrEnum):
    """The registry interfaces a placement can reference (the MENU).

    Two tiers: the legacy FACE interfaces (a mount with its own validator checker) and the
    standards-backed HARDWARE features (composition engine) placed from ``dsl.standards`` via
    ``layout.hardware`` and verified by the A4 conformance check. The hardware refs carry their
    component key in ``component_id`` ('M3', '608', '6x3') and the fit class in ``fit``."""

    # Legacy face interfaces (own validator checker).
    NEMA17_FACE = "nema17_face"  # 4x M3 on the 31mm square + central shaft bore
    FAN_FACE = "fan_face"  # 4x M3 on the fan square + central airflow bore (needs component_id)
    EXTRUSION_2020 = "extrusion_2020"  # 2x M5 T-nut pair (spacing_mm)
    # Standards-backed hardware features (composition engine; conformance-checked).
    CLEARANCE_HOLE = "clearance_hole"  # bolt pass-through (ISO 273); component_id = 'M3', fit class
    COUNTERBORE = "counterbore"  # clearance + SHCS head recess (ISO 4762)
    HEAT_SET_BOSS = "heat_set_boss"  # raised boss + insert pilot hole (manufacturer/provisional)
    CAPTIVE_NUT_POCKET = "captive_nut_pocket"  # hex-nut capture slot + bolt clearance (ISO 4032)
    BEARING_SEAT = "bearing_seat"  # shouldered boss-seat for a bearing's outer race (ISO 15)
    DOWEL_HOLE = "dowel_hole"  # locating-pin hole (ISO 2338)
    MAGNET_POCKET = "magnet_pocket"  # disc-magnet pocket (provisional)


class Face(StrEnum):
    """Which base-body face a placement sits on (sets the plane + drill orientation)."""

    PLATE_TOP = "plate_top"  # flat_plate +Z working face (decl at z=thickness, normal +Z)
    PLATE_BOTTOM = "plate_bottom"  # flat_plate -Z face, drilled +Z through (decl at z=0, normal -Z)
    BASE_TOP = "base_top"  # l_bracket base +Z (decl at z=thickness/2, normal +Z)
    UPRIGHT_FRONT = "upright_front"  # l_bracket upright +Y (decl on the wall, normal +Y)


class InterfacePlacement(BaseModel):
    """One registry face-interface placed on a base-body face. The interpreter derives
    the offset/dims when omitted (the ADR-0009 derive-by-default path)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: InterfaceRef
    face: Face
    offset_mm: float | None = Field(
        default=None,
        description="In-plane offset of the interface centre (X for flat_plate). "
        "OPTIONAL: null -> the interpreter derives the smallest legal offset.",
    )
    spacing_mm: float | None = Field(
        default=None, gt=0, description="M5 pair spacing for extrusion_2020 (null -> 20mm default)."
    )
    component_id: str | None = Field(
        default=None,
        description="Component id/key for a parameterised interface: a fan id, or a hardware key "
        "for a standards feature ('M3' for a clearance hole, '608' for a bearing, '6x3' a magnet).",
    )
    fit: str | None = Field(
        default=None,
        description="Fit class for a standards hardware feature: 'close'/'normal'/'coarse' "
        "(clearance, ISO 273) or 'press'/'slip' (bearing/dowel). Null -> the sensible default.",
    )
    name: str | None = Field(
        default=None,
        max_length=64,
        description="InterfaceDeclaration name override (else a default).",
    )


class BaseDimsSpec(BaseModel):
    """Explicit base-body dims — the backward-compat / byte-equivalence escape hatch
    (ADR-0014 Q2). All optional: when omitted the interpreter derives the tightest legal
    dims. ``flat_plate`` uses width/depth; ``l_bracket_bend`` uses the four base_/upright_
    fields. Explicit values WIN (matching the legacy specs' ``_resolve_*`` behaviour)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width_mm: float | None = Field(default=None, gt=0)
    depth_mm: float | None = Field(default=None, gt=0)
    base_width_mm: float | None = Field(default=None, gt=0)
    base_depth_mm: float | None = Field(default=None, gt=0)
    upright_width_mm: float | None = Field(default=None, gt=0)
    upright_height_mm: float | None = Field(default=None, gt=0)


class ComposablePlateSpec(BaseModel):
    """The unified plate-family spec. One member of the LayoutSpec union (ADR-0014)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.COMPOSABLE_PLATE] = Pattern.COMPOSABLE_PLATE
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 = Field(
        ...,
        description="Working bounding-box estimate (mm). The interpreter "
        "GROWS this to the derived footprint if it is too small -- it is an estimate, not a limit.",
    )
    max_envelope_mm: Vec3 | None = Field(
        default=None,
        description="User-stated HARD max bounding box, if any. Set deterministically from intent "
        "(NOT by the LLM); null = no hard limit (grow freely). The interpreter never grows past it.",
    )
    material: str = Field(..., min_length=1)
    min_wall_mm: float = Field(..., gt=0, description="Minimum wall/land thickness anywhere (mm).")
    max_overhang_deg: float = Field(default=45.0, gt=0, le=90)
    base: BaseArchetype = Field(..., description="The base body to place interfaces on.")
    base_dims: BaseDimsSpec | None = Field(
        default=None, description="Explicit dims (escape hatch); null -> derived."
    )
    interfaces: list[InterfacePlacement] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="The registry face-interfaces to place.",
    )
    layout: LayoutPattern | None = Field(
        default=None,
        description="How to ARRANGE repeated mounting features (holes / inserts / dowels / magnets): "
        "'corner' (the default 2x2-style bolt pattern), 'grid', 'row', or 'perimeter'. A single "
        "bearing seat is 'centered' by default; 'end_offset' pushes it to one end. Pick this ONLY "
        "when the request names an arrangement ('in a row', 'a 2x2 grid', 'around the edge', 'at the "
        "end'); otherwise leave null and the interpreter derives the sensible default. NEVER any "
        "coordinates -- the pattern is the only layout control; code derives every position.",
    )


__all__ = [
    "BaseArchetype",
    "BaseDimsSpec",
    "ComposablePlateSpec",
    "Face",
    "InterfacePlacement",
    "InterfaceRef",
    "LayoutPattern",
]
