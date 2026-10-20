"""Composable HARDWARE placements (the composition engine's vocabulary on a flat plate).

The composition engine places standards-backed hardware features (clearance holes, counterbored
screws, heat-set bosses, captive-nut pockets, bearing seats, dowel holes, magnet pockets) onto a
plate, drawing EVERY dimension from ``dsl.standards`` (never guessed). Kept out of
``composable_interpreter`` (which stays focused on the legacy face mounts, and is already large).

Three things per ref, all sourced from the same registry call so they cannot drift:
  - ``hardware_footprint_radius``  -> the in-plane radius the collision/edge-land check uses;
  - ``place_hardware_on_flat``     -> the PartDSL ops (depth features become bosses so they are
                                      safe on a thin min_wall plate -- no thin floors, no voids);
  - ``hardware_feature_of``        -> the (category, key, fit, nominal_mm) tuple for the A4
                                      conformance gate.
"""

from __future__ import annotations

import math
import re

from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.standards import (
    bearing,
    bearing_housing_bore,
    captive_nut_pocket,
    clearance_hole,
    counterbore,
    dowel_hole,
    heat_set_insert,
    magnet_pocket,
)
from forge_mind.layout.composable_spec import InterfacePlacement, InterfaceRef
from forge_mind.layout.hardware import (
    place_clearance_hole,
    place_counterbored_hole,
    place_dowel_hole,
    place_heat_set_boss,
)
from forge_mind.layout.interpreter import LayoutError
from forge_mind.types import Vec3

HARDWARE_REFS = frozenset(
    {
        InterfaceRef.CLEARANCE_HOLE,
        InterfaceRef.COUNTERBORE,
        InterfaceRef.HEAT_SET_BOSS,
        InterfaceRef.CAPTIVE_NUT_POCKET,
        InterfaceRef.BEARING_SEAT,
        InterfaceRef.DOWEL_HOLE,
        InterfaceRef.MAGNET_POCKET,
    }
)

# ref -> the A4 conformance category (the registry that owns its canonical dimension).
_CATEGORY: dict[InterfaceRef, str] = {
    InterfaceRef.CLEARANCE_HOLE: "clearance_hole",
    InterfaceRef.COUNTERBORE: "counterbore",
    InterfaceRef.HEAT_SET_BOSS: "heat_set_insert",
    InterfaceRef.CAPTIVE_NUT_POCKET: "captive_nut_pocket",
    InterfaceRef.BEARING_SEAT: "bearing_seat",
    InterfaceRef.DOWEL_HOLE: "dowel_hole",
    InterfaceRef.MAGNET_POCKET: "magnet_pocket",
}
_BEARING_BOSS_WALL_MM = 2.5  # material around a bearing seat's housing bore (the seat boss wall)


def _opkey(k: str) -> str:
    """A key safe for op_ids (M2.5 -> M2_5); the raw key is kept for registry lookups."""
    return re.sub(r"[^A-Za-z0-9]", "_", k)


def _key(p: InterfacePlacement) -> str:
    if not p.component_id:
        raise LayoutError(
            f"{p.ref.value} needs a component_id (the hardware key, e.g. 'M3' or '608')"
        )
    return p.component_id


def _clearance_fit(p: InterfacePlacement) -> str:
    return p.fit if p.fit in ("close", "normal", "coarse") else "normal"


def _press_fit(p: InterfacePlacement) -> str:
    return "slip" if p.fit == "slip" else "press"


def hardware_footprint_radius(p: InterfacePlacement, min_wall: float) -> float:
    """The outer in-plane radius of the placed feature -- what the collision + edge-land check
    keeps >= min_wall apart. Same registry call the geometry uses, so they cannot drift."""
    k = _key(p)
    if p.ref is InterfaceRef.CLEARANCE_HOLE:
        return clearance_hole(k, _clearance_fit(p)) / 2.0
    if p.ref is InterfaceRef.COUNTERBORE:
        return counterbore(k)[0] / 2.0
    if p.ref is InterfaceRef.DOWEL_HOLE:
        return dowel_hole(k, _press_fit(p)) / 2.0  # type: ignore[arg-type]
    if p.ref is InterfaceRef.HEAT_SET_BOSS:
        spec = heat_set_insert(k)
        return spec.hole_diameter_mm / 2.0 + max(spec.min_boss_wall_mm, min_wall)
    if p.ref is InterfaceRef.BEARING_SEAT:
        return bearing_housing_bore(k, _press_fit(p)) / 2.0 + max(_BEARING_BOSS_WALL_MM, min_wall)  # type: ignore[arg-type]
    if p.ref is InterfaceRef.MAGNET_POCKET:
        return magnet_pocket(k)[0] / 2.0 + min_wall
    if p.ref is InterfaceRef.CAPTIVE_NUT_POCKET:
        across_corners = captive_nut_pocket(k)[1]
        return across_corners / 2.0 + min_wall
    raise LayoutError(f"not a hardware ref: {p.ref!r}")


def hardware_rim_reach(p: InterfacePlacement, min_wall: float) -> float:
    """Reach used for a CIRCULAR rim (round-plate edge-land / disc sizing).

    A circular feature (a bearing seat, a hole, a magnet) reaches only its footprint radius, so this
    equals ``hardware_footprint_radius``. A captive-nut pocket is a rectangular BOX boss
    (``_captive_nut_boss`` builds ``across_flats + 2*w`` x ``across_corners + 2*w``), whose true
    reach to a round rim is the CORNER DIAGONAL, not the axis half-extent. Using the diagonal sizes
    the disc so the box corner clears the rim (the M8 captive-nut-on-round fix); it is a conservative
    bound (the box fits inside a disk of this radius about its centre), so it never false-passes.
    """
    if p.ref is InterfaceRef.CAPTIVE_NUT_POCKET:
        across_flats, across_corners, _depth = captive_nut_pocket(_key(p))
        # Half-extents match the built box: width = across_flats + 2*min_wall, depth = across_corners
        # + 2*min_wall (see _captive_nut_boss), so the corner is at hypot(half_w, half_d).
        return math.hypot(across_flats / 2.0 + min_wall, across_corners / 2.0 + min_wall)
    return hardware_footprint_radius(p, min_wall)


def hardware_feature_of(p: InterfacePlacement) -> tuple[str, str, str | None, float]:
    """(conformance category, key, fit, nominal built dimension mm) for the A4 conformance gate.
    The nominal is the SAME registry value the placement builds, so a placed feature conforms by
    construction -- the check confirms the registry has a spec + surfaces provisional provenance."""
    k = _key(p)
    cat = _CATEGORY[p.ref]
    if p.ref is InterfaceRef.CLEARANCE_HOLE:
        return cat, k, _clearance_fit(p), clearance_hole(k, _clearance_fit(p))
    if p.ref is InterfaceRef.COUNTERBORE:
        return cat, k, None, counterbore(k)[0]
    if p.ref is InterfaceRef.DOWEL_HOLE:
        return cat, k, _press_fit(p), dowel_hole(k, _press_fit(p))  # type: ignore[arg-type]
    if p.ref is InterfaceRef.HEAT_SET_BOSS:
        return cat, k, None, heat_set_insert(k).hole_diameter_mm
    if p.ref is InterfaceRef.BEARING_SEAT:
        return cat, k, _press_fit(p), bearing_housing_bore(k, _press_fit(p))  # type: ignore[arg-type]
    if p.ref is InterfaceRef.MAGNET_POCKET:
        return cat, k, None, magnet_pocket(k)[0]
    if p.ref is InterfaceRef.CAPTIVE_NUT_POCKET:
        return cat, k, None, captive_nut_pocket(k)[0]
    raise LayoutError(f"not a hardware ref: {p.ref!r}")


def place_hardware_on_flat(
    p: InterfacePlacement, center: tuple[float, float], w: float, occurrence: int
) -> list[Operation]:
    """PartDSL ops for one hardware feature centred at in-plane ``center`` (x, y) on a plate of
    thickness ``w`` (z in [0, w]). A centerline feature passes (x, 0.0) -> byte-identical to the
    former x-only signature; a patterned mounting feature passes its (x, y). Through features drill
    +Z; depth features (heat-set, bearing, magnet, captive nut) are BOSSES so they stay valid on a
    thin min_wall plate."""
    ox, oy = center
    k = _key(p)
    kid = _opkey(k)
    sfx = "" if occurrence == 0 else f"_{occurrence + 1}"
    mid, top = w / 2.0, w
    c_mid = Vec3(x=ox, y=oy, z=mid)

    if p.ref is InterfaceRef.CLEARANCE_HOLE:
        return place_clearance_hole(k, c_mid, w, op_id=f"clr_{kid}{sfx}", fit=_clearance_fit(p))
    if p.ref is InterfaceRef.COUNTERBORE:
        return place_counterbored_hole(
            k, c_mid, w, op_id_prefix=f"cbore_{kid}{sfx}", fit=_clearance_fit(p)
        )
    if p.ref is InterfaceRef.DOWEL_HOLE:
        return place_dowel_hole(k, c_mid, w, op_id=f"dowel_{kid}{sfx}", fit=_press_fit(p))  # type: ignore[arg-type]
    if p.ref is InterfaceRef.HEAT_SET_BOSS:
        spec = heat_set_insert(k)
        wall = max(spec.min_boss_wall_mm, w)
        cz = top + spec.length_mm / 2.0  # boss fully ABOVE the plate; the full plate is the floor
        return place_heat_set_boss(
            k, Vec3(x=ox, y=oy, z=cz), op_id_prefix=f"ins_{kid}{sfx}", boss_wall_mm=wall
        )
    if p.ref is InterfaceRef.BEARING_SEAT:
        return _bearing_boss_seat(k, ox, oy, top, _press_fit(p), sfx, w)
    if p.ref is InterfaceRef.MAGNET_POCKET:
        return _magnet_boss(k, ox, oy, top, w, sfx)
    if p.ref is InterfaceRef.CAPTIVE_NUT_POCKET:
        return _captive_nut_boss(k, ox, oy, top, w, sfx)
    raise LayoutError(f"not a hardware ref: {p.ref!r}")


def _bearing_boss_seat(
    k: str, ox: float, oy: float, top: float, fit: str, sfx: str, w: float
) -> list[Operation]:
    """A shouldered boss-seat on +Z: the bearing drops into the boss bore and rests on the plate top.
    The boss wall is >= the part min_wall (the seat is structural, not the bearing's own clearance)."""
    kid = _opkey(k)
    housing = bearing_housing_bore(k, fit)  # type: ignore[arg-type]
    width = bearing(k).width_mm
    cz = top + width / 2.0  # boss sits on the plate top, going up by the bearing width
    return [
        Cylinder(
            op_id=f"brgboss_{kid}{sfx}",
            mode=OpMode.ADD,
            position=Vec3(x=ox, y=oy, z=cz),
            radius=housing / 2.0 + max(_BEARING_BOSS_WALL_MM, w),
            height=width,
        ),
        Cylinder(
            op_id=f"brgbore_{kid}{sfx}",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=ox, y=oy, z=cz),
            radius=housing / 2.0,
            height=width,
        ),  # blind floor at the plate top = the shoulder
    ]


def _magnet_boss(k: str, ox: float, oy: float, top: float, w: float, sfx: str) -> list[Operation]:
    """A magnet holder boss on +Z with a pocket open at the boss top (floor = w thick)."""
    kid = _opkey(k)
    dia, depth = magnet_pocket(k)
    boss_h = depth + w
    boss_cz = top + boss_h / 2.0
    pocket_cz = top + boss_h - depth / 2.0  # pocket open at the boss top, floor at top + w
    return [
        Cylinder(
            op_id=f"magboss_{kid}{sfx}",
            mode=OpMode.ADD,
            position=Vec3(x=ox, y=oy, z=boss_cz),
            radius=dia / 2.0 + w,
            height=boss_h,
        ),
        Cylinder(
            op_id=f"magpkt_{kid}{sfx}",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=ox, y=oy, z=pocket_cz),
            radius=dia / 2.0,
            height=depth,
        ),
    ]


def _captive_nut_boss(
    k: str, ox: float, oy: float, top: float, w: float, sfx: str
) -> list[Operation]:
    """A boss on +Z that captures a hex nut (pocket open at the boss TOP) with the bolt clearance
    drilled from the part bottom up to the nut. A top boss (not a hanging one) so the plate never
    overhangs the boss -- it prints +Z up with no support."""
    kid = _opkey(k)
    across_flats, across_corners, depth = captive_nut_pocket(k)
    boss_h = depth + w  # nut pocket + a floor of w below it
    boss_cz = top + boss_h / 2.0  # boss on top, z in [top, top + boss_h]
    pocket_cz = top + boss_h - depth / 2.0  # pocket open at the boss top
    bolt_top = top + w  # the nut-pocket floor; bolt runs from z=0 (part bottom) up to here
    return [
        Box(
            op_id=f"nutboss_{kid}{sfx}",
            mode=OpMode.ADD,
            position=Vec3(x=ox, y=oy, z=boss_cz),
            width=across_flats + 2.0 * w,
            depth=across_corners + 2.0 * w,
            height=boss_h,
        ),
        Box(
            op_id=f"nutpkt_{kid}{sfx}",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=ox, y=oy, z=pocket_cz),
            width=across_flats,
            depth=across_corners,
            height=depth,
        ),
        Cylinder(
            op_id=f"nutbolt_{kid}{sfx}",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=ox, y=oy, z=bolt_top / 2.0),
            radius=clearance_hole(k, "normal") / 2.0,
            height=bolt_top,
        ),
    ]


__all__ = [
    "HARDWARE_REFS",
    "hardware_feature_of",
    "hardware_footprint_radius",
    "hardware_rim_reach",
    "place_hardware_on_flat",
]
