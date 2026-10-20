"""Placeable hardware interfaces (A2) — the composition-engine vocabulary.

Each ``place_*`` turns a standards-registry component (size + fit) into PartDSL operations, exactly
like ``layout.interfaces`` does for the NEMA/fan/extrusion faces. These are the placeable
INGREDIENTS the future composition engine will drop onto a base body: an M3 clearance hole, a
counterbored screw recess, a heat-set insert boss, a captive-nut pocket, a bearing seat, a dowel
hole, a magnet pocket. Dimensions come ONLY from ``dsl.standards`` (never guessed); geometry is
built from the existing ``Box`` / ``Cylinder`` primitives (no new primitive).

Convention: ``center`` is the feature's reference point; a hole/seat extends along local Z with
the given depth (the primitive convention). The caller (composition engine) positions + orients;
``op_id`` is supplied so ids stay unique. ADD ops (bosses) come before their SUBTRACT (hole).
"""

from __future__ import annotations

from typing import Literal

from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.standards import (
    Fit,
    bearing,
    bearing_housing_bore,
    captive_nut_pocket,
    clearance_hole,
    counterbore,
    dowel_hole,
    heat_set_insert,
    magnet_pocket,
    tap_drill,
)
from forge_mind.types import Vec3


def place_clearance_hole(
    size: str,
    center: Vec3,
    depth: float,
    *,
    op_id: str = "clearance_hole",
    fit: Fit | str = Fit.NORMAL,
) -> list[Operation]:
    """A bolt clearance hole (ISO 273 fit class)."""
    return [
        Cylinder(
            op_id=op_id,
            mode=OpMode.SUBTRACT,
            position=center,
            radius=clearance_hole(size, fit) / 2.0,
            height=depth,
        )
    ]


def place_tapped_hole(
    size: str, center: Vec3, depth: float, *, op_id: str = "tapped_hole"
) -> list[Operation]:
    """A tapping/pilot hole to cut (or self-tap) a coarse-pitch thread into the part."""
    return [
        Cylinder(
            op_id=op_id,
            mode=OpMode.SUBTRACT,
            position=center,
            radius=tap_drill(size) / 2.0,
            height=depth,
        )
    ]


def place_counterbored_hole(
    size: str,
    center: Vec3,
    depth: float,
    *,
    op_id_prefix: str = "cbore",
    fit: Fit | str = Fit.NORMAL,
) -> list[Operation]:
    """A clearance hole plus a coaxial counterbore recessing a socket-head cap screw flush at +Z."""
    cb_dia, cb_depth = counterbore(size)
    recess_z = center.z + (depth - cb_depth) / 2.0  # recess top flush with the clearance top
    return [
        Cylinder(
            op_id=f"{op_id_prefix}_clear",
            mode=OpMode.SUBTRACT,
            position=center,
            radius=clearance_hole(size, fit) / 2.0,
            height=depth,
        ),
        Cylinder(
            op_id=f"{op_id_prefix}_recess",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=center.x, y=center.y, z=recess_z),
            radius=cb_dia / 2.0,
            height=cb_depth,
        ),
    ]


def place_heat_set_boss(
    size: str, center: Vec3, *, op_id_prefix: str = "insert", boss_wall_mm: float | None = None
) -> list[Operation]:
    """A cylindrical boss with the heat-set insert's pilot hole (PROVISIONAL insert dims). The boss
    keeps >= ``min_boss_wall_mm`` of material around the insert; the hole is the insert length deep."""
    spec = heat_set_insert(size)
    wall = spec.min_boss_wall_mm if boss_wall_mm is None else boss_wall_mm
    return [
        Cylinder(
            op_id=f"{op_id_prefix}_boss",
            mode=OpMode.ADD,
            position=center,
            radius=spec.hole_diameter_mm / 2.0 + wall,
            height=spec.length_mm,
        ),
        Cylinder(
            op_id=f"{op_id_prefix}_hole",
            mode=OpMode.SUBTRACT,
            position=center,
            radius=spec.hole_diameter_mm / 2.0,
            height=spec.length_mm,
        ),
    ]


def place_captive_nut_pocket(
    size: str,
    center: Vec3,
    *,
    op_id_prefix: str = "nut",
    bolt_clearance_depth: float,
    clearance_mm: float = 0.2,
) -> list[Operation]:
    """A hex-nut capture pocket (a slot of width = across-flats, length = across-corners, depth =
    nut thickness) plus the bolt clearance hole through it (ISO 4032 nut)."""
    across_flats, across_corners, depth = captive_nut_pocket(size, clearance_mm)
    return [
        Box(
            op_id=f"{op_id_prefix}_pocket",
            mode=OpMode.SUBTRACT,
            position=center,
            width=across_flats,
            depth=across_corners,
            height=depth,
        ),
        Cylinder(
            op_id=f"{op_id_prefix}_bolt",
            mode=OpMode.SUBTRACT,
            position=center,
            radius=clearance_hole(size, Fit.NORMAL) / 2.0,
            height=bolt_clearance_depth,
        ),
    ]


def place_bearing_seat(
    designation: str,
    center: Vec3,
    *,
    op_id: str = "bearing_seat",
    fit: Literal["press", "slip"] = "press",
) -> list[Operation]:
    """A counterbored seat for a bearing's outer race (ISO 15 OD; PROVISIONAL FDM fit). Depth = the
    bearing width."""
    return [
        Cylinder(
            op_id=op_id,
            mode=OpMode.SUBTRACT,
            position=center,
            radius=bearing_housing_bore(designation, fit) / 2.0,
            height=bearing(designation).width_mm,
        )
    ]


def place_dowel_hole(
    size: str,
    center: Vec3,
    depth: float,
    *,
    op_id: str = "dowel_hole",
    fit: Literal["press", "slip"] = "press",
) -> list[Operation]:
    """A locating-dowel hole (ISO 2338 pin; PROVISIONAL FDM fit)."""
    return [
        Cylinder(
            op_id=op_id,
            mode=OpMode.SUBTRACT,
            position=center,
            radius=dowel_hole(size, fit) / 2.0,
            height=depth,
        )
    ]


def place_magnet_pocket(
    size: str, center: Vec3, *, op_id: str = "magnet_pocket"
) -> list[Operation]:
    """A slip-fit disc-magnet pocket (PROVISIONAL product size + fit)."""
    dia, depth = magnet_pocket(size)
    return [
        Cylinder(op_id=op_id, mode=OpMode.SUBTRACT, position=center, radius=dia / 2.0, height=depth)
    ]


__all__ = [
    "place_bearing_seat",
    "place_captive_nut_pocket",
    "place_clearance_hole",
    "place_counterbored_hole",
    "place_dowel_hole",
    "place_heat_set_boss",
    "place_magnet_pocket",
    "place_tapped_hole",
]
