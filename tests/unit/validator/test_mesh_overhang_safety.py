"""ADR-0018 Step-1 SAFETY BATTERY for the mesh overhang bridge exemption.

The absolute rule: a false-PASS (accepting an unprintable overhang) is FORBIDDEN; a
false-FAIL (over-flagging a bridgeable hole) is acceptable. The bridge exemption may
remove an overhang flag ONLY for a region that is concave AND narrow (major span <=
bridge limit) AND a two-sided bridge. These tests prove every real overhang -- curved,
flat, convex, wide-concave, or one-sided cantilever -- still FAILS, while a small
two-sided through-hole ceiling is exempted (so the rule is not vacuous).

Solids are hand-authored in build123d (the spike approach); the validator reads the STL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from build123d import Box, Circle, Cylinder, Plane, Pos, Rot, Spline, export_stl, sweep

from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.printability import run_printability_checks
from forge_mind.validator.report import CheckStatus

_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="overhang_safety", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=1.0, max_overhang_deg=45.0, max_bbox=Vec3(x=300.0, y=300.0, z=300.0)
        ),
    )


def _overhang_status(part: Any, tmp_path: Path, name: str) -> CheckStatus:
    stl = tmp_path / f"{name}.stl"
    export_stl(part, str(stl))
    checks, _fixes = run_printability_checks(stl, _program())
    overhang = next(c for c in checks if c.id == "printability.overhang_max_angle")
    return overhang.status


# --- the must-FLAG battery (zero false-PASS) --------------------------------


def test_swept_tube_curved_overhang_is_flagged(tmp_path: Path) -> None:
    # The spike's 145 mm^2 false-pass: a circular profile swept up and over. Concave but
    # WIDE (arch span >> bridge limit) -> the span guard keeps it flagged.
    path = Spline((0, 0, 0), (0, 0, 15), (15, 0, 25), (30, 0, 30), tangents=((0, 0, 1), (1, 0, 0)))
    tube = sweep(Plane.XY * Circle(4), path=path)
    assert _overhang_status(tube, tmp_path, "tube") == CheckStatus.FAIL


def test_flat_cantilever_overhang_is_flagged(tmp_path: Path) -> None:
    # A wide flat ledge on a narrow stem: planar 90 deg overhang, concavity ~0 -> flagged.
    ledge = (Pos(0, 0, 0) * Box(8, 8, 12)) + (Pos(0, 0, 8) * Box(30, 30, 4))
    assert _overhang_status(ledge, tmp_path, "ledge") == CheckStatus.FAIL


def test_convex_boss_underside_is_flagged(tmp_path: Path) -> None:
    # A solid horizontal pin on legs: its underside is CONVEX (not a hole) -> flagged.
    legs = (Pos(-12, 0, 0) * Box(4, 8, 12)) + (Pos(12, 0, 0) * Box(4, 8, 12))
    pin = Pos(0, 0, 14) * Rot(0, 90, 0) * Cylinder(radius=5, height=20)
    assert _overhang_status(legs + pin, tmp_path, "pin") == CheckStatus.FAIL


def test_wide_concave_bore_is_flagged(tmp_path: Path) -> None:
    # A big horizontal bore (r=15): the ceiling is concave but its span (~30 mm) far exceeds
    # the bridge limit -> the span guard keeps it flagged (it would need support).
    wide = Box(60, 20, 60) - (Pos(0, 0, 0) * Rot(90, 0, 0) * Cylinder(radius=15, height=20))
    assert _overhang_status(wide, tmp_path, "wide_bore") == CheckStatus.FAIL


def test_one_sided_cantilever_is_flagged(tmp_path: Path) -> None:
    # A cantilever bar (free at the +X end) with a transverse scoop: the bar's flat underside
    # is a real overhang -> flagged. A one-sided structure is never wholly exempted.
    wall = Pos(0, 0, 0) * Box(4, 10, 30)
    bar = Pos(9, 0, 20) * Box(18, 10, 4)
    onesided = (wall + bar) - (Pos(15, 0, 18.5) * Rot(90, 0, 0) * Cylinder(radius=2.5, height=12))
    assert _overhang_status(onesided, tmp_path, "one_sided") == CheckStatus.FAIL


# --- the exemption is real (not vacuous over-flagging) ----------------------


def test_small_two_sided_through_hole_is_exempt(tmp_path: Path) -> None:
    # A small horizontal M5-ish through-hole in a 6 mm wall: the ceiling is concave, narrow,
    # and a two-sided bridge -> exempted, so printability PASSES (bench-001 @ min_wall=6).
    wall = Box(40, 6, 30)
    hole = wall - (Pos(0, 0, 16) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=8))
    assert _overhang_status(hole, tmp_path, "through_hole") == CheckStatus.PASS


@pytest.mark.parametrize("wall_mm", [4.0, 6.0, 8.0])
def test_through_hole_exempt_across_wall_thickness(wall_mm: float, tmp_path: Path) -> None:
    # The bore-ceiling area scales with wall thickness (the area-gate failure mode); the
    # bridge exemption keys on span/shape, so a small through-hole stays exempt at any wall.
    wall = Box(40, wall_mm, 30)
    hole = wall - (Pos(0, 0, 16) * Rot(90, 0, 0) * Cylinder(radius=2.6, height=wall_mm + 4))
    assert _overhang_status(hole, tmp_path, f"hole_{wall_mm}") == CheckStatus.PASS
