"""ADR-0018 Step-2: min_wall safety regression (the hardening was DEFERRED).

The sustained-area hardening was NOT shipped: it false-passes a small UNDERSAMPLED real
thin feature (a 2x2x0.5 mm tab produces only isolated hits, which the criterion would skip
and then return a thick cluster -- a forbidden false-PASS). A small cosmetic edge sliver and
a small real thin feature are not distinguishable by cluster geometry, so min_wall ships
UNCHANGED (global minimum of sampled ray casts) -- conservative/safe (it over-fails cosmetic
slivers, never false-passes a thin wall).

These tests pin that safety on the ADR-0018 Q2 cases the rejected anti-parallel filter would
have false-passed: a hole-to-edge land (flat <-> cylinder) and a tapered shell, plus a plain
thin web, must all FAIL. They guard against any future min_wall change that trades a safe
false-FAIL for a false-PASS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from build123d import Box, Cylinder, Plane, Polygon, Pos, export_stl, extrude

from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.feature_checks import run_feature_checks
from forge_mind.validator.report import CheckStatus

_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _program(req: float) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="min_wall_safety", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=req, max_overhang_deg=45.0, max_bbox=Vec3(x=300.0, y=300.0, z=300.0)
        ),
    )


def _min_wall_status(part: Any, req: float, tmp_path: Path, name: str) -> CheckStatus:
    stl = tmp_path / f"{name}.stl"
    export_stl(part, str(stl))
    checks, _fixes = run_feature_checks(stl, _program(req))
    wall = next(c for c in checks if c.id == "features.min_wall_thickness")
    return wall.status


def test_thin_web_fails(tmp_path: Path) -> None:
    # Two blocks joined by a 1 mm web, requirement 2 mm.
    web = (
        (Pos(-8, 0, 0) * Box(8, 20, 20))
        + (Pos(8, 0, 0) * Box(8, 20, 20))
        + (Pos(0, 0, 0) * Box(8, 20, 1))
    )
    assert _min_wall_status(web, 2.0, tmp_path, "web") == CheckStatus.FAIL


def test_hole_to_edge_land_fails(tmp_path: Path) -> None:
    # The case anti-parallel filtering would false-pass (ADR-0018 Q2): a flat exterior face
    # against a cylindrical hole leaving a ~1.5 mm land < the 2 mm requirement.
    land = Box(20, 20, 10) - (Pos(5.5, 0, 0) * Cylinder(radius=3, height=10))
    assert _min_wall_status(land, 2.0, tmp_path, "land") == CheckStatus.FAIL


def test_tapered_shell_fails(tmp_path: Path) -> None:
    # A draft-angled wall 1 mm thick at the top, 3 mm at the bottom: also NOT anti-parallel,
    # must still fail at the 2 mm requirement.
    sect = Polygon((0, 0), (3, 0), (1, 25), (0, 25), align=None)
    taper = extrude(Plane.XZ * sect, amount=30)
    assert _min_wall_status(taper, 2.0, tmp_path, "taper") == CheckStatus.FAIL
