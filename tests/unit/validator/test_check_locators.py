"""Check locators (validator-UX slice 1): a validation check carries an OPTIONAL, coordinate-based
``Locator`` so the viewer can point at the finding on the 3D part. min_wall emits the thinnest-wall
POINT; overhang emits the worst region's BBOX. Honest by construction: present ONLY when the check
genuinely computed a locus, coordinate-based (NOT mesh indices, which would not survive the STL->GLB
export boundary), and never fabricated for a check that doesn't know where the problem is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from build123d import Box, Pos, export_stl

from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.feature_checks import run_feature_checks
from forge_mind.validator.intrinsic import validate_intrinsic
from forge_mind.validator.printability import run_printability_checks
from forge_mind.validator.report import CheckStatus

_DUMMY = DslBox(
    op_id="placeholder", mode=OpMode.ADD, position=Vec3(x=0, y=0, z=0), width=1, depth=1, height=1
)


def _program(min_wall: float = 2.0) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="locator", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_DUMMY],
        interfaces=[],
        constraints=Constraints(
            min_wall_thickness=min_wall,
            max_overhang_deg=45.0,
            max_bbox=Vec3(x=300.0, y=300.0, z=300.0),
        ),
    )


def _thin_web() -> Any:
    # Two thick blocks joined by a 1 mm web — the web is the only thin wall, at |z| < 0.5.
    return (
        (Pos(-8, 0, 0) * Box(8, 20, 20))
        + (Pos(8, 0, 0) * Box(8, 20, 20))
        + (Pos(0, 0, 0) * Box(8, 20, 1))
    )


def _flat_ledge() -> Any:
    # A wide flat ledge on a narrow stem: a planar overhang whose underside is ~30x30 mm near z=8.
    return (Pos(0, 0, 0) * Box(8, 8, 12)) + (Pos(0, 0, 8) * Box(30, 30, 4))


def test_min_wall_fail_carries_point_locator(tmp_path: Path) -> None:
    stl = tmp_path / "web.stl"
    export_stl(_thin_web(), str(stl))
    checks, _ = run_feature_checks(stl, _program(2.0))
    wall = next(c for c in checks if c.id == "features.min_wall_thickness")
    assert wall.status == CheckStatus.FAIL
    assert wall.locator is not None
    assert wall.locator.point is not None
    assert wall.locator.bbox_min is None and wall.locator.bbox_max is None  # a point finding
    assert wall.locator.lens == "thickness"
    assert wall.locator.frame == "stl_z_up"
    # the thinnest wall is the 1 mm web at |z| < 0.5 -> the locator localizes there, not on a block
    assert abs(wall.locator.point.z) < 2.0
    assert abs(wall.locator.point.x) < 6.0


def test_overhang_fail_carries_bbox_locator(tmp_path: Path) -> None:
    stl = tmp_path / "ledge.stl"
    export_stl(_flat_ledge(), str(stl))
    checks, _ = run_printability_checks(stl, _program())
    ov = next(c for c in checks if c.id == "printability.overhang_max_angle")
    assert ov.status == CheckStatus.FAIL
    assert ov.locator is not None
    assert ov.locator.bbox_min is not None and ov.locator.bbox_max is not None
    assert ov.locator.point is None  # a region finding
    assert ov.locator.lens == "overhang"
    assert ov.locator.frame == "stl_z_up"
    bmin, bmax = ov.locator.bbox_min, ov.locator.bbox_max
    # the overhang is the ledge UNDERSIDE: a ~30x30 mm flat face at the bottom of the ledge
    # (Box(30,30,4) centered at z=8 -> its downward face is at z=6), NOT the ledge's mid-height.
    assert (bmax.x - bmin.x) > 15.0 and (bmax.y - bmin.y) > 15.0
    assert bmin.z < 7.0 and bmax.z < 7.5


def test_overhang_pass_has_no_locator(tmp_path: Path) -> None:
    # A plain box flat on the bed: no overhang -> nothing to point at -> no locator (never fabricated).
    stl = tmp_path / "box.stl"
    export_stl(Box(20, 20, 10), str(stl))
    checks, _ = run_printability_checks(stl, _program())
    ov = next(c for c in checks if c.id == "printability.overhang_max_angle")
    assert ov.status == CheckStatus.PASS
    assert ov.locator is None


def test_intrinsic_report_flows_the_locator(tmp_path: Path) -> None:
    # End-to-end: the /check report (IntrinsicReport.checks are domain Check objects) carries the
    # locator, so the no-intent pre-print verdict can point the viewer at the thinnest wall.
    stl = tmp_path / "web.stl"
    export_stl(_thin_web(), str(stl))
    report = validate_intrinsic(stl)
    wall = next((c for c in report.checks if c.id == "features.min_wall_thickness"), None)
    assert wall is not None
    assert wall.locator is not None and wall.locator.point is not None
    # serializes cleanly for the HTTP layer (coordinate-based, JSON-safe)
    dumped = report.model_dump(mode="json")
    wall_json = next(c for c in dumped["checks"] if c["id"] == "features.min_wall_thickness")
    assert wall_json["locator"]["point"] is not None
    assert wall_json["locator"]["lens"] == "thickness"
