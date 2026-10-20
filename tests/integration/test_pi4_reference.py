"""bench-002 deterministic foundation: the hand-authored Pi 4 open-tray reference
compiles and passes the new pi4_mount + pi4_ports validator checks, and the
checks correctly fail on broken variants. No LLM, no interpreter.
"""

from __future__ import annotations

from pathlib import Path

from examples.handwritten_pi4_case import build_pi4_case_reference

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate

REQUIRED_PI4_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.pi4_mount.mount_holes_present",
    "interfaces.pi4_mount.hole_rectangle_58x49",
    "interfaces.pi4_mount.standoff_land",
    "interfaces.pi4_ports.long_edge_opening",
    "interfaces.pi4_ports.short_edge_opening",
]


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    model = compile_program(program, work)
    return validate(model, program)


def test_pi4_reference_passes_all_checks(tmp_path: Path) -> None:
    report = _validate(build_pi4_case_reference(), tmp_path)
    failed = set(report.failed_check_ids())
    assert not failed, f"unexpected failures: {failed}"
    for cid in REQUIRED_PI4_CHECKS:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    assert report.passed


def test_standoff_land_resolved_above_min_wall(tmp_path: Path) -> None:
    # The ADR-0005 resolution: OD 7 gives a 2.15mm land >= 2mm min_wall.
    report = _validate(build_pi4_case_reference(), tmp_path)
    land = report.check("interfaces.pi4_mount.standoff_land")
    assert land is not None and land.status.value == "pass"
    assert "2.1" in (land.actual or "")


def test_wrong_standoff_spacing_fails_pi4_mount(tmp_path: Path) -> None:
    # A 50mm hole spacing is not the Pi 4 58x49 rectangle.
    report = _validate(build_pi4_case_reference(hole_dx=50.0), tmp_path)
    rect = report.check("interfaces.pi4_mount.hole_rectangle_58x49")
    assert rect is not None and rect.status.value == "fail"


def test_thin_standoff_land_fails_pi4_mount(tmp_path: Path) -> None:
    # OD 6 (radius 3.0) gives only a 1.65mm land, below the 2mm min_wall.
    report = _validate(build_pi4_case_reference(standoff_r=3.0), tmp_path)
    land = report.check("interfaces.pi4_mount.standoff_land")
    assert land is not None and land.status.value == "fail"


def test_missing_short_port_fails_pi4_ports(tmp_path: Path) -> None:
    report = _validate(build_pi4_case_reference(include_short_port=False), tmp_path)
    short = report.check("interfaces.pi4_ports.short_edge_opening")
    assert short is not None and short.status.value == "fail"
    # The long-edge opening is still present, so it should still pass.
    long_open = report.check("interfaces.pi4_ports.long_edge_opening")
    assert long_open is not None and long_open.status.value == "pass"
