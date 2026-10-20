"""bench-006 Stage-1 deterministic foundation: the hand-authored fan-mount -> 2020
adapter-plate references compile and pass the validator, and the new ``fan_face``
checker catches fan-SPECIFIC defects (wrong bolt square, occluding airflow bore).

No LLM, no interpreter, no new pattern -- the fan mount reuses the adapter_plate
shape (flat plate + two face interfaces); the only new validator code is the
``fan_face`` interface checker (ADR-0011).
"""

from __future__ import annotations

from pathlib import Path

from examples.handwritten_fan_2020_adapter import build_fan_mount_reference

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate

# The 4 validator tiers + the new fan_face interface checks the reference must pass.
REQUIRED_FAN_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.fan_face.bolt_holes_present",
    "interfaces.fan_face.bolt_pattern_square",
    "interfaces.fan_face.airflow_bore_present",
    "interfaces.extrusion_face.two_m5_holes",
    "interfaces.extrusion_face.m5_spacing",
]


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    return validate(compile_program(program, work), program)


# ---------------------------------------------------------------------------
# Both reference sizes pass every required check.
# ---------------------------------------------------------------------------


def test_60mm_reference_passes_all_required_checks(tmp_path: Path) -> None:
    report = _validate(build_fan_mount_reference(size="60mm"), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    for cid in REQUIRED_FAN_CHECKS:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_40mm_reference_passes_all_required_checks(tmp_path: Path) -> None:
    # The tight feasibility-probe case: a 2 mm wall gives a ~2.03 mm bore-to-bolt
    # land, just above min_wall. Passes -- but see the bench-006 plan: a 4 mm wall
    # at this bore is infeasible.
    report = _validate(build_fan_mount_reference(size="40mm"), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    for cid in REQUIRED_FAN_CHECKS:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_fan_face_checks_pass(tmp_path: Path) -> None:
    report = _validate(build_fan_mount_reference(size="60mm"), tmp_path)
    for cid in (
        "interfaces.fan_face.bolt_holes_present",
        "interfaces.fan_face.bolt_pattern_square",
        "interfaces.fan_face.airflow_bore_present",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


# ---------------------------------------------------------------------------
# Broken variants: the fan_face checker catches fan-SPECIFIC errors. The matcher
# is permissive (any plate-shaped part routes here), so validation must be strict
# about real fan correctness.
# ---------------------------------------------------------------------------


def test_wrong_bolt_square_fails_fan_face(tmp_path: Path) -> None:
    # Build the 60 mm part but drill the bolts on a 46 mm square instead of the
    # declared 50 mm. 46 mm keeps all 4 holes present and clear of the airflow bore
    # (so bolt_holes_present still passes), isolating the defect to the pattern
    # SIZE -- exactly the fan-specific error the checker must catch.
    bad = build_fan_mount_reference(size="60mm", bolt_square=46.0)
    report = _validate(bad, tmp_path)
    present = report.check("interfaces.fan_face.bolt_holes_present")
    square = report.check("interfaces.fan_face.bolt_pattern_square")
    assert present is not None and present.status.value == "pass"  # 4 holes are there
    assert square is not None and square.status.value == "fail"  # but not a 50mm square


def test_oversized_airflow_bore_eats_bolt_land(tmp_path: Path) -> None:
    # A 60 mm fan whose airflow bore is grown to r=34 mm reaches past the M3 inner
    # edge (33.76 mm) -> it intersects/overlaps the bolt holes, destroying the land.
    # The min_wall tier catches the vanished web (and/or the bolt count drops as the
    # bore swallows holes); either way the part must FAIL, not silently pass.
    bad = build_fan_mount_reference(size="60mm", airflow_bore_r=34.0)
    report = _validate(bad, tmp_path)
    assert not report.passed, "an airflow bore that eats the bolt land must fail validation"
