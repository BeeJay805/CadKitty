"""Defect-injection validator mutation suite (ADR-0035 B5).

The product wedge is "no false PASS" -- but validator SENSITIVITY (do bad parts fail?) had
never been measured, only good-part pass rates. Each test injects ONE program-level defect
into a known-good fixture and asserts the EXPECTED verdict, computed from the spec'd tolerance
-- never assumed. Known escapes are pinned as strict xfails with the ADR reference, so they
flip green (and force removing the pin) the moment the catching check is promoted/added.

Defect x catcher matrix (this pass):

  defect                      | hard catch today        | advisory (INFO) catch (B1)
  ----------------------------+-------------------------+----------------------------------
  declared nominal corrupt    | conformance FAIL        | --
  hole cut undersize          | (escape, pinned)        | conformance.measured MEASURED-MISSING
  hole never cut / dropped    | (escape, pinned)        | conformance.measured MEASURED-MISSING
  holes built as solid pins   | (escape, pinned)        | conformance.measured MEASURED-MISSING
  whole part at 0.1x scale    | features.min_wall FAIL  | mesh.bbox_scale SHRUNKEN
  wall below declared minimum | features.min_wall FAIL  | --
  interface holes as pins     | XFAIL (deferred full-B1: interface checker concavity)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.validator.run import validate

pytestmark = pytest.mark.slow  # ~10 compile+validate cycles; keep out of the hot loop


def _golden_plate(
    *,
    hole_mode: OpMode = OpMode.SUBTRACT,
    hole_r: float = 1.7,
    n_holes: int = 4,
    declared_mm: float = 3.4,
    scale: float = 1.0,
) -> PartProgram:
    """The fixture: a 60x40x4 plate, 4 M3 clearance holes, declared hardware. All defects are
    single-parameter mutations of this."""
    s = scale
    ops: list[Any] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            width=60.0 * s,
            depth=40.0 * s,
            height=4.0 * s,
            position=Vec3(z=2.0 * s),
        )
    ]
    corners = [(-20.0, -12.0), (-20.0, 12.0), (20.0, -12.0), (20.0, 12.0)]
    for i, (x, y) in enumerate(corners[:n_holes]):
        ops.append(
            Cylinder(
                op_id=f"hole_{i}",
                mode=hole_mode,
                position=Vec3(x=x * s, y=y * s, z=2.0 * s),
                radius=hole_r * s,
                height=8.0 * s,
            )
        )
    return PartProgram(
        metadata=Metadata(name="mutation_fixture"),
        operations=ops,
        constraints=Constraints(min_wall_thickness=2.0, max_bbox=Vec3(x=60.0, y=40.0, z=4.0)),
        hardware_features=[
            HardwareFeature(category="clearance_hole", key="M3", nominal_mm=declared_mm, where="m")
            for _ in range(4)
        ],
    )


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    return validate(compile_program(program, work), program)


def _measured(report: ValidationReport) -> str:
    check = report.check("conformance.measured.clearance_hole.M3")
    assert check is not None, "the measured join must always emit on declared round features"
    assert check.status == CheckStatus.INFO
    return check.message


# --------------------------------------------------------------- baseline ---


def test_baseline_fixture_is_clean(tmp_path: Path) -> None:
    report = _validate(_golden_plate(), tmp_path)
    assert report.passed, report.failed_check_ids()
    assert "MEASURED-TRUE" in _measured(report)


# --------------------------------------------------- defects caught HARD today ---


def test_corrupt_declared_nominal_fails_conformance(tmp_path: Path) -> None:
    # Declared 3.0 vs registry 3.4 (tol 0.05): the declaration-level gate bites. Regression pin.
    report = _validate(_golden_plate(declared_mm=3.0), tmp_path)
    assert not report.passed
    assert any(c.startswith("conformance.clearance_hole") for c in report.failed_check_ids())


def test_wall_below_declared_minimum_fails(tmp_path: Path) -> None:
    # Plate thinned to 1mm under a declared 2mm minimum: the feature tier bites. Regression pin.
    program = _golden_plate()
    thin = program.model_copy(
        update={
            "operations": [
                Box(
                    op_id="plate",
                    mode=OpMode.ADD,
                    width=60.0,
                    depth=40.0,
                    height=1.0,
                    position=Vec3(z=0.5),
                )
            ],
            "hardware_features": [],
        }
    )
    report = _validate(thin, tmp_path)
    assert "features.min_wall_thickness" in report.failed_check_ids()


def test_scale_mutant_is_caught_twice(tmp_path: Path) -> None:
    # 0.1x: walls (0.4mm) fail the declared 2mm minimum (hard, today) AND the new INFO scale
    # check names it SHRUNKEN (the catch that survives even when no min_wall is declared).
    report = _validate(_golden_plate(scale=0.1), tmp_path)
    assert "features.min_wall_thickness" in report.failed_check_ids()
    scale = report.check("mesh.bbox_scale")
    assert scale is not None and "SHRUNKEN" in (scale.actual or "")


# ------------------------------------- escapes, now VISIBLE via the B1 slice ---


def test_undersize_hole_escapes_hard_gates_but_is_measured(tmp_path: Path) -> None:
    """Cut 3.0mm, declared 3.4mm: every hard gate passes (the audit's flagship false-PASS).
    The B1 measured join makes it VISIBLE; promotion to FAIL is behind the ADR-0035 criterion."""
    report = _validate(_golden_plate(hole_r=1.5), tmp_path)
    assert report.passed  # the pinned escape, honestly stated
    assert "MEASURED-MISSING" in _measured(report)


def test_dropped_hole_escapes_hard_gates_but_is_measured(tmp_path: Path) -> None:
    """3 of 4 declared holes built: hard gates pass; the measured join counts 3 < 4."""
    report = _validate(_golden_plate(n_holes=3), tmp_path)
    assert report.passed  # the pinned escape
    assert "MEASURED-MISSING" in _measured(report)


def test_solid_pins_escape_hard_gates_but_are_measured(tmp_path: Path) -> None:
    """Holes 'built' as ADD pins: hard gates pass (mesh is valid, declaration conforms);
    concavity excludes every pin, so the measured join reads 0 of 4. The pins protrude UP only
    (the part still prints flat) so the printability tier cannot catch this defect for the
    wrong reason -- the missing holes are the isolated defect under test."""
    ops: list[Any] = [
        Box(
            op_id="plate", mode=OpMode.ADD, width=60.0, depth=40.0, height=4.0, position=Vec3(z=2.0)
        )
    ]
    for i, (x, y) in enumerate([(-20.0, -12.0), (-20.0, 12.0), (20.0, -12.0), (20.0, 12.0)]):
        ops.append(
            Cylinder(
                op_id=f"pin_{i}",
                mode=OpMode.ADD,
                position=Vec3(x=x, y=y, z=5.0),  # spans 2..8: embedded 2mm, protrudes 4mm up
                radius=1.7,
                height=6.0,
            )
        )
    program = PartProgram(
        metadata=Metadata(name="pins_fixture"),
        operations=ops,
        constraints=Constraints(min_wall_thickness=2.0),
        hardware_features=[
            HardwareFeature(category="clearance_hole", key="M3", nominal_mm=3.4, where="m")
            for _ in range(4)
        ],
    )
    report = _validate(program, tmp_path)
    assert report.passed, report.failed_check_ids()  # the pinned escape
    assert "MEASURED-MISSING" in _measured(report)


# ------------------------------------------------- deferred escape (full B1) ---


def test_nema17_interface_with_solid_pins_should_fail(tmp_path: Path) -> None:
    # Was a strict-xfail "deferred B1 escape": 4 solid pins (convex) satisfied bolt_holes_present
    # because the interface tier ignored concavity. Fixed by ADR-0065 / audit H1 (holes_matching now
    # requires concave is True), so this now correctly FAILS.
    program = PartProgram(
        metadata=Metadata(name="nema17_pins"),
        operations=[
            Box(
                op_id="p", mode=OpMode.ADD, width=60.0, depth=60.0, height=5.0, position=Vec3(z=2.5)
            ),
            # central pilot bore: REAL (so only the bolt holes are the defect under test)
            Cylinder(
                op_id="bore", mode=OpMode.SUBTRACT, position=Vec3(z=2.5), radius=11.0, height=9.0
            ),
            # the 4 "bolt holes" as solid PINS on the 31mm square
            *[
                Cylinder(
                    op_id=f"pin_{i}",
                    mode=OpMode.ADD,
                    position=Vec3(x=x, y=y, z=6.5),
                    radius=1.7,
                    height=3.0,
                )
                for i, (x, y) in enumerate(
                    [(-15.5, -15.5), (-15.5, 15.5), (15.5, -15.5), (15.5, 15.5)]
                )
            ],
        ],
        interfaces=[
            InterfaceDeclaration(
                name="nema17_face",
                interface_type=InterfaceType.NEMA17_FACE,
                location=Vec3(x=0.0, y=0.0, z=5.0),
                normal=Vec3(x=0.0, y=0.0, z=1.0),
            )
        ],
    )
    report = _validate(program, tmp_path)
    assert not report.passed  # solid pins are not concave holes -> interface fails (ADR-0035 B1)
