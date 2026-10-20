"""Adversarial oracle probe (RSI oracle-trust track) — attack the validator with cheating geometry.

The mutation suite (test_mutation_suite.py) injects single defects into ONE fixture. This battery is
its adversarial sibling: it constructs parts an optimizer would use to FOOL the validator (a hole that
misses the body, an oversize hole, detached debris, a physically-absurd sheet) and asserts the
COMPUTED verdict -- never assumed. Where the validator disagrees with ground truth, the case pins the
escape HONESTLY (exactly as the mutation suite does), so the gap is visible and any future hard-gate
promotion forces removing the pin. This is the manual seed of the systematic false-PASS search the
autonomous-improvement loop will need (an RSI loop WILL hunt these blind spots; we find them first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.validator.run import validate

pytestmark = pytest.mark.slow  # compile+validate cycles; keep out of the hot loop


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    return validate(compile_program(program, work), program)


def _measured_msg(report: ValidationReport) -> str | None:
    """The declared-vs-MEASURED join message for the M3 clearance holes (INFO), or None if absent."""
    check = report.check("conformance.measured.clearance_hole.M3")
    if check is None:
        return None
    assert check.status == CheckStatus.INFO
    return check.message


def _plate_with_holes(*, hole_r: float, positions: list[tuple[float, float]]) -> PartProgram:
    """A 60x40x4 plate with M3 clearance holes declared, cut at ``hole_r`` and ``positions``. The
    declaration is always a valid M3 (so conformance passes); the GEOMETRY is what the case attacks."""
    ops: list[Any] = [
        Box(
            op_id="plate", mode=OpMode.ADD, width=60.0, depth=40.0, height=4.0, position=Vec3(z=2.0)
        )
    ]
    for i, (x, y) in enumerate(positions):
        ops.append(
            Cylinder(
                op_id=f"hole_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=x, y=y, z=2.0),
                radius=hole_r,
                height=8.0,
            )
        )
    return PartProgram(
        metadata=Metadata(name="adversarial_plate"),
        operations=ops,
        constraints=Constraints(min_wall_thickness=2.0, max_bbox=Vec3(x=60.0, y=40.0, z=4.0)),
        hardware_features=[
            HardwareFeature(category="clearance_hole", key="M3", nominal_mm=3.4, where="m")
            for _ in positions
        ],
    )


_CORNERS = [(-20.0, -12.0), (-20.0, 12.0), (20.0, -12.0), (20.0, 12.0)]


# --------------------------------------------------- hard gates that BITE (regression pins) ---


def test_detached_debris_fails_single_body(tmp_path: Path) -> None:
    """A valid plate plus a small cube floating 100mm away: two disconnected shells. The mesh tier's
    single-body gate must FAIL it -- a part smuggling detached debris is not one buildable object."""
    program = PartProgram(
        metadata=Metadata(name="debris"),
        operations=[
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                width=60.0,
                depth=40.0,
                height=4.0,
                position=Vec3(z=2.0),
            ),
            Box(
                op_id="debris",
                mode=OpMode.ADD,
                width=2.0,
                depth=2.0,
                height=2.0,
                position=Vec3(x=100.0, y=100.0, z=1.0),
            ),
        ],
        constraints=Constraints(min_wall_thickness=2.0),
    )
    report = _validate(program, tmp_path)
    assert "mesh.single_body" in report.failed_check_ids()


# --------------------------------------- declared-feature-not-realized (the flagship escape class) ---


def test_holes_that_miss_the_body_escape_hard_gates(tmp_path: Path) -> None:
    """Four M3 holes DECLARED but positioned far outside the 60x40 footprint: every SUBTRACT is a
    boolean no-op, so the plate is solid and the holes are absent. The hard gates pass (a clean solid
    that conforms to its declaration); only the INFO measured-join sees the absence. A POSITIONAL miss
    -- the mutation suite covers undersize/dropped/solid-pin, not this one."""
    report = _validate(_plate_with_holes(hole_r=1.7, positions=[(200.0, 200.0)] * 4), tmp_path)
    assert report.passed  # the escape, stated honestly
    msg = _measured_msg(report)
    assert msg is not None and "MEASURED-MISSING" in msg


def test_oversize_holes_escape_hard_gates(tmp_path: Path) -> None:
    """Holes cut at 5mm (r=2.5) while declared M3 (3.4mm): the holes EXIST but at the wrong diameter.
    The declaration conforms (3.4 is a valid M3) and the mesh is a clean solid, so the hard gates pass;
    the measured-join looks for a 3.4mm concave cylinder and does not find one. The complement of the
    suite's UNDERSIZE escape -- a bolt would sit loose, yet nothing hard-fails."""
    report = _validate(_plate_with_holes(hole_r=2.5, positions=_CORNERS), tmp_path)
    assert report.passed  # the escape
    msg = _measured_msg(report)
    assert msg is not None and "MEASURED-MISSING" in msg


# ------------------------------------------ a CLOSED blind spot: the printable-thickness floor ---


def test_degenerate_sheet_fails_the_absolute_thickness_floor(tmp_path: Path) -> None:
    """CLOSED (was a strict-xfail blind spot): a 60x40 plate only 0.05mm thick with no min_wall declared
    passes the mesh solid gates, and the feature tier has no declared floor to enforce -- yet it is
    physically unbuildable (under one FDM layer). The unconditional mesh.min_printable_thickness floor
    now FAILS it, independent of any declared min_wall. A regression pin for the closed escape."""
    program = PartProgram(
        metadata=Metadata(name="sheet"),
        operations=[
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                width=60.0,
                depth=40.0,
                height=0.05,
                position=Vec3(z=0.025),
            )
        ],
        constraints=Constraints(),  # no min_wall, no bbox -> the absolute floor is the only catcher
    )
    report = _validate(program, tmp_path)
    assert not report.passed
    assert "mesh.min_printable_thickness" in report.failed_check_ids()
