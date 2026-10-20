"""Systematic false-PASS sweep (RSI Rung-2) -- parametrize the adversarial oracle's frontier.

``test_adversarial_oracle.py`` hand-picks individual escapes (a hole that misses the body, an
oversize hole, a degenerate sheet). This is the SYSTEMATIC sibling the BACKLOG's "harden the oracle"
rung asks for: it sweeps each defect DIMENSION across its range and asserts the COMPUTED verdict at
every point -- never an assumed one -- so the boundary between "a gate bites" and "only the advisory
join notices" is pinned, and any point that slips through BOTH is found.

Each sweep point reduces to one of:

  - HARD-FAIL        a tier gate fails the report outright (the validator wins).
  - MEASURED-MISSING hard gates pass; the advisory measured-join (INFO) flags the declared-vs-realized
                     gap (the known frontier -- advisory, not yet a hard gate).
  - MEASURED-TRUE    the body contains the declared feature as measured.
  - UNMEASURABLE     concavity / geometry could not be determined (stated, never guessed).

A MEASURED-TRUE verdict on a feature that is NOT actually the declared feature is a FALSE-PASS: it
is PINNED honestly here (NORTH_STAR Section 6) so the blind spot is visible. Pinning is the loop's job;
a hardening that removes the pin is a separate, council-gated change -- this file never softens a gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.conformance import MEASURED_TOL_MM, HardwareFeature
from forge_mind.validator.mesh_checks import MIN_PRINTABLE_THICKNESS_MM
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.validator.run import validate

pytestmark = pytest.mark.slow  # compile+validate cycles; keep out of the hot loop

_M3_CLEARANCE_MM = 3.4  # registry M3 normal clearance; expected bore radius 1.7 mm
_M3_R = _M3_CLEARANCE_MM / 2.0
_PLATE_W, _PLATE_D, _PLATE_H = 60.0, 40.0, 4.0
_MEASURED_ID = "conformance.measured.clearance_hole.M3"


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    return validate(compile_program(program, work), program)


def _classify(report: ValidationReport) -> str:
    """Reduce a report to one verdict token, COMPUTED from the actual checks (never assumed)."""
    if not report.passed:
        return "HARD-FAIL"
    check = report.check(_MEASURED_ID)
    if check is None:
        return "NO-MEASURED-CHECK"
    assert check.status == CheckStatus.INFO  # the join is advisory by construction
    for token in ("MEASURED-TRUE", "MEASURED-MISSING", "UNMEASURABLE"):
        if token in check.message:
            return token
    return "UNKNOWN"


def _plate(
    *, ops_extra: list[Any], min_wall: float = 2.0, declare_hole: bool = True
) -> PartProgram:
    """A 60x40x4 plate plus ``ops_extra``, with one M3 clearance hole DECLARED when ``declare_hole``.
    The declaration is always a valid M3 (4b conformance always passes); the GEOMETRY is the variable
    under test."""
    ops: list[Any] = [
        Box(
            op_id="plate",
            mode=OpMode.ADD,
            width=_PLATE_W,
            depth=_PLATE_D,
            height=_PLATE_H,
            position=Vec3(z=_PLATE_H / 2),
        ),
        *ops_extra,
    ]
    return PartProgram(
        metadata=Metadata(name="sweep_plate"),
        operations=ops,
        constraints=Constraints(
            min_wall_thickness=min_wall, max_bbox=Vec3(x=_PLATE_W, y=_PLATE_D, z=_PLATE_H)
        ),
        hardware_features=(
            [
                HardwareFeature(
                    category="clearance_hole", key="M3", nominal_mm=_M3_CLEARANCE_MM, where="m"
                )
            ]
            if declare_hole
            else []
        ),
    )


def _through_hole(*, radius: float, x: float = 0.0, y: float = 0.0) -> Cylinder:
    """A through bore (height = 2x plate) at (x, y)."""
    return Cylinder(
        op_id="hole",
        mode=OpMode.SUBTRACT,
        position=Vec3(x=x, y=y, z=_PLATE_H / 2),
        radius=radius,
        height=_PLATE_H * 2,
    )


def _blind_hole(*, depth: float, radius: float = _M3_R) -> Cylinder:
    """A blind pocket ``depth`` mm down from the TOP face (depth < plate height = NOT through). The
    cutter pokes 1mm above the plate so the top opens cleanly; it bottoms out at H - depth."""
    bottom = _PLATE_H - depth
    top = _PLATE_H + 1.0
    return Cylinder(
        op_id="hole",
        mode=OpMode.SUBTRACT,
        position=Vec3(x=0.0, y=0.0, z=(bottom + top) / 2.0),
        radius=radius,
        height=top - bottom,
    )


# ----------------------------------------------------------- Sweep 1: radial position of the hole ---


@pytest.mark.parametrize(
    ("x_off", "expect"),
    [
        (0.0, "MEASURED-TRUE"),  # centered, fully on the body
        (10.0, "MEASURED-TRUE"),  # on the body
        (25.0, "MEASURED-TRUE"),  # still fully inside (edge at 30; bore spans 23.3..26.7)
        (
            50.0,
            "MEASURED-MISSING",
        ),  # fully off the 60-wide plate -> boolean no-op, the known escape
        (200.0, "MEASURED-MISSING"),  # far off -> solid plate
    ],
)
def test_sweep_hole_radial_offset(x_off: float, expect: str, tmp_path: Path) -> None:
    """As a correctly-sized M3 bore slides off the footprint, the body stops containing it. A hole on
    the body reads MEASURED-TRUE; one fully off reads MEASURED-MISSING (hard gates pass -- the
    declared-vs-realized escape). The invariant: a misplaced bore NEVER reads MEASURED-TRUE."""
    report = _validate(_plate(ops_extra=[_through_hole(radius=_M3_R, x=x_off)]), tmp_path)
    assert _classify(report) == expect


# --------------------------------------------------------- Sweep 2: bore diameter vs the declaration ---


@pytest.mark.parametrize(
    ("radius", "expect"),
    [
        (_M3_R, "MEASURED-TRUE"),  # exact
        (_M3_R + 0.02, "MEASURED-TRUE"),  # within MEASURED_TOL_MM/2 (0.025) on radius
        (_M3_R + 0.04, "MEASURED-MISSING"),  # outside tol -> wrong size, flagged
        (_M3_R - 0.10, "MEASURED-MISSING"),  # undersize -> flagged
        (2.5, "MEASURED-MISSING"),  # 5mm bore declared M3 -> bolt sits loose, flagged
    ],
)
def test_sweep_bore_diameter_mismatch(radius: float, expect: str, tmp_path: Path) -> None:
    """The declaration is a valid M3 (4b always PASSES); only the measured-join sees the wrong
    diameter. The boundary is MEASURED_TOL_MM/2 on the radius -- pin it so a future tol change is
    a deliberate, visible edit."""
    assert MEASURED_TOL_MM == 0.05  # the boundary this sweep pins
    report = _validate(_plate(ops_extra=[_through_hole(radius=radius)]), tmp_path)
    assert _classify(report) == expect


# ------------------------------------------ Sweep 3: bore DEPTH -- the through-ness blind spot ---


@pytest.mark.parametrize("depth", [1.0, 2.0])
def test_sweep_blind_hole_now_reads_missing(depth: float, tmp_path: Path) -> None:
    """The ADR-0054 through-ness escape, CLOSED (ADR-0082): a clearance hole is a THROUGH
    feature; the measured-join now probes both ends of the bore's centerline
    (``geometry.stack_pierces``) and a blind pocket -- floor material beyond one end -- no
    longer satisfies it. The report still PASSES overall (the join is INFO-tier; no gate was
    added), but the verdict is the honest MEASURED-MISSING, which drives the author feedback
    and the agentic completeness gate."""
    report = _validate(_plate(ops_extra=[_blind_hole(depth=depth)]), tmp_path)
    assert report.passed  # INFO-tier join: still no hard gate (unchanged, deliberate)
    assert _classify(report) == "MEASURED-MISSING"  # the blind pocket no longer reads through


@pytest.mark.parametrize("depth", [_PLATE_H, _PLATE_H * 2])
def test_sweep_through_hole_measures_true(depth: float, tmp_path: Path) -> None:
    """The honest companion: a bore that DOES go through reads MEASURED-TRUE legitimately. Without
    this, the FALSE-PASS pin above could be mistaken for the join simply always saying TRUE."""
    report = _validate(_plate(ops_extra=[_blind_hole(depth=depth)]), tmp_path)
    assert report.passed
    assert _classify(report) == "MEASURED-TRUE"


# --------------------------------- Sweep 4: the absolute printable-thickness floor boundary ---


@pytest.mark.parametrize(
    ("height", "fails"),
    [
        (0.05, True),  # half a layer -> unbuildable
        (0.08, True),  # below the 0.1mm floor
        (MIN_PRINTABLE_THICKNESS_MM, False),  # exactly one layer -> the floor is inclusive
        (0.5, False),  # comfortably printable
    ],
)
def test_sweep_printable_thickness_floor(height: float, fails: bool, tmp_path: Path) -> None:
    """A featureless plate with NO declared min_wall: the feature tier has no floor to enforce, so the
    UNCONDITIONAL mesh.min_printable_thickness (0.1mm) is the only catcher. Pin its exact boundary."""
    program = PartProgram(
        metadata=Metadata(name="sheet"),
        operations=[
            Box(
                op_id="plate",
                mode=OpMode.ADD,
                width=_PLATE_W,
                depth=_PLATE_D,
                height=height,
                position=Vec3(z=height / 2.0),
            )
        ],
        constraints=Constraints(),  # no min_wall, no bbox -> the absolute floor is the only catcher
    )
    report = _validate(program, tmp_path)
    if fails:
        assert "mesh.min_printable_thickness" in report.failed_check_ids()
    else:
        assert "mesh.min_printable_thickness" not in report.failed_check_ids()
