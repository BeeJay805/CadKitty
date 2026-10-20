"""End-to-end prismatic slide (ADR-0010 generalization): graph -> solve -> compile -> validate.

The proof that the assembly spine is NOT a hinge template: a linear piston-in-tube
slide, composed from generic archetypes with a NEW joint type (prismatic), passes
the SAME relational + kinematic validator -- and its defect variants fail for the
correct, isolated reasons (interference -> no fit, welded -> no DOF, over-travel ->
escapes the tube).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.assembly.joints import AssemblyGraph
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import ValidationReport
from tests.fixtures.assembly.slide_graphs import (
    good_slide_graph,
    interference_slide_graph,
    over_travel_slide_graph,
    welded_slide_graph,
)

pytestmark = pytest.mark.slow


def _run(graph: AssemblyGraph, tmp_path: Path) -> ValidationReport:
    return validate_assembly(compile_assembly(solve(graph), tmp_path))


def test_good_slide_passes_every_check(tmp_path: Path) -> None:
    report = _run(good_slide_graph(), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    for cid in ("assembly.pin_coaxial", "assembly.pin_clearance", "assembly.range_of_motion"):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    for body in ("tube", "piston"):
        check = report.check(f"assembly.body.{body}.single_body")
        assert check is not None and check.status.value == "pass", body


def test_interference_slide_fails_clearance(tmp_path: Path) -> None:
    report = _run(interference_slide_graph(), tmp_path)
    assert not report.passed
    assert "assembly.pin_clearance" in report.failed_check_ids()  # piston fatter than bore


def test_welded_slide_fails_only_motion(tmp_path: Path) -> None:
    report = _run(welded_slide_graph(), tmp_path)
    assert not report.passed
    assert report.failed_check_ids() == ["assembly.range_of_motion"]
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and "FIXED" in (motion.actual or "")


def test_over_travel_slide_fails_only_motion_by_escape(tmp_path: Path) -> None:
    report = _run(over_travel_slide_graph(), tmp_path)
    assert not report.passed
    # The chain gate reports the escape on the per-joint sweep check AND the
    # aggregate range_of_motion (same physical fact). Nothing UNRELATED fails.
    failed = set(report.failed_check_ids())
    assert "assembly.range_of_motion" in failed
    assert all(
        f == "assembly.range_of_motion" or f.startswith("assembly.sweep.") for f in failed
    ), failed
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and "slides out" in (motion.actual or "")
    clearance = report.check("assembly.pin_clearance")
    assert clearance is not None and clearance.status.value == "pass"  # the piston itself fits
