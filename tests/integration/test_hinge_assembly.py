"""End-to-end assembly spine (ADR-0010): graph -> solve -> compile -> validate.

THE SPIKE DELIVERABLE: a hand-authored 2-leaf hinge, composed from generic
archetypes, compiles to three single-body STLs and passes a relational + kinematic
validator -- and three single-defect variants each fail for the correct, isolated
reason (welded -> no motion, sub-clearance -> no fit, binding -> collision).

Marked slow: each test compiles 3 bodies (build123d/OCCT) and samples a motion
sweep, so it is excluded from the fast unit pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.assembly.joints import AssemblyGraph
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import ValidationReport
from tests.fixtures.assembly.hinge_graphs import (
    binding_hinge_graph,
    good_hinge_graph,
    sub_clearance_hinge_graph,
    welded_hinge_graph,
)

pytestmark = pytest.mark.slow


def _run(graph: AssemblyGraph, tmp_path: Path) -> ValidationReport:
    return validate_assembly(compile_assembly(solve(graph), tmp_path))


def test_good_hinge_passes_every_check(tmp_path: Path) -> None:
    report = _run(good_hinge_graph(), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    assert report.failed_check_ids() == []
    for cid in ("assembly.pin_coaxial", "assembly.pin_clearance", "assembly.range_of_motion"):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    # Every body is a single watertight shell (the multi-body relaxation, per body).
    for body in ("leafA", "leafB", "pin"):
        check = report.check(f"assembly.body.{body}.single_body")
        assert check is not None and check.status.value == "pass", body


def test_welded_hinge_fails_only_motion(tmp_path: Path) -> None:
    report = _run(welded_hinge_graph(), tmp_path)
    assert not report.passed
    assert report.failed_check_ids() == ["assembly.range_of_motion"]  # isolated defect
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and "FIXED" in (motion.actual or "")


def test_sub_clearance_hinge_fails_only_clearance(tmp_path: Path) -> None:
    report = _run(sub_clearance_hinge_graph(), tmp_path)
    assert not report.passed
    assert report.failed_check_ids() == ["assembly.pin_clearance"]  # isolated defect
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and motion.status.value == "pass"  # geometry still moves


def test_binding_hinge_fails_only_motion_by_collision(tmp_path: Path) -> None:
    report = _run(binding_hinge_graph(), tmp_path)
    assert not report.passed
    # The one physical defect (knuckle interference) surfaces on the per-joint sweep,
    # the aggregate range_of_motion, AND static_clearance (a binding hinge already
    # interferes at the home pose). Nothing UNRELATED fails.
    failed = set(report.failed_check_ids())
    assert "assembly.range_of_motion" in failed
    assert failed <= {
        "assembly.range_of_motion",
        "assembly.sweep.leafA__leafB",
        "assembly.static_clearance",
    }, failed
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and "hits" in (motion.actual or "")
    clearance = report.check("assembly.pin_clearance")
    assert clearance is not None and clearance.status.value == "pass"  # the pin itself is fine
