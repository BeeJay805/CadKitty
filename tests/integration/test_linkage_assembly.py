"""End-to-end 2-bar linkage (ADR-0010, third mechanism family): a kinematic chain.

Proves the assembly spine composes and validates a SERIAL chain of joints -- base ->
bar1 -> bar2, each link pivoting on the previous (moving) one -- through the SAME
relational + kinematic validator, with defects failing in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.assembly.joints import AssemblyGraph
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import ValidationReport
from tests.fixtures.assembly.linkage_graphs import (
    good_linkage_graph,
    interference_linkage_graph,
    seized_linkage_graph,
)

pytestmark = pytest.mark.slow


def _run(graph: AssemblyGraph, tmp_path: Path) -> ValidationReport:
    return validate_assembly(compile_assembly(solve(graph), tmp_path))


def test_good_linkage_passes_every_check(tmp_path: Path) -> None:
    report = _run(good_linkage_graph(), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    for cid in ("assembly.pin_coaxial", "assembly.pin_clearance", "assembly.range_of_motion"):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    for body in ("base", "bar1", "bar2", "pin1", "pin2"):
        check = report.check(f"assembly.body.{body}.single_body")
        assert check is not None and check.status.value == "pass", body


def test_interference_linkage_fails_clearance(tmp_path: Path) -> None:
    report = _run(interference_linkage_graph(), tmp_path)
    assert not report.passed
    assert "assembly.pin_clearance" in report.failed_check_ids()  # one pin fatter than its bore


def test_seized_linkage_fails_only_motion(tmp_path: Path) -> None:
    report = _run(seized_linkage_graph(), tmp_path)
    assert not report.passed
    assert report.failed_check_ids() == ["assembly.range_of_motion"]  # both pivots fixed -> no DOF
