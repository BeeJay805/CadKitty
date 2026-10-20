"""bench-007 tier-1 deterministic foundation: the hand-authored actuated NEMA 17
arm joint static-validates (ADR-0012).

The first ACTUATED assembly: motor (non-printed hardware node) + output_arm (grips the
5 mm shaft, revolves about it) + mount (fixed to the motor face). Tier 1 is STATIC --
coupling correct + revolute coaxial with the real shaft + per-body shells -- and the
deferred motor-sweep WARN is scored as a known-deferral via passes_criteria, never a
phantom failure and never a fake clear.

Marked slow: compiles the printed bodies (build123d/OCCT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from tests.fixtures.assembly.arm_joint_graphs import (
    ARM_JOINT_ACCEPTED_WARNS,
    ARM_JOINT_REQUIRED_CHECKS,
    arm_joint_rooted_at_mount,
    good_arm_joint_graph,
)

pytestmark = pytest.mark.slow


def test_arm_joint_scores_pass_under_tier1_criteria(tmp_path: Path) -> None:
    report = validate_assembly(compile_assembly(solve(good_arm_joint_graph()), tmp_path))
    # Tier-1 honest scoring: required checks PASS, the deferred motor-sweep WARN accepted.
    assert report.passes_criteria(
        required_checks=ARM_JOINT_REQUIRED_CHECKS,
        accepted_warns=ARM_JOINT_ACCEPTED_WARNS,
    ), f"failed={report.failed_check_ids()} warned={report.warned_check_ids()}"
    for cid in ARM_JOINT_REQUIRED_CHECKS:
        c = report.check(cid)
        assert c is not None and c.status.value == "pass", cid


def test_arm_joint_valid_regardless_of_root(tmp_path: Path) -> None:
    # Regression (bench-007 dispatch finding): rooting the SAME joint at the mount must
    # still pass -- a fixed mount-to-motor-face joint is not an axial constraint, so it
    # must NOT be held to shaft coaxiality. (Before the fix this FAILed hardware_coaxial
    # at 20.9mm because the bracket's mating port is off the motor axis.)
    report = validate_assembly(compile_assembly(solve(arm_joint_rooted_at_mount()), tmp_path))
    assert report.passes_criteria(
        required_checks=ARM_JOINT_REQUIRED_CHECKS,
        accepted_warns=ARM_JOINT_ACCEPTED_WARNS,
    ), f"failed={report.failed_check_ids()} warned={report.warned_check_ids()}"
    hc = report.check("assembly.hardware_coaxial")
    assert hc is not None and hc.status.value == "pass"


def test_motor_is_not_printed(tmp_path: Path) -> None:
    model = compile_assembly(solve(good_arm_joint_graph()), tmp_path)
    # The motor is a HARDWARE node (never a printed body / deliverable), but it now
    # carries a compiled collision ENVELOPE so the sweep can include it.
    assert {b.id for b in model.bodies} == {"output_arm", "mount"}
    assert [h.id for h in model.hardware] == ["motor"]
    motor = model.find_hardware("motor")
    assert motor is not None and motor.mesh is not None  # envelope, not a deliverable


def test_motor_sweep_is_earned_not_skipped(tmp_path: Path) -> None:
    # The ADR-0012 deferral is CLOSED: the motor envelope mesh lets the swept-collision
    # gate actually run on the hardware-anchored revolute. The claim must be EARNED --
    # a per-joint sweep check exists and PASSes; "clear" is never reported by skipping.
    report = validate_assembly(compile_assembly(solve(good_arm_joint_graph()), tmp_path))
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and motion.status.value == "pass", motion
    sweep = report.check("assembly.sweep.motor__output_arm")
    assert sweep is not None and sweep.status.value == "pass", sweep
    assert "chain vs rest" in (sweep.actual or "")


def test_revolute_axis_coaxial_with_motor_shaft(tmp_path: Path) -> None:
    # The actuation claim: the revolute axis IS the real shaft axis (falls out of the
    # frame-mate solver -- the arm hub mates motor.shaft).
    model = compile_assembly(solve(good_arm_joint_graph()), tmp_path)
    motor = model.find_hardware("motor")
    arm = model.body("output_arm")
    assert motor is not None
    shaft = next(p for p in motor.ports if p.name == "shaft")
    hub = next(p for p in arm.ports if p.name == "hub")
    shaft_axis = motor.transform[:3, :3] @ np.array(shaft.axis.as_tuple())
    hub_axis = arm.transform[:3, :3] @ np.array(hub.axis.as_tuple())
    cos = abs(
        float(np.dot(shaft_axis / np.linalg.norm(shaft_axis), hub_axis / np.linalg.norm(hub_axis)))
    )
    assert cos > 0.999, f"arm hub axis not coaxial with the motor shaft (cos={cos:.4f})"
