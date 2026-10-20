"""Non-printed hardware-node de-risk spike (ADR-0012).

The smallest assembly that exercises a non-printed hardware node: a `nema17_motor`
hardware node + ONE trivial printed `rod` frame-mated to the motor's SHAFT axis. Proves
the node is accepted, the compiler emits only the printed part (the motor produces no
STL), the validator runs without printability on the motor, and -- the Q3 payoff --
the printed part lands COAXIAL with the motor shaft axis straight out of the solver.

Marked slow: compiles a body (build123d/OCCT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _motor_and_rod_graph():  # type: ignore[no-untyped-def]
    """A nema17_motor hardware node (root) + a printed sleeve revolving about the shaft.

    A REVOLUTE joint (the actuated arm's real joint): the sleeve (a stand-in output
    hub) rotates about the motor shaft. PHYSICALLY TRUE since the hardware envelope
    landed: the hub is a TUBE whose bore clears the real 5 mm shaft (the original
    solid-rod stand-in occupied the same space as the shaft -- a real interpenetration
    the gate now catches) and the mate is "align" so the hub sits on the shaft side.
    """
    from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode

    nodes = [
        PartNode(
            id="motor",
            archetype="hardware",
            printed=False,
            params={"hardware": "nema17_motor"},
        ),
        # The canonical printed hub: base-anchored, so an align-mate at the shaft
        # port (the boss-top plane) puts it fully ABOVE the boss with the bore
        # clearing the real 2.5 mm shaft.
        PartNode(
            id="hub",
            archetype="output_arm",
            params={
                "length_x": 20.0,
                "width_y": 8.0,
                "thickness": 4.0,
                "hub_outer_r": 6.0,
                "hub_height": 10.0,
            },
        ),
    ]
    joints = [
        Joint(
            type=JointType.REVOLUTE,
            a="motor.shaft",
            b="hub.hub",
            range_deg=(0.0, 360.0),
            mate="align",
        ),
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="motor")


# ---------------------------------------------------------------------------
# (a) the graph accepts the non-printed node
# ---------------------------------------------------------------------------


def test_graph_accepts_non_printed_hardware_node() -> None:
    graph = _motor_and_rod_graph()
    motor = next(n for n in graph.nodes if n.id == "motor")
    assert motor.printed is False
    assert motor.params["hardware"] == "nema17_motor"
    # A normal node defaults to printed=True (additive: existing graphs unchanged).
    hub = next(n for n in graph.nodes if n.id == "hub")
    assert hub.printed is True


# ---------------------------------------------------------------------------
# (b) the compiler emits ONLY the printed part (the motor produces no STL)
# ---------------------------------------------------------------------------


def test_compiler_keeps_hardware_out_of_the_bodies(tmp_path: Path) -> None:
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly

    model = compile_assembly(solve(_motor_and_rod_graph()), tmp_path)
    # Exactly one printed body (the hub); the motor is a HARDWARE entry -- never a
    # deliverable body -- that now carries a compiled collision ENVELOPE so the
    # swept gate can include it (the ADR-0012 deferral is closed).
    assert [b.id for b in model.bodies] == ["hub"]
    assert [h.id for h in model.hardware] == ["motor"]
    motor = model.find_hardware("motor")
    assert motor is not None and motor.mesh is not None and motor.kind == "nema17_motor"
    assert model.body("hub").cad.stl_path.exists()


# ---------------------------------------------------------------------------
# (c) the validator passes without running printability on the motor
# ---------------------------------------------------------------------------


def test_validator_skips_printability_on_hardware(tmp_path: Path) -> None:
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly
    from forge_mind.validator.assembly_checks import validate_assembly

    report = validate_assembly(compile_assembly(solve(_motor_and_rod_graph()), tmp_path))
    # No per-body mesh check exists for the motor (it was never compiled)...
    assert report.check("assembly.body.motor.single_body") is None
    # ...but the printed hub IS mesh-checked.
    hub_shell = report.check("assembly.body.hub.single_body")
    assert hub_shell is not None and hub_shell.status.value == "pass"
    # No HARD failures: the validator handled the mesh-less motor gracefully.
    assert report.failed_check_ids() == [], report.failed_check_ids()


def test_hardware_coaxial_check_actually_validates(tmp_path: Path) -> None:
    # Codex P1: the hardware-anchored joint must be VALIDATED (frame coaxial), not
    # silently skipped. The validator emits assembly.hardware_coaxial and it PASSES
    # because the hub is mated to the motor shaft frame.
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly
    from forge_mind.validator.assembly_checks import validate_assembly

    report = validate_assembly(compile_assembly(solve(_motor_and_rod_graph()), tmp_path))
    hw = report.check("assembly.hardware_coaxial")
    assert hw is not None and hw.status.value == "pass", hw


def test_hardware_anchored_motion_is_swept_not_skipped(tmp_path: Path) -> None:
    # Codex P2's intent (never falsely clear) is now EXCEEDED: the motor envelope
    # mesh lets the swept-collision gate actually RUN on the hardware-anchored
    # revolute. The claim is earned -- a per-joint sweep check exists and PASSes.
    # (The meshless-hardware WARN path is pinned in tests/unit/assembly/test_arm_lane.)
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly
    from forge_mind.validator.assembly_checks import validate_assembly

    report = validate_assembly(compile_assembly(solve(_motor_and_rod_graph()), tmp_path))
    motion = report.check("assembly.range_of_motion")
    assert motion is not None and motion.status.value == "pass", motion
    sweep = report.check("assembly.sweep.motor__hub")
    assert sweep is not None and sweep.status.value == "pass", sweep


# ---------------------------------------------------------------------------
# (Q3 payoff) the printed part lands COAXIAL with the motor shaft axis -- for free
# ---------------------------------------------------------------------------


def test_coaxial_to_shaft_falls_out_of_the_solver(tmp_path: Path) -> None:
    from forge_mind.assembly.solver import solve
    from forge_mind.compiler.compile_assembly import compile_assembly

    model = compile_assembly(solve(_motor_and_rod_graph()), tmp_path)
    motor = model.find_hardware("motor")
    assert motor is not None
    hub = model.body("hub")

    # Motor shaft axis in world: the 'shaft' port direction through the motor transform.
    shaft_port = next(p for p in motor.ports if p.name == "shaft")
    shaft_axis = motor.transform[:3, :3] @ np.array(shaft_port.axis.as_tuple())
    shaft_origin = (motor.transform @ np.array([*shaft_port.origin.as_tuple(), 1.0]))[:3]

    # Hub axis in world: its 'hub' port direction through the hub transform (the
    # fixture's printed hub is an output_arm; its mate port is named "hub").
    hub_port = next(p for p in hub.ports if p.name == "hub")
    hub_axis = hub.transform[:3, :3] @ np.array(hub_port.axis.as_tuple())
    hub_origin = (hub.transform @ np.array([*hub_port.origin.as_tuple(), 1.0]))[:3]

    # Coaxial: axes parallel (|dot| ~ 1) AND hub origin lies on the shaft line.
    cos = abs(
        float(np.dot(shaft_axis / np.linalg.norm(shaft_axis), hub_axis / np.linalg.norm(hub_axis)))
    )
    assert cos > 0.999, f"hub axis not parallel to shaft axis (cos={cos:.4f})"
    w = hub_origin - shaft_origin
    perp = w - np.dot(w, shaft_axis / np.linalg.norm(shaft_axis)) * (
        shaft_axis / np.linalg.norm(shaft_axis)
    )
    assert float(np.linalg.norm(perp)) < 1e-6, "hub origin not on the motor shaft axis"
