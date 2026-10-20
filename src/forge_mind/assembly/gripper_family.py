"""The two-finger gripper family: a deterministic gear-coupled gripper (MG996R).

One servo, two printed fingers with full involute gear discs at their roots, a cited
ISO 2338 dowel as the idler pivot. The graph is a TREE for placement (palm -> servo ->
drive finger; palm -> dowel -> idler finger) plus ONE GEAR_MESH constraint edge that
couples the two finger rotations -- the solver never places across it, and the sweep
drives the coupled trajectory (driver theta, follower -theta) instead of sweeping
either finger alone (ADR-0044).

Zero LLM calls end to end; every claim is the assembly gate's to issue.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.agentic.gears import mesh_clock_deg
from forge_mind.assembly.gripper_parts import (
    GRIPPER_BOSS_BORE_BOTTOM_MM,
    GRIPPER_DOWEL_LENGTH_MM,
    GRIPPER_DOWEL_SIZE,
    GRIPPER_FINGER_PLANE_MM,
    GRIPPER_GEAR_MODULE,
    GRIPPER_GEAR_TEETH,
    GripperDesign,
)
from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode


@dataclass(frozen=True)
class GripperSpec:
    """The family's parametric surface. ``payload`` names the CITED payload-object
    registry key the fingers are sized for; ``payload_declared`` records whether the
    USER's prompt named that object (the aperture gate only fires on a declared
    payload -- a bare 'gripper' prompt makes no aperture claim)."""

    payload: str = "beverage_can_355ml"
    payload_declared: bool = False
    name: str = "gripper"


def build_gripper_graph(spec: GripperSpec) -> AssemblyGraph:
    """The gripper as an AssemblyGraph. Placement tree: palm roots everything; the
    GEAR_MESH edge only couples the two finger rotations (never places)."""
    design = GripperDesign(spec.payload)
    lo, hi = design.drive_range_deg()

    nodes = [
        PartNode(id="palm", archetype="gripper_palm", params={}),
        PartNode(
            id="servo_g",
            archetype="hardware",
            params={"hardware": "mg996r_servo"},
            printed=False,
        ),
        PartNode(
            id="dowel",
            archetype="hardware",
            params={
                "hardware": "dowel_pin",
                "size": GRIPPER_DOWEL_SIZE,
                "length_mm": GRIPPER_DOWEL_LENGTH_MM,
                "pivot_z_mm": GRIPPER_FINGER_PLANE_MM - GRIPPER_BOSS_BORE_BOTTOM_MM,
            },
            printed=False,
        ),
        PartNode(
            id="finger_drive",
            archetype="gripper_finger",
            params={"side": "drive", "payload": spec.payload},
        ),
        PartNode(
            id="finger_idler",
            archetype="gripper_finger",
            params={"side": "idler", "payload": spec.payload},
        ),
    ]
    joints = [
        Joint(type=JointType.FIXED, a="palm.cradle_mount", b="servo_g.mount", mate="align"),
        Joint(type=JointType.FIXED, a="palm.dowel_seat", b="dowel.seat", mate="align"),
        Joint(
            type=JointType.REVOLUTE,
            a="servo_g.horn",
            b="finger_drive.hub",
            range_deg=(lo, hi),
            clearance_class="running_fit",
            mate="align",
        ),
        Joint(
            type=JointType.REVOLUTE,
            a="dowel.pivot",
            b="finger_idler.hub",
            range_deg=(-hi, -lo),  # the mirrored travel (follower = -driver)
            clearance_class="running_fit",
            mate="align",
        ),
        Joint(
            type=JointType.GEAR_MESH,
            a="finger_drive.hub",
            b="finger_idler.hub",
            module=GRIPPER_GEAR_MODULE,
            teeth_a=GRIPPER_GEAR_TEETH,
            teeth_b=GRIPPER_GEAR_TEETH,
            clock_a_deg=0.0,
            clock_b_deg=mesh_clock_deg(GRIPPER_GEAR_TEETH),
            mate="align",
        ),
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="palm")


def gripper_joint_display_names() -> dict[str, str]:
    """Human names for the moving joints, keyed by the moving part id."""
    return {"finger_drive": "grip (drive)", "finger_idler": "grip (idler)"}


__all__ = ["GripperSpec", "build_gripper_graph", "gripper_joint_display_names"]
