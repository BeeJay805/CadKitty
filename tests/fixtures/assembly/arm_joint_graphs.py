"""Hand-authored reference actuated arm-joint AssemblyGraph (bench-007, ADR-0012).

The first ACTUATED assembly: a non-printed ``nema17_motor`` hardware node + a printed
``output_arm`` that grips the 5 mm shaft (shaft_coupling) and revolves about the shaft
axis, plus a printed ``mount`` fixed to the motor face. Tier 1 is STATIC -- the
revolute's swept-collision vs the motor body is deferred (the motor has no mesh), so
the range_of_motion check WARNs and is scored via passes_criteria(accepted_warns=...).

Reuses the assembly spine end to end: the hardware node (spike), the frame-mate solver
(coaxiality falls out for free), the revolute/fixed joints, and the 5th-tier validator.
The ``mount`` is a ``flat_link`` (a plain printed bar with an axis-+Z port) -- tier-1
only needs a printed body fixed to the motor face; a richer NEMA17-face bracket is the
LLM-front's job, not the static reference.
"""

from __future__ import annotations

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode

# output_arm: a beam with a shaft hub. hub_outer_r leaves a >= min_wall boss wall
# around the 2.7 mm grip bore; the beam reaches out from the hub.
_ARM = {
    "length_x": 60.0,
    "width_y": 12.0,
    "thickness": 5.0,
    "hub_outer_r": 7.0,
    "hub_height": 12.0,
}


def good_arm_joint_graph() -> AssemblyGraph:
    """A correct actuated joint: motor (root, non-printed) + arm gripping + revolving
    about the shaft + a printed face plate bolted to the motor face.

    PHYSICALLY TRUE since the hardware envelope landed: ``mate="align"`` everywhere (an
    oppose-mate flipped the arm/mount INTO the motor body -- invisible while hardware
    had no mesh, a real static-clearance failure now); the mount is a ``face_plate``
    that actually carries the cited NEMA-17 pattern (4x M3 on the 31 mm square) and a
    pilot bore the 22 mm boss registers in -- a bare bar bolted to a face it has no
    holes for fails the mount-pattern gate, as it should."""
    nodes = [
        PartNode(
            id="motor", archetype="hardware", printed=False, params={"hardware": "nema17_motor"}
        ),
        PartNode(id="output_arm", archetype="output_arm", params=dict(_ARM)),
        PartNode(id="mount", archetype="face_plate", params={}),
    ]
    joints = [
        # The arm hub slides onto the shaft (coaxial) AND revolves about it (the DOF).
        Joint(type=JointType.COAXIAL_INSERT, a="motor.shaft", b="output_arm.hub", mate="align"),
        Joint(
            type=JointType.REVOLUTE,
            a="motor.shaft",
            b="output_arm.hub",
            range_deg=(0.0, 360.0),
            mate="align",
        ),
        # The plate bolts to the motor face (rigid, on the cited pattern).
        Joint(type=JointType.FIXED, a="motor.face", b="mount.face_mount", mate="align"),
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="motor")


def arm_joint_rooted_at_mount() -> AssemblyGraph:
    """Same actuated joint, but rooted at the mount (the intuitive 'ground' the LLM
    often picks). The relative geometry is identical -- only the world frame differs --
    so this MUST still validate: a ``fixed`` mount-to-motor-face joint is not an axial
    constraint and must not be held to shaft coaxiality (bench-007 dispatch finding)."""
    g = good_arm_joint_graph()
    return g.model_copy(update={"root": "mount"})


# The acceptance gate for passes_criteria. The motor now carries an envelope mesh,
# so the sweep RUNS (the ADR-0012 deferral is closed): range_of_motion is REQUIRED
# PASS, the mount must carry the cited face pattern, and no WARN is accepted.
ARM_JOINT_REQUIRED_CHECKS = [
    "assembly.body.output_arm.single_body",
    "assembly.body.mount.single_body",
    "assembly.hardware_coaxial",
    "assembly.static_clearance",
    "assembly.mount_pattern.motor.face",
    "assembly.range_of_motion",
    "interfaces.output_arm.bore_grips_shaft",
    "interfaces.output_arm.set_screw_present",
]
ARM_JOINT_ACCEPTED_WARNS: list[str] = []


__all__ = [
    "ARM_JOINT_ACCEPTED_WARNS",
    "ARM_JOINT_REQUIRED_CHECKS",
    "arm_joint_rooted_at_mount",
    "good_arm_joint_graph",
]
