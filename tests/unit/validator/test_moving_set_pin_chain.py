"""Regression: the serial pin-jointed chain moving set (motion_sweep.downstream_ids).

The M2 arm shipped `articulates:false` on a yaw "collision" that was a validator
artifact: the placement adapter directs every pin-in-bore revolute pin -> link for BOTH
links a pin joins, so a directed a->b closure from a moving link reached NOTHING it
carries -- each joint's moving set collapsed to a singleton and the lone link swept into
the pins + further links riding on it (phantom self-collisions). `downstream_ids` now
takes the undirected tree-cut (component of b after removing the swept edge), carrying
the whole outboard subtree, with a directed-closure fallback for joints on a closed loop.
"""

from __future__ import annotations

from forge_mind.assembly.joints import Joint, JointType
from forge_mind.validator.motion_sweep import _directed_downstream, downstream_ids

REV = JointType.REVOLUTE
COAX = JointType.COAXIAL_INSERT
GEAR = JointType.GEAR_MESH


def _arm_joints() -> list[Joint]:
    """The M2 arm topology exactly as the adapter directs it: every pin is the a-side
    (source) of both the revolute (to the link it drives) and the coaxial (to the link it
    rides). base is the only true source that is also ground."""
    return [
        Joint(type=REV, a="base.yaw_boss", b="turntable.yaw_bore", range_deg=(0.0, 180.0)),
        Joint(
            type=REV, a="shoulder_pin.journal", b="upper_arm.shoulder_bore", range_deg=(0.0, 120.0)
        ),
        Joint(type=COAX, a="shoulder_pin.ends", b="turntable.shoulder_bore"),
        Joint(type=REV, a="elbow_pin.journal", b="forearm.elbow_bore", range_deg=(0.0, 120.0)),
        Joint(type=COAX, a="elbow_pin.ends", b="upper_arm.elbow_bore"),
        Joint(
            type=REV, a="wrist_pin.journal", b="gripper_mount.wrist_bore", range_deg=(0.0, 120.0)
        ),
        Joint(type=COAX, a="wrist_pin.ends", b="forearm.wrist_bore"),
    ]


def _rev(joints: list[Joint], b_part: str) -> Joint:
    return next(j for j in joints if j.type is JointType.REVOLUTE and j.b_part() == b_part)


def test_each_joint_carries_its_full_outboard_subtree() -> None:
    """Moving a link carries every pin + link that physically rides on it, not just the
    single b-link. These are the exact sets that make the arm articulate honestly."""
    js = _arm_joints()
    assert downstream_ids(js, _rev(js, "turntable")) == {
        "turntable",
        "shoulder_pin",
        "upper_arm",
        "elbow_pin",
        "forearm",
        "wrist_pin",
        "gripper_mount",
    }
    assert downstream_ids(js, _rev(js, "upper_arm")) == {
        "upper_arm",
        "elbow_pin",
        "forearm",
        "wrist_pin",
        "gripper_mount",
    }
    assert downstream_ids(js, _rev(js, "forearm")) == {"forearm", "wrist_pin", "gripper_mount"}
    assert downstream_ids(js, _rev(js, "gripper_mount")) == {"gripper_mount"}


def test_the_phantom_partners_now_ride_the_mover() -> None:
    """The precise regression: the parts the arm was reported to "collide" with are now
    inside the moving set (they ride the mover), so they can never be a static partner."""
    js = _arm_joints()
    # yaw was reported hitting the shoulder_pin/upper_arm it carries:
    yaw_moving = downstream_ids(js, _rev(js, "turntable"))
    assert {"shoulder_pin", "upper_arm"} <= yaw_moving
    # elbow was reported hitting its own gripper_mount/wrist_pin:
    elbow_moving = downstream_ids(js, _rev(js, "forearm"))
    assert {"wrist_pin", "gripper_mount"} <= elbow_moving


def test_directed_closure_would_collapse_to_singletons() -> None:
    """Documents the bug and guards against a 'simplification' back to the directed
    closure: on this pin->link topology the old logic sees only the single b-link."""
    js = _arm_joints()
    for b in ("turntable", "upper_arm", "forearm"):
        assert _directed_downstream(js, _rev(js, b)) == {b}
        assert downstream_ids(js, _rev(js, b)) != _directed_downstream(js, _rev(js, b))


def test_loop_joint_falls_back_to_directed_unchanged() -> None:
    """A joint on a closed loop has no well-defined subtree (the cut does not split the
    graph): downstream_ids must return the directed closure BYTE-IDENTICALLY, so the
    four-bar / slider-crank / epicyclic coupled drivers are untouched."""
    four_bar = [
        Joint(type=REV, a="ground.p1", b="crank.p1", range_deg=(0.0, 360.0)),
        Joint(type=REV, a="crank.p2", b="coupler.p1"),
        Joint(type=REV, a="coupler.p2", b="rocker.p1"),
        Joint(type=REV, a="rocker.p2", b="ground.p2"),
    ]
    ground_crank = four_bar[0]
    assert downstream_ids(four_bar, ground_crank) == _directed_downstream(four_bar, ground_crank)


def test_gear_mesh_partner_is_not_dragged_through_the_mesh() -> None:
    """GEAR_MESH couples rotations, it does not carry one gear on the other; the undirected
    cut must still exclude the mesh edge so a driven gear's subtree is just itself."""
    train = [
        Joint(type=REV, a="frame.axleA", b="gearA.bore", range_deg=(0.0, 360.0)),
        Joint(type=REV, a="frame.axleB", b="gearB.bore", range_deg=(0.0, 360.0)),
        Joint(type=GEAR, a="gearA.teeth", b="gearB.teeth", module=1.0, teeth_a=20, teeth_b=20),
    ]
    driver = train[0]
    assert downstream_ids(train, driver) == {"gearA"}
    assert "gearB" not in downstream_ids(train, driver)
