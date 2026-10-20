"""ADR-0130: typed arch-to-plan binding -- joint identity echoed, actuation reconciled.

H4 (TEST_LOG L98/L99) measured the gap this closes: the architecture declared 4 actuators
with full drive bindings, the prompt demanded the servo joints verbatim, and the plan still
died at the gate -- prose does not bind (instance #7). The binding is now typed: the plan
joint realizing an architecture joint carries ``Joint.name``; ``_plan_actuation_issues``
reconciles every ``ActuatorPlacement`` against the joint graph deterministically.

The real-artifact tests consume verbatim copies of the archived H4 pair (law 4: a test
that fabricates its input proves the matcher, not the system).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.agentic.functional import (
    FunctionalPlan,
    PartPlan,
    _plan_actuation_issues,
    _plan_topology_issues,
)
from forge_mind.agentic.mech_arch import (
    ActuatorPlacement,
    MechArchitecture,
    MechJoint,
    arch_context_block,
)
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3

_H4 = Path(__file__).resolve().parents[2] / "fixtures" / "h4"


def _port(name: str) -> Port:
    return Port(
        name=name, origin=Vec3(x=0, y=0, z=0), axis=Vec3(x=1, y=0, z=0), normal=Vec3(x=0, y=0, z=1)
    )


def _mech_joint(name: str, kind: str = "actuated") -> MechJoint:
    return MechJoint(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        axis_dir="+X",
        station_mm=(0.0, 0.0, 100.0),
        bearing="Ø6.0 pin journal in Ø6.35 bore land, running fit 0.35",
        retention="printed shoulder + push-on cap",
    )


def _placement(joint: str, hw_id: str, horn_link: str) -> ActuatorPlacement:
    return ActuatorPlacement(
        joint=joint, hw_id=hw_id, position_mm=(0.0, 0.0, 100.0), shaft_dir="+X", horn_link=horn_link
    )


def _arch(joints: list[MechJoint], actuators: list[ActuatorPlacement]) -> MechArchitecture:
    return MechArchitecture(
        object_name="rig",
        joints=joints,
        actuators=actuators,
        fastening_system="M3 self-tappers into 4.0 bosses",
        seam_strategy="tongue-and-groove",
        wiring_route="spine channel",
        mirrored_pairs=[],
    )


def _mini_arch() -> MechArchitecture:
    return _arch([_mech_joint("shoulder")], [_placement("shoulder", "servo_a", "arm")])


def _mini_plan(joints: list[Joint], extra_parts: list[PartPlan] | None = None) -> FunctionalPlan:
    return FunctionalPlan(
        object_name="rig",
        parts=[
            PartPlan(name="frame", prompt="frame", ports=[_port("bay"), _port("hub")]),
            PartPlan(name="arm", prompt="arm", ports=[_port("horn_boss"), _port("hub")]),
            *(extra_parts or []),
        ],
        assembly_note="arm on frame",
        joints=joints,
    )


def _mount(a: str = "frame.bay") -> Joint:
    return Joint(type=JointType.FIXED, a=a, b="servo_a.mount")


def _horn(name: str | None = "shoulder", b: str = "arm.horn_boss") -> Joint:
    return Joint(type=JointType.REVOLUTE, name=name, a="servo_a.horn", b=b, range_deg=(-45, 45))


# ---- the real H4 artifact ----------------------------------------------------------------


def _h4_pair() -> tuple[FunctionalPlan, MechArchitecture]:
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints

    plan = FunctionalPlan.model_validate_json((_H4 / "plan.json").read_text(encoding="utf-8"))
    arch = MechArchitecture.model_validate_json(
        (_H4 / "mech_arch.json").read_text(encoding="utf-8")
    )
    return _canonicalize_plan_endpoints(plan), arch


def test_h4_real_arch_still_validates_under_the_new_invariants() -> None:
    """Schema tightening must break no archived artifact (all five real arches audited)."""
    _, arch = _h4_pair()
    assert len(arch.actuators) == 4


def test_h4_archived_plan_gets_one_structured_correction_per_actuated_joint() -> None:
    """The pre-ADR plan carries no names, so all four actuated joints are unrealized --
    and each correction is copy-actionable (names the joint, the horn, the driven link)."""
    plan, arch = _h4_pair()
    issues = _plan_actuation_issues(plan, arch)
    for a in arch.actuators:
        mine = [i for i in issues if f"'{a.joint}'" in i]
        assert mine, f"no correction for {a.joint}"
        assert any(f"{a.hw_id}.horn" in i and a.horn_link in i and "name=" in i for i in mine)


def test_h4_minimally_corrected_plan_passes_topology_and_actuation() -> None:
    """The plan H4's replan SHOULD have produced: drop the 4 printed twins, name the 4
    horn joints. Everything else is the archived artifact verbatim."""
    plan, arch = _h4_pair()
    twins = {a.hw_id for a in arch.actuators}
    by_horn = {f"{a.hw_id}.horn": a.joint for a in arch.actuators}
    corrected = plan.model_copy(
        update={
            "parts": [p for p in plan.parts if p.name not in twins],
            "joints": [
                j.model_copy(update={"name": by_horn[j.a]}) if j.a in by_horn else j
                for j in plan.joints
            ],
        }
    )
    hw_ports = {a.hw_id: {"horn", "mount"} for a in arch.actuators}
    assert _plan_topology_issues(corrected, hw_ports) == []
    assert _plan_actuation_issues(corrected, arch) == []


# ---- the six holes, synthetically pinned -------------------------------------------------


def test_complete_binding_passes() -> None:
    plan = _mini_plan([_mount(), _horn()])
    assert _plan_actuation_issues(plan, _mini_arch()) == []


def test_missing_horn_joint_demands_the_exact_row() -> None:
    plan = _mini_plan([_mount()])
    issues = _plan_actuation_issues(plan, _mini_arch())
    assert any("name='shoulder'" in i and "servo_a.horn" in i and "'arm'" in i for i in issues)


def test_mere_mount_occurrence_is_not_binding() -> None:
    """Hole 1: a servo present only through its mount used to satisfy the topology gate."""
    plan = _mini_plan(
        [_mount(), Joint(type=JointType.REVOLUTE, a="frame.hub", b="arm.hub")]  # horn: nowhere
    )
    assert _plan_topology_issues(plan, {"servo_a": {"horn", "mount"}}) == []  # the old blind spot
    assert _plan_actuation_issues(plan, _mini_arch())  # the reconciliation sees it


def test_right_endpoints_without_the_name_get_a_name_it_correction() -> None:
    plan = _mini_plan([_mount(), _horn(name=None)])
    issues = _plan_actuation_issues(plan, _mini_arch())
    assert len(issues) == 1 and "name='shoulder'" in issues[0]


def test_horn_on_the_wrong_link_names_the_declared_one() -> None:
    """Hole 2: a horn bolted to the wrong link must not pass."""
    plan = _mini_plan([_mount(), _horn(b="frame.hub")])
    issues = _plan_actuation_issues(plan, _mini_arch())
    assert any("'arm'" in i and "frame" in i for i in issues)


def test_missing_mount_is_demanded_on_the_reference_side() -> None:
    plan = _mini_plan([_horn()])
    issues = _plan_actuation_issues(plan, _mini_arch())
    assert any("servo_a.mount" in i and "rigid mount" in i for i in issues)


def test_same_rigid_body_mount_is_a_dead_drive() -> None:
    """Hole 5: case and horn on the SAME rigid link -- directly or through a fixed chain."""
    direct = _mini_plan([_mount(a="arm.hub"), _horn()])
    assert any("SAME rigid body" in i for i in _plan_actuation_issues(direct, _mini_arch()))
    bracket = PartPlan(name="bracket", prompt="bracket", ports=[_port("pad"), _port("bay")])
    chained = _mini_plan(
        [
            Joint(type=JointType.FIXED, a="arm.hub", b="bracket.pad"),
            _mount(a="bracket.bay"),
            _horn(),
        ],
        extra_parts=[bracket],
    )
    assert any("SAME rigid body" in i for i in _plan_actuation_issues(chained, _mini_arch()))


def test_invented_and_duplicate_plan_names_are_refused() -> None:
    plan = _mini_plan(
        [
            _mount(),
            _horn(),
            Joint(type=JointType.REVOLUTE, name="elbow", a="frame.hub", b="arm.hub"),
        ]
    )
    assert any("'elbow'" in i for i in _plan_actuation_issues(plan, _mini_arch()))
    dup = _mini_plan(
        [
            _mount(),
            _horn(),
            Joint(type=JointType.REVOLUTE, name="shoulder", a="frame.hub", b="arm.hub"),
        ]
    )
    assert any("2 plan joints" in i for i in _plan_actuation_issues(dup, _mini_arch()))


def test_no_actuators_is_vacuous() -> None:
    plan = _mini_plan([Joint(type=JointType.REVOLUTE, a="frame.hub", b="arm.hub")])
    arch = _arch([_mech_joint("hinge", kind="passive")], [])
    assert _plan_actuation_issues(plan, arch) == []


def test_driven_link_absent_from_the_plan_is_named() -> None:
    plan = FunctionalPlan(
        object_name="rig",
        parts=[
            PartPlan(name="frame", prompt="frame", ports=[_port("bay")]),
            PartPlan(name="lid", prompt="lid", ports=[_port("rim")]),  # not the driven link
        ],
        assembly_note="rig",
        joints=[
            _mount(),
            Joint(type=JointType.REVOLUTE, name="shoulder", a="servo_a.horn", b="arm.horn_boss"),
        ],
    )
    issues = _plan_actuation_issues(plan, _mini_arch())
    assert any("'arm'" in i and "not a plan part" in i for i in issues)


def test_near_miss_demands_the_horn_interface_not_just_a_name() -> None:
    """Review-caught: an unnamed joint through the WRONG interface (mount, not horn) must
    get the full row correction -- 'just name it' would stage the diagnosis across two
    gates and burn the single replan."""
    wrong_iface = _mini_plan(
        [
            _mount(),
            Joint(type=JointType.REVOLUTE, a="servo_a.mount", b="arm.hub", range_deg=(-45, 45)),
        ]
    )
    issues = _plan_actuation_issues(wrong_iface, _mini_arch())
    assert any("add a revolute joint carrying name='shoulder'" in i for i in issues)
    assert not any("set name=" in i for i in issues)


def test_mount_landing_on_hardware_or_itself_is_not_anchored() -> None:
    """Review-caught false PASS: a mount joint ending on another hardware entity (or on
    the actuator itself) anchored nothing yet counted as mounted."""
    self_mount = _mini_plan(
        [
            Joint(type=JointType.FIXED, a="servo_a.horn", b="servo_a.mount"),
            _horn(),
        ]
    )
    issues = _plan_actuation_issues(self_mount, _mini_arch())
    assert any("must land on a printed housing part" in i for i in issues)

    arch2 = _arch(
        [_mech_joint("shoulder"), _mech_joint("elbow")],
        [_placement("shoulder", "servo_a", "arm"), _placement("elbow", "servo_b", "frame")],
    )
    mount_to_mount = _mini_plan(
        [
            Joint(type=JointType.FIXED, a="servo_a.mount", b="servo_b.mount"),
            _horn(),
            Joint(
                type=JointType.REVOLUTE,
                name="elbow",
                a="servo_b.horn",
                b="frame.hub",
                range_deg=(-45, 45),
            ),
        ]
    )
    issues = _plan_actuation_issues(mount_to_mount, arch2)
    assert sum("must land on a printed housing part" in i for i in issues) == 2


def test_absent_driven_link_reports_every_binding_defect_in_one_pass() -> None:
    """Review-caught staged diagnosis: the absent-link case must not suppress the horn
    and mount checks -- the bounded replan runs once, so the correction must be complete."""
    plan = _mini_plan([Joint(type=JointType.REVOLUTE, a="frame.hub", b="arm.hub")])
    arch = _arch([_mech_joint("shoulder")], [_placement("shoulder", "servo_a", "paddle")])
    issues = _plan_actuation_issues(plan, arch)
    assert any("'paddle' is not a plan part" in i for i in issues)
    assert any("not realized" in i for i in issues)
    assert any("no rigid mount" in i for i in issues)


# ---- arch invariants (holes 3 + 6) -------------------------------------------------------


def test_arch_refuses_duplicate_joint_names() -> None:
    with pytest.raises(ValidationError, match="duplicate joint name"):
        _arch([_mech_joint("shoulder"), _mech_joint("shoulder")], [])


def test_arch_refuses_duplicate_hw_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate hw_id"):
        _arch(
            [_mech_joint("shoulder"), _mech_joint("hip")],
            [_placement("shoulder", "servo_a", "arm"), _placement("hip", "servo_a", "leg")],
        )


def test_arch_refuses_placement_on_undeclared_or_passive_joint() -> None:
    with pytest.raises(ValidationError, match="undeclared joint"):
        _arch([_mech_joint("shoulder")], [_placement("elbow", "servo_a", "arm")])
    with pytest.raises(ValidationError, match="passive"):
        _arch([_mech_joint("shoulder", kind="passive")], [_placement("shoulder", "servo_a", "arm")])


def test_arch_refuses_unplaced_or_double_placed_actuated_joints() -> None:
    with pytest.raises(ValidationError, match="no actuator placement"):
        _arch([_mech_joint("shoulder")], [])
    with pytest.raises(ValidationError, match="placement"):
        _arch(
            [_mech_joint("shoulder")],
            [_placement("shoulder", "servo_a", "arm"), _placement("shoulder", "servo_b", "arm")],
        )


# ---- the model-facing half ---------------------------------------------------------------


def test_arch_context_block_carries_the_typed_rows() -> None:
    """Structure the model copies verbatim: per actuator, the exact joint rows to emit,
    the name to echo, and the no-twin rule."""
    block = arch_context_block(_mini_arch())
    assert "name='shoulder'" in block
    assert "servo_a.horn" in block and "servo_a.mount" in block
    assert "'arm'" in block
    assert "parts" in block  # the do-not-redeclare rule
