"""Plan-time topology gate (the i4-finish audit's headline fix).

The live i4 plan had rod3/rod4/piston3/piston4 in ZERO joints -- the old 3-port cap made a
4-cylinder engine undeclarable, the parts dangled, and the sweep hit them as phantom static
obstacles ~90 minutes in. The gate catches an incomplete joint graph in milliseconds, one
bounded re-decompose fixes it, and a still-broken plan is an honest no_build NOW.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.agentic import functional as fm
from forge_mind.agentic.functional import FunctionalPlan, PartPlan, _plan_topology_issues
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3


def _port(name: str) -> Port:
    return Port(
        name=name, origin=Vec3(x=0, y=0, z=0), axis=Vec3(x=1, y=0, z=0), normal=Vec3(x=0, y=0, z=1)
    )


def _rev(a: str, b: str) -> Joint:
    return Joint(type=JointType.REVOLUTE, a=a, b=b, range_deg=(0, 360))


def test_ports_cap_allows_a_four_cylinder_crankshaft() -> None:
    """The audit's root cause: max_length=3 could not hold a crank's mains + 4 crankpins."""
    crank = PartPlan(
        name="crankshaft",
        prompt="a crank",
        ports=[_port(n) for n in ("main", "pin1", "pin2", "pin3", "pin4", "nose", "tail")],
    )
    assert len(crank.ports) == 7


def test_rigid_plan_has_no_topology_to_check() -> None:
    plan = FunctionalPlan(
        object_name="hammer",
        parts=[PartPlan(name="head", prompt="head"), PartPlan(name="handle", prompt="handle")],
        assembly_note="head on handle",
        joints=[],
    )
    assert _plan_topology_issues(plan) == []


def test_complete_joint_graph_passes() -> None:
    plan = FunctionalPlan(
        object_name="pivot toy",
        parts=[
            PartPlan(name="base", prompt="base", ports=[_port("hub")]),
            PartPlan(name="arm", prompt="arm", ports=[_port("bore")]),
        ],
        assembly_note="arm pivots on base",
        joints=[_rev("base.hub", "arm.bore")],
    )
    assert _plan_topology_issues(plan) == []


def test_dangling_part_missing_port_and_unknown_part_are_all_named() -> None:
    """The exact live-run failure class: parts in zero joints + endpoints on undeclared ports."""
    plan = FunctionalPlan(
        object_name="inline-2",
        parts=[
            PartPlan(name="block", prompt="block", ports=[_port("main_bearing")]),
            PartPlan(name="crankshaft", prompt="crank", ports=[_port("main_journal")]),
            PartPlan(name="rod2", prompt="rod"),  # dangling: appears in no joint
        ],
        assembly_note="engine",
        joints=[
            _rev("block.main_bearing", "crankshaft.main_journal"),
            _rev("crankshaft.crankpin1", "rod1.big_end"),  # crankpin1 undeclared + rod1 unknown
        ],
    )
    issues = " | ".join(_plan_topology_issues(plan))
    assert "rod2" in issues and "NO joint" in issues
    assert "does not declare port 'crankpin1'" in issues
    assert "rod1.big_end" in issues and "does not declare" in issues


def test_port_match_is_slug_insensitive() -> None:
    """'Main Axis' on the joint must find the declared port 'main-axis' (adapter parity)."""
    plan = FunctionalPlan(
        object_name="toy",
        parts=[
            PartPlan(name="base", prompt="base", ports=[_port("main-axis")]),
            PartPlan(name="arm", prompt="arm", ports=[_port("bore")]),
        ],
        assembly_note="arm on base",
        joints=[_rev("base.Main Axis", "arm.bore")],
    )
    assert _plan_topology_issues(plan) == []


def _fake_pipeline(prompt: str, out_dir: Path, **_k: object) -> SimpleNamespace:
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "p.stl"
    stl.write_text("x")
    return SimpleNamespace(
        final_stl=stl,
        final_step=None,
        final_render=None,
        total_cost_usd=0.0,
        prompt=prompt,
        validation=None,
        geometry_passed=True,
        critic_approved=True,
        stopped_reason="accepted",
        iterations=[object()],
        critique=None,
        brief=None,
    )


def test_broken_topology_gets_one_retry_then_honest_no_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = FunctionalPlan(
        object_name="engine",
        parts=[
            PartPlan(name="block", prompt="block", ports=[_port("hub")]),
            PartPlan(name="crank", prompt="crank", ports=[_port("journal")]),
            PartPlan(name="rod3", prompt="rod"),  # dangling
        ],
        assembly_note="engine",
        joints=[_rev("block.hub", "crank.journal")],
    )
    calls = {"n": 0, "callouts": []}

    def _decompose(*_a: object, extra_callout: str | None = None, **_k: object):
        calls["n"] += 1
        calls["callouts"].append(extra_callout)
        return broken, SimpleNamespace(cost_usd=0.0)

    monkeypatch.setattr(fm, "decompose_functional", _decompose)
    monkeypatch.setattr(fm, "run_agentic_pipeline", _fake_pipeline)
    brief = SimpleNamespace(object_name="engine", material=None, visual_description="x")
    res = fm.produce_functional_part("an engine", tmp_path, client=object(), brief=brief)
    assert calls["n"] == 2  # one retry, no more
    assert calls["callouts"][1] and "PLAN TOPOLOGY ERROR" in calls["callouts"][1]
    assert res.disposition == "no_build"
    assert res.failure is not None and "plan topology" in res.failure
    assert (tmp_path / "plan.json").exists()  # the surviving (broken) plan is still recorded


def test_retry_that_fixes_topology_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = FunctionalPlan(
        object_name="toy",
        parts=[
            PartPlan(name="base", prompt="base", ports=[_port("hub")]),
            PartPlan(name="arm", prompt="arm", ports=[_port("bore")]),
            PartPlan(name="badge", prompt="badge"),  # dangling on the first try
        ],
        assembly_note="toy",
        joints=[_rev("base.hub", "arm.bore")],
    )
    fixed = broken.model_copy(
        update={
            "parts": [
                broken.parts[0],
                broken.parts[1],
                PartPlan(name="badge", prompt="badge", ports=[_port("mount")]),
            ],
            "joints": [
                _rev("base.hub", "arm.bore"),
                Joint(type=JointType.FIXED, a="base.hub", b="badge.mount"),
            ],
        }
    )
    seq = [broken, fixed]

    def _decompose(*_a: object, **_k: object):
        return seq.pop(0), SimpleNamespace(cost_usd=0.0)

    monkeypatch.setattr(fm, "decompose_functional", _decompose)
    monkeypatch.setattr(fm, "run_agentic_pipeline", _fake_pipeline)
    monkeypatch.setattr(
        fm, "acceptance_judge", lambda *a, **k: (None, SimpleNamespace(cost_usd=0.0))
    )
    brief = SimpleNamespace(object_name="toy", material=None, visual_description="x")
    res = fm.produce_functional_part("a toy", tmp_path, client=object(), brief=brief)
    assert res.disposition != "no_build" or res.failure is None or "topology" not in res.failure
    assert not seq  # both plans consumed (retry happened and was accepted)


def test_shorthand_endpoints_are_canonicalized_unambiguously() -> None:
    """The pump lesson (row 61): 'housing.bore' with a declared 'volute_housing' is an
    unambiguous shorthand -> rewritten; two housings make it ambiguous -> left for the
    gate to report."""
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints

    plan = FunctionalPlan(
        object_name="pump",
        parts=[
            PartPlan(name="volute_housing", prompt="housing", ports=[_port("bearing_bore")]),
            PartPlan(name="impeller", prompt="impeller", ports=[_port("shaft")]),
        ],
        assembly_note="impeller in housing",
        joints=[_rev("housing.bearing_bore", "impeller.shaft")],
    )
    fixed = _canonicalize_plan_endpoints(plan)
    assert fixed.joints[0].a == "volute_housing.bearing_bore"
    assert _plan_topology_issues(fixed) == []

    ambiguous = plan.model_copy(
        update={
            "parts": [
                *plan.parts,
                PartPlan(name="front_housing", prompt="fh", ports=[_port("bearing_bore")]),
            ]
        }
    )
    same = _canonicalize_plan_endpoints(ambiguous)
    assert same.joints[0].a == "housing.bearing_bore"  # ambiguous: untouched
    assert _plan_topology_issues(same)  # ...and the gate still reports it


_H4 = Path(__file__).resolve().parents[2] / "fixtures" / "h4"


def _h4_plan_and_hw_ports() -> tuple[FunctionalPlan, dict[str, set[str]]]:
    """The REAL H4 artifact (verbatim copy of E:/ForgeMind/benchmarks/2026-07-25_humanoid_h4/H4,
    TEST_LOG L98), with the hardware port map derived exactly as the pipeline derives it."""
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints
    from forge_mind.agentic.mech_arch import MechArchitecture, hardware_decls_from
    from forge_mind.structures.servo_bridge import placed_hardware_for

    plan = FunctionalPlan.model_validate_json((_H4 / "plan.json").read_text(encoding="utf-8"))
    arch = MechArchitecture.model_validate_json(
        (_H4 / "mech_arch.json").read_text(encoding="utf-8")
    )
    hw_ports = {
        decl.hw_id: {p.name for p in placed_hardware_for(decl).ports}
        for decl in hardware_decls_from(arch)
    }
    return _canonicalize_plan_endpoints(plan), hw_ports


def test_h4_hardware_collision_names_the_real_defect() -> None:
    """L98's measured failure: 4 servos re-declared as printed parts were reported
    'in NO joint' (each sat in 2 joints) -- an unactionable lie the bounded replan
    could not fix. The gate must name the COLLISION instead, once per twin."""
    plan, hw_ports = _h4_plan_and_hw_ports()
    servos = {"servo_shoulder_L", "servo_shoulder_R", "servo_hip_L", "servo_hip_R"}
    assert servos <= {p.name for p in plan.parts}  # the artifact really has the twins
    assert servos <= set(hw_ports)  # ...and the arch really derives same-named hardware

    issues = _plan_topology_issues(plan, hw_ports)
    collision = [i for i in issues if "re-declares bought hardware" in i]
    assert len(collision) == 4  # one per twin, no duplicates
    assert {n for n in servos if any(f"'{n}'" in i for i in collision)} == servos
    for i in collision:
        assert "REMOVE" in i and "horn" in i and "mount" in i  # actionable + names the ports
    # the false correction is gone: no servo is reported dangling
    assert not any("NO joint" in i and any(n in i for n in servos) for i in issues)
    assert len(issues) == 4  # nothing else fails topology in the real plan


def test_h4_plan_without_hardware_is_topology_clean() -> None:
    """The counterfactual that confirmed L98's suspicion: the same plan's joint graph is
    COMPLETE as written -- every 'dangling' report was an artifact of the collision."""
    plan, _ = _h4_plan_and_hw_ports()
    assert _plan_topology_issues(plan, None) == []


def test_collision_is_reported_even_on_a_jointless_plan() -> None:
    """Review-caught: the no-joints early return must not skip the collision rule --
    a jointless plan still authors its printed twin on top of the placed hardware."""
    plan = FunctionalPlan(
        object_name="rig",
        parts=[
            PartPlan(name="frame", prompt="frame"),
            PartPlan(name="servo_x", prompt="twin"),
        ],
        assembly_note="rig",
        joints=[],
    )
    issues = _plan_topology_issues(plan, {"servo_x": {"horn", "mount"}})
    assert len(issues) == 1 and "re-declares bought hardware 'servo_x'" in issues[0]
    assert _plan_topology_issues(plan, None) == []  # no hardware -> rigid object, no topology


def test_collision_exemption_does_not_swallow_genuine_dangling() -> None:
    """A collided twin is exempt from the dangling rule; a genuinely dangling part
    next to it is still named."""
    plan = FunctionalPlan(
        object_name="rig",
        parts=[
            PartPlan(name="frame", prompt="frame", ports=[_port("bay")]),
            PartPlan(name="servo_x", prompt="twin", ports=[_port("body")]),  # collides
            PartPlan(name="bracket", prompt="loose"),  # genuinely dangling
        ],
        assembly_note="rig",
        joints=[
            Joint(type=JointType.FIXED, a="frame.bay", b="servo_x.mount"),
            _rev("servo_x.horn", "frame.bay"),
        ],
    )
    issues = _plan_topology_issues(plan, {"servo_x": {"horn", "mount"}})
    assert sum("re-declares bought hardware 'servo_x'" in i for i in issues) == 1
    assert not any("'servo_x' appears in NO joint" in i for i in issues)
    assert any("'bracket' appears in NO joint" in i for i in issues)


def test_exact_names_never_rewritten() -> None:
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints

    plan = FunctionalPlan(
        object_name="toy",
        parts=[
            PartPlan(name="base", prompt="base", ports=[_port("hub")]),
            PartPlan(name="arm", prompt="arm", ports=[_port("bore")]),
        ],
        assembly_note="arm on base",
        joints=[_rev("base.hub", "arm.bore")],
    )
    assert _canonicalize_plan_endpoints(plan) is plan
