"""MechArchitecture stage (FORGE_MECH_ARCH): the typed interface control document.

The load-bearing properties: placements DERIVE HardwareDecls (cards and gates measure
against the same document), the transform maps the servo local frame (shaft on local
+Z) to the declared world direction exactly, and the binding context block carries
every decision class the plan gates audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from forge_mind.agentic.mech_arch import (
    ActuatorPlacement,
    MechArchitecture,
    MechJoint,
    arch_context_block,
    arch_transform,
    hardware_decls_from,
    plan_mech_architecture,
)

_DIRS = {
    "+X": (1, 0, 0),
    "-X": (-1, 0, 0),
    "+Y": (0, 1, 0),
    "-Y": (0, -1, 0),
    "+Z": (0, 0, 1),
    "-Z": (0, 0, -1),
}


def _arch() -> MechArchitecture:
    return MechArchitecture(
        object_name="desktop humanoid",
        joints=[
            MechJoint(
                name="shoulder_L",
                kind="actuated",
                axis_dir="+Y",
                station_mm=(0.0, 50.0, 290.0),
                bearing="Ø6.0 pin journal in Ø6.35 bore land, running fit 0.35",
                retention="printed shoulder + push-on cap",
            ),
            MechJoint(
                name="knee_L",
                kind="passive",
                axis_dir="+Y",
                station_mm=(0.0, 26.0, 105.0),
                bearing="Ø6.0 journal, running fit 0.35",
                retention="e-clip groove 0.9 wide",
            ),
        ],
        actuators=[
            ActuatorPlacement(
                joint="shoulder_L",
                hw_id="servo_shoulder_L",
                position_mm=(0.0, 40.0, 290.0),
                shaft_dir="+Y",
                horn_link="upper_arm_L",
            )
        ],
        fastening_system="M3 self-tappers into 4.0 mm bosses; M4 at servo lugs",
        seam_strategy="chest/back tongue-and-groove on x=0, 4x M3 bosses",
        wiring_route="limb channels to the torso bay via 8 mm grommets",
        mirrored_pairs=[["thigh_L", "thigh_R"]],
    )


@pytest.mark.parametrize("d", sorted(_DIRS))
def test_arch_transform_maps_shaft_and_position_exactly(d: str) -> None:
    t = arch_transform((10.0, 20.0, 30.0), d)
    assert np.allclose(t[:3, 3], (10.0, 20.0, 30.0))
    assert np.allclose(t[:3, 2], _DIRS[d])  # local +Z (the shaft) -> declared direction
    r = t[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)  # orthonormal
    assert np.isclose(np.linalg.det(r), 1.0)  # right-handed, no reflection


def test_hardware_decls_derive_from_placements() -> None:
    decls = hardware_decls_from(_arch())
    assert len(decls) == 1
    d = decls[0]
    assert d.kind == "mg996r_servo" and d.hw_id == "servo_shoulder_L"
    assert np.allclose(np.asarray(d.transform)[:3, 2], (0, 1, 0))  # shaft on +Y


def test_context_block_carries_every_decision_class() -> None:
    block = arch_context_block(_arch())
    assert "BINDING" in block
    assert "shoulder_L (actuated)" in block and "knee_L (passive)" in block
    assert "running fit 0.35" in block and "push-on cap" in block
    assert "servo_shoulder_L" in block and "horn couples to 'upper_arm_L'" in block
    assert "FASTENING SYSTEM: M3 self-tappers" in block
    assert "MIRRORED PAIRS" in block and "thigh_L/thigh_R" in block


def test_schema_forbids_stray_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MechJoint.model_validate(
            {
                "name": "x",
                "kind": "passive",
                "axis_dir": "+Y",
                "station_mm": (0, 0, 0),
                "bearing": "b",
                "retention": "r",
                "surprise": 1,
            }
        )


def test_plan_mech_architecture_rides_doctrine_and_brief() -> None:
    captured: dict = {}

    class _Fake:
        def generate_structured(self, *, system, user, output_model, model, max_tokens):
            captured.update(system=system, user=user, output_model=output_model)
            return _arch(), SimpleNamespace(cost_usd=0.0)

    arch, _u = plan_mech_architecture(
        "a desktop humanoid",
        client=_Fake(),
        model=None,
        brief_text="smooth tapered limb shells",
        doctrine="\nENGINEERING DOCTRINE (test marker)",
    )
    assert arch.object_name == "desktop humanoid"
    assert "CHIEF MECHANICAL ARCHITECT" in captured["system"]
    assert "ENGINEERING DOCTRINE (test marker)" in captured["system"]
    assert "smooth tapered limb shells" in captured["user"]
    assert captured["output_model"] is MechArchitecture


# --- THE DRIVE RELATIONSHIP IS SEVERED HERE ----------------------------------------------------
# Reconciled 2026-07-25, then CORRECTED the same day by an adversarial review that was right.
#
# The 2026-07-24 handoff proposed "plumbing a hardware channel through the plan schema so the
# planner has somewhere to declare actuators". A channel does exist -- THIS stage -- and H3 shipped
# with zero actuators because FORGE_MECH_ARCH was OFF (absent from H3/env_snapshot.json, no
# mech_arch.json written). But "a channel exists" is NOT "actuators drive joints", and the first
# version of these tests overclaimed exactly that.
#
# What is actually true: `ActuatorPlacement` knows `joint` and `horn_link` ("servo_shoulder_L
# drives shoulder_L via horn upper_arm_L"), and `hardware_decls_from` DISCARDS both -- HardwareDecl
# carries only kind/hw_id/transform. So no drive relationship reaches any gate, and
# `undriven_dof_lower_bound` is pure counting (`moving - len(hardware)`), which cannot tell six
# servos driving six joints from six servos floating unattached. Its own docstring says so and
# says it never gates; the defect is that nothing else checks the binding either.
#
# These tests pin BOTH halves -- the counting bound AND the severed binding -- so the gap is
# executable. See docs/research/2026-07-25-reporting-false-pass-class-exhausted.md.

_PROBES = Path("E:/ForgeMind/benchmarks/2026-07-23_humanoid_probe/probes")

_ARCHIVE_ONLY = pytest.mark.skipif(
    not _PROBES.exists(),
    # NOTE: this archive is NOT in the repo, so these cases SKIP in CI. They are a local
    # calibration pin, not a green-gate guarantee -- do not cite them as CI-enforced.
    reason="humanoid probe archive not present (local calibration only; skipped in CI)",
)


def _probe(probe: str) -> tuple[Any, MechArchitecture]:
    from forge_mind.agentic.functional import FunctionalPlan

    return (
        FunctionalPlan.model_validate_json(
            (_PROBES / probe / "plan.json").read_text(encoding="utf-8")
        ),
        MechArchitecture.model_validate_json(
            (_PROBES / probe / "mech_arch.json").read_text(encoding="utf-8")
        ),
    )


def test_the_derivation_drops_the_arch_side_drive_relationship() -> None:
    """The architecture states WHICH joint each actuator drives and through WHICH horn link;
    neither survives into the HardwareDecl. NOTE this is NOT the whole story -- the plan has its
    own, working binding carrier (see the next test) -- so this alone is not the defect."""
    arch = _arch()
    placement = arch.actuators[0]
    assert placement.joint == "shoulder_L" and placement.horn_link == "upper_arm_L"
    decl = hardware_decls_from(arch)[0]
    assert not hasattr(decl, "joint") and not hasattr(decl, "horn_link")
    assert set(type(decl).__dataclass_fields__) == {"kind", "hw_id", "transform"}


def test_a_declared_actuator_that_no_joint_references_is_not_flagged() -> None:
    """THE ACTUAL DEFECT, characterized: an architecture-to-plan RECONCILIATION gap.

    The binding carrier exists and works -- `FunctionalPlan.joints` may name `<hw_id>.<port>` as
    an endpoint (ADR-0124), `_plan_topology_issues` validates it against the bridge's real ports,
    `from_part_results` preserves it, and the coaxial / mount-pattern / sweep checks consume it.
    The shipped pan-tilt plan uses it: `pan_servo.horn -> pan_bracket.hub`.

    What is missing is any check that a plan USES it. The architecture's actuator placements reach
    the planner as PROSE (`arch_context_block`) and `arch` is otherwise consulted only to write
    mech_arch.json and derive HardwareDecls -- nothing reconciles "the arch declared 6 actuators"
    against "the plan bound 0 of them". Hardware is even explicitly EXEMPT from the dangling-part
    rule that catches this for printed parts.

    Measured consequence (archived probes P1/P2): 11 actuators declared, ZERO joints referencing
    any of them, no issue raised, and `undriven_dof_min` still improves 9->3 and 9->4.

    Same failure mode as the L90 mirror-pairs finding: doctrine aimed at the model does not bind.

    If this test starts FAILING, the reconciliation gate now exists -- delete the test and say so.
    """
    from forge_mind.agentic.functional import (
        FunctionalPlan,
        PartPlan,
        _plan_topology_issues,
    )
    from forge_mind.assembly.joints import Joint, JointType, Port
    from forge_mind.types import Vec3

    hub = Port(
        name="hub",
        origin=Vec3(x=0.0, y=0.0, z=0.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    plan = FunctionalPlan(
        object_name="arm that declares a servo and connects it to nothing",
        parts=[
            PartPlan(name="upper_arm_L", prompt="the upper arm card", ports=[hub]),
            PartPlan(name="forearm_L", prompt="the forearm card", ports=[hub]),
        ],
        assembly_note="forearm pivots on the upper arm",
        joints=[
            Joint(
                type=JointType.REVOLUTE,
                a="upper_arm_L.hub",
                b="forearm_L.hub",
                range_deg=(0.0, 90.0),
            )
        ],
    )
    arch = _arch()  # declares servo_shoulder_L driving shoulder_L via horn upper_arm_L
    hw_ids = {d.hw_id for d in hardware_decls_from(arch)}

    assert hw_ids == {"servo_shoulder_L"}
    assert not any(h in f"{j.a}{j.b}" for j in plan.joints for h in hw_ids), (
        "fixture premise: the plan binds no servo"
    )
    # The servo the architecture declared is bound to nothing, and the plan gate is silent.
    issues = _plan_topology_issues(plan, {"servo_shoulder_L": {"horn", "mount"}})
    assert not any("servo_shoulder_L" in i for i in issues), issues


def test_the_undriven_bound_cannot_see_whether_anything_is_connected() -> None:
    """Companion to the above, and the reason it matters: the bound is `moving - len(hardware)`,
    so hardware that drives NOTHING lowers it exactly as much as hardware that drives everything.
    A mannequin with a servo glued to its back reads the same as a working mechanism."""
    from forge_mind.agentic.functional import FunctionalPlan, PartPlan, undriven_dof_lower_bound
    from forge_mind.assembly.joints import Joint, JointType, Port
    from forge_mind.structures.servo_bridge import HardwareDecl
    from forge_mind.types import Vec3

    pivot = Port(
        name="pivot",
        origin=Vec3(x=0.0, y=0.0, z=10.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    plan = FunctionalPlan(
        object_name="one-joint mechanism",
        parts=[
            PartPlan(name="base", prompt="the base card", ports=[pivot]),
            PartPlan(name="lever", prompt="the lever card", ports=[pivot]),
        ],
        assembly_note="lever pivots on base",
        joints=[
            Joint(type=JointType.REVOLUTE, a="base.pivot", b="lever.pivot", range_deg=(0.0, 90.0))
        ],
    )
    driving = hardware_decls_from(_arch())  # placed at the shoulder, names its joint + horn
    glued_on = [HardwareDecl(kind="mg996r_servo", hw_id="attached_to_nothing")]

    assert len(driving) == len(glued_on) == 1
    assert undriven_dof_lower_bound(plan, None) == 1
    assert undriven_dof_lower_bound(plan, driving) == 0
    assert undriven_dof_lower_bound(plan, glued_on) == 0, "identical -- the bound counts, only"


@_ARCHIVE_ONLY
@pytest.mark.parametrize(
    ("probe", "actuators", "undriven_without", "undriven_with", "joints_referencing_a_servo"),
    [("P1", 6, 9, 3, 0), ("P2", 5, 9, 4, 0)],
)
def test_the_counting_bound_moves_while_no_joint_references_any_servo(
    probe: str,
    actuators: int,
    undriven_without: int,
    undriven_with: int,
    joints_referencing_a_servo: int,
) -> None:
    """Both halves on the REAL archived architectures: declaring actuators moves the reported
    bound (9 -> 3 / 9 -> 4), and ZERO of the plan's joints reference any servo hw_id while it
    moves. The number improves without anything being connected -- that is the gap, measured."""
    from forge_mind.agentic.functional import undriven_dof_lower_bound

    plan, arch = _probe(probe)
    hw = hardware_decls_from(arch)
    assert len(hw) == actuators, "one HardwareDecl per declared actuator placement"
    assert undriven_dof_lower_bound(plan, None) == undriven_without, "H3's situation"
    assert undriven_dof_lower_bound(plan, hw) == undriven_with

    hw_ids = {a.hw_id for a in arch.actuators}
    referencing = [j for j in plan.joints if any(h in f"{j.a}{j.b}" for h in hw_ids)]
    assert len(referencing) == joints_referencing_a_servo, referencing


@_ARCHIVE_ONLY
@pytest.mark.parametrize("probe", ["P1", "P2"])
def test_registry_lookup_resolves_but_yields_no_collision_envelope(probe: str) -> None:
    """Registry KEY lookup succeeds for all 11 placements -- but without a workdir this is the
    frames-only path and every mesh is None, so nothing here proves envelope-sweep visibility.
    Compiling the envelopes is the production path (`placed_hardware_for(decl, workdir=...)`);
    asserting a validator result through it is the work this test deliberately does NOT claim."""
    from forge_mind.structures.servo_bridge import placed_hardware_for

    _plan, arch = _probe(probe)
    for decl in hardware_decls_from(arch):
        placed = placed_hardware_for(decl)
        assert placed is not None, decl.hw_id
        assert getattr(placed, "mesh", None) is None, "frames-only path: no envelope compiled"


def test_the_schema_caps_can_express_a_real_flat_six() -> None:
    """ADR-0128. The caps must SCALE TOGETHER -- the code's own law, learned when a 3-port cap
    made a 4-cylinder engine undeclarable while parts/joints had room.

    Sized off a real architecture this engine produced (2026-07-25 F1 probe), not a guess, and
    then CORRECTED: the architecture's mirrored_pairs lists 28 parts, but pins appear there as
    JOINTS while `_DECOMPOSE_SYSTEM` requires "each pin as its OWN part". So the roster is
    2 case + 2 decks + 6 barrels + 6 pistons + 6 conrods + 6 rod caps + crank + hand-crank
    + **6 wrist pins** = 36, not the 30 a first pass at this cap assumed. Reading the arch alone
    undercounts by exactly the six parts the prompt insists on.

    At the old 16/16/8 a flat-six was not hard, it was INEXPRESSIBLE, so decompose had to drop
    half the engine while the object kept the name "flat six" -- a false-PASS at the plan tier.
    """
    from forge_mind.agentic.functional import FunctionalPlan, PartPlan, PlanRevision
    from forge_mind.agentic.mech_arch import MechArchitecture

    def cap(model, field):
        for meta in model.model_fields[field].metadata:
            if hasattr(meta, "max_length"):
                return meta.max_length
        raise AssertionError(f"{model.__name__}.{field} has no max_length")

    # 36 parts: the 30 in the architecture roster PLUS 6 wrist pins, which the prompt requires
    # to be their own parts.
    #
    # 41 joints, COUNTED rather than mirrored off the part cap -- a first pass set the joint cap
    # to 40 to match `parts` and it was one short of this very case. The complete-joint-graph rule
    # requires every part in >= 1 joint:
    #   30 cylinder-chain: per cylinder crankpin, rod cap, wrist-pin<->conrod, wrist-pin<->piston,
    #      piston slide, x6
    #   11 structural: case halves, crank main, 2 decks, 6 barrels, hand-crank
    flat_six_parts, flat_six_joints, crank_ports = 36, 41, 11

    assert cap(FunctionalPlan, "parts") >= flat_six_parts
    assert cap(FunctionalPlan, "joints") >= flat_six_joints
    assert cap(PartPlan, "ports") >= crank_ports, "the crankshaft must declare its own throws"
    assert cap(MechArchitecture, "joints") >= 23, "the measured architecture joint count"

    # The revision cap is deliberately BELOW the part cap: replacing every card is a re-plan,
    # not a repair. But it must still cover one full bank plus the crank.
    assert 13 <= cap(PlanRevision, "revised_parts") < cap(FunctionalPlan, "parts")


def test_the_decompose_prompt_does_not_contradict_the_schema_caps() -> None:
    """THE FAILURE THIS CATCHES: raising a schema cap is COSMETIC if the prompt still states the
    old limit, because the model obeys the prose it is given, not the pydantic Field. The ports
    cap was raised 8 -> 12 while `_DECOMPOSE_SYSTEM` and the topology-retry message both still
    said "up to 8 ports per part", so the planner would have kept emitting the same truncated
    engine and no gate reconciles the architecture roster against the plan.
    """
    import re

    from forge_mind.agentic.functional import _DECOMPOSE_SYSTEM, PartPlan

    def cap(model, field):
        for meta in model.model_fields[field].metadata:
            if hasattr(meta, "max_length"):
                return meta.max_length
        raise AssertionError("no max_length")

    from forge_mind.agentic.functional import FunctionalPlan

    stated_ports = [int(n) for n in re.findall(r"up to (\d+) ports per part", _DECOMPOSE_SYSTEM)]
    assert stated_ports, "the prompt must state a port budget at all"
    for n in stated_ports:
        assert n == cap(PartPlan, "ports"), (
            f"prompt says {n} ports/part, schema allows {cap(PartPlan, 'ports')} -- "
            "a stale prompt limit silently caps the planner below the schema"
        )

    # The PART budget is the one that recurred: the ports string was fixed while this one was
    # missed, so the planner was still told "up to 10" against a 40-part schema.
    stated_parts = [
        int(n) for n in re.findall(r"up to (\d+) only for a genuine", _DECOMPOSE_SYSTEM)
    ]
    assert stated_parts, "the prompt must state a part budget at all"
    for n in stated_parts:
        assert n == cap(FunctionalPlan, "parts"), (
            f"prompt says {n} parts, schema allows {cap(FunctionalPlan, 'parts')}"
        )


def test_dependent_caps_scale_with_the_part_cap() -> None:
    """The coupling, DERIVED from the part cap instead of hand-counted.

    This bug recurred three times in one session: `parts` was raised and `ports`,
    `FunctionalPlan.joints`, or `MechArchitecture.joints` was left behind. Each time the fix was
    a new hand-counted constant, which is exactly what drifts. So the requirement is computed
    here from whatever `parts` currently is.

    The densest mechanism a plan can hold is an N-cylinder engine at 5 parts per cylinder
    (barrel, piston, wrist pin, conrod, big-end cap) plus ~6 fixed parts (case halves,
    crankshaft, hand-crank, base, rib). Its crankshaft is the most port-dense part in the
    engine: N crankpins + (N/2 + 1) mains + 1 hand-crank hub.
    """
    from forge_mind.agentic.functional import FunctionalPlan, PartPlan

    def cap(model, field):
        for meta in model.model_fields[field].metadata:
            if hasattr(meta, "max_length"):
                return meta.max_length
        raise AssertionError(f"{model.__name__}.{field} has no max_length")

    parts_cap = cap(FunctionalPlan, "parts")
    cylinders = (parts_cap - 6) // 5  # the largest engine the part cap admits

    # Crankshaft ports for that engine.
    needed_ports = cylinders + (cylinders // 2 + 1) + 1
    assert cap(PartPlan, "ports") >= needed_ports, (
        f"{parts_cap} parts admits a {cylinders}-cylinder engine whose crankshaft needs "
        f"{needed_ports} ports, but the port cap is {cap(PartPlan, 'ports')} -- the topology gate "
        "requires every joint endpoint's port, so that assembly cannot be expressed"
    )

    # Plan joints for that engine. The structural term is NOT constant -- barrels attach to decks
    # one per cylinder, so it scales. A "+11" constant lifted from the flat-six case understates
    # a flat-ten by four and would let a future cap reduction pass this test while making the
    # admitted engine inexpressible.
    #   6 per cylinder: crankpin, rod cap, wrist-pin<->conrod, wrist-pin<->piston, piston slide,
    #                   barrel<->deck
    #   5 fixed:        case halves, crank<->case main, 2 decks<->case, hand-crank<->crank
    # Self-check: at 6 cylinders this yields 41, the hand-counted flat-six figure.
    def joints_for(n_cylinders: int) -> int:
        return n_cylinders * 6 + 5

    # TWO anchors, because one cannot pin a line. The regression this guards against is the
    # constant-structural-term formula `n*5 + 11`, and that agrees with the correct `n*6 + 5` at
    # exactly six cylinders (both give 41) -- so a flat-six-only anchor is blind to it. The second
    # anchor is where they diverge: at ten cylinders the wrong formula says 61, the right one 65.
    assert joints_for(6) == 41, "must reproduce the measured flat-six joint count"
    assert joints_for(10) == 65, "must reproduce the flat-ten count (n*5+11 would give 61)"

    needed_joints = joints_for(cylinders)
    assert cap(FunctionalPlan, "joints") >= needed_joints, (
        f"{parts_cap} parts admits {cylinders} cylinders needing {needed_joints} joints, "
        f"cap is {cap(FunctionalPlan, 'joints')}"
    )

    # The architecture is never richer than the plan's joint graph. EQUALITY, not >=: the claim
    # is that these are pinned together, and `>=` would silently allow either lowering
    # FunctionalPlan.joints or raising MechArchitecture.joints alone -- letting the arch emit a
    # roster the plan schema cannot represent. Exceeding the arch cap fails SILENTLY (the stage
    # is try/except'd and degrades to the archless pipeline), so this must catch drift both ways.
    assert cap(MechArchitecture, "joints") == cap(FunctionalPlan, "joints")
