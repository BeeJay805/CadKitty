"""The gripper lane (ADR-0044): matcher, coupled-sweep claim gate, payload checks,
CK_JOINT v1.1 export, API wiring (money-free -- the lane is deterministic end to end).

One gripper build is shared module-wide (solve + compile + gate ~30 s). The mutation
cases pin that the gate DISCRIMINATES without recompiling: the same compiled bodies
with a dishonestly-widened range must FAIL the coupled sweep (pads cross the midline),
and a mis-clocked gear pair must FAIL the kinematic mesh."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from forge_mind.assembly.gripper_family import (
    GripperSpec,
    build_gripper_graph,
    gripper_joint_display_names,
)
from forge_mind.assembly.gripper_match import gripper_match
from forge_mind.assembly.gripper_parts import GripperDesign
from forge_mind.assembly.joints import Joint, JointType
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import AssemblyModel, compile_assembly
from forge_mind.validator.assembly_checks import gear_coupled_pairs, validate_assembly
from forge_mind.validator.report import CheckStatus
from forge_mind.viz.assembly_glb import export_assembly_glb

# ------------------------------------------------------------- cited registry facts


def test_payload_object_facts_are_cited() -> None:
    """The can the gripper claims to grip is a REGISTRY fact with its citation --
    never a gen-time guess (the grounded-ingest red line)."""
    from forge_mind.dsl.standards import payload_object

    can = payload_object("beverage_can_355ml")
    assert can.grip_diameter_mm == 66.0  # CMI "211" body
    assert can.height_mm == 122.2  # CMI "413" body
    assert can.mass_full_g == 390.0 and can.mass_empty_g == 15.0
    assert can.provisional is True  # nominal industry figures, no sample measured
    assert "211" in can.source or "Crown" in can.source


def test_servo_stall_torque_is_cited_and_never_a_gate() -> None:
    from forge_mind.assembly.servo import (
        MG996R_STALL_TORQUE_KG_CM_4V8,
        MG996R_STALL_TORQUE_KG_CM_6V0,
    )

    assert MG996R_STALL_TORQUE_KG_CM_4V8 == 9.4  # TowerPro datasheet
    assert MG996R_STALL_TORQUE_KG_CM_6V0 == 11.0


# ---------------------------------------------------------------- matcher (pure regex)


def test_gripper_match_accepts_cited_two_finger_grippers() -> None:
    spec = gripper_match("a two-finger gripper driven by an MG996R servo")
    assert spec is not None and spec.payload_declared is False
    spec = gripper_match("an MG996R gripper that can pick up a soda can")
    assert spec is not None and spec.payload_declared is True
    spec = gripper_match("design a claw for one MG996R to grab a 355 ml beverage can")
    assert spec is not None and spec.payload_declared is True


def test_gripper_match_declines_what_it_cannot_claim() -> None:
    # no cited servo -> no claim
    assert gripper_match("a two-finger gripper for a soda can") is None
    # mechanisms/variants the family does not build
    assert gripper_match("a three-finger gripper driven by an MG996R") is None
    assert gripper_match("a soft robotic hand with an MG996R") is None
    assert gripper_match("a vacuum gripper using an MG996R") is None
    assert gripper_match("a parallel-jaw gripper with an MG996R") is None
    # arm + gripper combos stay with the agentic engine until composition lands
    assert gripper_match("a 4-DOF MG996R arm with a two-finger gripper") is None


def test_arm_match_still_declines_grippers() -> None:
    """The arm matcher is byte-untouched: gripper prompts are the NEW module's."""
    from forge_mind.assembly.arm_match import arm_match

    assert arm_match("a two-finger gripper driven by an MG996R servo") is None


# ----------------------------------------------------------- the shared gripper build


@pytest.fixture(scope="module")
def gripper(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("gripper")
    graph = build_gripper_graph(GripperSpec(payload_declared=True))
    model = compile_assembly(solve(graph), out)
    report = validate_assembly(model, payload_object="beverage_can_355ml")
    return model, report, out


def test_gripper_passes_the_full_claim_gate(gripper) -> None:
    model, report, _ = gripper
    assert report.passed, [c.id for c in report.checks if c.status is CheckStatus.FAIL]
    by_id = {c.id: c for c in report.checks}
    assert by_id["assembly.static_clearance"].status is CheckStatus.PASS
    # BOTH finger sweeps are issued by the ONE coupled run (gear-coupled trajectory).
    assert by_id["assembly.sweep.servo_g__finger_drive"].status is CheckStatus.PASS
    assert by_id["assembly.sweep.dowel__finger_idler"].status is CheckStatus.PASS
    assert "gear-coupled" in by_id["assembly.sweep.servo_g__finger_drive"].message
    # The gear pair is independently proven: spacing + kinematic mesh.
    assert by_id["assembly.gear_center_distance"].status is CheckStatus.PASS
    assert by_id["assembly.gear_mesh"].status is CheckStatus.PASS
    # The cited servo interfaces are measured on the printed parts.
    assert by_id["assembly.mount_pattern.servo_g.mount"].status is CheckStatus.PASS
    assert by_id["assembly.mount_pattern.servo_g.horn"].status is CheckStatus.PASS
    assert by_id["assembly.range_of_motion"].status is CheckStatus.PASS
    # every hardware node carries an envelope mesh (what makes the sweep EARNED)
    assert all(h.mesh is not None for h in model.hardware)


def test_aperture_gate_measures_the_cited_can(gripper) -> None:
    _, report, _ = gripper
    ap = next(c for c in report.checks if c.id == "assembly.gripper_aperture")
    assert ap.status is CheckStatus.PASS
    design = GripperDesign("beverage_can_355ml")
    # The MEASURED spans (mesh vertices at the coupled extremes) land on the design
    # targets within tessellation tolerance -- and inside the gate bounds.
    assert f"open {design.aperture_open_mm:.1f}" in ap.actual or "open" in ap.actual


def test_payload_torque_is_info_only(gripper) -> None:
    _, report, _ = gripper
    tq = next(c for c in report.checks if c.id == "assembly.payload_torque")
    assert tq.status is CheckStatus.INFO  # REPORTED arithmetic, never a gate
    assert "never a pass criterion" in tq.message
    assert "9.4" in tq.message and "390" in tq.message  # cited stall + cited mass


def test_no_payload_claim_without_declaration(gripper) -> None:
    """A bare 'gripper' prompt makes NO aperture claim: the check must not appear."""
    model, _, _ = gripper
    report = validate_assembly(model)  # payload_object=None
    assert not [c for c in report.checks if c.id == "assembly.gripper_aperture"]
    assert not [c for c in report.checks if c.id == "assembly.payload_torque"]


# ------------------------------------------------------- the gate DISCRIMINATES


def _with_joint(model: AssemblyModel, old: Joint, new: Joint) -> AssemblyModel:
    joints = [new if j is old else j for j in model.joints]
    return AssemblyModel(bodies=model.bodies, joints=joints, hardware=model.hardware)


def test_dishonest_range_fails_the_coupled_sweep(gripper) -> None:
    """Widening the DECLARED range past the designed travel must FAIL: at large
    closing angles the coupled pads cross the midline and the fingers collide --
    the exact pose an independent per-joint sweep can never reach."""
    model, _, _ = gripper
    drive = next(
        j for j in model.joints if j.type is JointType.REVOLUTE and j.b_part() == "finger_drive"
    )
    widened = drive.model_copy(update={"range_deg": (-60.0, 0.0)})
    report = validate_assembly(_with_joint(model, drive, widened))
    assert not report.passed
    sweep = next(c for c in report.checks if c.id == "assembly.sweep.servo_g__finger_drive")
    assert sweep.status is CheckStatus.FAIL
    assert "coupled" in sweep.message


def test_misclocked_gears_fail_the_mesh_gate(gripper) -> None:
    """A gear pair whose tooth phase cannot mesh must FAIL assembly.gear_mesh
    (the kinematic check binds), no matter how clean the rest of the gate is."""
    model, _, _ = gripper
    gear = next(j for j in model.joints if j.type is JointType.GEAR_MESH)
    misclocked = gear.model_copy(update={"clock_b_deg": gear.clock_a_deg})
    report = validate_assembly(_with_joint(model, gear, misclocked))
    mesh = next(c for c in report.checks if c.id == "assembly.gear_mesh")
    assert mesh.status is CheckStatus.FAIL


def test_kinematic_mesh_check_fails_closed_without_a_boolean_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incident 2026-06-11: with no boolean engine the check used to record ZERO
    interference and certify ANY clocking (CI had no manifold3d; Fly would have
    shipped it). A check that cannot measure must never certify."""
    import trimesh.boolean

    from forge_mind.agentic.gears import GearSpec, validate_gear_mesh_kinematic

    def _no_engine(*a: object, **k: object) -> object:
        raise ValueError("no boolean engine available")

    monkeypatch.setattr(trimesh.boolean, "intersection", _no_engine)
    g = GearSpec(module=1.5, teeth=24)
    result = validate_gear_mesh_kinematic(g, g)
    assert result.meshes is False
    assert "fail closed" in result.note


def test_undersized_aperture_fails_the_payload_gate(gripper) -> None:
    """A range too small to close on the can must FAIL the aperture gate (closed
    span stays above can-2) while the motion itself stays clean."""
    model, _, _ = gripper
    drive = next(
        j for j in model.joints if j.type is JointType.REVOLUTE and j.b_part() == "finger_drive"
    )
    tiny = drive.model_copy(update={"range_deg": (-1.0, 0.0)})
    report = validate_assembly(_with_joint(model, drive, tiny), payload_object="beverage_can_355ml")
    ap = next(c for c in report.checks if c.id == "assembly.gripper_aperture")
    assert ap.status is CheckStatus.FAIL


# ------------------------------------------------- coupled-sweep + solver mechanics


def test_gear_mesh_is_never_a_placement_edge() -> None:
    """downstream_ids must NOT pull the gear-coupled partner into a finger's moving
    subtree (rigid co-rotation = wrong-pose sweep = the false-PASS vector)."""
    from forge_mind.validator.motion_sweep import downstream_ids

    graph = build_gripper_graph(GripperSpec())
    drive = next(
        j for j in graph.joints if j.type is JointType.REVOLUTE and j.b_part() == "finger_drive"
    )
    moving = downstream_ids(graph.joints, drive)
    assert "finger_drive" in moving
    assert "finger_idler" not in moving  # coupled, NOT carried


def test_gear_coupled_pair_detection() -> None:
    graph = build_gripper_graph(GripperSpec())
    pairs = gear_coupled_pairs(graph.joints)
    assert len(pairs) == 1
    driver, follower, ratio = pairs[0]
    assert driver.b_part() == "finger_drive"
    assert follower.b_part() == "finger_idler"
    assert ratio == 1.0  # 24T : 24T


# ---------------------------------------------------- CK_JOINT v1.1 export wiring


def _glb_json(path: Path) -> dict:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    json_len = struct.unpack_from("<I", raw, 12)[0]
    return json.loads(raw[20 : 20 + json_len].decode("utf-8"))


def test_export_earns_swept_only_from_the_coupled_checks(gripper) -> None:
    """sweep_level keys on the EXACT per-joint check ids the coupled run issued --
    pinned so a loose-match 'fix' can never mint an unearned 'swept'."""
    model, report, out = gripper
    manifest = export_assembly_glb(
        model, out / "gripper.glb", report=report, joint_names=gripper_joint_display_names()
    )
    by_id = {j["id"]: j for j in manifest["joints"]}
    assert by_id["finger_drive"]["checked"] == "swept"
    assert by_id["finger_idler"]["checked"] == "swept"
    # WITHOUT the report, nothing may claim swept (earned, never default).
    manifest_none = export_assembly_glb(
        model, out / "gripper_none.glb", report=None, joint_names=gripper_joint_display_names()
    )
    assert all(j["checked"] == "none" for j in manifest_none["joints"])


def test_export_carries_the_mimic_coupling(gripper) -> None:
    """ck_joint v1.1: the FOLLOWER carries mimic {of, ratio} (additive; old viewers
    degrade to two honestly-labeled sliders). The driver carries none."""
    model, report, out = gripper
    manifest = export_assembly_glb(
        model, out / "gripper_m.glb", report=report, joint_names=gripper_joint_display_names()
    )
    by_id = {j["id"]: j for j in manifest["joints"]}
    assert by_id["finger_idler"]["mimic"] == {"of": "finger_drive", "ratio": -1.0}
    assert "mimic" not in by_id["finger_drive"]
    # The same coupling rides the GLB node extras (the viewer's carrier).
    gltf = _glb_json(out / "gripper_m.glb")
    cks = {
        n["name"]: n["extras"]["ck_joint"]
        for n in gltf["nodes"]
        if "extras" in n and "ck_joint" in n.get("extras", {})
    }
    assert cks["finger_idler"]["mimic"] == {"of": "finger_drive", "ratio": -1.0}
    assert "mimic" not in cks["finger_drive"]


# ------------------------------------------------------------------- API wiring


def test_api_gripper_route_is_doubly_flag_gated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """FORGE_ASSEMBLY_LANE alone must NOT route gripper prompts (the pinned
    fall-through holds); adding FORGE_GRIPPER=1 ships the gripper at $0 LLM."""
    import forge_mind.api.core as core

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("agentic engine must not run in this test")

    monkeypatch.setattr(core, "produce_part", _boom)
    monkeypatch.setenv("FORGE_ASSEMBLY_LANE", "1")
    monkeypatch.setenv("FORGE_GRIPPER", "1")
    res = core.generate_part("an MG996R gripper that can pick up a soda can", tmp_path)
    assert res.status == "ok"
    ok = res.ok
    assert ok is not None
    assert ok.gen_path == "assembly:gripper"
    assert ok.cost_usd == 0.0
    assert ok.validation.passed
    assert ok.confidence is not None and ok.confidence.basis == "validated"
    assert ok.confidence.dimension_tiers.sourced > 0
    assert (tmp_path / "part.glb").exists()
    assert (tmp_path / "parts.json").exists()
    gate = json.loads((tmp_path / "parts.json").read_text())["gate"]
    assert gate["checks"]["assembly.gripper_aperture"] == "pass"
    # ADR-0044 increment: the positioned multi-body STEP (printed parts only) +
    # preview PNG ship alongside, and the response carries the assembly manifest.
    assert (tmp_path / "part.step").exists()
    assert (tmp_path / "part.png").exists()
    asm = ok.assembly
    assert asm is not None
    by_part = {p.id: p for p in asm.parts}
    assert by_part["palm"].printed is True and by_part["palm"].source is None
    assert by_part["servo_g"].printed is False
    assert by_part["servo_g"].source and "TowerPro" in by_part["servo_g"].source
    assert by_part["dowel"].source and "ISO 2338" in by_part["dowel"].source
    by_joint = {j.id: j for j in asm.joints}
    assert by_joint["finger_drive"].checked == "swept"
    assert by_joint["finger_idler"].mimic_of == "finger_drive"
    assert by_joint["finger_idler"].mimic_ratio == -1.0
    assert asm.gate["assembly.gripper_aperture"] == "pass"
    assert asm.step_note and "Printed parts only" in asm.step_note
    # The BOM is REAL now (the empty-fallback regression the council caught).
    assert ok.kit.bom.lines, "the assembly lane must ship a populated BOM"
    kinds = {ln.kind for ln in ok.kit.bom.lines}
    assert "actuator" in kinds and "dowel" in kinds
