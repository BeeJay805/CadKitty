"""The servo-arm lane: matcher, claim gate, CK_JOINT export, API wiring (money-free).

One pan-tilt build is shared module-wide (solve + compile + gate ~ seconds); the
mutation case pins that the gate DISCRIMINATES -- a deliberately low-slung yoke over an
oversized base must fail BOTH the static fit and the chain sweep. No LLM runs anywhere
in this file (the lane is deterministic end to end)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from forge_mind.assembly.arm_family import ServoArmSpec, build_arm_graph
from forge_mind.assembly.arm_match import arm_match, joint_display_names
from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import CheckStatus
from forge_mind.viz.assembly_glb import export_assembly_glb, sweep_level

# ---------------------------------------------------------------- matcher (pure regex)


def test_arm_match_accepts_cited_serial_chains() -> None:
    spec = arm_match("a 4-DOF desktop robot arm driven by MG996R servos")
    assert spec is not None and spec.dof == 4
    spec = arm_match("a pan-tilt camera head for two MG996R servos")
    assert spec is not None and spec.dof == 2 and spec.name == "pan_tilt"
    spec = arm_match("design a turntable using an MG996R servo")
    assert spec is not None and spec.dof == 1


def test_arm_match_paraphrase_stable() -> None:
    a = arm_match("Build me a 3-DOF robotic arm using MG996R servos for my desk")
    b = arm_match("For my desk, using MG996R servos, build a robotic arm with 3 DOF")
    assert a is not None and b is not None and a.dof == b.dof == 3


def test_arm_match_declines_what_it_cannot_claim() -> None:
    # no cited servo -> no claim
    assert arm_match("a 4-DOF desktop robot arm") is None
    # a different mechanism class
    assert arm_match("a two-finger gripper driven by an MG996R servo") is None
    # over-constrained: 4 DOF but only two servos for an all-servo chain
    assert arm_match("a 4-DOF arm for two MG996R servos") is None
    # other ecosystems stay with their own lanes
    assert arm_match("a NEMA 17 motor mount arm") is None


# ----------------------------------------------------------- the shared pan-tilt build


@pytest.fixture(scope="module")
def pan_tilt(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("pan_tilt")
    model = compile_assembly(solve(build_arm_graph(ServoArmSpec(dof=2, name="pan_tilt"))), out)
    report = validate_assembly(model)
    return model, report, out


def test_pan_tilt_passes_the_full_claim_gate(pan_tilt) -> None:
    model, report, _ = pan_tilt
    assert report.passed, [c.id for c in report.checks if c.status is CheckStatus.FAIL]
    by_id = {c.id: c for c in report.checks}
    # the claims that make "validated" mean something, individually present + PASS:
    assert by_id["assembly.static_clearance"].status is CheckStatus.PASS
    assert by_id["assembly.sweep.servo_1__link_1"].status is CheckStatus.PASS
    assert by_id["assembly.sweep.servo_2__link_2"].status is CheckStatus.PASS
    assert by_id["assembly.mount_pattern.servo_1.mount"].status is CheckStatus.PASS
    assert by_id["assembly.mount_pattern.servo_1.horn"].status is CheckStatus.PASS
    assert by_id["assembly.mount_pattern.servo_2.mount"].status is CheckStatus.PASS
    assert by_id["assembly.mount_pattern.servo_2.horn"].status is CheckStatus.PASS
    assert by_id["assembly.range_of_motion"].status is CheckStatus.PASS
    # both servos carry envelope meshes (the thing that makes the sweep EARNED)
    assert all(h.mesh is not None for h in model.hardware)


def test_ck_joint_glb_export_matches_the_frozen_contract(pan_tilt) -> None:
    model, report, out = pan_tilt
    glb = out / "assembly.glb"
    manifest = export_assembly_glb(model, glb, report=report, joint_names=joint_display_names(2))
    raw = glb.read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF" and version == 2
    clen, ctype = struct.unpack_from("<II", raw, 12)
    assert ctype == 0x4E4F534A
    gltf = json.loads(raw[20 : 20 + clen].decode("utf-8"))
    nodes = gltf["nodes"]
    joints = [
        n["extras"]["ck_joint"]
        for n in nodes
        if isinstance(n.get("extras"), dict) and "ck_joint" in n["extras"]
    ]
    assert len(joints) == 2
    for ck in joints:
        # the frozen v1 contract fields
        assert ck["type"] == "revolute"
        assert len(ck["axis"]) == 3
        assert ck["limits_deg"] == [-90.0, 90.0]
        # EARNED: the gate passed, so (and only so) the level is "swept"
        assert ck["checked"] == "swept"
    names = {ck["name"] for ck in joints}
    assert names == {"pan", "tilt"}
    # 5 mesh nodes: base, 2 servos, 2 links -- per-part identity preserved
    assert sum(1 for n in nodes if "mesh" in n) == 5
    assert {j["checked"] for j in manifest["joints"]} == {"swept"}


def test_sweep_level_is_earned_never_asserted(pan_tilt) -> None:
    _, report, _ = pan_tilt
    assert sweep_level(report, "servo_1", "link_1") == "swept"
    # an unknown joint has no check -> "none"; no report -> "none"
    assert sweep_level(report, "ghost", "part") == "none"
    assert sweep_level(None, "servo_1", "link_1") == "none"


def test_mutant_arm_fails_static_fit_and_chain_sweep(tmp_path: Path) -> None:
    """The discriminating case: a low tilt axis over an oversized base. The OLD pairwise
    motion check could not see servo_2 (not a pan-joint endpoint) hitting servo_1; the
    chain gate must."""
    nodes = [
        PartNode(
            id="base",
            archetype="servo_base",
            params={"plate_x": 200.0, "plate_y": 200.0},
        ),
        PartNode(
            id="servo_1",
            archetype="hardware",
            params={"hardware": "mg996r_servo"},
            printed=False,
        ),
        PartNode(
            id="link_1",
            archetype="servo_link",
            params={"length_x": 24.0, "end": "cradle_perp", "cradle_height_z": 10.0},
        ),
        PartNode(
            id="servo_2",
            archetype="hardware",
            params={"hardware": "mg996r_servo"},
            printed=False,
        ),
        PartNode(id="link_2", archetype="servo_link", params={"length_x": 60.0, "end": "tool"}),
    ]
    joints = [
        Joint(type=JointType.FIXED, a="base.cradle_mount", b="servo_1.mount", mate="align"),
        Joint(
            type=JointType.REVOLUTE,
            a="servo_1.horn",
            b="link_1.hub",
            range_deg=(-90.0, 90.0),
            clearance_class="running_fit",
            mate="align",
        ),
        Joint(type=JointType.FIXED, a="link_1.cradle_mount", b="servo_2.mount", mate="align"),
        Joint(
            type=JointType.REVOLUTE,
            a="servo_2.horn",
            b="link_2.hub",
            range_deg=(-90.0, 90.0),
            clearance_class="running_fit",
            mate="align",
        ),
    ]
    bad = AssemblyGraph(nodes=nodes, joints=joints, root="base")
    report = validate_assembly(compile_assembly(solve(bad), tmp_path))
    assert not report.passed
    by_id = {c.id: c for c in report.checks}
    assert by_id["assembly.static_clearance"].status is CheckStatus.FAIL
    assert by_id["assembly.range_of_motion"].status is CheckStatus.FAIL


def test_meshless_hardware_never_claims_swept(tmp_path: Path) -> None:
    """Strip a servo's envelope: the sweep must WARN (not earned), never report clear."""
    model = compile_assembly(solve(build_arm_graph(ServoArmSpec(dof=2, name="pan_tilt"))), tmp_path)
    model.hardware[0].mesh = None  # simulate a frames-only (legacy ADR-0012) node
    report = validate_assembly(model)
    rom = next(c for c in report.checks if c.id == "assembly.range_of_motion")
    assert rom.status is CheckStatus.WARN
    assert "not swept" in rom.message.lower() or "NOT earned" in rom.message
    assert sweep_level(report, "servo_1", "link_1") == "none"


def test_sweep_level_demotes_when_corner_pose_chain_check_fails() -> None:
    # Regression (audit H6): a PASSING per-joint sweep does not prove the COUPLED range is clear. If
    # the corner-pose chain check for this joint's moving part FAILED, the joint must NOT be "swept".
    from types import SimpleNamespace

    def ck(cid: str, status: CheckStatus) -> SimpleNamespace:
        return SimpleNamespace(id=cid, status=status)

    failed_corner = SimpleNamespace(
        checks=[
            ck("assembly.sweep.servo_1__link_1", CheckStatus.PASS),
            # the chain collides at combined corner poses even though the per-joint sweep passed
            ck("assembly.sweep_corner.link_1__link_2", CheckStatus.FAIL),
        ]
    )
    assert sweep_level(failed_corner, "servo_1", "link_1") == "none"
    # with the corner check passing (or absent), the passing per-joint sweep IS swept
    clean = SimpleNamespace(checks=[ck("assembly.sweep.servo_1__link_1", CheckStatus.PASS)])
    assert sweep_level(clean, "servo_1", "link_1") == "swept"
