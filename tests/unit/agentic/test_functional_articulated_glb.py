"""The agentic functional path emits the FROZEN ck_joint GLB contract: an assembled
mechanism's GLB carries parented per-part nodes + extras.ck_joint (checked EARNED, never
asserted), so the web viewer's articulation rig can MOVE an agentic assembly. Also pins
the artifact wiring: FunctionalResult.assembled_glb reaches PartResult.artifacts['glb']
(without it the API rebuilt a rigid GLB from the flat STL)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from build123d import Box, Pos, export_step, export_stl

from forge_mind.agentic.assembly_adapter import from_part_results
from forge_mind.agentic.functional import _articulated_glb, _validate_motion
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3

pytestmark = pytest.mark.slow  # builds tiny B-reps (build123d/OCCT)


def _write_part(tmp: Path, name: str, solid: Any) -> SimpleNamespace:
    stl = tmp / f"{name}.stl"
    step = tmp / f"{name}.step"
    export_stl(solid, str(stl))
    export_step(solid, str(step))
    return SimpleNamespace(final_stl=stl, final_step=step)


def _arm_fixture(tmp: Path):
    """A base slab (z 0..10) + an arm link pivoting at the origin-adjacent post top."""
    base = Pos(0, 0, 5) * Box(40, 40, 10)
    arm = Pos(15, 0, 14) * Box(30, 8, 8)  # adjacent above the base, pivot at its left end
    part_results = [
        ("base", _write_part(tmp, "base", base)),
        ("arm", _write_part(tmp, "arm", arm)),
    ]
    pivot = {
        "origin": Vec3(x=0.0, y=0.0, z=14.0),
        "axis": Vec3(x=0.0, y=0.0, z=1.0),
        "normal": Vec3(x=1.0, y=0.0, z=0.0),
    }
    ports = {
        "base": [Port(name="pivot", **pivot)],
        "arm": [Port(name="pivot", **pivot)],
    }
    joints = [Joint(type=JointType.REVOLUTE, a="base.pivot", b="arm.pivot", range_deg=(0.0, 90.0))]
    return part_results, joints, ports


def _glb_json(glb_path: Path) -> dict[str, Any]:
    raw = glb_path.read_bytes()
    assert raw[:4] == b"glTF"
    json_len = struct.unpack_from("<I", raw, 12)[0]
    return json.loads(raw[20 : 20 + json_len].decode("utf-8"))


def test_articulated_glb_carries_ck_joint_nodes(tmp_path: Path) -> None:
    part_results, joints, ports = _arm_fixture(tmp_path)
    model = from_part_results(part_results, joints, ports)  # type: ignore[arg-type]
    assert len(model.bodies) == 2 and len(model.joints) == 1

    glb = tmp_path / "assembled.glb"
    manifest = _articulated_glb(model, None, glb)
    assert manifest is not None and glb.exists()

    gltf = _glb_json(glb)
    names = {n.get("name") for n in gltf["nodes"]}
    assert {"base", "arm", "assembly_yup_root"} <= names
    ck_nodes = [n for n in gltf["nodes"] if "extras" in n and "ck_joint" in n["extras"]]
    assert len(ck_nodes) == 1
    ck = ck_nodes[0]["extras"]["ck_joint"]
    assert ck["type"] == "revolute"
    assert ck["limits_deg"] == [0.0, 90.0]
    # EARNED, never asserted: no validation report ran -> "none", and the viewer says so.
    assert ck["checked"] == "none"
    assert manifest["joints"][0]["checked"] == "none"


def test_validate_motion_returns_model_and_report(tmp_path: Path) -> None:
    part_results, joints, ports = _arm_fixture(tmp_path)
    plan = SimpleNamespace(
        joints=joints,
        parts=[
            SimpleNamespace(name="base", ports=ports["base"]),
            SimpleNamespace(name="arm", ports=ports["arm"]),
        ],
    )
    motion, model, report = _validate_motion(part_results, plan)  # type: ignore[arg-type]
    assert motion is not None and model is not None and report is not None
    assert motion["joints_declared"] == 1
    # the export consumes exactly these: a swept-and-passed joint earns "swept"
    glb = tmp_path / "assembled.glb"
    manifest = _articulated_glb(model, report, glb)
    assert manifest is not None
    checked = manifest["joints"][0]["checked"]
    assert checked in {"swept", "none"}  # earned from the report, never invented
    if motion["motion_status"] == "pass":
        assert checked == "swept"


def test_rigid_plan_returns_none_triple() -> None:
    plan = SimpleNamespace(joints=[], parts=[])
    assert _validate_motion([], plan) == (None, None, None)  # type: ignore[arg-type]


def test_functional_artifacts_carry_the_glb(tmp_path: Path) -> None:
    from forge_mind.agentic.result import _functional_to_part_result

    glb = tmp_path / "assembled.glb"
    glb.write_bytes(b"glTF....")
    fres = SimpleNamespace(
        out_dir=tmp_path,
        object_name="arm",
        parts=[],
        assembled_stl=None,
        assembled_step=None,
        assembled_glb=glb,
        assembled_render=None,
        verdict=None,
        disposition="shipped_with_caveats",
        total_cost_usd=0.0,
        whole_brief=None,
        fit=None,
        motion=None,
    )
    pr = _functional_to_part_result(fres, "an arm", 0.0)  # type: ignore[arg-type]
    assert pr.artifacts.get("glb") == str(glb)
