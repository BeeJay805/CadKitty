"""The measured-motion -> ck_joint GLB bridge (ADR-0086 slice 2): animatable claims become
joint extras with the EARNED verification level and the MEASURED arc as slider limits;
locked/refused bodies render rigid."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.physics.report.motion_project import project_motion
from forge_mind.viz.motion_glb import export_motion_glb


def _glb_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    magic, version, _length = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF" and version == 2
    clen, ctype = struct.unpack_from("<II", raw, 12)
    assert ctype == 0x4E4F534A  # JSON chunk
    out: dict[str, Any] = json.loads(raw[20 : 20 + clen].decode("utf-8"))
    return out


def test_single_rotor_bakes_a_spin_joint(tmp_path: Path) -> None:
    ring = trimesh.creation.annulus(r_min=5.0, r_max=15.0, height=8.0)
    motion, _ = project_motion(ring)
    assert motion is not None
    glb = tmp_path / "part.glb"
    assert export_motion_glb(glb, [ring], list(motion.claims)) is True
    doc = _glb_json(glb)
    assert [n["name"] for n in doc["nodes"]] == ["body_0"]
    ck = doc["nodes"][0]["extras"]["ck_joint"]
    assert ck["type"] == "revolute"
    assert ck["limits_deg"] == [-180.0, 180.0]
    # a lone body has no collision partner: no sweep ran and none is implied
    assert ck["checked"] == "none"
    # the Z-up spin axis (0,0,1) maps to Y-up (0,1,0) — the same map the geometry gets
    axis = np.asarray(ck["axis"], dtype=float)
    assert np.allclose(np.abs(axis), [0.0, 1.0, 0.0], atol=1e-3)


def test_pair_bakes_rotor_joint_with_measured_arc_and_pose_level(tmp_path: Path) -> None:
    # the arc-limited spider-vs-wall pair from the T2 suite
    hub = trimesh.creation.cylinder(radius=6.0, height=3.0)
    parts = [hub]
    for k in range(3):
        arm = trimesh.creation.box(extents=(11.0, 2.4, 3.0))
        arm.apply_translation((10.5, 0.0, 0.0))  # spans x 5..16: overlaps INTO the hub
        arm.apply_transform(
            trimesh.transformations.rotation_matrix(np.deg2rad(120.0 * k), (0, 0, 1))
        )
        parts.append(arm)
    rotor = trimesh.boolean.union(parts)
    wall = trimesh.creation.box(extents=(4.0, 4.0, 10.0))
    wall.apply_translation((13.0 * np.cos(np.deg2rad(60.0)), 13.0 * np.sin(np.deg2rad(60.0)), 0.0))
    mesh = trimesh.util.concatenate([rotor, wall])
    motion, _ = project_motion(mesh)
    assert motion is not None and motion.claims[0].status == "arc_limited"
    claim = motion.claims[0]

    # the same welded body derivation the server wiring uses, so body_<i> ids agree
    welded = mesh.copy()
    welded.merge_vertices()
    bodies = [b for b in welded.split(only_watertight=False) if len(b.faces) > 0]
    glb = tmp_path / "part.glb"
    assert export_motion_glb(glb, bodies, list(motion.claims)) is True
    doc = _glb_json(glb)
    assert [n["name"] for n in doc["nodes"]] == ["body_0", "body_1"]
    by_name = {n["name"]: n for n in doc["nodes"]}
    rotor_node = by_name[claim.moving[0]]
    stator_node = by_name[claim.static[0]]
    ck = rotor_node["extras"]["ck_joint"]
    assert claim.allowed_arc_deg is not None
    assert ck["limits_deg"] == [claim.allowed_arc_deg[0], claim.allowed_arc_deg[1]]
    assert ck["checked"] == "poses:24"  # the EARNED sweep level, verbatim
    assert "extras" not in stator_node  # the stator never gets a joint


def test_locked_claims_produce_no_joints(tmp_path: Path) -> None:
    shaft = trimesh.creation.cylinder(radius=5.5, height=30.0)
    ring = trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0)
    mesh = trimesh.util.concatenate([shaft, ring])
    motion, _ = project_motion(mesh)
    assert motion is not None and motion.claims[0].status == "locked"
    bodies = [b for b in mesh.split(only_watertight=False) if len(b.faces) > 0]
    glb = tmp_path / "part.glb"
    assert export_motion_glb(glb, bodies, list(motion.claims)) is True
    doc = _glb_json(glb)
    assert all("extras" not in n for n in doc["nodes"])  # rigid render: no fake movement


def test_bearing_bakes_driver_plus_mimic_rollers(tmp_path: Path) -> None:
    outer = trimesh.creation.annulus(r_min=14.0, r_max=18.0, height=6.0)
    inner = trimesh.creation.annulus(r_min=4.0, r_max=8.0, height=6.0)
    bodies = [outer, inner]
    for k in range(6):
        ball = trimesh.creation.icosphere(subdivisions=2, radius=2.8)
        a = np.deg2rad(60.0 * k)
        ball.apply_translation((11.0 * np.cos(a), 11.0 * np.sin(a), 0.0))
        bodies.append(ball)
    mesh = trimesh.util.concatenate(bodies)
    motion, _ = project_motion(mesh)
    assert motion is not None and motion.claims[0].kind == "coupled_rotation"
    claim = motion.claims[0]

    welded = mesh.copy()
    welded.merge_vertices()
    split = [b for b in welded.split(only_watertight=False) if len(b.faces) > 0]
    glb = tmp_path / "part.glb"
    assert export_motion_glb(glb, split, list(motion.claims)) is True
    doc = _glb_json(glb)
    by_name = {n["name"]: n for n in doc["nodes"]}
    driver = by_name[claim.moving[0]]["extras"]["ck_joint"]
    assert driver["id"] == claim.id and driver["checked"] == "poses:24"
    assert claim.couplings is not None
    mimic_count = 0
    for c in claim.couplings:
        ck = by_name[c.body]["extras"]["ck_joint"]
        assert ck["mimic"]["of"] == claim.id
        assert abs(ck["mimic"]["ratio"] - c.ratio) < 1e-9
        assert ck["checked"] == "poses:24"  # certified by the SAME coupled sweep
        mimic_count += 1
    assert mimic_count == 6
    assert "extras" not in by_name[claim.static[0]]  # the outer race stays rigid


def test_vertex_colors_carried_for_single_body(tmp_path: Path) -> None:
    ring = trimesh.creation.annulus(r_min=5.0, r_max=15.0, height=8.0)
    motion, _ = project_motion(ring)
    assert motion is not None
    colors = np.tile(np.array([200, 40, 40, 255], dtype=np.uint8), (len(ring.vertices), 1))
    glb = tmp_path / "part.glb"
    assert export_motion_glb(glb, [ring], list(motion.claims), colors) is True
    doc = _glb_json(glb)
    assert "COLOR_0" in doc["meshes"][0]["primitives"][0]["attributes"]
