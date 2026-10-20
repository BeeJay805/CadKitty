"""CK_JOINT GLB export: the articulated-assembly contract with the viewer (frozen v1).

Emits a binary glTF whose node tree IS the kinematic tree: every part is a named node
(mesh in NODE-LOCAL coordinates), a part placed by a moving joint has its PIVOT AT THE
NODE ORIGIN and carries ``extras.ck_joint`` exactly per the contract frozen on the
coordination board (web/src/lib/joints.ts is the loader-side reference):

    { id, name?, type: "revolute"|"prismatic", axis: [x,y,z] node-local unit,
      limits_deg?/limits_mm?, home?, checked: "swept"|"poses:N"|"none" }

Honesty: ``checked`` is EARNED, never asserted -- it is derived from the validation
report's per-joint ``assembly.sweep.<a>__<b>`` check and emits "swept" ONLY when that
check ran and PASSED. Everything else ships "none"; the viewer shows the absence.

Frame design: every node's rotation is identity in the kernel's Z-up world; moving
nodes translate to their pivot, so axes are world directions verbatim. One root node
carries the Z-up -> Y-up matrix (glTF convention), and the loader's matrixWorld
composition orients everything -- the same transform ``viz.glb.to_gltf_yup`` applies
to single parts.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.assembly.joints import Joint
from forge_mind.compiler.compile_assembly import AssemblyModel
from forge_mind.validator.assembly_checks import gear_coupled_pairs
from forge_mind.validator.report import CheckStatus, ValidationReport

_PRINTED_COLOR = (0.91, 0.64, 0.23, 1.0)  # CadKitty amber
_HARDWARE_COLOR = (0.28, 0.30, 0.33, 1.0)  # servo/motor dark gray
# Z-up (kernel) -> Y-up (glTF): rotate -90 deg about X, as viz.glb.to_gltf_yup.
_ZUP_TO_YUP = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]  # column-major glTF


def _pad(data: bytes, align: int, fill: bytes) -> bytes:
    rem = len(data) % align
    return data if rem == 0 else data + fill * (align - rem)


def sweep_level(report: ValidationReport | None, a_part: str, b_part: str) -> str:
    """The EARNED motion-verification level for a joint: "swept" iff this joint's per-joint sweep
    PASSED *and* no coupled corner-pose chain check for its moving part FAILED; "none" otherwise.

    A passing per-joint (pairwise) sweep does NOT prove the joint is clear over its full COUPLED
    range: chained joints can collide only at combined corner poses, caught by a separate
    ``assembly.sweep_corner.<parent>__<child>`` check. Labeling the joint "swept" while that corner
    check FAILED over-claims motion verification (audit H6, ADR-0065), so a failed corner check
    naming this joint's moving part demotes it to "none"."""
    if report is None:
        return "none"
    want = f"assembly.sweep.{a_part}__{b_part}"
    swept = next((c.status is CheckStatus.PASS for c in report.checks if c.id == want), False)
    if not swept:
        return "none"
    for c in report.checks:
        if (
            c.id.startswith("assembly.sweep_corner.")
            and c.status is not CheckStatus.PASS
            and b_part in c.id[len("assembly.sweep_corner.") :].split("__")
        ):
            return "none"
    return "swept"


def export_assembly_glb(
    model: AssemblyModel,
    glb_path: Path,
    *,
    report: ValidationReport | None = None,
    joint_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write the assembly as a CK_JOINT GLB; return the parts/joints manifest.

    ``joint_names`` optionally maps a moving joint's b-part id to a display name
    (e.g. link_1 -> "pan"). The manifest mirrors what was written (per-part node,
    printed flag, hardware kind; per-joint axis/origin/limits/checked) so the API
    layer and the bench scorer can read the export without re-parsing the GLB.
    """
    names = joint_names or {}
    placing: dict[str, Joint] = {}  # b part id -> joint that placed it
    for joint in model.joints:
        if joint.type.value == "gear_mesh":
            continue  # a constraint edge, never a placement edge (matches the solver)
        placing.setdefault(joint.b_part(), joint)
    # Gear-coupled pairs (ADR-0044): the FOLLOWER's ck_joint carries an additive
    # ``mimic`` field (ck_joint v1.1) -- follower_angle = ratio * driver_angle, ratio
    # negative for an external mesh. Old viewers ignore unknown keys and degrade to
    # two honestly-labeled independent sliders.
    mimic_of: dict[str, tuple[str, float]] = {}
    for d, f, ratio in gear_coupled_pairs(model.joints):
        mimic_of[f.b_part()] = (d.b_part(), -ratio)

    parts: list[tuple[str, np.ndarray, trimesh.Trimesh, bool, str | None]] = []
    for b in model.bodies:
        parts.append((b.id, np.asarray(b.transform, dtype=float), b.mesh, True, None))
    for h in model.hardware:
        if h.mesh is not None:
            parts.append((h.id, np.asarray(h.transform, dtype=float), h.mesh, False, h.kind))
    by_id = {p[0]: p for p in parts}

    def world_port(joint: Any) -> tuple[np.ndarray, np.ndarray]:
        a_tf, a_ports = model.placed_ports(joint.a_part())
        port = next(p for p in a_ports if p.name == joint.a_port())
        m = np.asarray(a_tf, dtype=float)
        o = (m @ np.array([*port.origin.as_tuple(), 1.0]))[:3]
        d = m[:3, :3] @ np.asarray(port.axis.as_tuple(), dtype=float)
        n = float(np.linalg.norm(d))
        return o, (d / n if n > 1e-12 else d)

    # Node frames: identity rotation; origin = pivot for moving parts, parent origin
    # otherwise; root parts sit at the world origin.
    moving_types = {"revolute", "prismatic"}
    origins: dict[str, np.ndarray] = {}
    children: dict[str | None, list[str]] = {}
    joints_manifest: list[dict[str, Any]] = []

    def origin_of(pid: str) -> np.ndarray:
        if pid in origins:
            return origins[pid]
        j = placing.get(pid)
        if j is None:
            origins[pid] = np.zeros(3)
        elif j.type.value in moving_types:
            o, _ = world_port(j)
            origins[pid] = o
        else:
            origins[pid] = origin_of(j.a_part())
        return origins[pid]

    for pid, *_ in parts:
        j = placing.get(pid)
        parent = j.a_part() if j is not None and j.a_part() in by_id else None
        children.setdefault(parent, []).append(pid)
        origin_of(pid)

    # ---- glTF construction -------------------------------------------------------
    bin_parts: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    materials = [
        {
            "name": "printed",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(_PRINTED_COLOR),
                "metallicFactor": 0.1,
                "roughnessFactor": 0.6,
            },
        },
        {
            "name": "hardware",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(_HARDWARE_COLOR),
                "metallicFactor": 0.4,
                "roughnessFactor": 0.5,
            },
        },
    ]

    def add_view(data: bytes, target: int) -> int:
        offset = sum(len(b) for b in bin_parts)
        bin_parts.append(_pad(data, 4, b"\x00"))
        buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
        )
        return len(buffer_views) - 1

    def add_mesh(pid: str, verts: np.ndarray, mesh: trimesh.Trimesh, printed: bool) -> int:
        v32 = verts.astype("<f4")
        n32 = np.asarray(mesh.vertex_normals, dtype="<f4")
        idx = np.asarray(mesh.faces, dtype="<u4").reshape(-1)
        pos_view = add_view(v32.tobytes(), 34962)
        nrm_view = add_view(n32.tobytes(), 34962)
        idx_view = add_view(idx.tobytes(), 34963)
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": len(v32),
                "type": "VEC3",
                "min": [float(x) for x in v32.min(axis=0)],
                "max": [float(x) for x in v32.max(axis=0)],
            }
        )
        pos_acc = len(accessors) - 1
        accessors.append(
            {
                "bufferView": nrm_view,
                "componentType": 5126,
                "count": len(n32),
                "type": "VEC3",
            }
        )
        nrm_acc = len(accessors) - 1
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": 5125,
                "count": len(idx),
                "type": "SCALAR",
            }
        )
        idx_acc = len(accessors) - 1
        meshes.append(
            {
                "name": pid,
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                        "indices": idx_acc,
                        "material": 0 if printed else 1,
                    }
                ],
            }
        )
        return len(meshes) - 1

    node_index: dict[str, int] = {}

    def build_node(pid: str) -> int:
        part_id, transform, mesh, printed, _kind = by_id[pid]
        world = mesh.copy()
        world.apply_transform(transform)
        verts = np.asarray(world.vertices, dtype=float) - origins[pid]
        mesh_idx = add_mesh(part_id, verts, world, printed)
        j = placing.get(pid)
        parent_origin = (
            origins[j.a_part()] if (j is not None and j.a_part() in by_id) else np.zeros(3)
        )
        translation = origins[pid] - parent_origin
        node: dict[str, Any] = {"name": part_id, "mesh": mesh_idx}
        if np.any(np.abs(translation) > 1e-9):
            node["translation"] = [float(x) for x in translation]
        if j is not None and j.type.value in moving_types:
            _, axis = world_port(j)
            checked = sweep_level(report, j.a_part(), j.b_part())
            ck: dict[str, Any] = {
                "id": pid,
                "name": names.get(pid, pid),
                "type": j.type.value,
                "axis": [float(x) for x in axis],
                "home": 0,
                "checked": checked,
            }
            if j.type.value == "revolute":
                lo, hi = j.range_deg or (0.0, 180.0)
                ck["limits_deg"] = [float(lo), float(hi)]
            else:
                lo, hi = j.range_mm or (0.0, 10.0)
                ck["limits_mm"] = [float(lo), float(hi)]
            if pid in mimic_of:
                of_id, ratio = mimic_of[pid]
                ck["mimic"] = {"of": of_id, "ratio": float(ratio)}
            node["extras"] = {"ck_joint": ck}
            joints_manifest.append(
                {
                    "id": pid,
                    "name": names.get(pid, pid),
                    "type": j.type.value,
                    "parent": j.a_part(),
                    "child": pid,
                    "axis_world_zup": [float(x) for x in axis],
                    "origin_world_zup": [float(x) for x in origins[pid]],
                    "limits": ck.get("limits_deg", ck.get("limits_mm")),
                    "checked": checked,
                    **(
                        {"mimic": {"of": mimic_of[pid][0], "ratio": float(mimic_of[pid][1])}}
                        if pid in mimic_of
                        else {}
                    ),
                }
            )
        nodes.append(node)
        idx = len(nodes) - 1
        node_index[pid] = idx
        kids = [build_node(c) for c in children.get(pid, [])]
        if kids:
            node["children"] = kids
        return idx

    roots = [build_node(pid) for pid in children.get(None, [])]
    nodes.append({"name": "assembly_yup_root", "matrix": list(_ZUP_TO_YUP), "children": roots})

    gltf = {
        "asset": {"version": "2.0", "generator": "forge-mind assembly_glb (ck_joint v1)"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": sum(len(b) for b in bin_parts)}],
    }

    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")
    bin_chunk = _pad(b"".join(bin_parts), 4, b"\x00")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with glb_path.open("wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, total))
        fh.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fh.write(json_chunk)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        fh.write(bin_chunk)

    return {
        "format": "ck_joint_v1",
        "parts": [
            {
                "id": pid,
                "node": node_index[pid],
                "printed": printed,
                "hardware_kind": kind,
                "source": hardware_source(kind),
            }
            for pid, _tf, _mesh, printed, kind in parts
        ],
        "joints": joints_manifest,
    }


def hardware_source(kind: str | None) -> str | None:
    """The registry citation for a hardware kind (None for printed parts). ONE
    resolver so the manifest, the BOM, and the provenance panel can never cite
    different sources for the same part."""
    if kind is None:
        return None
    if kind == "mg996r_servo":
        from forge_mind.assembly.servo import MG996R_SOURCE

        return MG996R_SOURCE
    if kind == "nema17_motor":
        return "NEMA ICS 16 (size 17 face) + common datasheet body length"
    if kind == "dowel_pin":
        from forge_mind.dsl.standards import DOWEL_PINS

        return DOWEL_PINS.get("6").source
    return None


__all__ = ["export_assembly_glb", "hardware_source", "sweep_level"]
