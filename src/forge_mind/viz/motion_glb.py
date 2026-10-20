"""Measured motion -> the frozen ck_joint GLB contract (ADR-0086 slice 2).

Uploads have no declared joints, but slice 1 MEASURES motion claims for them. This writer
bakes those claims into the same ``extras.ck_joint`` node metadata the engine's declared
mechanisms already use — so the web viewer's EXISTING articulation rig (JointPanel sliders,
forward kinematics, honesty banner) animates uploaded parts with zero new frontend surface,
and the slider limits ARE the certified arc: the user physically cannot scrub into a pose
the sweep measured as colliding.

Honesty mapping (the no-fake-movement law):
  * only ``free`` / ``arc_limited`` claims produce a joint; ``locked`` / ``refused`` bodies
    render rigid (the MotionProof suite carries the measured reason);
  * ``checked`` reports the EARNED level verbatim: ``poses:<N>`` for a swept pair claim,
    ``none`` for a single-body spin (there is no collision partner — no sweep ran, and we
    never imply one did);
  * ``limits_deg`` for an ``arc_limited`` claim is exactly the measured clear arc.

Bodies arrive Z-up (the analysis frame); geometry AND claim axes are converted with the
same Z-up -> Y-up map the flat viewer GLB uses, so nothing changes frames twice. Node names
are ``body_<i>`` — identical to the MotionReport contract's body ids.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.motion.pair import SWEEP_POSES
from forge_mind.physics.report.schema import MotionClaim
from forge_mind.viz.glb import to_gltf_yup

_MATERIAL = {
    "name": "printed",
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.72, 0.75, 0.78, 1.0],
        "metallicFactor": 0.1,
        "roughnessFactor": 0.6,
    },
}


def _pad(data: bytes, align: int, fill: bytes) -> bytes:
    rem = len(data) % align
    return data if rem == 0 else data + fill * (align - rem)


def _yup_vec(v: tuple[float, float, float] | list[float]) -> list[float]:
    """The same -90 deg-about-X map ``to_gltf_yup`` applies to geometry: (x,y,z)->(x,z,-y)."""
    return [float(v[0]), float(v[2]), -float(v[1])]


_KIND_NAMES = {"spin": "Spin", "pair_rotation": "Rotation", "coupled_rotation": "Drive"}


def _ck_joint(claim: MotionClaim) -> dict[str, Any] | None:
    """The DRIVER's ck_joint extras for an ANIMATABLE claim (None -> rigid render)."""
    if claim.status not in ("free", "arc_limited") or claim.axis is None:
        return None
    if claim.allowed_arc_deg is not None:
        lo, hi = float(claim.allowed_arc_deg[0]), float(claim.allowed_arc_deg[1])
    else:
        lo, hi = -180.0, 180.0
    axis = claim.axis
    return {
        "id": claim.id,
        "name": _KIND_NAMES.get(claim.kind, "Rotation"),
        "type": "revolute",
        "axis": _yup_vec((axis.direction.x, axis.direction.y, axis.direction.z)),
        "origin": _yup_vec((axis.origin.x, axis.origin.y, axis.origin.z)),
        "limits_deg": [lo, hi],
        "home": 0,
        # EARNED verification level, verbatim (the viewer surfaces it, never upgrades it):
        # swept claims report their pose count; a lone body has no collision partner, so no
        # sweep ran and none is implied.
        "checked": ("none" if claim.kind == "spin" else f"poses:{SWEEP_POSES}"),
    }


def _mimic_joints(claim: MotionClaim, driver_ck: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """ck_joint extras for each coupled follower (ck_joint v1.1 mimic): the follower turns
    ``ratio`` times the driver about its OWN measured axis — a bearing's rollers orbit the
    main axis, a meshed gear counter-rotates about its own center. The viewer's existing
    mimic handling drives them from the ONE driver slider."""
    out: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(claim.couplings or []):
        out[c.body] = {
            "id": f"{claim.id}_f{i}",
            "name": "Coupled",
            "type": "revolute",
            "axis": _yup_vec((c.axis.direction.x, c.axis.direction.y, c.axis.direction.z)),
            "origin": _yup_vec((c.axis.origin.x, c.axis.origin.y, c.axis.origin.z)),
            "limits_deg": [-360.0, 360.0],  # driven: range follows the driver via the ratio
            "home": 0,
            "checked": driver_ck["checked"],  # certified by the SAME coupled sweep
            "mimic": {"of": str(driver_ck["id"]), "ratio": float(c.ratio)},
        }
    return out


def export_motion_glb(
    glb_path: Path,
    bodies_zup: list[trimesh.Trimesh],
    claims: list[MotionClaim],
    vertex_colors: np.ndarray | None = None,
) -> bool:
    """Write ``glb_path`` as per-body nodes with ck_joint extras from the measured claims.

    ``vertex_colors`` (RGBA uint8, per-vertex) is carried as COLOR_0 for the single-body
    case (the stress lens bake). Atomic (tmp + replace): on ANY failure the existing flat
    GLB stays and the viewer renders the part rigid — honest degradation, never corruption.
    Returns True when the articulated GLB was written."""
    tmp = glb_path.with_name(glb_path.name + ".motion.tmp")
    try:
        joints_by_body: dict[str, dict[str, Any]] = {}
        for claim in claims:
            ck = _ck_joint(claim)
            if ck is None:
                continue
            mimics = _mimic_joints(claim, ck)
            for body_id in claim.moving:
                # coupled followers get their mimic joint; every other mover is the driver
                joints_by_body[body_id] = mimics.get(body_id, ck)

        bin_parts: list[bytes] = []
        buffer_views: list[dict[str, Any]] = []
        accessors: list[dict[str, Any]] = []
        meshes: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []

        def add_view(data: bytes, target: int) -> int:
            offset = sum(len(b) for b in bin_parts)
            bin_parts.append(_pad(data, 4, b"\x00"))
            buffer_views.append(
                {"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target}
            )
            return len(buffer_views) - 1

        for i, body in enumerate(bodies_zup):
            yb = to_gltf_yup(body.copy())
            v32 = np.asarray(yb.vertices, dtype="<f4")
            n32 = np.asarray(yb.vertex_normals, dtype="<f4")
            idx = np.asarray(yb.faces, dtype="<u4").reshape(-1)
            attributes: dict[str, int] = {}
            accessors.append(
                {
                    "bufferView": add_view(v32.tobytes(), 34962),
                    "componentType": 5126,
                    "count": len(v32),
                    "type": "VEC3",
                    "min": [float(x) for x in v32.min(axis=0)],
                    "max": [float(x) for x in v32.max(axis=0)],
                }
            )
            attributes["POSITION"] = len(accessors) - 1
            accessors.append(
                {
                    "bufferView": add_view(n32.tobytes(), 34962),
                    "componentType": 5126,
                    "count": len(n32),
                    "type": "VEC3",
                }
            )
            attributes["NORMAL"] = len(accessors) - 1
            if vertex_colors is not None and len(bodies_zup) == 1:
                c8 = np.ascontiguousarray(np.asarray(vertex_colors, dtype=np.uint8)[:, :4])
                if len(c8) == len(v32):
                    accessors.append(
                        {
                            "bufferView": add_view(c8.tobytes(), 34962),
                            "componentType": 5121,
                            "count": len(c8),
                            "type": "VEC4",
                            "normalized": True,
                        }
                    )
                    attributes["COLOR_0"] = len(accessors) - 1
            accessors.append(
                {
                    "bufferView": add_view(idx.tobytes(), 34963),
                    "componentType": 5125,
                    "count": len(idx),
                    "type": "SCALAR",
                }
            )
            meshes.append(
                {
                    "name": f"body_{i}",
                    "primitives": [
                        {
                            "attributes": attributes,
                            "indices": len(accessors) - 1,
                            "material": 0,
                        }
                    ],
                }
            )
            node: dict[str, Any] = {"name": f"body_{i}", "mesh": len(meshes) - 1}
            ck = joints_by_body.get(f"body_{i}")
            if ck is not None:
                node["extras"] = {"ck_joint": ck}
            nodes.append(node)

        blob = b"".join(bin_parts)
        gltf = {
            "asset": {"version": "2.0", "generator": "forge-mind motion_glb (ck_joint v1.1)"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}],
            "nodes": nodes,
            "meshes": meshes,
            "materials": [_MATERIAL],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(blob)}],
        }
        js = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4, b" ")
        header = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
        payload = (
            header
            + struct.pack("<II", len(js), 0x4E4F534A)
            + js
            + struct.pack("<II", len(blob), 0x004E4942)
            + blob
        )
        tmp.write_bytes(payload)
        tmp.replace(glb_path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False  # the flat GLB stays; the viewer renders the part rigid (honest)


__all__ = ["export_motion_glb"]
