"""Render one staged, verified occurrence mesh without opening native CAD files."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import trimesh

from forge_mind.viz.render import render_workspace_preview

MAX_TOPOLOGY_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_BODIES = 4096
MAX_ENTITIES = 200_000
MAX_VERTICES = 1_000_000
MAX_TRIANGLES = 500_000
REQUEST_KEYS = {
    "revision_id",
    "occurrence_path",
    "body_index",
    "topology_sha256",
    "step_sha256",
    "document_sha256",
}


def _regular_bytes(path: Path, maximum: int) -> bytes:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not 0 < info.st_size <= maximum
    ):
        raise ValueError("Expected a bounded regular file")
    data = path.read_bytes()
    if not 0 < len(data) <= maximum:
        raise ValueError("File exceeded its bound")
    return data


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON key")
        result[name] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError("Nonfinite JSON number: " + value)


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _index(value: Any, maximum: int) -> int:
    if type(value) is not int or not 0 <= value < maximum:
        raise ValueError("Invalid mesh or body index")
    return int(value)


def _point(value: Any) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(v) not in {int, float} or not math.isfinite(v) for v in value)
    ):
        raise ValueError("Invalid finite mesh vertex")
    return [float(v) for v in value]


class TopologyMeshLimitError(ValueError):
    """Valid display input exceeds a rendering resource limit."""


def selected_topology_meshes(
    topology: dict[str, Any], body_indices: list[int]
) -> tuple[Any, int, int]:
    """Validate the entire topology once and combine only the requested world bodies."""
    if not isinstance(body_indices, list) or not body_indices:
        raise ValueError("Select at least one explicit topology body")
    if topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Unsupported topology coordinate frame")
    counts = topology.get("topology_counts")
    if not isinstance(counts, dict):
        raise ValueError("Missing body count metadata")
    count = counts.get("bodies")
    if type(count) is int and count > MAX_BODIES:
        raise TopologyMeshLimitError("Topology body count exceeds its display limit")
    if type(count) is not int or not 1 <= count <= MAX_BODIES:
        raise ValueError("Invalid body count")
    for body_index in body_indices:
        _index(body_index, count)
    if len(set(body_indices)) != len(body_indices):
        raise ValueError("Duplicate selected topology body")
    entities = topology.get("entities")
    meshes = topology.get("meshes")
    if (isinstance(entities, list) and len(entities) > MAX_ENTITIES) or (
        isinstance(meshes, list) and len(meshes) > MAX_ENTITIES
    ):
        raise TopologyMeshLimitError("Topology entity count exceeds its display limit")
    if (
        not isinstance(entities, list)
        or not 1 <= len(entities) <= MAX_ENTITIES
        or not isinstance(meshes, list)
        or not 1 <= len(meshes) <= MAX_ENTITIES
    ):
        raise ValueError("Missing or oversized topology mesh metadata")
    by_id: dict[str, dict[str, Any]] = {}
    bodies: dict[int, dict[str, Any]] = {}
    for row in entities:
        if not isinstance(row, dict):
            raise ValueError("Invalid topology entity")
        identity = row.get("id")
        if (
            not isinstance(identity, str)
            or not identity
            or len(identity) > 128
            or identity in by_id
        ):
            raise ValueError("Duplicate or invalid entity identity")
        index = _index(row.get("body_index"), count)
        if row.get("kind") not in {"body", "face", "edge"}:
            raise ValueError("Unsupported entity kind")
        by_id[identity] = row
        if row["kind"] == "body":
            if index in bodies:
                raise ValueError("Ambiguous body metadata")
            bodies[index] = row
    if set(bodies) != set(range(count)):
        raise ValueError("Missing body metadata")
    for row in bodies.values():
        if (
            not isinstance(row.get("signature"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", row["signature"])
            or row.get("body_signature") != row["signature"]
        ):
            raise ValueError("Invalid body signature metadata")
        lower, upper = np.asarray(_point(row.get("min"))), np.asarray(_point(row.get("max")))
        if (upper < lower).any():
            raise ValueError("Unordered body bounds")
    for row in by_id.values():
        if row.get("body_signature") != bodies[row["body_index"]]["signature"]:
            raise ValueError("Entity has ambiguous body signature")
    selected: dict[int, list[Any]] = {index: [] for index in body_indices}
    seen: set[str] = set()
    vertices_count = triangles_count = 0
    for row in meshes:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or row["id"] not in by_id
        ):
            raise ValueError("Mesh has no authoritative entity")
        identity = row["id"]
        entity = by_id[identity]
        index = _index(row.get("body_index"), count)
        if identity in seen or index != entity["body_index"] or entity["kind"] == "edge":
            raise ValueError("Duplicate or ambiguous mesh metadata")
        seen.add(identity)
        points, triangles = row.get("vertices"), row.get("triangles")
        if (
            not isinstance(points, list)
            or not points
            or not isinstance(triangles, list)
            or not triangles
        ):
            raise ValueError("Empty or invalid mesh")
        vertices_count += len(points)
        triangles_count += len(triangles)
        if vertices_count > MAX_VERTICES or triangles_count > MAX_TRIANGLES:
            raise TopologyMeshLimitError("Mesh count bound exceeded")
        finite_points = [_point(point) for point in points]
        coordinates = np.asarray(finite_points)
        body = bodies[index]
        lower, upper = np.asarray(body["min"]), np.asarray(body["max"])
        tolerance = max(float(np.linalg.norm(upper - lower)) * 1e-6, 1e-5)
        if (coordinates < lower - tolerance).any() or (coordinates > upper + tolerance).any():
            raise ValueError("Mesh lies outside its body metadata bounds")
        for face in triangles:
            if not isinstance(face, list) or len(face) != 3:
                raise ValueError("Invalid triangle")
            for value in face:
                _index(value, len(points))
        if index in selected:
            selected[index].append((entity["kind"], identity, finite_points, triangles))
    required: dict[int, set[str]] = {index: set() for index in body_indices}
    expected_kind = "body" if topology.get("selection_scope") == "body" else "face"
    if topology.get("selection_scope") not in {"body", "face_edge_body"}:
        raise ValueError("Mesh selection scope is ambiguous")
    for key, row in by_id.items():
        if row["body_index"] in required and row["kind"] == expected_kind:
            required[row["body_index"]].add(key)
    combined = []
    for index in sorted(selected):
        items = selected[index]
        if {item[0] for item in items} != {expected_kind}:
            raise ValueError("Selected body meshes are missing or ambiguous")
        if {item[1] for item in items} != required[index]:
            raise ValueError("Selected body mesh coverage is incomplete")
        combined.extend(items)
    meshes_out = []
    discarded = 0
    for _, _, vertices, triangles in combined:
        # Binary STL uses float32. Validate the displayed precision before export.
        points_array = np.asarray(vertices, dtype=np.float64).astype(np.float32).astype(np.float64)
        if not np.isfinite(points_array).all():
            raise ValueError("Vertices cannot be represented in display STL")
        mesh = trimesh.Trimesh(vertices=points_array, faces=triangles, process=False)
        areas = mesh.area_faces
        if not np.isfinite(areas).all():
            raise ValueError("Nonfinite display triangle area")
        # Valid curved tessellations can have collapsed pole triangles. Remove
        # only exact zero-area display faces; do not repair the source surface.
        keep = areas > 0
        discarded += int(np.count_nonzero(~keep))
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        if not len(mesh.faces):
            raise ValueError("Selected mesh has no nondegenerate display triangles")
        meshes_out.append(mesh)
    return trimesh.util.concatenate(meshes_out), int(count), discarded


def _selected_mesh(topology: dict[str, Any], body_index: int) -> tuple[Any, int, int]:
    """Preserve the original single-occurrence selection boundary."""
    return selected_topology_meshes(topology, [body_index])


def render_occurrence(request_path: Path) -> dict[str, Any]:
    """Render inside the service-owned staging directory; authority comes from its request."""
    request = _json(_regular_bytes(request_path, MAX_REQUEST_BYTES))
    if set(request) not in (REQUEST_KEYS, REQUEST_KEYS | {"focus"}):
        raise ValueError("Unexpected occurrence-render request keys")
    if not isinstance(request["revision_id"], str) or not re.fullmatch(
        r"[a-f0-9]{32}", request["revision_id"]
    ):
        raise ValueError("Invalid revision identifier")
    if not isinstance(request["occurrence_path"], str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]{0,63}(?:/[A-Za-z][A-Za-z0-9_]{0,63}){0,63}",
        request["occurrence_path"],
    ):
        raise ValueError("Invalid occurrence path")
    for key in ("topology_sha256", "step_sha256", "document_sha256"):
        if not isinstance(request[key], str) or not re.fullmatch(r"[a-f0-9]{64}", request[key]):
            raise ValueError("Invalid source digest")
    directory = request_path.parent
    topology_bytes = _regular_bytes(directory / "topology.json", MAX_TOPOLOGY_BYTES)
    if hashlib.sha256(topology_bytes).hexdigest() != request["topology_sha256"]:
        raise ValueError("Staged topology hash mismatch")
    topology = _json(topology_bytes)
    tessellation = topology.get("pick_tessellation")
    if not isinstance(tessellation, dict):
        raise ValueError("Missing tessellation metadata")
    for name in ("relative_linear_deflection", "angular_deflection_radians"):
        value = tessellation.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid tessellation deflection")
    if not isinstance(tessellation.get("purpose"), str) or len(tessellation["purpose"]) > 512:
        raise ValueError("Missing tessellation purpose")
    resolved_focus = None
    if "focus" in request:
        from forge_mind.workspace.occurrence_focus import (
            OccurrenceFocus,
            focus_view_metadata,
            resolve_focus,
            validate_focus_frame,
        )

        resolved_focus = validate_focus_frame(request["focus"])
        expected_focus = resolve_focus(
            topology,
            request["body_index"],
            OccurrenceFocus.model_validate(resolved_focus["request"]),
        )
        if resolved_focus != expected_focus:
            raise ValueError("Staged occurrence focus does not match saved selection")
    mesh, count, discarded = _selected_mesh(topology, request["body_index"])
    image_path, receipt_path = directory / "inspection.png", directory / "receipt.json"
    if image_path.exists() or receipt_path.exists():
        raise ValueError("Occurrence-render outputs already exist")
    with TemporaryDirectory(prefix="occurrence-render-", dir=directory) as scratch:
        temporary = Path(scratch)
        stl, image = temporary / "selected.stl", temporary / "inspection.png"
        if resolved_focus is None:
            mesh.export(stl, file_type="stl")
            render_workspace_preview(
                stl,
                image,
                frame_bounds=mesh.bounds,
                inspection_label="Selected occurrence inspection · other bodies omitted",
                frame_scope="selected occurrence frame",
            )
        else:
            from forge_mind.viz.focus import render_occurrence_focus

            focus_view = render_occurrence_focus(mesh, image, resolved_focus)
            if focus_view != focus_view_metadata(resolved_focus):
                raise ValueError("Occurrence focus renderer returned different view metadata")
        image_bytes = _regular_bytes(image, MAX_IMAGE_BYTES)
    receipt = {
        **request,
        "image": {"file": "inspection.png", "sha256": hashlib.sha256(image_bytes).hexdigest()},
        "frame": "cad_z_up_mm_world",
        "selected_body_count": 1,
        "omitted_body_count": count - 1,
        "omitted_body_indices": [index for index in range(count) if index != request["body_index"]],
        "frame_bounds": {"min": mesh.bounds[0].tolist(), "max": mesh.bounds[1].tolist()},
        "display": {
            "purpose": "Selected occurrence mesh display only; not manufacturing measurement or complete assembly",
            "source_purpose": tessellation["purpose"],
            "relative_linear_deflection": tessellation["relative_linear_deflection"],
            "angular_deflection_radians": tessellation["angular_deflection_radians"],
            "vertex_precision": "binary STL float32",
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "discarded_degenerate_triangles": discarded,
            "triangle_filter": "Exact zero-area display triangles omitted; no surface repair or coverage certification",
        },
        "source_binding": "Service supplies verified revision, occurrence, STEP and document binding; worker verifies staged topology digest",
    }
    if resolved_focus is not None:
        receipt["focus"] = resolved_focus
        receipt["focus_view"] = focus_view
    receipt_bytes = (json.dumps(receipt, allow_nan=False) + "\n").encode()
    if len(receipt_bytes) > MAX_RECEIPT_BYTES:
        raise ValueError("Occurrence-render receipt exceeded its bound")
    with image_path.open("xb") as output:
        output.write(image_bytes)
    with receipt_path.open("xb") as output:
        output.write(receipt_bytes)
    return receipt


def main() -> int:
    if len(sys.argv) != 2:
        raise ValueError("Use occurrence_render REQUEST_JSON")
    render_occurrence(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
