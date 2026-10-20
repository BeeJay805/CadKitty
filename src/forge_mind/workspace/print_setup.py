"""Isolated saved-geometry print-plane observer and rigid STL derivative writer."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from forge_mind.workspace.definition_meshes import MANIFEST_NAME, validate_definition_mesh
from forge_mind.workspace.document import digest
from forge_mind.workspace.print_setup_models import (
    ALGORITHM,
    RANKING,
    SCOPE,
    PrintCandidate,
    PrintPartInspection,
    PrintPlane,
    PrintSetupRequest,
    PrintSetupResult,
    PrintSource,
)
from forge_mind.workspace.processing import MAX_JOB_BYTES, regular_file

RIGID_GUARD = 1e-10  # Dimensionless arithmetic guard, not a physical allowance.
_STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> bytes:
    if not regular_file(path, MAX_JOB_BYTES):
        raise ValueError("Print setup input is missing, linked or oversized")
    with path.open("rb") as stream:
        data = stream.read(MAX_JOB_BYTES + 1)
    if len(data) > MAX_JOB_BYTES:
        raise ValueError("Print setup input exceeds its byte budget")
    return data


def stl_triangles(data: bytes) -> Any:
    """Read the complete ordered triangle inventory, without mesh repair or welding."""
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + count * 50:
            triangles = np.frombuffer(data, dtype=_STL_DTYPE, count=count, offset=84)[
                "vertices"
            ].astype(float)
        else:
            triangles = _ascii_triangles(data)
    else:
        triangles = _ascii_triangles(data)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("Print setup requires a nonempty complete STL triangle inventory")
    if not np.isfinite(triangles).all():
        raise ValueError("STL coordinates must be finite")
    lengths = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("STL contains collapsed or invalid triangles; no repair was attempted")
    return triangles


def _ascii_triangles(data: bytes) -> Any:
    try:
        lines = [line.strip().lower() for line in data.decode("ascii").splitlines() if line.strip()]
        if not lines or not lines[0].startswith("solid") or not lines[-1].startswith("endsolid"):
            raise ValueError("ASCII STL requires complete solid delimiters")
        facets = lines[1:-1]
        if not facets or len(facets) % 7:
            raise ValueError("ASCII STL has an incomplete facet inventory")
        triangles = []
        for index in range(0, len(facets), 7):
            facet = facets[index : index + 7]
            normal = facet[0].split()
            if (
                len(normal) != 5
                or normal[:2] != ["facet", "normal"]
                or facet[1] != "outer loop"
                or facet[5:] != ["endloop", "endfacet"]
            ):
                raise ValueError("ASCII STL has malformed facets")
            vertices = []
            for line in facet[2:5]:
                tokens = line.split()
                if len(tokens) != 4 or tokens[0] != "vertex":
                    raise ValueError("ASCII STL has malformed vertices")
                vertices.append([float(value) for value in tokens[1:]])
            triangles.append(vertices)
        return np.asarray(triangles, dtype=float)
    except UnicodeDecodeError as exc:
        raise ValueError("STL is neither an exact binary inventory nor complete ASCII STL") from exc


def _mesh(triangles: Any) -> Any:
    import trimesh

    # Exact coordinate equality associates connectivity; no coordinates are changed.
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices=vertices, faces=inverse.reshape(-1, 3), process=False)


def mesh_inventory(triangles: Any, expected_bodies: int) -> tuple[Any, list[Any], dict[str, Any]]:
    mesh = _mesh(triangles)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Saved mesh is not watertight with consistent winding and positive volume")
    if not math.isfinite(float(mesh.volume)) or not np.isfinite(mesh.center_mass).all():
        raise ValueError("Saved mesh volume or uniform-material centroid is nonfinite")
    shells = list(mesh.split(only_watertight=False, repair=False))
    positive = [shell for shell in shells if shell.volume > 0]
    if any(not shell.is_watertight or not shell.is_winding_consistent for shell in shells):
        raise ValueError("A saved connected shell is not a closed consistently wound surface")
    if len(positive) != expected_bodies:
        raise ValueError("Positive closed mesh shells do not match the saved native body count")
    if expected_bodies == 1:
        bodies = [mesh]  # Includes negatively wound enclosed cavity shells.
    elif len(shells) == len(positive):
        bodies = positive
    else:
        raise ValueError("Multiple made bodies with enclosed cavity shells need body association")
    return (
        mesh,
        bodies,
        {
            "triangle_count": len(triangles),
            "exact_coordinate_vertex_count": len(mesh.vertices),
            "body_count": expected_bodies,
            "closed_shell_count": len(shells),
            "watertight": True,
            "winding_consistent": True,
            "volume_mm3": float(mesh.volume),
            "uniform_material_centroid_mm": mesh.center_mass.tolist(),
            "bounds_mm": mesh.bounds.tolist(),
            "connectivity_scope": "Exact saved vertex equality; no coordinate repair or tolerance welding.",
        },
    )


def native_planes(path: Path, expected_bodies: int) -> list[PrintPlane]:
    """Native imports occur only in the admitted isolated worker."""
    from build123d import import_step
    from OCP.BRep import BRep_Tool

    shape = import_step(str(path))
    if not shape.is_valid or len(shape.solids()) != expected_bodies:
        raise ValueError("Saved STEP is invalid or its made body count changed")
    planes = []
    for index, face in enumerate(shape.faces()):
        if "PLANE" not in str(face.geom_type):
            continue
        normal = np.asarray(list(face.normal_at()), dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
            raise ValueError("Native planar face has no finite normal")
        normal /= np.linalg.norm(normal)
        planes.append(
            PrintPlane(
                face_index=index,
                point_mm=[float(v) for v in face.center()],
                outward_normal=normal.tolist(),
                area_mm2=float(face.area),
                kernel_tolerance_mm=float(BRep_Tool.Tolerance_s(face.wrapped)),
            )
        )
    return planes


def validate_transform(transform: Any) -> Any:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Print transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=RIGID_GUARD)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=RIGID_GUARD)
        or abs(float(np.linalg.det(rotation)) - 1) > RIGID_GUARD
    ):
        raise ValueError("Print transform must be proper rigid: no scale, shear or reflection")
    return matrix


def plate_transform(plane: PrintPlane) -> Any:
    normal = np.asarray(plane.outward_normal, dtype=float)
    if abs(float(np.linalg.norm(normal)) - 1) > RIGID_GUARD:
        raise ValueError("Measured planar normal must be unit length")
    target = np.array([0.0, 0.0, -1.0])
    dot = float(np.clip(normal @ target, -1, 1))
    cross = np.cross(normal, target)
    sine = float(np.linalg.norm(cross))
    if sine <= np.finfo(float).eps * 8:
        if dot > 0:
            rotation = np.eye(3)
        else:
            basis = np.eye(3)[int(np.argmin(np.abs(normal)))]
            axis = np.cross(normal, basis)
            axis /= np.linalg.norm(axis)
            rotation = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        axis = cross / sine
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + sine * skew + (1 - dot) * (skew @ skew)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -(rotation @ np.asarray(plane.point_mm, dtype=float))
    return validate_transform(matrix)


def transformed(triangles: Any, matrix: Any) -> Any:
    matrix = validate_transform(matrix)
    return triangles @ matrix[:3, :3].T + matrix[:3, 3]


def arithmetic_guard(triangles: Any, plane: PrintPlane) -> dict[str, float]:
    coordinates = np.asarray(triangles, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        floats = coordinates.astype(np.float32)
        spacing = np.abs(np.spacing(floats).astype(float))
    if not np.isfinite(floats).all() or not np.isfinite(spacing).all():
        raise ValueError("Source coordinates exceed finite binary-STL precision")
    scale = max(1.0, float(np.abs(coordinates).max()), max(abs(v) for v in plane.point_mm))
    roundoff = 128 * float(np.finfo(float).eps) * scale
    # Corroborating an analytic plane against saved float32 vertices needs an
    # explicit native/serialization uncertainty. This is not bed or fit clearance.
    coordinate_guard = float(spacing.max()) * math.sqrt(3) / 2
    return {
        "arithmetic_mm": roundoff,
        "source_float32_half_ulp_projection_mm": coordinate_guard,
        "kernel_plane_tolerance_mm": plane.kernel_tolerance_mm,
        "plane_correspondence_mm": roundoff + coordinate_guard + plane.kernel_tolerance_mm,
        "dimensionless_rigid_guard": RIGID_GUARD,
    }


def _contact(triangles: Any, centroid: Any, guard: float) -> dict[str, Any]:
    from scipy.spatial import ConvexHull, QhullError

    z = triangles[:, :, 2]
    minimum = float(z.min())
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    contact = (np.abs(z).max(axis=1) <= guard) & (cross[:, 2] < -RIGID_GUARD * lengths)
    area = float((lengths[contact] / 2).sum())
    inside: bool | None = None
    margin: float | None = None
    patches = 0
    if np.any(contact):
        contact_mesh = _mesh(triangles[contact])
        patches = len(contact_mesh.split(only_watertight=False, repair=False))
        xy = np.unique(triangles[contact, :, :2].reshape(-1, 2), axis=0)
        if len(xy) >= 3:
            try:
                hull = ConvexHull(xy)
                distances = hull.equations[:, :2] @ centroid[:2] + hull.equations[:, 2]
                margin = -float(distances.max())
                inside = True if margin > guard else False if margin < -guard else None
            except QhullError:
                inside = None
    return {
        "minimum_plate_z_mm": minimum,
        "below_plate_vertex_count": int(np.count_nonzero(z < -guard)),
        "checked_triangle_vertex_count": int(z.size),
        "contact_triangle_count": int(contact.sum()),
        "contact_area_mm2": area,
        "contact_patch_count": patches,
        "uniform_material_centroid_plate_mm": np.asarray(centroid, dtype=float).tolist(),
        "centroid_inside_contact_hull": inside,
        "centroid_hull_margin_mm": margin,
        "numeric_guard_mm": guard,
        "scope": "Complete delivered triangle inventory. Contact hull may span holes or disconnected patches; not adhesion or process success.",
    }


def measure_candidate(
    triangles: Any, bodies: list[Any], source: PrintSource, plane: PrintPlane
) -> PrintCandidate:
    matrix = plate_transform(plane)
    guards = arithmetic_guard(triangles, plane)
    guard = guards["plane_correspondence_mm"]
    posed = transformed(triangles, matrix)
    observations = []
    for index, body in enumerate(bodies):
        body_triangles = transformed(np.asarray(body.triangles), matrix)
        centroid = np.asarray(body.center_mass) @ matrix[:3, :3].T + matrix[:3, 3]
        observation = _contact(body_triangles, centroid, guard)
        observation["body_index"] = index
        observation["source_volume_mm3"] = float(body.volume)
        observations.append(observation)
    centroid = np.asarray(_mesh(triangles).center_mass) @ matrix[:3, :3].T + matrix[:3, 3]
    whole = _contact(posed, centroid, guard)
    ready = all(
        item["below_plate_vertex_count"] == 0
        and item["contact_area_mm2"] > 0
        and item["centroid_inside_contact_hull"] is True
        for item in observations
    )
    details = {
        "whole_mesh": whole,
        "bodies": observations,
        "numeric_guards": guards,
        "ordered_source_triangles_sha256": sha(np.asarray(triangles, dtype="<f8").tobytes()),
    }
    record = {
        "plane": plane.model_dump(),
        "transform": matrix.tolist(),
        "status": "measured" if ready else "unresolved",
        "reason": "Measured mesh placement; physical printing remains unverified."
        if ready
        else "Intrusion, missing planar contact, or unresolved/outside centroid support at this plane.",
        "observations": details,
    }
    return PrintCandidate.model_validate(
        {
            **record,
            "candidate_hash": digest(
                {"algorithm": ALGORITHM, "source": source.model_dump(), **record}
            ),
        }
    )


def inspect_mesh(
    triangles: Any, source: PrintSource, planes: list[PrintPlane]
) -> PrintPartInspection:
    _, bodies, inventory = mesh_inventory(triangles, source.body_count)
    candidates = [measure_candidate(triangles, bodies, source, plane) for plane in planes]
    measured = [candidate for candidate in candidates if candidate.status == "measured"]
    measured.sort(
        key=lambda candidate: (
            -candidate.observations["whole_mesh"]["contact_area_mm2"],
            candidate.plane.face_index,
        )
    )
    selected = measured[0].candidate_hash if measured else None
    return PrintPartInspection(
        source=source,
        status="measured" if selected else "unresolved",
        candidates=candidates,
        recommended_candidate_hash=selected,
        mesh=inventory,
        reason="Recommended by declared geometric contact ranking; not physical optimality."
        if selected
        else "No measured planar candidate has complete mesh placement and centroid-hull support.",
    )


def binary_stl(triangles: Any) -> bytes:
    if len(triangles) > 0xFFFFFFFF:
        raise ValueError("Triangle inventory exceeds the binary-STL format")
    records = np.zeros(len(triangles), dtype=_STL_DTYPE)
    with np.errstate(over="ignore", invalid="ignore"):
        records["vertices"] = triangles
    if not np.isfinite(records["vertices"]).all():
        raise ValueError("Print pose exceeds finite binary-STL coordinate range")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("Rigid print pose contains collapsed triangles")
    records["normal"] = normals / lengths[:, None]
    header = b"Forge-Mind rigid plate pose; units mm; physical readiness unverified".ljust(80, b" ")
    return header + struct.pack("<I", len(triangles)) + records.tobytes()


def oriented_stl(
    triangles: Any, source: PrintSource, candidate: PrintCandidate
) -> tuple[bytes, dict[str, Any]]:
    if candidate.status != "measured":
        raise ValueError("Cannot export an unresolved print candidate")
    record = candidate.model_dump(
        exclude={"candidate_hash", "source_frame", "target_frame", "units"}
    )
    if candidate.candidate_hash != digest(
        {"algorithm": ALGORITHM, "source": source.model_dump(), **record}
    ):
        raise ValueError("Print candidate hash does not match its exact measured inputs")
    if sha(np.asarray(triangles, dtype="<f8").tobytes()) != candidate.observations.get(
        "ordered_source_triangles_sha256"
    ):
        raise ValueError(
            "Print candidate does not bind this complete ordered source triangle inventory"
        )
    matrix = validate_transform(candidate.transform)
    posed = transformed(triangles, matrix)
    data = binary_stl(posed)
    delivered = stl_triangles(data)
    source_mesh, _, before = mesh_inventory(triangles, source.body_count)
    after_mesh, after_bodies, after = mesh_inventory(delivered, source.body_count)
    rounding = float(np.abs(delivered - posed).max())
    ulp = float(np.abs(np.spacing(posed.astype(np.float32)).astype(float)).max()) / 2
    arithmetic = arithmetic_guard(triangles, candidate.plane)["arithmetic_mm"]
    if rounding > ulp + arithmetic:
        raise ValueError("Serialized print triangles exceed their float32 rounding budget")
    inverse = (delivered - matrix[:3, 3]) @ matrix[:3, :3]
    inverse_error = float(np.abs(inverse - triangles).max())
    inverse_guard = math.sqrt(3) * rounding + arithmetic
    if inverse_error > inverse_guard:
        raise ValueError(
            "Print mesh inverse triangle inventory differs beyond arithmetic uncertainty"
        )
    volume_difference = abs(float(after_mesh.volume) - float(source_mesh.volume))
    if volume_difference > max(0.001, abs(float(source_mesh.volume)) * 0.001):
        raise ValueError(
            "Print mesh serialized volume differs from its source export consistency budget"
        )
    before_shells = _shell_volumes(source_mesh)
    after_shells = _shell_volumes(after_mesh)
    if len(before_shells) != len(after_shells) or any(
        abs(a - b) > max(0.001, abs(a) * 0.001)
        for a, b in zip(before_shells, after_shells, strict=True)
    ):
        raise ValueError(
            "A serialized print shell changed volume beyond its export consistency budget"
        )
    guard = (
        candidate.observations["numeric_guards"]["plane_correspondence_mm"]
        + math.sqrt(3) * rounding
    )
    contacts = [
        _contact(np.asarray(body.triangles), body.center_mass, guard) for body in after_bodies
    ]
    if any(
        item["below_plate_vertex_count"] != 0
        or item["contact_area_mm2"] <= 0
        or item["centroid_inside_contact_hull"] is not True
        for item in contacts
    ):
        raise ValueError("Serialized print mesh no longer satisfies measured plate placement")
    return data, {
        "sha256": sha(data),
        "bytes": len(data),
        "candidate_hash": candidate.candidate_hash,
        "source_stl_sha256": source.stl_sha256,
        "part_id": source.part_id,
        "quantity": source.quantity,
        "frame": "plate_z_up",
        "units": "mm",
        "transform": candidate.transform,
        "triangle_inventory_preserved": len(delivered) == len(triangles),
        "inverse_maximum_coordinate_error_mm": inverse_error,
        "inverse_numeric_guard_mm": inverse_guard,
        "float32_maximum_rounding_mm": rounding,
        "float32_half_ulp_mm": ulp,
        "source_mesh": before,
        "derived_mesh": after,
        "volume_difference_mm3": volume_difference,
        "source_shell_volumes_mm3": before_shells,
        "derived_shell_volumes_mm3": after_shells,
        "delivered_body_plate_observations": contacts,
        "physical_readiness": "unverified",
    }


def _shell_volumes(mesh: Any) -> list[float]:
    import trimesh

    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    ordered = sorted(components, key=lambda indices: int(min(indices)))
    return [float(_mesh(np.asarray(mesh.triangles)[indices]).volume) for indices in ordered]


def run_print_setup(directory: Path, request: PrintSetupRequest) -> dict[str, Any]:
    """Only owned staged bytes are parsed; saved revision files are never written."""
    document = _read(directory / "document.json")
    manifest = _read(directory / MANIFEST_NAME)
    parts = {}
    triangles_by_part = {}
    for part_id in request.part_ids:
        step_name, stl_name = f"definition_{part_id}.step", f"definition_{part_id}.stl"
        step, stl = _read(directory / step_name), _read(directory / stl_name)
        entry = validate_definition_mesh(
            document, manifest, request.revision_id, part_id, request.artifact_hashes, step, stl
        )
        source = PrintSource(
            revision_id=request.revision_id,
            part_id=part_id,
            quantity=entry["quantity"],
            document_sha256=sha(document),
            manifest_sha256=sha(manifest),
            step_file=step_name,
            step_sha256=sha(step),
            stl_file=stl_name,
            stl_sha256=sha(stl),
            stl_bytes=len(stl),
            body_count=entry["body_count"],
        )
        triangles = stl_triangles(stl)
        parts[part_id] = inspect_mesh(
            triangles, source, native_planes(directory / step_name, source.body_count)
        )
        triangles_by_part[part_id] = triangles
    status: Literal["measured", "prepared", "unresolved"] = (
        "measured" if all(part.status == "measured" for part in parts.values()) else "unresolved"
    )
    derived = {}
    files = {}
    if request.operation == "prepare" and status == "measured":
        # Compute and validate every derivative before writing any output file.
        for part_id, part in parts.items():
            candidate = next(
                c for c in part.candidates if c.candidate_hash == part.recommended_candidate_hash
            )
            data, receipt = oriented_stl(triangles_by_part[part_id], part.source, candidate)
            name = f"print_{part_id}.stl"
            derived[name] = data
            files[name] = receipt
        for name, data in derived.items():
            with (directory / name).open("xb") as output:
                output.write(data)
        status = "prepared"
    result = PrintSetupResult(
        revision_id=request.revision_id,
        operation=request.operation,
        status=status,
        parts=parts,
        files=files,
        ranking=RANKING,
        scope=SCOPE,
    )
    return result.model_dump(mode="json")


def main(request_path: Path) -> None:
    request = PrintSetupRequest.model_validate(json.loads(_read(request_path)))
    result = run_print_setup(request_path.parent, request)
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(data) > MAX_JOB_BYTES:
        raise ValueError("Print setup report exceeds its byte budget")
    with (request_path.parent / "report.json").open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
