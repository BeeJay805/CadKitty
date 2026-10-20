"""Revision-local picking and conservative exact serialized B-rep correspondence."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from io import BytesIO
from typing import Any

from build123d import Edge, Face, Solid, export_brep
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools

from forge_mind.workspace.capacity import enforce_limit, optional_limit

# Optional deployment policies, independent of kernel validity and worker watchdogs.
FEATURE_FACE_LIMIT = optional_limit("FEATURE_FACE_LIMIT")
FEATURE_EDGE_LIMIT = optional_limit("FEATURE_EDGE_LIMIT")
ASSEMBLY_FACE_LIMIT = optional_limit("ASSEMBLY_FACE_LIMIT")
ASSEMBLY_EDGE_LIMIT = optional_limit("ASSEMBLY_EDGE_LIMIT")
PATTERN_COUNT_LIMIT = optional_limit("PATTERN_COUNT_LIMIT")
PATTERN_INSTANCE_LIMIT = optional_limit("PATTERN_INSTANCE_LIMIT")


def same_exported_solid(a: Any, other: Any) -> bool:
    """Same imported topology and rigid placement, including nested location chains."""
    if (
        not a.wrapped.IsPartner(other.wrapped)
        or a.wrapped.Orientation() != other.wrapped.Orientation()
    ):
        return False
    left, right = a.wrapped.Location().Transformation(), other.wrapped.Location().Transformation()
    return all(
        abs(left.Value(i, j) - right.Value(i, j)) <= 1e-10 for i in range(1, 4) for j in range(1, 5)
    )


def signature(shape: Any) -> str:
    """Complete supported analytic boundaries, otherwise strict kernel bytes.

    Area/centroid/bounds alone NEVER authorize correspondence. Numeric canonicalization
    is 1e-8 mm, below the kernel modeling tolerance, not a user acceptance allowance.
    """
    canonical = analytic_record(shape)
    if canonical is not None:
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    stream = BytesIO()
    if not export_brep(shape, stream):
        raise ValueError("Kernel could not serialize selected geometry")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _point(values: Any) -> list[float]:
    return [round(float(v), 8) + 0.0 for v in values]


def _axis(values: Any) -> list[float]:
    point = _point(values)
    for value in point:
        if abs(value) > 1e-8:
            return point if value > 0 else [-v + 0.0 for v in point]
    raise ValueError("Degenerate analytic axis")


def analytic_record(shape: Any) -> Any:
    if isinstance(shape, Edge):
        kind = str(shape.geom_type)
        if "LINE" in kind:
            return ["line", sorted([_point(shape.position_at(0)), _point(shape.position_at(1))])]
        if "CIRCLE" in kind:
            circle = BRepAdaptor_Curve(shape.wrapped).Circle()
            center, direction = circle.Location(), circle.Axis().Direction()
            radius = circle.Radius()
            record = [
                "circle",
                _point((center.X(), center.Y(), center.Z())),
                _axis((direction.X(), direction.Y(), direction.Z())),
                round(radius, 8),
            ]
            if abs(shape.length - 2 * math.pi * radius) > 1e-7:
                record += [
                    sorted([_point(shape.position_at(0)), _point(shape.position_at(1))]),
                    _point(shape.position_at(0.5)),
                    round(shape.length, 8),
                ]
            return record
        return None
    if isinstance(shape, Face):
        surface = BRepAdaptor_Surface(shape.wrapped)
        kind = str(shape.geom_type)
        loops = []
        for wire in shape.wires():
            edges = [analytic_record(edge) for edge in wire.edges()]
            if any(edge is None for edge in edges):
                return None
            loops.append(sorted(edges, key=lambda e: json.dumps(e, sort_keys=True)))
        if "PLANE" in kind:
            normal = _point(shape.normal_at())
            center = list(shape.center())
            # Plane equation plus the COMPLETE bounded wires determines planar trim.
            return [
                "plane",
                normal,
                round(sum(a * b for a, b in zip(normal, center, strict=True)), 8),
                sorted(loops, key=lambda e: json.dumps(e, sort_keys=True)),
            ]
        if "CYLINDER" in kind:
            cyl = surface.Cylinder()
            radius = cyl.Radius()
            if abs(surface.LastUParameter() - surface.FirstUParameter() - 2 * math.pi) > 1e-7:
                return None
            span = surface.LastVParameter() - surface.FirstVParameter()
            if abs(shape.area - 2 * math.pi * radius * span) > 1e-7 * max(1, shape.area):
                return None
            direction, origin = cyl.Axis().Direction(), cyl.Location()
            axis = _axis((direction.X(), direction.Y(), direction.Z()))
            point = [origin.X(), origin.Y(), origin.Z()]
            along = sum(a * b for a, b in zip(axis, point, strict=True))
            foot = _point([p - a * along for p, a in zip(point, axis, strict=True)])
            # Seam lines are parameterization artifacts; full cylinder trim is its circles.
            circles = sorted(
                [analytic_record(e) for e in shape.edges() if "CIRCLE" in str(e.geom_type)],
                key=lambda e: json.dumps(e, sort_keys=True),
            )
            if len(circles) != 2:
                return None
            center = list(shape.center())
            sample = list(shape.position_at(0.37, 0.5))
            outward = (
                sum(
                    (p - c) * n
                    for p, c, n in zip(sample, center, shape.normal_at(sample), strict=True)
                )
                > 0
            )
            return ["full_cylinder", axis, foot, round(radius, 8), circles, outward]
        return None
    if isinstance(shape, Solid):
        faces = [analytic_record(face) for face in shape.faces()]
        if any(face is None for face in faces):
            return None
        return ["solid", sorted(faces, key=lambda e: json.dumps(e, sort_keys=True))]
    return None


def resolve(reference: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    matches = [
        e
        for e in topology.get("entities", [])
        if e["kind"] == reference["kind"]
        and e["signature"] == reference["signature"]
        and e["body_signature"] == reference.get("body_signature")
    ]
    return {
        "status": "resolved" if len(matches) == 1 else "ambiguous" if matches else "unresolved",
        "entity": matches[0] if len(matches) == 1 else None,
    }


def selected_shapes(shape: Any, kind: str, signatures: list[str]) -> list[Any]:
    if not signatures:
        raise ValueError(f"Select explicit {kind} geometry at the input feature stage")
    items = list(shape.faces() if kind == "face" else shape.edges())
    mapping: dict[str, list[Any]] = {}
    for item in items:
        mapping.setdefault(signature(item), []).append(item)
    result = []
    for value in signatures:
        matches = mapping.get(value, [])
        if len(matches) != 1:
            raise ValueError(
                "Selection is missing or ambiguous at this feature's input stage; select it again"
            )
        result.append(matches[0])
    return result


def topology_index(shape: Any, *, assembly: bool = False) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    solids = list(shape.solids())
    face_count, edge_count = len(shape.faces()), len(shape.edges())
    detailed = face_count > (4500 if assembly else 1200) or edge_count > (
        14000 if assembly else 4000
    )
    body_picking = assembly or detailed
    enforce_limit(len(solids), optional_limit("BODY_LIMIT"), "indexed bodies")
    enforce_limit(face_count, ASSEMBLY_FACE_LIMIT if assembly else FEATURE_FACE_LIMIT, "faces")
    enforce_limit(edge_count, ASSEMBLY_EDGE_LIMIT if assembly else FEATURE_EDGE_LIMIT, "edges")
    triangle_limit = optional_limit("PICK_TRIANGLE_LIMIT")
    total_triangles = 0
    # Index all geometry before meshing: serialization must not contain a new triangulation.
    faces_to_mesh: list[tuple[Any, dict[str, Any]]] = []
    for body_index, solid in enumerate(solids):
        body_sig = signature(solid)
        objects_by_kind = [("body", [solid])]
        if not detailed:
            objects_by_kind.extend([("face", solid.faces()), ("edge", solid.edges())])
        for kind, objects in objects_by_kind:
            for obj in objects:
                sig = body_sig if kind == "body" else signature(obj)
                key = kind + "_" + sig[:20]
                ordinal = counts.get(key, 0)
                counts[key] = ordinal + 1
                bounds = obj.bounding_box()
                center = obj.center()
                record: dict[str, Any] = {
                    "id": f"{key}_{ordinal}",
                    "kind": kind,
                    "signature": sig,
                    "body_index": body_index,
                    "body_signature": body_sig,
                    "center": list(center),
                    "min": list(bounds.min),
                    "max": list(bounds.max),
                    "measure": float(
                        obj.volume if kind == "body" else obj.area if kind == "face" else obj.length
                    ),
                    "unit": "mm^3" if kind == "body" else "mm^2" if kind == "face" else "mm",
                }
                if kind == "face":
                    record["surface"] = str(obj.geom_type)
                    try:
                        record["normal"] = list(obj.normal_at())
                    except Exception:
                        record["normal"] = None
                    if not body_picking:
                        faces_to_mesh.append((obj, record))
                elif kind == "body" and body_picking:
                    faces_to_mesh.append((obj, record))
                elif kind == "edge":
                    record["curve"] = str(obj.geom_type)
                    record["polyline"] = [list(obj.position_at(i / 32)) for i in range(33)]
                entities.append(record)
    mesh_profiles = [(0.2, 0.3), (0.5, 0.5), (1.0, 0.8)] if detailed else [(0.05, 0.15)]
    for linear_tolerance, angular_tolerance in mesh_profiles:
        meshes = []
        total_triangles = 0
        for face, record in faces_to_mesh:
            mesh_shape = deepcopy(face) if detailed else face
            if detailed:
                # OCCT otherwise reuses finer cached triangles after an STL export.
                # Clean only a copy: exact shape and canonical references stay intact.
                BRepTools.Clean_s(mesh_shape.wrapped)
            vertices, triangles = mesh_shape.tessellate(linear_tolerance, angular_tolerance)
            total_triangles += len(triangles)
            if triangle_limit is not None and total_triangles > triangle_limit:
                break
            meshes.append(
                {
                    "id": record["id"],
                    "body_index": record["body_index"],
                    "vertices": [list(v) for v in vertices],
                    "triangles": triangles,
                }
            )
        if triangle_limit is None or total_triangles <= triangle_limit:
            break
    if triangle_limit is not None and total_triangles > triangle_limit:
        enforce_limit(total_triangles, triangle_limit, "pick tessellation triangles")
    return {
        "version": 1,
        "frame": "cad_z_up_mm",
        "selection_scope": "body" if body_picking else "face_edge_body",
        "pick_tessellation": {
            "relative_linear_deflection": linear_tolerance,
            "angular_deflection_radians": angular_tolerance,
            "triangles": total_triangles,
            "triangle_limit": triangle_limit,
            "purpose": "Selection preview only; not manufacturing geometry",
        },
        "topology_counts": {"bodies": len(solids), "faces": face_count, "edges": edge_count},
        "detail_policy": {
            "mode": "body_index" if detailed else "full_index",
            "reason": "Detailed geometry: face/edge pick records and polylines deferred; exact STEP and body signatures preserved."
            if detailed
            else None,
            "face_edge_picking_available": not body_picking,
            "exact_geometry_preserved": True,
        },
        "entities": entities,
        "meshes": meshes,
        "reference_method": "Complete canonical planar/line/circular/full-cylinder boundaries at 1e-8 mm; strict B-rep bytes for unsupported geometry. Unique across bodies, no nearest-match fallback. CAD Z-up millimeters.",
    }
