"""Assembly tessellation identity without welding contacts between distinct solids."""

from __future__ import annotations

import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.workspace.tessellation import STL_ANGULAR_TOLERANCE, STL_LINEAR_TOLERANCE


def triangle_inventory(triangles: Any) -> Counter[bytes]:
    """Exact oriented float32 multiset, canonicalized in bounded vector batches.

    Only cyclic rotation is permitted: reversing winding, deleting a triangle,
    changing a coordinate or changing multiplicity must change the inventory.
    Copies keep signed-zero normalization from mutating the caller's mesh.
    """
    values = np.asarray(triangles, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("Expected tessellation triangles with shape (n, 3, 3)")
    inventory: Counter[bytes] = Counter()
    for offset in range(0, len(values), 32768):
        batch = values[offset : offset + 32768].copy()
        if not np.isfinite(batch).all():
            raise ValueError("Nonfinite tessellation coordinates")
        batch[batch == 0] = 0  # normalize negative zero in STL representation
        collapsed = (
            np.all(batch[:, 0] == batch[:, 1], axis=1)
            | np.all(batch[:, 1] == batch[:, 2], axis=1)
            | np.all(batch[:, 2] == batch[:, 0], axis=1)
        )
        batch = batch[~collapsed]
        rows = np.arange(len(batch))
        start = np.zeros(len(batch), dtype=np.intp)
        for vertex in (1, 2):
            current, candidate = batch[rows, start], batch[:, vertex]
            smaller = (candidate[:, 0] < current[:, 0]) | (
                (candidate[:, 0] == current[:, 0])
                & (
                    (candidate[:, 1] < current[:, 1])
                    | ((candidate[:, 1] == current[:, 1]) & (candidate[:, 2] < current[:, 2]))
                )
            )
            start[smaller] = vertex
        canonical = batch[rows[:, None], (start[:, None] + np.arange(3)) % 3]
        inventory.update(triangle.tobytes() for triangle in canonical)
    return inventory


def assembly_mesh_evidence(
    part: Any,
    mesh: Any,
    settings: tuple[float, float] = (STL_LINEAR_TOLERANCE, STL_ANGULAR_TOLERANCE),
) -> dict[str, Any]:
    from build123d import export_stl

    expected: Counter[bytes] = Counter()
    bodies = []
    with tempfile.TemporaryDirectory(prefix="cad-mesh-") as temporary:
        for index, solid in enumerate(part.solids()):
            path = Path(temporary) / "body.stl"
            export_stl(solid, path, tolerance=settings[0], angular_tolerance=settings[1])
            body: Any = trimesh.load(path, force="mesh")
            raw: Any = trimesh.load(path, force="mesh", process=False)
            expected.update(triangle_inventory(raw.triangles))
            # Match trusted export handling of collapsed OCCT pole triangles;
            # never remove a triangle with three distinct vertices or fill holes.
            faces = body.faces
            body.update_faces(
                (faces[:, 0] != faces[:, 1])
                & (faces[:, 0] != faces[:, 2])
                & (faces[:, 1] != faces[:, 2])
            )
            body.remove_unreferenced_vertices()
            bbox = solid.bounding_box()
            bounds = [[bbox.min.X, bbox.min.Y, bbox.min.Z], [bbox.max.X, bbox.max.Y, bbox.max.Z]]
            valid = bool(
                solid.is_valid
                and solid.volume > 0
                and body.is_watertight
                and body.is_winding_consistent
                and body.volume > 0
                and math.isclose(body.volume, solid.volume, rel_tol=0.001, abs_tol=0.001)
            )
            bodies.append(
                {
                    "body_index": index,
                    "valid": valid,
                    "watertight": bool(body.is_watertight),
                    "step_bounds_within_0_01_mm": bool(
                        np.allclose(body.bounds, bounds, rtol=0, atol=0.01)
                    ),
                    "maximum_sampled_bounds_difference_mm": float(
                        np.max(np.abs(body.bounds - bounds))
                    ),
                }
            )
    matches = expected == triangle_inventory(mesh.triangles)
    return {
        "valid": bool(bodies) and all(b["valid"] for b in bodies) and matches,
        "triangle_inventory_matches": matches,
        "bodies": bodies,
        "scope": "Each exported STEP solid and its independently generated tessellation; oriented float32 triangle multiset matches aggregate STL, including multiplicity. Curved mesh extrema are sampled approximations, not manufacturing tolerances. Does not establish clearance or printability of a welded aggregate.",
    }


def reference_tessellation_matches(
    part: Any,
    mesh: Any,
    settings: tuple[float, float] = (STL_LINEAR_TOLERANCE, STL_ANGULAR_TOLERANCE),
) -> dict[str, Any]:
    """Prove sampled mesh identity without inflating an exact-CAD dimensional tolerance."""
    from build123d import export_stl

    with tempfile.TemporaryDirectory(prefix="cad-reference-") as temporary:
        path = Path(temporary) / "reference.stl"
        export_stl(part, path, tolerance=settings[0], angular_tolerance=settings[1])
        reference: Any = trimesh.load(path, force="mesh", process=False)
        matches = triangle_inventory(reference.triangles) == triangle_inventory(mesh.triangles)
    return {
        "triangle_inventory_matches": matches,
        "linear_deflection_setting": settings[0],
        "linear_deflection_is_relative": True,
        "angular_deflection_rad": settings[1],
        "scope": "Independent tessellation of the final STEP with identical exporter settings; exact oriented float32 triangle multiset including multiplicity. Sampled curved bounds are not a dimensional or manufacturing tolerance.",
    }
