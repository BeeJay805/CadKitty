"""Explicit private CAD-aware meshing; no healing, fallback, or solver-limit changes."""

from __future__ import annotations

import hashlib
import importlib
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from forge_mind.workspace.capacity import enforce_limit, optional_limit

_GMSH_LOCK = threading.Lock()
_MESH_RELATIVE_TOLERANCE = 0.005
_DEVIATION_DIAGONAL_FRACTION = 0.001


def _mapped_elements(
    tags: Any, coordinates: Any, element_types: Any, nodes: Any
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Gmsh tags are identifiers, never array offsets. Accept only linear tetrahedra."""
    tags = np.asarray(tags)
    points = np.asarray(coordinates, dtype=float)
    if (
        tags.ndim != 1
        or len(tags) < 4
        or tags.dtype.kind not in "iu"
        or len(np.unique(tags)) != len(tags)
        or np.any(tags <= 0)
        or points.size != 3 * len(tags)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Invalid or duplicate Gmsh node tags/coordinates")
    if list(element_types) != [4] or len(nodes) != 1:
        raise ValueError("CAD mesher must return only first-order four-node tetrahedra")
    raw = np.asarray(nodes[0])
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size % 4:
        raise ValueError("Malformed Gmsh tetrahedral connectivity")
    if raw.size < 4:
        raise ValueError("CAD mesh must contain at least one tetrahedron")
    enforce_limit(raw.size // 4, optional_limit("STRUCTURAL_TET_LIMIT"), "CAD tetrahedra")
    lookup = {int(tag): i for i, tag in enumerate(tags)}
    try:
        cells = np.array([lookup[int(tag)] for tag in raw], dtype=np.int64).reshape(-1, 4)
    except KeyError as exc:
        raise ValueError("Tetrahedron references an unknown Gmsh node") from exc
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("Tetrahedron repeats a corner node")
    if len(np.unique(cells)) != len(tags):
        raise ValueError("CAD mesh contains nodes outside its volume mesh")
    return points.reshape(-1, 3), cells


def _mesh_facts(
    points: NDArray[np.float64], cells: NDArray[np.int64]
) -> tuple[dict[str, Any], set[tuple[int, ...]]]:
    """Check budgets and connectivity before allocating any FEM basis or sparse matrix."""
    enforce_limit(len(cells), optional_limit("STRUCTURAL_TET_LIMIT"), "CAD tetrahedra")
    edges = np.concatenate(
        [cells[:, pair] for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    dofs = 3 * (len(points) + len(edges))
    enforce_limit(dofs, optional_limit("STRUCTURAL_DOF_LIMIT"), "CAD P2 DOFs")
    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    corners = points[cells]
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(volumes).all() or np.any(volumes <= max(1e-18, diagonal**3 * 1e-15)):
        raise ValueError("CAD mesh contains degenerate tetrahedra")
    faces = np.concatenate(
        [cells[:, triple] for triple in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    )
    unique, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    if counts.max() > 2:
        raise ValueError("CAD mesh has non-manifold tetrahedral faces")
    # Point/edge contact alone is not a connected three-dimensional material mesh.
    owners: dict[tuple[int, ...], int] = {}
    for row, face in enumerate(faces):
        key = tuple(sorted(map(int, face)))
        element = row % len(cells)
        if key in owners:
            parent[find(element)] = find(owners[key])
        else:
            owners[key] = element
    if len({find(i) for i in range(len(cells))}) != 1:
        raise ValueError("CAD tetrahedral mesh is disconnected across volume faces")
    boundary = {tuple(map(int, face)) for face in unique[counts == 1]}
    return {
        "vertices": len(points),
        "tetrahedra": len(cells),
        "edges": len(edges),
        "p2_displacement_dofs": dofs,
        "mesh_volume_mm3": float(volumes.sum()),
        "minimum_tet_volume_mm3": float(volumes.min()),
        "boundary_triangles": len(boundary),
    }, boundary


def _face_samples(
    face: Any, points: NDArray[np.float64], triangles: NDArray[np.int64], tolerance: float
) -> dict[str, Any]:
    """Check the trimmed face, including its holes, at all triangle nodes/mids/centroids."""
    tri = points[triangles]
    samples = np.unique(
        np.concatenate(
            [
                tri.reshape(-1, 3),
                (tri[:, 0] + tri[:, 1]) / 2,
                (tri[:, 1] + tri[:, 2]) / 2,
                (tri[:, 2] + tri[:, 0]) / 2,
                tri.mean(axis=1),
            ]
        ),
        axis=0,
    )
    maximum = 0.0
    for sample in samples:
        distance = float(face.distance_to(tuple(sample)))
        if not math.isfinite(distance) or distance > tolerance:
            raise ValueError(
                f"Boundary triangle sample deviates from its trimmed CAD face: {distance:.9g} mm; limit {tolerance:.9g} mm"
            )
        maximum = max(maximum, distance)
    area = float(
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    )
    relative = abs(area / float(face.area) - 1)
    if relative > _MESH_RELATIVE_TOLERANCE:
        raise ValueError(
            f"Per-face mesh area differs from trimmed CAD area by {relative:.6g}; limit {_MESH_RELATIVE_TOLERANCE}"
        )
    return {
        "sample_count": len(samples),
        "maximum_sampled_deviation_mm": maximum,
        "mesh_area_mm2": area,
        "cad_area_mm2": float(face.area),
        "relative_area_error": relative,
    }


def _boundary_receipts(
    gmsh: Any,
    solid: Any,
    points: NDArray[np.float64],
    tags: Any,
    expected_boundary: set[tuple[int, ...]],
    tolerance: float,
    diagonal: float,
) -> list[dict[str, Any]]:
    import build123d as b

    original_faces = list(solid.faces())
    entities = gmsh.model.getEntities(2)
    if len(entities) != len(original_faces):
        raise ValueError(
            "Imported CAD face count differs from selected source; no healing is permitted"
        )
    lookup = {int(tag): i for i, tag in enumerate(tags)}
    matched: set[int] = set()
    covered: set[tuple[int, ...]] = set()
    receipts = []
    comparison_tolerance = max(diagonal * 1e-7, 1e-6)
    for _, tag in entities:
        center = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
        area = float(gmsh.model.occ.getMass(2, tag))
        bounds = np.asarray(gmsh.model.getBoundingBox(2, tag))
        matches = []
        for index, face in enumerate(original_faces):
            bbox = face.bounding_box()
            if (
                abs(area / float(face.area) - 1) <= 1e-7
                and np.linalg.norm(center - tuple(face.center(b.CenterOf.MASS)))
                <= comparison_tolerance
                and np.allclose(bounds, [*bbox.min, *bbox.max], rtol=0, atol=comparison_tolerance)
            ):
                matches.append(index)
        if len(matches) != 1 or matches[0] in matched:
            raise ValueError(
                "Imported CAD face cannot be matched uniquely to the original trimmed face"
            )
        index = matches[0]
        matched.add(index)
        types, _, connectivity = gmsh.model.mesh.getElements(2, tag)
        if list(types) != [2] or len(connectivity) != 1 or not len(connectivity[0]):
            raise ValueError("Every CAD face must retain a nonempty first-order triangle mesh")
        try:
            triangles = np.array(
                [lookup[int(node)] for node in connectivity[0]], dtype=np.int64
            ).reshape(-1, 3)
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid CAD face mesh connectivity") from exc
        inventory = [tuple(map(int, row)) for row in np.sort(triangles, axis=1)]
        if len(set(inventory)) != len(inventory) or covered.intersection(inventory):
            raise ValueError("Duplicate CAD boundary triangles")
        covered.update(inventory)
        face = original_faces[index]
        checks = _face_samples(face, points, triangles, tolerance)
        receipts.append(
            {
                "source_face_index": index,
                "gmsh_face_tag": int(tag),
                "source_surface_type": str(face.geom_type),
                "source_center_of_mass_mm": list(face.center(b.CenterOf.MASS)),
                "triangle_count": len(triangles),
                **checks,
            }
        )
    if covered != expected_boundary:
        raise ValueError("CAD face coverage does not equal the tetrahedral exterior")
    return receipts


def mesh_cad_solid(
    solid: Any, mesh_factor: float, directory: Path
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, Any]]:
    """Mesh one exact selected solid, in world coordinates, inside the bounded worker.

    The caller supplies a fresh directory for each coarse/fine mesh. Geometry checks
    are sampled approximation checks, not a certificate of global geometric error.
    """
    import build123d as b

    if not math.isfinite(mesh_factor) or not 0.03 <= mesh_factor <= 0.3:
        raise ValueError("CAD mesh factor must be from 0.03 to 0.3")
    if not solid.is_valid or len(solid.solids()) != 1 or not float(solid.volume) > 0:
        raise ValueError("CAD mesher requires one valid positive-volume selected solid")
    try:
        gmsh = importlib.import_module("gmsh")
    except (ImportError, OSError) as exc:
        raise RuntimeError("Optional CAD mesher dependency gmsh is unavailable") from exc
    directory.mkdir(parents=True, exist_ok=True)
    selected = directory / "selected.step"
    if selected.exists() or selected.is_symlink():
        raise ValueError("CAD mesher requires a fresh worker directory")
    b.export_step(solid, selected)
    source_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    bounds = solid.bounding_box()
    diagonal = float(np.linalg.norm(tuple(bounds.size)))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Selected CAD extent must be finite and positive")
    tolerance = max(diagonal * _DEVIATION_DIAGONAL_FRACTION, 1e-6)
    options = {
        "General.Terminal": 0,
        "General.NumThreads": 2,
        "Mesh.MeshSizeMax": diagonal * mesh_factor,
        "Mesh.MeshSizeFromCurvature": 32,
        "Mesh.MinimumCirclePoints": 48,
        "Mesh.MeshSizeExtendFromBoundary": 0,
        "Mesh.ElementOrder": 1,
        "Mesh.Algorithm3D": 1,
    }
    start = time.monotonic()
    with _GMSH_LOCK:
        if gmsh.isInitialized():
            raise RuntimeError("CAD mesher cannot share an initialized Gmsh session")
        try:
            gmsh.initialize([], readConfigFiles=False)
            for key, value in options.items():
                gmsh.option.setNumber(key, value)
            gmsh.model.occ.importShapes(str(selected), highestDimOnly=True)
            gmsh.model.occ.synchronize()
            volumes = gmsh.model.getEntities(3)
            if len(volumes) != 1:
                raise ValueError("STEP import did not preserve exactly one CAD volume")
            tag = volumes[0][1]
            imported_volume = float(gmsh.model.occ.getMass(3, tag))
            imported_bounds = gmsh.model.getBoundingBox(3, tag)
            if abs(imported_volume / float(solid.volume) - 1) > 1e-7 or not np.allclose(
                imported_bounds, [*bounds.min, *bounds.max], rtol=0, atol=max(diagonal * 1e-7, 1e-6)
            ):
                raise ValueError("Imported CAD volume/bounds differ from selected source")
            gmsh.model.mesh.generate(3)
            mesh_seconds = time.monotonic() - start
            tags, coordinates, _ = gmsh.model.mesh.getNodes()
            types, _, connectivity = gmsh.model.mesh.getElements(3, tag)
            points, cells = _mapped_elements(tags, coordinates, types, connectivity)
            facts, exterior = _mesh_facts(points, cells)
            volume_error = abs(facts["mesh_volume_mm3"] / float(solid.volume) - 1)
            if volume_error > _MESH_RELATIVE_TOLERANCE:
                raise ValueError(
                    f"Tetrahedral volume differs from CAD by {volume_error:.6g}; limit {_MESH_RELATIVE_TOLERANCE}"
                )
            face_checks = _boundary_receipts(
                gmsh, solid, points, tags, exterior, tolerance, diagonal
            )
            receipt = {
                "name": "Gmsh",
                "kind": "gmsh_occ",
                "mesher": "gmsh_occ",
                "version": gmsh.__version__,
                "options": options,
                "mesh_factor": mesh_factor,
                "selected_step_sha256": source_hash,
                "selected_step": selected.name,
                "coordinate_frame": "world_mm",
                "cad_volume_mm3": float(solid.volume),
                "imported_volume_mm3": imported_volume,
                "relative_volume_error": volume_error,
                "relative_volume_tolerance": _MESH_RELATIVE_TOLERANCE,
                "sampled_deviation_tolerance_mm": tolerance,
                "deviation_diagonal_fraction": _DEVIATION_DIAGONAL_FRACTION,
                "mesh_seconds": mesh_seconds,
                "total_seconds": time.monotonic() - start,
                "faces": face_checks,
                **facts,
                "scope": "Exact STEP import, first-order tetrahedral geometry with sampled trimmed-face approximation checks. All boundary vertices, edge midpoints and triangle centroids checked; no continuous Hausdorff error bound or physical qualification. No healing or fallback.",
            }
            return points, cells, receipt
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"CAD mesher failed: {exc}") from exc
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()
