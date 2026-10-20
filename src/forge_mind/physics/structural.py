"""Bounded, explicit component linear statics inside the workspace's isolated worker.

All geometry, vectors and tensors here use the caller's analysis frame (mm, N, MPa).
Quadrature extrema are sampled values, never bounds on the continuous stress field.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
import warnings
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from forge_mind.physics.fea import _FEA_LOCK, _boundary_facets, _facet_area
from forge_mind.workspace.capacity import enforce_limit, optional_limit


def _mesh_resolution(points: Any, cells: Any) -> dict[str, Any]:
    """Record actual geometry, independent of node IDs, element order and winding."""
    corners = np.asarray(points, dtype=float)[np.asarray(cells)]
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    diameter_squared = np.max(
        [
            np.sum((corners[:, i] - corners[:, j]) ** 2, axis=1)
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        ],
        axis=0,
    )
    order = np.lexsort((corners[:, :, 2], corners[:, :, 1], corners[:, :, 0]), axis=1)
    records = np.take_along_axis(corners, order[:, :, None], axis=1).reshape(-1, 12)
    records = records[np.lexsort(tuple(records[:, i] for i in reversed(range(12))))]
    records[records == 0] = 0  # Normalize signed zero before canonical byte hashing.
    return {
        "volume_mesh_sha256": hashlib.sha256(records.astype("<f8").tobytes()).hexdigest(),
        "volume_weighted_rms_tet_diameter_mm": float(
            np.sqrt(np.average(diameter_squared, weights=volumes))
        ),
        "maximum_tet_diameter_mm": float(np.sqrt(diameter_squared.max())),
    }


def _von_mises(tensor: Any) -> Any:
    dev = tensor - np.eye(3)[:, :, None, None] * np.trace(tensor, axis1=0, axis2=1) / 3
    return np.sqrt(1.5 * np.sum(dev * dev, axis=(0, 1)))


def _solve_field(
    mesh: Any,
    *,
    force_n: float,
    direction: Any,
    e_mpa: float,
    nu: float,
    fixed_end: Literal["minimum", "maximum"],
    planes: tuple[float, float],
    tolerance: float,
    original_areas: tuple[float, float],
) -> dict[str, Any]:
    """Pure P2 solver entry for reproducible analytic tests; no native mesher."""
    from scipy.sparse.linalg import MatrixRankWarning
    from skfem import Basis, ElementTetP2, ElementVector, FacetBasis, LinearForm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    enforce_limit(mesh.t.shape[1], optional_limit("STRUCTURAL_TET_LIMIT"), "structural tetrahedra")
    # Count P2 DOFs before creating quadrature arrays or sparse stiffness entries.
    dofs = 3 * (mesh.p.shape[1] + mesh.edges.shape[1])
    enforce_limit(dofs, optional_limit("STRUCTURAL_DOF_LIMIT"), "structural displacement DOFs")
    points = np.asarray(mesh.p)
    corners = points[:, mesh.t]
    volumes = np.abs(np.linalg.det(np.moveaxis(corners[:, 1:] - corners[:, :1], 2, 0))) / 6
    if not np.isfinite(points).all() or not np.isfinite(volumes).all() or np.any(volumes <= 1e-15):
        raise ValueError("Non-finite or degenerate tetrahedral mesh")
    predicates = [lambda x, z=z: abs(x[2] - z) <= tolerance for z in planes]
    cap_facets = [_boundary_facets(mesh, pred) for pred in predicates]
    areas = [_facet_area(mesh, pred) for pred in predicates]
    area_error = [
        abs(area / original - 1) for area, original in zip(areas, original_areas, strict=True)
    ]
    if min(areas) <= 0 or max(area_error) > 0.005:
        raise ValueError(
            "Remeshed full planar cap area differs from declared boundary by over 0.5%"
        )
    fixed_index = 0 if fixed_end == "minimum" else 1
    fixed_facets, loaded_facets = cap_facets[fixed_index], cap_facets[1 - fixed_index]
    if np.intersect1d(fixed_facets, loaded_facets).size:
        raise ValueError("Fixed and loaded boundaries overlap")
    scalar = Basis(mesh, ElementTetP2(), intorder=5)
    element = ElementVector(ElementTetP2())
    # Positive-weight degree-five quadrature also makes display recovery convex.
    basis = Basis(mesh, element, intorder=5)
    lam, mu = lame_parameters(e_mpa, nu)
    stiffness = linear_elasticity(lam, mu).assemble(basis)
    fixed = basis.get_dofs(facets=fixed_facets).all()
    traction = np.asarray(direction) * force_n / areas[1 - fixed_index]

    @LinearForm  # type: ignore[untyped-decorator]
    def load(v: Any, w: Any) -> Any:
        return traction[0] * v[0] + traction[1] * v[1] + traction[2] * v[2]

    rhs = load.assemble(FacetBasis(mesh, element, facets=loaded_facets))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        displacement = solve(*condense(stiffness, rhs, D=fixed))
    if not np.isfinite(displacement).all():
        raise RuntimeError("Non-finite displacement from linear solver")
    residual = stiffness @ displacement - rhs
    free = np.setdiff1d(np.arange(basis.N), fixed)
    relative_residual = float(np.linalg.norm(residual[free])) / max(
        float(np.linalg.norm(rhs)), 1e-30
    )
    indices = np.asarray(basis.split_indices())
    nodal_displacement = displacement[indices].T
    applied = rhs[indices].T
    reaction = np.zeros_like(rhs)
    reaction[fixed] = residual[fixed]
    reactions = reaction[indices].T
    applied_force, reaction_force = applied.sum(axis=0), reactions.sum(axis=0)
    # Moments about the mesh centroid avoid origin-dependent cancellation diagnostics.
    origin = scalar.doflocs.mean(axis=1)
    arms = scalar.doflocs.T - origin
    applied_moment = np.cross(arms, applied).sum(axis=0)
    reaction_moment = np.cross(arms, reactions).sum(axis=0)
    length = float(np.linalg.norm(np.ptp(points, axis=1)))
    force_error = float(np.linalg.norm(applied_force + reaction_force) / force_n)
    moment_error = float(np.linalg.norm(applied_moment + reaction_moment) / (force_n * length))
    load_error = float(np.linalg.norm(applied_force - np.asarray(direction) * force_n) / force_n)
    if max(relative_residual, force_error, moment_error, load_error) > 1e-7:
        raise RuntimeError("Linear solve failed residual or force/moment equilibrium checks")

    gradient = np.asarray(basis.interpolate(displacement).grad)
    strain = (gradient + gradient.swapaxes(0, 1)) / 2
    stress = 2 * mu * strain + lam * np.eye(3)[:, :, None, None] * np.trace(
        strain, axis1=0, axis2=1
    )
    vm = _von_mises(stress)
    strain_dev = strain - np.eye(3)[:, :, None, None] * np.trace(strain, axis1=0, axis2=1) / 3
    equivalent = np.sqrt(2 / 3 * np.sum(strain_dev**2, axis=(0, 1)))
    if not np.isfinite(stress).all() or not np.isfinite(strain).all():
        raise RuntimeError("Non-finite stress or strain")
    # Display only: volume-weighted element quadrature mean, recovered to P2 nodes.
    element_vm = np.sum(vm * basis.dx, axis=1) / np.sum(basis.dx, axis=1)
    element_strain = np.sum(equivalent * basis.dx, axis=1) / np.sum(basis.dx, axis=1)
    recovered = np.zeros((2, scalar.N))
    weights = np.zeros(scalar.N)
    for node_ids in scalar.element_dofs:
        np.add.at(weights, node_ids, volumes)
        np.add.at(recovered[0], node_ids, element_vm * volumes)
        np.add.at(recovered[1], node_ids, element_strain * volumes)
    recovered /= weights
    edges = {
        tuple(sorted(edge)): int(scalar.edge_dofs[0, i]) for i, edge in enumerate(mesh.edges.T)
    }
    triangles: list[tuple[int, int, int]] = []
    fixed_triangles: list[int] = []
    loaded_triangles: list[int] = []
    fixed_set, loaded_set = set(fixed_facets), set(loaded_facets)
    for facet in mesh.boundary_facets():
        a, b, c = map(int, mesh.facets[:, facet])
        adjacent = int(mesh.f2t[0, facet])
        inward = points[:, mesh.t[:, adjacent]].mean(axis=1) - points[:, [a, b, c]].mean(axis=1)
        normal = np.cross(points[:, b] - points[:, a], points[:, c] - points[:, a])
        if np.dot(normal, inward) > 0:
            b, c = c, b
        ab, bc, ca = (edges[tuple(sorted(pair))] for pair in ((a, b), (b, c), (c, a)))
        start = len(triangles)
        triangles.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        if facet in fixed_set:
            fixed_triangles.extend(range(start, start + 4))
        if facet in loaded_set:
            loaded_triangles.extend(range(start, start + 4))
    used, inverse = np.unique(np.asarray(triangles), return_index=False, return_inverse=True)
    coordinates = np.asarray(basis.global_coordinates()).transpose(1, 2, 0).reshape(-1, 3)
    peak = int(np.argmax(vm))
    magnitude = np.linalg.norm(nodal_displacement, axis=1)
    maximum = int(np.argmax(magnitude))
    # Dimensionless volume/edge shape indicator: vanishes for slivers, not a proof of accuracy.
    edge_lengths = np.linalg.norm(points[:, mesh.edges[0]] - points[:, mesh.edges[1]], axis=0)
    quality = 6 * np.sqrt(2) * volumes / np.max(edge_lengths[mesh.t2e], axis=0) ** 3
    return {
        "summary": {
            "max_displacement_mm": float(magnitude[maximum]),
            "peak_von_mises_mpa": float(vm.flat[peak]),
            "display_max_von_mises_mpa": float(recovered[0, used].max()),
            "displacement_at_mm": scalar.doflocs[:, maximum].tolist(),
            "stress_at_mm": coordinates[peak].tolist(),
            "peak_stress_tensor_mpa": stress.transpose(2, 3, 0, 1).reshape(-1, 3, 3)[peak].tolist(),
            "peak_strain_tensor": strain.transpose(2, 3, 0, 1).reshape(-1, 3, 3)[peak].tolist(),
        },
        "field": {
            "volume_positions_mm": points.T.tolist(),
            "tetrahedra": np.asarray(mesh.t).T.tolist(),
            "quadrature_element_ids": np.repeat(np.arange(mesh.t.shape[1]), vm.shape[1]).tolist(),
            "positions_mm": scalar.doflocs[:, used].T.tolist(),
            "triangles": inverse.reshape(-1, 3).tolist(),
            "displacements_mm": nodal_displacement[used].tolist(),
            "von_mises_mpa": recovered[0, used].tolist(),
            "strain_equivalent": recovered[1, used].tolist(),
            "fixed_triangles": fixed_triangles,
            "loaded_triangles": loaded_triangles,
            "quadrature_positions_mm": coordinates.tolist(),
            "stress_tensors_mpa": stress.transpose(2, 3, 0, 1).reshape(-1, 3, 3).tolist(),
            "strain_tensors": strain.transpose(2, 3, 0, 1).reshape(-1, 3, 3).tolist(),
        },
        "diagnostics": {
            "status": "pass" if quality.min() >= 1e-4 else "unverified",
            "relative_residual": relative_residual,
            "force_balance_relative_error": force_error,
            "moment_balance_relative_error": moment_error,
            "applied_force_relative_error": load_error,
            "applied_force_n": applied_force.tolist(),
            "reaction_force_n": reaction_force.tolist(),
            "applied_moment_n_mm": applied_moment.tolist(),
            "reaction_moment_n_mm": reaction_moment.tolist(),
            "moment_origin_mm": origin.tolist(),
            "fixed_area_mm2": areas[fixed_index],
            "loaded_area_mm2": areas[1 - fixed_index],
            "boundary_area_relative_errors": area_error,
            "boundary_tolerance_mm": tolerance,
            "tetrahedra": int(mesh.t.shape[1]),
            **_mesh_resolution(mesh.p.T, mesh.t.T),
            "displacement_dofs": int(basis.N),
            "minimum_tet_quality": float(quality.min()),
            "minimum_tet_volume_mm3": float(volumes.min()),
            "max_displacement_to_span": float(magnitude.max()) / length,
            "peak_equivalent_infinitesimal_strain": float(equivalent.max()),
        },
        "limitations": [
            "Isolated component, homogeneous isotropic small-strain linear elasticity; no assembly contact or load transfer.",
            "Quadrature stress maximum and P2-node displacement maximum are sampled, not continuous error bounds.",
            "Displayed stress/strain are volume-weighted element-mean nodal recovery, not acceptance extrema.",
            "No buckling, fatigue, plasticity, print anisotropy or physical qualification; clamp singularities may not converge.",
        ],
        "solver": {
            "method": "P2 tetrahedral isotropic linear elasticity",
            "version": importlib.metadata.version("scikit-fem"),
            "stress_measure": "von Mises at integration points",
            "quadrature_order": 5,
            "strain_measure": "sqrt(2/3 * deviatoric infinitesimal strain contraction)",
            "display_recovery": "volume-weighted adjacent-element quadrature means at P2 nodes",
            "frame": "analysis",
            "units": {"length": "mm", "force": "N", "stress": "MPa", "strain": "dimensionless"},
        },
    }


@lru_cache(maxsize=1)
def _validation_cases() -> tuple[dict[str, Any], ...]:
    """Exercise this exact field solver, not the separate legacy heatmap solver."""
    from skfem import MeshTet

    mesh = MeshTet.init_tensor(np.linspace(0, 2, 3), np.linspace(0, 2, 3), np.linspace(0, 20, 11))
    outcomes = []
    for name, direction, expected_displacement in (
        ("axial", (0, 0, 1), 0.01 * 20 / (1000 * 4)),
        ("bending", (1, 0, 0), 0.01 * 20**3 / (3 * 1000 * (2 * 2**3 / 12))),
    ):
        result = _solve_field(
            mesh,
            force_n=0.01,
            direction=direction,
            e_mpa=1000,
            nu=0,
            fixed_end="minimum",
            planes=(0, 20),
            tolerance=1e-8,
            original_areas=(4, 4),
        )
        actual = result["summary"]["max_displacement_mm"]
        error = abs(actual / expected_displacement - 1)
        tolerance = 1e-5 if name == "axial" else 0.06
        outcomes.append(
            {
                "case": name,
                "quantity": "max_displacement_mm",
                "actual": actual,
                "expected": expected_displacement,
                "relative_error": error,
                "tolerance": tolerance,
                "passed": error <= tolerance,
            }
        )
        if name == "axial":
            stress = result["summary"]["peak_von_mises_mpa"]
            stress_error = abs(stress / (0.01 / 4) - 1)
            outcomes.append(
                {
                    "case": name,
                    "quantity": "peak_von_mises_mpa",
                    "actual": stress,
                    "expected": 0.01 / 4,
                    "relative_error": stress_error,
                    "tolerance": 1e-5,
                    "passed": stress_error <= 1e-5,
                }
            )
    return tuple(outcomes)


def structural_field(
    mesh: Any,
    *,
    force_n: float,
    direction: Any,
    e_mpa: float,
    nu: float,
    mesh_factor: float,
    fixed_end: Literal["minimum", "maximum"],
    tetra_mesh: Any | None = None,
    mesher_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mesh and solve explicit full extreme-Z planar caps. Requires an isolated worker."""
    vector = np.asarray(direction, dtype=float)
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or not np.isfinite(np.linalg.norm(vector))
        or np.linalg.norm(vector) < 1e-12
    ):
        raise ValueError("Load direction must be a finite nonzero three-vector")
    if not all(np.isfinite([force_n, e_mpa, nu, mesh_factor])) or not (
        force_n > 0 and e_mpa > 0 and -1 < nu < 0.5 and 0.03 <= mesh_factor <= 0.3
    ):
        raise ValueError("Invalid force, isotropic material or mesh factor")
    if fixed_end not in {"minimum", "maximum"}:
        raise ValueError("fixed_end must be minimum or maximum")
    if os.getenv("FORGE_FEA_STRESS", "1") == "0":
        raise RuntimeError("FEA disabled")
    validation_cases = _validation_cases()
    if not all(case["passed"] for case in validation_cases):
        raise RuntimeError("Structural field solver analytic validation gate failed")
    enforce_limit(
        len(mesh.vertices),
        optional_limit("STRUCTURAL_SURFACE_VERTEX_LIMIT"),
        "structural surface vertices",
    )
    enforce_limit(
        len(mesh.faces), optional_limit("STRUCTURAL_SURFACE_FACE_LIMIT"), "structural surface faces"
    )
    verts, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)
    if not np.isfinite(verts).all() or not mesh.is_watertight or len(mesh.split()) != 1:
        raise ValueError("Structural analysis requires one finite watertight connected component")
    planes = (float(verts[:, 2].min()), float(verts[:, 2].max()))
    tolerance = max(float(np.linalg.norm(np.ptp(verts, axis=0))) * 2e-4, 1e-7)
    if planes[1] - planes[0] <= 2 * tolerance:
        raise ValueError("Analysis-Z span is too small to separate boundary caps")
    tri = verts[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    originals = tuple(
        float(areas[np.all(abs(tri[:, :, 2] - z) <= 1e-8, axis=1)].sum()) for z in planes
    )
    if min(originals) <= 0:
        raise ValueError("Both extreme analysis-Z boundaries must contain planar faces")
    from skfem import MeshTet

    if tetra_mesh is None:
        import pytetwild

        # Native mesher writes to cwd. The outer process provides resource bounds.
        with _FEA_LOCK, tempfile.TemporaryDirectory(prefix="structural_") as scratch:
            previous = os.getcwd()
            try:
                os.chdir(scratch)
                points, tetrahedra = pytetwild.tetrahedralize(
                    verts, faces, edge_length_fac=mesh_factor, epsilon=1e-4, num_threads=2
                )[:2]
            finally:
                os.chdir(previous)
        if len(tetrahedra) < 1:
            raise ValueError("Native tetrahedral mesh has no elements")
        enforce_limit(
            len(tetrahedra), optional_limit("STRUCTURAL_TET_LIMIT"), "structural tetrahedra"
        )
        tetra_mesh = MeshTet(np.asarray(points).T, np.asarray(tetrahedra).T)
        mesher_metadata = {
            "name": "TetWild",
            "version": importlib.metadata.version("pytetwild"),
            "mesh_factor": mesh_factor,
            "epsilon_relative_to_diagonal": 1e-4,
            "threads": 2,
        }
    result = _solve_field(
        tetra_mesh,
        force_n=force_n,
        direction=vector / np.linalg.norm(vector),
        e_mpa=e_mpa,
        nu=nu,
        fixed_end=fixed_end,
        planes=planes,
        tolerance=tolerance,
        original_areas=(originals[0], originals[1]),
    )
    result["solver"]["validation_cases"] = list(validation_cases)
    result["solver"]["mesher"] = mesher_metadata
    result["solver"]["mesher_version"] = (mesher_metadata or {}).get("version")
    result["solver"]["mesh_factor"] = mesh_factor
    return result
