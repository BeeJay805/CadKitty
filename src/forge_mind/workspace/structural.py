"""Resolve exported component geometry and retain both mesh refinement calculations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.capacity import enforce_limit, optional_limit
from forge_mind.workspace.structural_models import STRUCTURAL_SCOPE, StructuralStudy


def write_field(path: Path, field: dict[str, Any]) -> None:
    """Validate complete associations and enforce only configured artifact policies."""
    from forge_mind.workspace.structural_service import validate_field

    validate_field(field)
    encoded = json.dumps(field, separators=(",", ":"), allow_nan=False)
    enforce_limit(
        len(encoded.encode()), optional_limit("STRUCTURAL_FIELD_BYTES"), "structural field bytes"
    )
    path.write_text(encoded, encoding="utf-8")


def analysis_frame(axis: Any) -> np.ndarray:
    z = np.asarray(axis, dtype=float)
    z /= math.hypot(*z)
    hint = np.eye(3)[int(np.argmin(abs(z)))]
    x = hint - np.dot(hint, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def compare_study(
    study: StructuralStudy, coarse: dict[str, Any], fine: dict[str, Any]
) -> dict[str, Any]:
    def change(key: str) -> float:
        return float(
            abs(fine["summary"][key] - coarse["summary"][key])
            / max(abs(fine["summary"][key]), 1e-12)
        )

    displacement_delta = change("max_displacement_mm")
    stress_delta = change("peak_von_mises_mpa")
    reasons = []
    coarse_mesh, fine_mesh = coarse.get("diagnostics", {}), fine.get("diagnostics", {})
    mesh_keys = (
        "volume_mesh_sha256",
        "volume_weighted_rms_tet_diameter_mm",
        "displacement_dofs",
        "tetrahedra",
    )
    refinement: dict[str, Any] = {"status": "unverified", "diameter_ratio": None}
    if not all(key in mesh for mesh in (coarse_mesh, fine_mesh) for key in mesh_keys):
        reasons.append("Actual volume-mesh refinement evidence is unavailable")
    else:
        coarse_h = coarse_mesh["volume_weighted_rms_tet_diameter_mm"]
        fine_h = fine_mesh["volume_weighted_rms_tet_diameter_mm"]
        ratio = fine_h / coarse_h if coarse_h > 0 else None
        refinement["diameter_ratio"] = ratio
        if (
            coarse_mesh["volume_mesh_sha256"] == fine_mesh["volume_mesh_sha256"]
            or fine_mesh["displacement_dofs"] <= coarse_mesh["displacement_dofs"]
            or fine_mesh["tetrahedra"] <= coarse_mesh["tetrahedra"]
            or ratio is None
            or not 0 < ratio < 1 - 1e-6
        ):
            reasons.append(
                "Fine mesh did not demonstrate distinct geometry, increased DOFs/elements and smaller volume-weighted element diameter; changing requested size alone is not refinement evidence"
            )
        else:
            refinement["status"] = "pass"
    if displacement_delta > study.convergence_tolerance:
        reasons.append(
            "Displacement has not stabilized within the declared mesh-refinement tolerance"
        )
    if stress_delta > study.convergence_tolerance:
        reasons.append(
            "Sampled peak stress has not stabilized; local stress concentrations may be mesh-sensitive"
        )
    for label, run in (("coarse", coarse), ("fine", fine)):
        if run.get("diagnostics", {}).get("status") != "pass":
            reasons.append(label.capitalize() + " solver numerical-quality checks did not pass")
    quality = {
        "status": "unverified" if reasons else "pass",
        "coarse": {
            "summary": coarse["summary"],
            "diagnostics": coarse.get("diagnostics"),
            "solver": coarse.get("solver"),
        },
        "fine": {
            "summary": fine["summary"],
            "diagnostics": fine.get("diagnostics"),
            "solver": fine.get("solver"),
        },
        "displacement_relative_change": displacement_delta,
        "stress_relative_change": stress_delta,
        "tolerance": study.convergence_tolerance,
        "refinement": refinement,
        "reasons": reasons,
        "scope": "Two-mesh comparison with observed global refinement and algebraic equilibrium checks, not a local or rigorous discretization error bound",
    }
    peak = fine["summary"]["peak_von_mises_mpa"]
    ratio = (
        study.material.yield_strength_mpa / peak
        if study.material.yield_strength_mpa and peak > 0
        else None
    )
    comparisons = []
    for identifier, title, value, limit, unit, violated, missing in (
        (
            "deflection",
            "Maximum computed displacement",
            fine["summary"]["max_displacement_mm"],
            study.max_deflection_mm,
            "mm",
            study.max_deflection_mm is not None
            and fine["summary"]["max_displacement_mm"] > study.max_deflection_mm,
            "No maximum displacement has been declared",
        ),
        (
            "yield_ratio",
            "Yield-to-sampled-stress ratio",
            ratio,
            study.min_safety_factor,
            "ratio",
            ratio is not None
            and study.min_safety_factor is not None
            and ratio < study.min_safety_factor,
            "An explicit yield strength, nonzero computed stress and minimum ratio are required",
        ),
    ):
        comparisons.append(
            {
                "id": identifier,
                "title": title,
                "status": "fail"
                if violated
                else "pass"
                if value is not None and limit is not None and not reasons
                else "unverified",
                "measured": value,
                "limit": limit,
                "unit": unit,
                "reason": "Exceeds declared criterion"
                if violated
                else missing
                if value is None or limit is None
                else "; ".join(reasons)
                if reasons
                else "Within declared limits for this idealized component calculation only",
            }
        )
    return {
        "summary": {**fine["summary"], "yield_to_sampled_stress_ratio": ratio},
        "comparisons": comparisons,
        "numerical_quality": quality,
    }


def world_field(field: dict[str, Any], frame: np.ndarray, origin: np.ndarray) -> None:
    """Positions translate; displacement vectors and second-order tensors do not."""
    field.update(schema_version=1, frame="cad_z_up_mm")
    for key in ("positions_mm", "quadrature_positions_mm", "volume_positions_mm"):
        if key in field:
            field[key] = (np.asarray(field[key]) @ frame.T + origin).tolist()
    field["displacements_mm"] = (np.asarray(field["displacements_mm"]) @ frame.T).tolist()
    for key in ("stress_tensors_mpa", "strain_tensors"):
        if key in field:
            field[key] = np.einsum("ij,njk,lk->nil", frame, np.asarray(field[key]), frame).tolist()
    field["analysis_frame"] = {"origin_mm": origin.tolist(), "basis_columns": frame.tolist()}


def run_structural_study(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import build123d as b
    import trimesh

    from forge_mind.physics.structural import structural_field
    from forge_mind.workspace.assembly import named_solids, occurrence_label
    from forge_mind.workspace.tessellation import STL_ANGULAR_TOLERANCE, STL_LINEAR_TOLERANCE
    from forge_mind.workspace.topology import same_exported_solid, signature

    study = StructuralStudy.model_validate(payload["study"])
    source = Path(payload["source"])
    revision = payload["revision"]
    expected_hash = revision["measurement"]["artifact_hashes"]["part.step"]
    if file_sha256(source / "part.step") != expected_hash:
        raise ValueError("Source STEP changed before component analysis")
    shape = b.import_step(str(source / "part.step"))
    solids = list(shape.solids())
    if not shape.is_valid or not solids:
        raise ValueError("Structural study requires valid exported solids")
    selected: Any
    if study.target.body_index is not None:
        if study.target.body_index >= len(solids):
            raise ValueError("Selected analysis body is absent")
        selected = solids[study.target.body_index]
    else:
        selected = named_solids(shape).get(occurrence_label(str(study.target.occurrence_path)))
        if selected is None:
            raise ValueError("Selected analysis occurrence is absent")
    indices = [i for i, solid in enumerate(solids) if same_exported_solid(selected, solid)]
    if len(indices) != 1 or len(selected.solids()) != 1:
        raise ValueError("Analysis target must resolve uniquely to one connected exported body")
    target = {
        "body_index": indices[0],
        "occurrence_path": study.target.occurrence_path,
        "body_signature": signature(selected),
        "name": study.target.occurrence_path or f"Body {indices[0] + 1}",
    }
    result: dict[str, Any] = {
        "status": "failed",
        "target": target,
        "summary": {},
        "comparisons": [],
        "numerical_quality": {"status": "unverified", "reasons": []},
        "scope": STRUCTURAL_SCOPE,
        "limitations": [STRUCTURAL_SCOPE],
    }
    try:
        vertices, triangles = selected.tessellate(STL_LINEAR_TOLERANCE, STL_ANGULAR_TOLERANCE)
        enforce_limit(
            len(vertices),
            optional_limit("STRUCTURAL_SURFACE_VERTEX_LIMIT"),
            "structural surface vertices",
        )
        enforce_limit(
            len(triangles),
            optional_limit("STRUCTURAL_SURFACE_FACE_LIMIT"),
            "structural surface faces",
        )
        points = np.array([tuple(v) for v in vertices])
        origin = np.asarray(tuple(selected.bounding_box().center()), dtype=float)
        frame = analysis_frame(study.axis)
        mesh = trimesh.Trimesh(
            vertices=(points - origin) @ frame, faces=np.asarray(triangles), process=True
        )
        if not mesh.is_watertight or not mesh.is_volume or len(mesh.split()) != 1:
            raise ValueError("Selected body did not yield one closed connected analysis surface")
        direction = np.asarray(study.direction, dtype=float)
        direction /= math.hypot(*direction)
        runs = []
        for stage, factor in zip(("coarse", "fine"), study.mesh_factors, strict=True):
            mesh_options: dict[str, Any] = {}
            if study.mesher == "cad":
                from skfem import MeshTet

                from forge_mind.physics.cad_meshing import mesh_cad_solid

                stage_directory = directory / stage
                stage_directory.mkdir(exist_ok=True)
                cad_points, cad_tets, mesh_receipt = mesh_cad_solid(
                    selected, factor, stage_directory
                )
                np.savez_compressed(
                    stage_directory / "mesh.npz",
                    positions_mm=cad_points,
                    tetrahedra=cad_tets,
                )
                mesh_receipt["mesh_sha256"] = file_sha256(stage_directory / "mesh.npz")
                mesh_receipt["artifact_frame"] = "cad_z_up_mm"
                mesh_options = {
                    "tetra_mesh": MeshTet(((cad_points - origin) @ frame).T, cad_tets.T),
                    "mesher_metadata": mesh_receipt,
                }
            run = structural_field(
                mesh,
                force_n=study.force_n,
                direction=tuple(direction @ frame),
                e_mpa=study.material.elastic_modulus_mpa,
                nu=study.material.poisson_ratio,
                mesh_factor=factor,
                fixed_end=study.fixed_end,
                **mesh_options,
            )
            runs.append(run)
        coarse, fine = runs
        result.update(compare_study(study, coarse, fine))
        field = fine["field"]
        world_field(field, frame, origin)
        field["force_n"] = study.force_n
        field["force_direction"] = direction.tolist()
        # Locations retain their frame. Fields transform as vectors, never with translation.
        for summary in (
            result["summary"],
            result["numerical_quality"]["coarse"]["summary"],
            result["numerical_quality"]["fine"]["summary"],
        ):
            for key in ("displacement_at_mm", "stress_at_mm"):
                if key in summary:
                    summary[key] = (np.asarray(summary[key]) @ frame.T + origin).tolist()
            summary["frame"] = "cad_z_up_mm"
            for key in ("peak_stress_tensor_mpa", "peak_strain_tensor"):
                if key in summary:
                    summary[key] = (frame @ np.asarray(summary[key]) @ frame.T).tolist()
        result["analysis_frame"] = field["analysis_frame"]
        for stage_name in ("coarse", "fine"):
            result["numerical_quality"][stage_name]["diagnostics"]["frame"] = "analysis_local_mm"
        write_field(directory / "field.json", field)
        result.update(
            status="computed",
            solver=fine.get("solver"),
            limitations=[STRUCTURAL_SCOPE, *fine.get("limitations", [])],
            field_summary={
                "vertices": len(field["positions_mm"]),
                "triangles": len(field["triangles"]),
                "stress_association": "Recovered nodal display values; numerical comparisons use sampled integration-point peak",
            },
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        result.update(
            status="failed",
            reason=str(exc),
            summary={},
            comparisons=[],
            numerical_quality={"status": "unverified", "reasons": [str(exc)]},
            limitations=[STRUCTURAL_SCOPE, str(exc)],
        )
    return result
