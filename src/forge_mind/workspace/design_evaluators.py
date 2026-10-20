"""Registered instruments consume exported geometry, never model-authored scores."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.processing import process_revision

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.design_study_models import StudyEvaluator


def metric_catalog(kind: str, settings: dict[str, Any] | None = None) -> list[dict[str, str]]:
    if kind == "interface_geometry":
        from forge_mind.workspace.interface_geometry import interface_metrics

        return interface_metrics(settings)
    entries = {
        "geometry": [
            ("volume_mm3", "Solid volume", "mm³"),
            ("surface_area_mm2", "Surface area", "mm²"),
            ("extent_x_mm", "X extent", "mm"),
            ("extent_y_mm", "Y extent", "mm"),
            ("extent_z_mm", "Z extent", "mm"),
            ("solid_count", "Solid count", "count"),
        ],
        "structure": [
            ("max_displacement_mm", "Maximum displacement", "mm"),
            ("peak_von_mises_mpa", "Peak von Mises stress", "MPa"),
        ],
        "flow": [("drag_n", "Drag force", "N"), ("drag_coefficient", "Drag coefficient", "1")],
    }
    return [
        {"key": kind + "." + k, "label": label, "unit": unit} for k, label, unit in entries[kind]
    ]


def evaluator_discovery() -> list[dict[str, Any]]:
    import os
    import shutil

    from forge_mind.workspace.flow_models import FlowSettings
    from forge_mind.workspace.interface_geometry import InterfaceGeometrySettings
    from forge_mind.workspace.structural_models import STRUCTURAL_SCOPE, StructuralStudy

    executable = os.environ.get("CADKITTY_SU2_EXE") or shutil.which("SU2_CFD")
    return [
        {
            "kind": "geometry",
            "available": True,
            "metrics": metric_catalog("geometry"),
            "schema": {},
            "scope": "Measurements of the actual exported solids. No inferred physical performance.",
        },
        {
            "kind": "interface_geometry",
            "available": True,
            "metrics": [],
            "metric_templates": [
                "interface_geometry.<pair_id>_distance_mm",
                "interface_geometry.<pair_id>_overlap_mm3",
            ],
            "schema": InterfaceGeometrySettings.model_json_schema(),
            "scope": (
                "Named saved occurrence pairs: minimum solid distance and Boolean overlap volume. "
                "Reference geometry and explicitly fixed occurrences stay frozen. Zero distance "
                "does not distinguish contact from interference. Static nominal geometry only; "
                "no fit class, continuous motion, load or physical qualification."
            ),
        },
        {
            "kind": "structure",
            "available": importlib.util.find_spec("skfem") is not None,
            "metrics": metric_catalog("structure"),
            "schema": StructuralStudy.model_json_schema(),
            "scope": STRUCTURAL_SCOPE,
        },
        {
            "kind": "flow",
            "available": bool(executable and Path(executable).is_file()),
            "metrics": metric_catalog("flow"),
            "schema": FlowSettings.model_json_schema(),
            "scope": "SU2 viscous external flow in an explicitly declared regime. Numerical validity and applicability are checked separately from physical qualification.",
        },
    ]


def evaluate_design(
    service: WorkspaceService,
    sid: str,
    revision: dict[str, Any],
    evaluator: StudyEvaluator,
    directory: Path,
    cancelled: Callable[[], bool],
    sensitivity_tolerance: float = 0.02,
    *,
    flow_context: dict[str, Any] | None = None,
    flow_minimum_level: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    source = service.store.directory(sid, revision["id"])
    directory.mkdir(parents=True, exist_ok=True)
    if evaluator.kind == "geometry":
        result = process_revision(
            directory,
            {"operation": "study_geometry", "source_step": str(source / "part.step")},
            None,
            cancelled,
        )
    elif evaluator.kind == "interface_geometry":
        from forge_mind.validation_provenance import file_sha256

        expected_hashes = {
            name: revision["measurement"]["artifact_hashes"][name]
            for name in ("part.step", "document.json")
        }
        if any(file_sha256(source / name) != value for name, value in expected_hashes.items()):
            raise ValueError("Interface evaluator source differs from the saved revision")
        result = process_revision(
            directory,
            {
                "operation": "study_interface_geometry",
                "source": str(source),
                "settings": evaluator.settings,
            },
            None,
            cancelled,
        )
        if result.get("source_hashes") != expected_hashes or any(
            file_sha256(source / name) != value for name, value in expected_hashes.items()
        ):
            raise ValueError("Interface evaluator source changed during measurement")
    elif evaluator.kind == "flow":
        from forge_mind.workspace.flow_adaptation import adaptive_flow
        from forge_mind.workspace.flow_models import FlowSettings

        result = adaptive_flow(
            source / "part.step",
            directory,
            FlowSettings.model_validate(evaluator.settings),
            sensitivity_tolerance,
            cancelled,
            cache_root=directory.parent.parent / "flow-cache",
            identity=flow_context or {"revision_id": revision["id"]},
            minimum_level=flow_minimum_level,
        )
    else:
        result = process_revision(
            directory,
            {
                "operation": "structural_study",
                "source": str(source),
                "revision": revision,
                "study": evaluator.settings,
            },
            None,
            cancelled,
        )
        summary = result.get("summary", {})
        quality = result.get("numerical_quality", {})
        declared = {
            name
            for name, setting in (
                ("deflection", "max_deflection_mm"),
                ("yield_ratio", "min_safety_factor"),
            )
            if evaluator.settings.get(setting) is not None
        }
        comparisons = {c["id"]: c for c in result.get("comparisons", [])}
        metrics = ("max_displacement_mm", "peak_von_mises_mpa")
        uncertainty = {
            metric: abs(quality["coarse"]["summary"][metric] - quality["fine"]["summary"][metric])
            for metric in metrics
            if metric in quality.get("coarse", {}).get("summary", {})
            and metric in quality.get("fine", {}).get("summary", {})
        }
        result = {
            **result,
            "eligible": result.get("status") == "computed"
            and quality.get("status") == "pass"
            and all(comparisons.get(name, {}).get("status") == "pass" for name in declared)
            and all(metric in uncertainty for metric in metrics),
            "metrics": {m: summary[m] for m in metrics if m in summary},
            "metric_uncertainty": uncertainty,
        }
    raw = result.get("metrics", {})
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in raw.values()):
        raise ValueError("Instrument returned a nonfinite or nonnumeric metric")
    result["metrics"] = {evaluator.kind + "." + k: float(v) for k, v in raw.items()}
    return result


def geometry_worker(step_path: Path) -> dict[str, Any]:
    import build123d as b

    shape = b.import_step(str(step_path))
    solids = list(shape.solids())
    if not shape.is_valid or not solids or any(not s.is_valid or s.volume <= 0 for s in solids):
        raise ValueError("Geometry evaluator requires valid positive-volume exported solids")
    box = shape.bounding_box()
    return {
        "status": "complete",
        "eligible": True,
        "metrics": {
            "volume_mm3": sum(s.volume for s in solids),
            "surface_area_mm2": sum(s.area for s in solids),
            "extent_x_mm": box.size.X,
            "extent_y_mm": box.size.Y,
            "extent_z_mm": box.size.Z,
            "solid_count": len(solids),
        },
        "scope": "Exact exported B-rep measures. Assembly volume sums component volumes, including overlaps; it is not usable internal volume or a physical qualification.",
    }


def flow_worker(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from forge_mind.validation_provenance import file_sha256
    from forge_mind.workspace.flow_models import FlowSettings
    from forge_mind.workspace.flow_solver import evaluate_flow, solver_executable

    settings = FlowSettings.model_validate(payload["settings"])
    source = Path(payload["source_step"])
    source_hash = file_sha256(source)
    solver_hash = file_sha256(solver_executable())
    runs: list[dict[str, Any]] = []
    for stage in ("coarse", "fine", "domain"):
        candidate = settings.model_copy(deep=True)
        candidate.mesh.refinement = "coarse" if stage == "coarse" else "fine"
        if stage == "domain":
            candidate.mesh.domain_padding_factor *= 1.5
        result = evaluate_flow(source, directory / stage, candidate, lambda: False)
        if (
            result.get("provenance", {}).get("solver_sha256") != solver_hash
            or file_sha256(solver_executable()) != solver_hash
        ):
            raise ValueError("Flow solver changed during the frozen comparison")
        runs.append({"stage": stage, "result": result})
        (directory / (stage + "-receipt.json")).write_text(
            json.dumps(result, allow_nan=False), encoding="utf-8"
        )
        if not result.get("eligible"):
            break
    final = runs[-1]["result"]
    sensitivity: dict[str, Any] = {
        "eligible": False,
        "tolerance": payload["sensitivity_tolerance"],
        "scope": "Observed sensitivity across two meshes and an enlarged domain, not a formal uncertainty bound.",
    }
    uncertainty = {}
    if len(runs) == 3 and all(r["result"].get("eligible") for r in runs):
        coarse, fine, domain = [r["result"] for r in runs]
        for metric in ("drag_n", "drag_coefficient"):
            mesh_delta = abs(coarse["metrics"][metric] - fine["metrics"][metric])
            domain_delta = abs(fine["metrics"][metric] - domain["metrics"][metric])
            uncertainty[metric] = mesh_delta + domain_delta
            sensitivity[metric] = {
                "mesh_delta": mesh_delta,
                "domain_delta": domain_delta,
                "relative_change": uncertainty[metric] / max(abs(domain["metrics"][metric]), 1e-12),
            }
        mesh_distinct = coarse.get("provenance", {}).get("mesh_sha256") != fine.get(
            "provenance", {}
        ).get("mesh_sha256")
        sensitivity["eligible"] = mesh_distinct and all(
            sensitivity[m]["relative_change"] <= payload["sensitivity_tolerance"]
            for m in uncertainty
        )
    eligible = final.get("eligible") is True and sensitivity["eligible"]
    field_stage = None
    field_scope = None
    # Only inspect stages produced by this invocation, with a matching source
    # and artifact receipt. An incomplete refinement may still retain an earlier
    # diagnostic field without borrowing another attempt's or candidate's data.
    for run in reversed(runs):
        field = directory / run["stage"] / "surface_field.json"
        field_hash = run["result"].get("artifacts", {}).get("surface_field.json", {}).get("sha256")
        if (
            not field_hash
            or not field.is_file()
            or run["result"].get("provenance", {}).get("source_step_sha256") != source_hash
            or file_sha256(field) != field_hash
        ):
            continue
        from forge_mind.workspace.processing import copy_artifact

        copy_artifact(field, directory / "surface_field.json")
        if file_sha256(directory / "surface_field.json") != field_hash:
            raise ValueError("Flow diagnostic field changed during capture")
        field_stage = run["stage"]
        if len(runs) < 3 or not all(r["result"].get("eligible") for r in runs):
            field_scope = f"diagnostic {field_stage} field; refinement incomplete"
        elif not eligible:
            field_scope = f"diagnostic {field_stage} field; sensitivity unverified"
        else:
            field_scope = (
                f"{field_stage} field; observed sensitivity checked; physical validity unverified"
            )
        break
    return {
        **final,
        "eligible": eligible,
        "status": "complete" if eligible else "unverified",
        "sensitivity": sensitivity,
        "metric_uncertainty": uncertainty,
        "refinement_runs": runs,
        "field_stage": field_stage,
        "field_scope": field_scope,
    }
