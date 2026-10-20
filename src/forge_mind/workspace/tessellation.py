"""Trusted, volume-checked STEP tessellation and reproducible exporter settings."""

from __future__ import annotations

import json
import math
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge_mind.validation_provenance import file_sha256

STL_LINEAR_TOLERANCE = 0.005
STL_ANGULAR_TOLERANCE = 0.05
_EDGE_SAMPLE_LIMIT = 2
_FAILURE_MESSAGE_LIMIT = 1400  # The worker error reader retains 1,600 characters.


def _unique_occurrence_label(
    names: dict[str, Any], solids: list[Any], index: int, same: Callable[[Any, Any], bool]
) -> tuple[str | None, str]:
    labels = [label for label, solid in names.items() if same(solids[index], solid)]
    if not labels:
        return None, "not_named"
    if len(labels) != 1 or sum(same(solid, names[labels[0]]) for solid in solids) != 1:
        return None, "ambiguous"
    label = labels[0]
    if len(label.encode("utf-8")) > 128:
        return None, "label_exceeds_diagnostic_bound"
    return label, "bound_exported_occurrence_label"


def _exported_body_identity(part: Any, solids: list[Any], index: int) -> tuple[str | None, str]:
    # Failure-only lookup; exact exported topology/placement, never bounds matching.
    try:
        from forge_mind.workspace.assembly import named_solids
        from forge_mind.workspace.topology import same_exported_solid

        return _unique_occurrence_label(named_solids(part), solids, index, same_exported_solid)
    except Exception as exc:
        return None, "unavailable:" + type(exc).__name__


def _mesh_failure_message(
    mesh: Any, index: int, mesh_volume: float, identity: tuple[str | None, str]
) -> str:
    """Explain the existing refusal from loaded mesh data; never change geometry."""
    import numpy as np

    conditions = []
    if not mesh.is_watertight:
        conditions.append("not_watertight")
    if not mesh.is_winding_consistent:
        conditions.append("winding_inconsistent")
    if not math.isfinite(mesh_volume):
        conditions.append("mesh_volume_nonfinite")
    elif mesh_volume <= 0:
        conditions.append("mesh_volume_nonpositive")
    diagnostic: dict[str, Any] = {
        "body_index": index,
        "occurrence_label": identity[0],
        "identity_status": identity[1],
        "failed_conditions": conditions,
        "mesh_volume_mm3": mesh_volume if math.isfinite(mesh_volume) else None,
        "frame": "input_step_mm",
        "scope": "Coordinates in current part.step: assembly-world or definition-local according to caller. Loaded STL edges, not exact STEP edges or proof of cause.",
    }
    try:
        edges = mesh.edges_unique
        incidence = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        for name, mask in (
            ("boundary_edges", incidence == 1),
            ("nonmanifold_edges", incidence > 2),
        ):
            count = int(mask.sum())
            sample = []
            for edge_index in np.flatnonzero(mask)[:_EDGE_SAMPLE_LIMIT]:
                coordinates = mesh.vertices[edges[edge_index]]
                if np.isfinite(coordinates).all():
                    sample.append(
                        {
                            "incidence": int(incidence[edge_index]),
                            "endpoints_mm": coordinates.tolist(),
                        }
                    )
            diagnostic[name] = {"count": count, "sample": sample, "omitted": count - len(sample)}
        diagnostic["edge_inventory_status"] = "available"
    except Exception as exc:
        # A diagnostic failure must not hide the original topology/volume refusal.
        diagnostic.pop("boundary_edges", None)
        diagnostic.pop("nonmanifold_edges", None)
        diagnostic["edge_inventory_status"] = "unavailable:" + type(exc).__name__
    prefix = "STEP tessellation has invalid body topology or volume: "
    while True:
        message = prefix + json.dumps(diagnostic, separators=(",", ":"), allow_nan=False)
        if len(message) <= _FAILURE_MESSAGE_LIMIT:
            return message
        for name in ("boundary_edges", "nonmanifold_edges"):
            group = diagnostic.get(name)
            if group and group["sample"]:
                group["sample"].pop()
                group["omitted"] += 1
                break
        else:
            diagnostic["occurrence_label"] = None
            diagnostic["identity_status"] = "label_exceeds_diagnostic_bound"


def remove_collapsed_triangles(mesh: Any) -> int:
    """Remove repeated-index pole triangles only; never repair holes or surfaces."""
    faces = mesh.faces
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    )
    count = int((~keep).sum())
    if count:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return count


def read_tessellation_settings(directory: Path) -> tuple[float, float]:
    path = directory / "tessellation.json"
    try:
        info = path.lstat()
    except FileNotFoundError:
        return STL_LINEAR_TOLERANCE, STL_ANGULAR_TOLERANCE
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
        raise ValueError("Invalid tessellation metadata file")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate tessellation metadata key")
            result[key] = item
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != {
        "version",
        "step_sha256",
        "linear_deflection",
        "angular_deflection",
    }:
        raise ValueError("Invalid tessellation metadata schema")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported tessellation metadata version")
    if value["step_sha256"] != file_sha256(directory / "part.step"):
        raise ValueError("Tessellation metadata does not match STEP")
    settings = []
    for key, maximum in (
        ("linear_deflection", STL_LINEAR_TOLERANCE),
        ("angular_deflection", STL_ANGULAR_TOLERANCE),
    ):
        number = value[key]
        if (
            type(number) not in (int, float)
            or not math.isfinite(number)
            or not 0 < number <= maximum
        ):
            raise ValueError("Invalid tessellation deflection")
        settings.append(float(number))
    return settings[0], settings[1]


def export_verified_tessellation(directory: Path) -> tuple[float, float, int]:
    """Refine failing body volumes under the caller's worker deadline/resource policies.

    Every body must satisfy the existing 0.1%/0.001 mm3 volume check. No aggregate
    volume can hide a small body's error. Invalid topology is never repaired here.
    """
    import trimesh
    from build123d import export_stl, import_step

    linear, angular = STL_LINEAR_TOLERANCE, STL_ANGULAR_TOLERANCE
    previous_errors: list[float] | None = None
    with tempfile.TemporaryDirectory(prefix="cad-quality-", dir=directory) as temporary:
        while True:
            # OCCT caches triangulations on shapes; start each attempt from final STEP.
            part = import_step(directory / "part.step")
            solids = list(part.solids())
            if not part.is_valid or not solids:
                raise ValueError("Tessellation requires valid positive STEP solids")
            errors = []
            failed = []
            for index, solid in enumerate(solids):
                volume = float(solid.volume)
                if not solid.is_valid or not math.isfinite(volume) or volume <= 0:
                    raise ValueError("Tessellation requires valid positive STEP solids")
                target = Path(temporary) / "body.stl"
                export_stl(solid, target, tolerance=linear, angular_tolerance=angular)
                body: Any = trimesh.load(target, force="mesh")
                remove_collapsed_triangles(body)
                mesh_volume = float(body.volume)
                if (
                    not body.is_watertight
                    or not body.is_winding_consistent
                    or not math.isfinite(mesh_volume)
                    or mesh_volume <= 0
                ):
                    raise ValueError(
                        _mesh_failure_message(
                            body, index, mesh_volume, _exported_body_identity(part, solids, index)
                        )
                    )
                errors.append(abs(mesh_volume - volume) / volume)
                if not math.isclose(mesh_volume, volume, rel_tol=0.001, abs_tol=0.001):
                    failed.append(index)
            if not failed:
                export_stl(
                    part, directory / "part.stl", tolerance=linear, angular_tolerance=angular
                )
                mesh: Any = trimesh.load(directory / "part.stl", force="mesh")
                collapsed = remove_collapsed_triangles(mesh)
                if collapsed:
                    mesh.export(directory / "part.stl")
                (directory / "tessellation.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "step_sha256": file_sha256(directory / "part.step"),
                            "linear_deflection": linear,
                            "angular_deflection": angular,
                        }
                    ),
                    encoding="utf-8",
                )
                return linear, angular, collapsed
            if previous_errors is not None and any(errors[i] >= previous_errors[i] for i in failed):
                raise ValueError("STEP tessellation volume refinement stagnated")
            previous_errors = errors
            refined = linear / 2, angular / 2
            if any(not math.isfinite(v) or v <= 0 for v in refined) or refined == (linear, angular):
                raise ValueError("STEP tessellation deflection is not representable")
            linear, angular = refined
