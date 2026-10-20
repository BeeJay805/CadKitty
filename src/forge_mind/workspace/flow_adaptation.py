"""Controlled, resumable numerical refinement. Sensitivity is not physical accuracy."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.flow_models import FlowSettings
from forge_mind.workspace.processing import copy_artifact

METRICS = ("drag_n", "drag_coefficient")
SCOPE = (
    "Observed mesh/domain sensitivity and final-window force spread, not a formal error "
    "bound, mesh-domain interaction bound, turbulence validation or physical qualification."
)


def effective_settings(settings: FlowSettings, mesh: int, domain: int) -> FlowSettings:
    """One fixed physical sizing function; domain growth never coarsens that function."""
    candidate = settings.model_copy(deep=True)
    candidate.mesh.reference_length_m = (
        settings.mesh.reference_length_m or settings.reference_length_m
    )
    candidate.mesh.refinement = "coarse" if mesh == 0 else "fine"
    candidate.mesh.fine_scale = settings.mesh.fine_scale ** max(mesh, 1)
    candidate.mesh.domain_padding_factor *= settings.adaptation.domain_scale**domain
    return FlowSettings.model_validate(candidate.model_dump())


def assess_cross(
    results: dict[tuple[int, int], dict[str, Any]],
    mesh: int,
    domain: int,
    tolerance: float,
    *,
    confirmation: bool,
) -> dict[str, Any]:
    count = 2 if confirmation else 1
    points = (
        [(mesh, domain)]
        + [(mesh - i, domain) for i in range(1, count + 1)]
        + [(mesh, domain - i) for i in range(1, count + 1)]
    )
    answer: dict[str, Any] = {
        "eligible": False,
        "tolerance": tolerance,
        "scope": SCOPE,
        "confirmation": confirmation,
        "mesh_level": mesh,
        "domain_level": domain,
        "points": [list(p) for p in points],
    }
    if any(p not in results or results[p].get("eligible") is not True for p in points):
        answer["reason"] = "Controlled comparison is incomplete or a solve is inadmissible"
        return answer
    hashes = [results[p].get("provenance", {}).get("mesh_sha256") for p in points]
    if any(not h for h in hashes) or len(set(hashes)) != len(hashes):
        answer["reason"] = "Refinement did not produce distinct verified meshes"
        return answer
    try:
        noises = [float(results[p]["convergence"]["force_relative_spread"]) for p in points]
        if any(not math.isfinite(n) or n < 0 for n in noises):
            raise ValueError("Invalid numerical noise")
        noise = max(noises)
        for metric in METRICS:
            values = {p: float(results[p]["metrics"][metric]) for p in points}
            if not all(math.isfinite(v) and v > 0 for v in values.values()):
                raise ValueError("Invalid flow objective")
            m_changes = [
                values[mesh - i, domain] - values[mesh - i - 1, domain] for i in range(count)
            ]
            d_changes = [
                values[mesh, domain - i] - values[mesh, domain - i - 1] for i in range(count)
            ]
            mesh_delta, domain_delta = max(map(abs, m_changes)), max(map(abs, d_changes))
            noise_delta = noise * max(values.values())
            oscillating = any(
                len(changes) == 2
                and changes[0] * changes[1] < 0
                and min(map(abs, changes)) > noise_delta
                for changes in (m_changes, d_changes)
            )
            answer[metric] = {
                "mesh_delta": mesh_delta,
                "domain_delta": domain_delta,
                "noise_delta": noise_delta,
                "mesh_signed_changes": m_changes,
                "domain_signed_changes": d_changes,
                "oscillating": oscillating,
                "absolute_allowance": mesh_delta + domain_delta + noise_delta,
                "relative_change": (mesh_delta + domain_delta + noise_delta) / values[mesh, domain],
            }
        answer["eligible"] = all(
            answer[m]["relative_change"] <= tolerance and not answer[m]["oscillating"]
            for m in METRICS
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        answer["reason"] = "Missing or invalid measured force/noise evidence"
    return answer


def adaptive_flow(
    source: Path,
    directory: Path,
    settings: FlowSettings,
    tolerance: float,
    cancelled: Callable[[], bool],
    *,
    cache_root: Path,
    identity: dict[str, Any],
    minimum_level: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    from forge_mind.workspace.flow_stage_cache import FlowStageCache

    cache = FlowStageCache(
        source,
        cache_root,
        {**identity, "settings": settings.model_dump(), "tolerance": tolerance},
        cancelled,
        settings.resources.max_total_seconds,
    )
    directory.mkdir(parents=True, exist_ok=True)
    runs: dict[tuple[int, int], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    sensitivity: dict[str, Any] = {"eligible": False, "tolerance": tolerance, "scope": SCOPE}
    stop_reason = None
    mesh, domain = max(1, minimum_level[0]), max(1, minimum_level[1])
    refinements = 0
    grading_control: dict[str, Any] | None = None

    def snapshot_result(result: dict[str, Any], location: Path, label: str, reused: bool) -> None:
        snapshot = directory / label
        snapshot.mkdir()
        for name, receipt in result.get("artifacts", {}).items():
            original = location / name
            if Path(name).name != name or file_sha256(original) != receipt["sha256"]:
                raise ValueError("Cached flow artifact changed during capture")
            copy_artifact(original, snapshot / name)
            if file_sha256(snapshot / name) != receipt["sha256"]:
                raise ValueError("Flow artifact changed during capture")
        (snapshot / "stage-receipt.json").write_text(
            json.dumps({"result": result, "reused": reused}, allow_nan=False), encoding="utf-8"
        )

    def solve(point: tuple[int, int]) -> None:
        if point in runs:
            return
        if cancelled():
            raise ValueError("Flow refinement cancelled; completed stages retained")
        candidate = effective_settings(settings, *point)
        cache.update_progress(mesh_level=point[0], domain_level=point[1])
        result, location, reused = cache.run(candidate, f"Mesh {point[0]} / domain {point[1]}")
        runs[point] = result
        # Freeze all consumed evidence into this candidate attempt. Cache changes can
        # never silently alter an already registered study evaluation.
        snapshot_result(result, location, f"m{point[0]}d{point[1]}", reused)
        if result.get("eligible") is not True:
            raise ValueError("Numerical stage failed: " + "; ".join(result.get("errors", [])))

    try:
        if settings.mesh.fine_scale > 1 / 1.1:
            raise ValueError("Adaptive mesh steps need a refinement ratio of at least 1.1")
        if settings.mesh.grading_rate > 0:
            # Declare and measure the grading perturbation separately at fixed
            # extent. It consumes the same sensitivity budget, never a free gain.
            solve((0, 0))
            uniform = effective_settings(settings, 0, 0)
            uniform.mesh.grading_rate = 0
            control, location, reused = cache.run(uniform, "Fixed-extent uniform grading control")
            snapshot_result(control, location, "grading-control", reused)
            if control.get("eligible") is not True:
                raise ValueError("Uniform grading control did not converge")
            grading_control = {"result": control, "relative_allowance": {}}
            for metric in METRICS:
                a, b = control["metrics"][metric], runs[0, 0]["metrics"][metric]
                noise = sum(
                    r["convergence"]["force_relative_spread"] * abs(r["metrics"][metric])
                    for r in (control, runs[0, 0])
                )
                allowance = (abs(a - b) + noise) / max(abs(a), 1e-12)
                grading_control["relative_allowance"][metric] = allowance
                if not math.isfinite(allowance) or allowance > tolerance / 2:
                    raise ValueError(
                        "Farfield grading perturbation exceeds half the frozen sensitivity budget"
                    )
        while True:
            for point in ((mesh - 1, domain), (mesh, domain - 1), (mesh, domain)):
                solve(point)
            sensitivity = assess_cross(runs, mesh, domain, tolerance, confirmation=False)
            if sensitivity["eligible"]:
                # Two successive changes per axis at the SAME terminal state.
                mesh, domain = max(2, mesh), max(2, domain)
                for point in (
                    (mesh - 2, domain),
                    (mesh - 1, domain),
                    (mesh, domain - 2),
                    (mesh, domain - 1),
                    (mesh, domain),
                ):
                    solve(point)
                sensitivity = assess_cross(runs, mesh, domain, tolerance, confirmation=True)
            if grading_control and "drag_n" in sensitivity:
                for metric in METRICS:
                    allowance = grading_control["relative_allowance"][metric]
                    sensitivity[metric]["grading_relative_allowance"] = allowance
                    sensitivity[metric]["relative_change"] += allowance
                    sensitivity[metric]["absolute_allowance"] += allowance * abs(
                        runs[mesh, domain]["metrics"][metric]
                    )
                sensitivity["eligible"] = sensitivity["eligible"] and all(
                    sensitivity[m]["relative_change"] <= tolerance for m in METRICS
                )
            history.append(sensitivity)
            cache.update_progress(sensitivity=sensitivity)
            if sensitivity["eligible"]:
                break
            if "reason" in sensitivity:
                stop_reason = sensitivity["reason"]
                break
            if any(sensitivity[m]["oscillating"] for m in METRICS):
                stop_reason = "Controlled differences reverse above measured solver noise"
                break
            if (
                settings.adaptation.max_refinements is not None
                and refinements >= settings.adaptation.max_refinements
            ):
                stop_reason = "Explicit refinement promotion budget exhausted"
                break
            current = sensitivity["drag_n"]
            if current["noise_delta"] / abs(runs[mesh, domain]["metrics"]["drag_n"]) >= tolerance:
                stop_reason = "Solver noise exhausts the frozen sensitivity budget; tighten numerical convergence"
                break
            axis = (
                "mesh_delta" if current["mesh_delta"] >= current["domain_delta"] else "domain_delta"
            )
            # Stop actual stagnation rather than silently running an unproductive
            # unbounded sequence. A first failure always gets further refinement.
            other_level = "domain_level" if axis == "mesh_delta" else "mesh_level"
            same_axis = [
                h
                for h in history[:-1]
                if h.get("promoted_axis") == axis and h[other_level] == sensitivity[other_level]
            ]
            if (
                len(same_axis) >= 2
                and all(current[axis] >= h["drag_n"][axis] for h in same_axis[-2:])
                and current[axis] > current["noise_delta"]
            ):
                stop_reason = (
                    "Observed sensitivity is not contracting after repeated controlled refinements"
                )
                break
            sensitivity["promoted_axis"] = axis
            mesh += axis == "mesh_delta"
            domain += axis == "domain_delta"
            refinements += 1
    except (ValueError, OSError, OverflowError) as exc:
        stop_reason = str(exc)

    terminal = (mesh, domain)
    final = runs.get(terminal, {})
    eligible = (
        stop_reason is None
        and sensitivity.get("eligible") is True
        and sensitivity.get("confirmation") is True
    )
    if not eligible:
        sensitivity = {**sensitivity, "eligible": False}
    field_stage = None
    # Final prescribed state first; an earlier field is explicitly diagnostic.
    field_points = list(reversed(runs))
    if terminal in field_points:
        field_points.remove(terminal)
        field_points.insert(0, terminal)
    for point in field_points:
        result = runs[point]
        receipt = result.get("artifacts", {}).get("surface_field.json")
        original = directory / f"m{point[0]}d{point[1]}" / "surface_field.json"
        if receipt and original.is_file() and file_sha256(original) == receipt["sha256"]:
            copy_artifact(original, directory / "surface_field.json")
            field_stage = f"mesh {point[0]} / domain {point[1]}"
            break
    cache.update_progress(
        phase="complete" if eligible else "stopped",
        stop_reason=stop_reason,
        sensitivity=sensitivity,
    )
    return {
        **final,
        "metrics": final.get("metrics", {}),
        "provenance": final.get(
            "provenance",
            {
                "source_step_sha256": getattr(cache, "source_hash", None),
                "solver_sha256": getattr(cache, "solver_hash", None),
            },
        ),
        "eligible": eligible,
        "status": "complete" if eligible else "unverified",
        "errors": [stop_reason] if stop_reason else final.get("errors", []),
        "sensitivity": sensitivity,
        "sensitivity_history": history,
        "grading_control": grading_control,
        "metric_uncertainty": {
            m: sensitivity[m]["absolute_allowance"] for m in METRICS if m in sensitivity
        },
        "final_validation": {
            "protocol": settings.adaptation.protocol,
            "mesh_level": mesh,
            "domain_level": domain,
            "reference_length_m": settings.mesh.reference_length_m or settings.reference_length_m,
            "confirmed": eligible,
        },
        "refinement_runs": [
            {"stage": f"m{m}d{d}", "result": result} for (m, d), result in runs.items()
        ],
        "field_stage": field_stage,
        "field_scope": (
            f"{field_stage}; observed sensitivity checked; physical validity unverified"
            if eligible
            else f"diagnostic {field_stage}; refinement unverified"
        )
        if field_stage
        else None,
    }
