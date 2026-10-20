"""Search anchors are work choices, never new feasibility or acceptance verdicts."""

from __future__ import annotations

import math
from typing import Any

from forge_mind.workspace.design_study_models import DesignStudySpec


def _number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _score(
    spec: DesignStudySpec, candidate: dict[str, Any]
) -> tuple[tuple[float, ...], tuple[float, ...], list[dict[str, Any]]] | None:
    if (
        candidate.get("status") != "complete"
        or candidate.get("eligible") is not True
        or not isinstance(candidate.get("id"), str)
        or not candidate["id"]
        or candidate.get("original_requirement_failures") != []
    ):
        return None
    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict):
        return None
    objectives = []
    for objective in spec.objectives:
        measured = _number(metrics.get(objective.metric))
        if measured is None:
            return None
        objectives.append(measured if objective.direction == "minimize" else -measured)
    residuals = []
    violations = []
    for constraint in spec.constraints:
        measured = _number(metrics.get(constraint.metric))
        if measured is None:
            return None
        residual = max(
            0.0,
            constraint.value - measured
            if constraint.operator == "ge"
            else measured - constraint.value,
        )
        if not math.isfinite(residual):
            return None
        residuals.append(residual)
        violations.append(
            {
                "metric": constraint.metric,
                "operator": constraint.operator,
                "value": constraint.value,
                "measured": measured,
                "residual": residual,
            }
        )
    return tuple(residuals), tuple(objectives), violations


def choose_search_anchor(
    spec: DesignStudySpec,
    candidates: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Use verified study reads; residuals retain metric units and frozen constraint order."""
    unavailable: dict[str, Any] = {
        "candidate_id": None,
        "status": "unavailable",
        "reason": "No complete eligible search anchor has all required finite metrics and no owner failures",
        "violations": [],
    }
    flow = any(evaluator.kind == "flow" for evaluator in spec.evaluators)
    if flow and comparison.get("conclusion") not in {"baseline_best", "improved"}:
        return {
            **unavailable,
            "reason": "Flow search requires a comparable feasible best from the existing verified comparison",
        }
    scored = [
        (candidate, score)
        for candidate in candidates
        if (score := _score(spec, candidate)) is not None
    ]
    best_id = comparison.get("best_candidate_id")
    for candidate, (residuals, _, violations) in scored:
        if candidate["id"] == best_id and candidate.get("feasible") is True and not any(residuals):
            return {
                "candidate_id": candidate["id"],
                "status": "feasible",
                "reason": "Use the existing comparison's feasible best; no new acceptance verdict",
                "violations": violations,
            }
    if flow or not scored:
        return unavailable
    candidate, (_, _, violations) = min(scored, key=lambda row: (row[1][0], row[1][1]))
    return {
        "candidate_id": candidate["id"],
        "status": "search_only",
        "reason": "Lexicographic constraint residuals, then objectives, then ledger order; feasibility is unchanged",
        "violations": violations,
    }
