"""Flow screening cannot become a ranked result at unmatched numerical resolution."""

from __future__ import annotations

import math
from typing import Any


def flow_result(candidate: dict[str, Any]) -> dict[str, Any]:
    return next((e["result"] for e in candidate.get("evaluations", []) if e["kind"] == "flow"), {})


def validation_level(candidate: dict[str, Any]) -> tuple[int, int] | None:
    validation = flow_result(candidate).get("final_validation", {})
    reference = validation.get("reference_length_m")
    if (
        validation.get("confirmed") is not True
        or validation.get("protocol") != "controlled_cross_v1"
        or type(reference) not in (float, int)
        or not math.isfinite(reference)
        or reference <= 0
    ):
        return None
    m, d = validation.get("mesh_level"), validation.get("domain_level")
    if type(m) is not int or type(d) is not int or m < 2 or d < 2:
        return None
    return m, d


def common_flow_validation(candidates: list[dict[str, Any]], baseline_id: str) -> bool:
    baseline = next((c for c in candidates if c["id"] == baseline_id), None)
    if baseline is None or validation_level(baseline) is None:
        return False
    signature = flow_result(baseline)["final_validation"]
    return all(
        validation_level(c) is not None and flow_result(c).get("final_validation") == signature
        for c in candidates
    )


def rank_stable(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Reject a resolved ranking reversal across the final controlled cross."""
    level = validation_level(baseline)
    if level is None:
        return False
    m, d = level
    points = [f"m{m}d{d}", f"m{m - 1}d{d}", f"m{m - 2}d{d}", f"m{m}d{d - 1}", f"m{m}d{d - 2}"]
    runs = [
        {r["stage"]: r["result"] for r in flow_result(c).get("refinement_runs", [])}
        for c in (baseline, candidate)
    ]
    if any(p not in r for r in runs for p in points):
        return False
    differences = []
    for p in points:
        values = [r[p]["metrics"]["drag_n"] for r in runs]
        noise = sum(
            abs(value) * r[p]["convergence"]["force_relative_spread"]
            for value, r in zip(values, runs, strict=True)
        )
        difference = values[0] - values[1]
        differences.append(difference if abs(difference) > noise else 0)
    return not (any(d > 0 for d in differences) and any(d < 0 for d in differences))
