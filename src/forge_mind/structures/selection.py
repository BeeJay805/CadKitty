"""Deterministic winner selection (promoted from scripts/arm_search/m1_search.py).

Survivors = candidates that BEAT the reference on every required axis, ranked by mass. Fail-closed
on any unproven axis: a ``None`` (unmeasured mass, unproven stiffness) never counts as a win.
Selection is pure code -- models never decide winners (the PrintCell constitution).
"""

from __future__ import annotations

from typing import Any

Candidate = dict[str, Any]


def beats_reference(r: Candidate) -> bool:
    """A candidate beats the reference iff it built + passed geometry + measured deflection is
    within the bound AND it is lighter. ``is True`` on the tri-state axes rejects None (unproven)."""
    return bool(
        r.get("built")
        and r.get("geometry_passed")
        and r.get("within_deflection_bound") is True
        and r.get("lighter_than_reference") is True
    )


def select_winner(results: list[Candidate]) -> dict[str, Any]:
    """Survivors that beat the reference, ranked by mass; the lightest survivor wins."""
    survivors = sorted(
        (r for r in results if beats_reference(r)),
        key=lambda r: float(r["mass_g"]),
    )
    return {
        "winner": survivors[0] if survivors else None,
        "n_survivors": len(survivors),
        "n_candidates": len(results),
        "field": results,
    }


__all__ = ["Candidate", "beats_reference", "select_winner"]
