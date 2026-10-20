"""structures.selection: deterministic winner selection, fail-closed on unproven axes.

Ported from tests/unit/arm_search/test_m1_search.py (ADR-0097 consolidation).
"""

from __future__ import annotations

from typing import Any

from forge_mind.structures.selection import beats_reference, select_winner


def _cand(**kw: Any) -> dict[str, Any]:
    base = {
        "built": True,
        "geometry_passed": True,
        "within_deflection_bound": True,
        "lighter_than_reference": True,
        "mass_g": 100.0,
    }
    base.update(kw)
    return base


def test_select_ranks_survivors_by_mass_and_ignores_losers() -> None:
    field = [
        _cand(mass_g=120.0),
        _cand(mass_g=110.0),
        _cand(lighter_than_reference=False, mass_g=160.0),
        _cand(within_deflection_bound=False, mass_g=90.0),
        _cand(built=False, mass_g=None),
        _cand(within_deflection_bound=None, mass_g=95.0),  # unproven -> fail-closed
    ]
    out = select_winner(field)
    assert out["n_candidates"] == 6
    assert out["n_survivors"] == 2
    assert out["winner"]["mass_g"] == 110.0


def test_select_no_survivors_is_none() -> None:
    out = select_winner([_cand(lighter_than_reference=False), _cand(built=False)])
    assert out["winner"] is None
    assert out["n_survivors"] == 0


def test_beats_reference_requires_all_four_and_is_fail_closed() -> None:
    assert beats_reference(_cand()) is True
    for kill in (
        {"built": False},
        {"geometry_passed": False},
        {"within_deflection_bound": False},
        {"within_deflection_bound": None},
        {"lighter_than_reference": False},
        {"lighter_than_reference": None},
    ):
        assert beats_reference(_cand(**kill)) is False
