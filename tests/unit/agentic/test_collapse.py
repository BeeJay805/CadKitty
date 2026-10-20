"""Envelope-collapse detection (L102 class): the lineage signature the existing restart
triggers cannot see — sandbox builds fine, geometry is a sliver of the stated envelope.
Cutoff 0.30 is data-grounded (67 archived built iterations: legit p10=0.78, collapses
<=0.114). The fixtures below use the REAL measured cases."""

from __future__ import annotations

from forge_mind.agentic.collapse import (
    build_ratio,
    envelope_collapsed,
    envelope_dims_mm,
    lineage_all_collapsed,
)


def test_real_collapse_cases_read_below_cutoff() -> None:
    """chest_plate iter_02 (kept-best!) and thigh_r's final sliver — the measured class."""
    chest = build_ratio([6.0, 4.5, 4.5], "34 x 105 x 188 mm")
    thigh = build_ratio([5.0, 5.0, 26.0], "34 x 44 x 55 mm")
    assert chest is not None and chest < 0.30
    assert thigh is not None and thigh < 0.30


def test_legitimate_builds_read_above_cutoff() -> None:
    """A part filling its envelope (the archived p10 was 0.78; median 1.0)."""
    r = build_ratio([180.0, 100.0, 30.0], "188 x 105 x 34 mm")
    assert r is not None and r > 0.78


def test_round_part_envelope_form_is_supported() -> None:
    """Review-caught: '76 mm dia x 14 mm tall' must parse (the ad-hoc regex missed it)."""
    assert envelope_dims_mm("76 mm dia x 14 mm tall") == (76.0, 76.0, 14.0)
    collapsed_puck = build_ratio([6.0, 6.0, 2.0], "76 mm dia x 14 mm tall")
    assert collapsed_puck is not None and collapsed_puck < 0.30


def test_streak_logic_two_consecutive_built_iterations() -> None:
    assert envelope_collapsed([0.03, 0.03]) is True
    assert envelope_collapsed([0.03]) is False  # one iteration is not a lineage signature
    assert envelope_collapsed([0.03, None, 0.03]) is False  # unbuilt iteration breaks it
    assert envelope_collapsed([1.0, 0.9]) is False
    assert envelope_collapsed([0.9, 0.03, 0.03]) is True  # only the tail matters


def test_unmeasurable_inputs_fail_open() -> None:
    assert build_ratio(None, "34 x 105 x 188 mm") is None
    assert build_ratio([1.0, 2.0, 3.0], None) is None
    assert build_ratio([1.0, 2.0, 3.0], "no numbers here") is None
    assert envelope_collapsed([]) is False


def test_lineage_all_collapsed_gates_the_incumbent_reset() -> None:
    """Review-probed defect guard: only an ALL-collapsed lineage may drop the incumbent —
    a mixed lineage keeps its genuine best; unbuilt iterations don't count either way."""
    assert lineage_all_collapsed([0.03, 0.05]) is True
    assert lineage_all_collapsed([None, 0.03]) is True  # the only BUILT iteration collapsed
    assert lineage_all_collapsed([0.9, 0.03, 0.03]) is False  # a real best exists — keep it
    assert lineage_all_collapsed([]) is False
    assert lineage_all_collapsed([None, None]) is False


def test_cutoff_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_COLLAPSE_RATIO", "0.05")
    assert envelope_collapsed([0.06, 0.06]) is False  # 0.06 above the tightened cutoff
    monkeypatch.setenv("FORGE_COLLAPSE_RATIO", "not-a-number")
    assert envelope_collapsed([0.29, 0.29]) is True  # falls back to 0.30
