"""Implicit routing: only lattice/TPMS prompts match; params parse with sane defaults (ADR-0026)."""

from __future__ import annotations

from forge_mind.implicit.router import build_implicit, implicit_match


def test_matches_gyroid_with_params() -> None:
    r = implicit_match("a 50mm gyroid lattice cube with 12 mm cell and 2 mm wall")
    assert r is not None
    assert r.archetype == "gyroid"
    assert r.size_mm == 50.0
    assert r.cell_mm == 12.0
    assert r.feature_mm == 2.0


def test_matches_strut_lattice() -> None:
    r = implicit_match("a strut lattice block, 40 mm, strut diameter 3 mm, 10mm cell")
    assert r is not None
    assert r.archetype == "strut_lattice"
    assert r.feature_mm == 3.0
    assert r.size_mm == 40.0


def test_schwarz_wins_over_generic() -> None:
    r = implicit_match("a schwarz-p tpms minimal surface structure")
    assert r is not None
    assert r.archetype == "schwarz_p"


def test_normal_parts_do_not_match() -> None:
    assert implicit_match("a NEMA-17 motor mount bracket") is None
    assert implicit_match("a phone stand that leans back 15 degrees") is None
    assert implicit_match("") is None
    assert implicit_match(None) is None


def test_defaults_when_unspecified() -> None:
    r = implicit_match("make me a gyroid")
    assert r is not None
    assert r.size_mm == 40.0
    assert r.cell_mm == 10.0
    assert r.feature_mm == 1.6


def test_cell_only_does_not_become_size() -> None:
    # "8mm cell" with no stated size must fall back to the default size, not adopt the cell value.
    r = implicit_match("a gyroid with an 8mm cell")
    assert r is not None
    assert r.cell_mm == 8.0
    assert r.size_mm == 40.0


def test_build_dispatch_produces_watertight_part() -> None:
    r = implicit_match("a 20 mm gyroid, 8mm cell, 1.4mm wall")
    assert r is not None
    part = build_implicit(r)
    assert part.archetype == "gyroid"
    assert part.watertight
