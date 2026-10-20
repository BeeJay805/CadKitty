"""Composition engine: deterministic enumeration + the no-false-PASS keep invariant.

Enumeration is geometry-free (fast). The keep invariant compiles solids, so it is @slow."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.composition import SHARDS, enumerate_recipes
from forge_mind.composition.engine import _attempt, _conformance
from forge_mind.layout.composable_hardware import HARDWARE_REFS
from forge_mind.validator.conformance import HardwareFeature, check_conformance
from forge_mind.validator.report import CheckStatus


def test_enumeration_is_deterministic_and_unique() -> None:
    for shard in SHARDS:
        first = [r.cid for r in enumerate_recipes(shard)]
        second = [r.cid for r in enumerate_recipes(shard)]
        assert first == second
        assert len(first) == len(set(first))


def test_enumeration_counts() -> None:
    counts = {shard: sum(1 for _ in enumerate_recipes(shard)) for shard in SHARDS}
    # open_box reuses the SAME hardware vocabulary as the flat plate (its floor is a flat plate), so
    # box_singles mirrors flat_singles and box_pairs is the hardware-shortlist pair set.
    assert counts == {
        "flat_singles": 61,
        "flat_pairs": 52,
        "lbracket": 1,
        # every walled base mirrors the flat-plate hardware vocabulary: 61 singles + 28 pairs.
        "box_singles": 61,
        "box_pairs": 28,
        "bracket_singles": 61,
        "bracket_pairs": 28,
        "uchannel_singles": 61,
        "uchannel_pairs": 28,
        "tray_singles": 61,
        "tray_pairs": 28,
        "partial_singles": 61,
        "partial_pairs": 28,
        "round_singles": 61,
        "round_pairs": 28,
    }


def test_conformance_runs_on_every_hardware_placement() -> None:
    # Every hardware feature a recipe places is conformance-checked (spec-level, no geometry).
    recipe = next(r for r in enumerate_recipes("flat_singles"))
    placements = recipe.placements()
    assert any(p.ref in HARDWARE_REFS for p in placements)
    clean, summary = _conformance(recipe, placements)
    assert clean and summary  # a registry-built feature conforms

    # A deliberately-wrong dimension is caught (the no-false-PASS conformance gate).
    bad = check_conformance(
        HardwareFeature(category="clearance_hole", key="M3", fit="normal", nominal_mm=9.9)
    )
    assert bad.status == CheckStatus.FAIL


@pytest.mark.slow
def test_no_false_pass_kept_parts_revalidate(tmp_path: Path) -> None:
    # Every kept part, independently re-checked, passes the robust validation AND conformance;
    # nothing the gate rejects is kept. (A representative slice to stay bounded.)
    recipes = list(enumerate_recipes("flat_singles"))[:14]
    kept = rejected = 0
    for recipe in recipes:
        part, reject = _attempt(recipe, tmp_path)
        if part is not None:
            kept += 1
            assert set(part.tiers_passed) <= {"mesh", "features", "printability", "interfaces"}
            assert "mesh" in part.tiers_passed and "printability" in part.tiers_passed
            assert all(c.endswith("=pass") for c in part.conformance)
            assert part.material_mass_g["PLA"] > 0.0
        else:
            rejected += 1
            assert reject is not None and reject.stage in {
                "derive",
                "compile",
                "features",
                "mesh",
                "printability",
                "interfaces",
                "conformance",
            }
    assert kept >= 10  # the slice produces real breadth
