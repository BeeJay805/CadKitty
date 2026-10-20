"""Tests for the held-out problem generator's HARD families (added 2026-07-04).

The pass@8 headroom probe (TEST_LOG row 25) came back null because the holdout was all easy
prismatic parts (pass@1 saturated at 100%). The hard families (flange / indexer / manifold) are
multi-hole-group parts mirroring the arbitrary_v1 secondary-hole-drop failure classes, so pass@1
drops into a band where a best-of-8 lift is measurable. These pin their shape + determinism +
disjointness so a regeneration can't silently drift back to easy.
"""

from __future__ import annotations

import random
import sys
from itertools import pairwise
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import gen_holdout_problems as gh  # noqa: E402


def test_hard_difficulty_uses_hard_families() -> None:
    prompts = gh._gen_cad(random.Random(7351), 12, "hard")
    classes = {p["class"] for p in prompts}
    assert classes == {"flange", "indexer", "manifold"}
    # every hard prompt has MORE THAN ONE hole group -- the completeness pressure that makes it hard
    assert all(len(p["expect"]["holes"]) >= 2 for p in prompts)
    # and at least one class carries THREE groups (bore + two hole sets)
    assert any(len(p["expect"]["holes"]) >= 3 for p in prompts)


def test_easy_difficulty_is_unchanged_default() -> None:
    prompts = gh._gen_cad(random.Random(7351), 12, "easy")
    assert {p["class"] for p in prompts} == {"bracket", "plate", "mount", "standoff"}
    # default arg matches "easy" (backward compatibility with existing snapshots)
    assert [p["id"] for p in gh._gen_cad(random.Random(7351), 8)] == [
        p["id"] for p in gh._gen_cad(random.Random(7351), 8, "easy")
    ]


def test_hard_rubrics_are_axial_and_satisfiable_shape() -> None:
    # Every hole group is a positive diameter with a positive min_count and a tolerance the bench
    # re-measures against -- no group is unmeasurable-by-construction (which would false-fail a
    # correct part). Diameter bands within a prompt must not overlap (else the count is ambiguous).
    for p in gh._gen_cad(random.Random(4242), 12, "hard"):
        groups = p["expect"]["holes"]
        for g in groups:
            assert g["diameter_mm"] > 0 and g["min_count"] >= 1 and g["tol_mm"] > 0
        bands = sorted(
            (g["diameter_mm"] - g["tol_mm"], g["diameter_mm"] + g["tol_mm"]) for g in groups
        )
        for lo_hi, next_lo_hi in pairwise(bands):
            assert lo_hi[1] < next_lo_hi[0], f"overlapping diameter bands in {p['id']}: {bands}"


def test_hard_set_is_deterministic() -> None:
    a = gh._gen_cad(random.Random(999), 9, "hard")
    b = gh._gen_cad(random.Random(999), 9, "hard")
    assert [p["prompt"] for p in a] == [p["prompt"] for p in b]


def test_hard_set_is_disjoint_from_frozen(tmp_path: Path) -> None:
    # the real disjointness gate (ids + near-dup Jaccard vs every frozen prompt) must pass on hard
    cad = gh._gen_cad(random.Random(7351), 12, "hard")
    inv = gh._gen_invention(random.Random(7351), 6)
    gh._assert_disjoint(cad, inv)  # raises SystemExit on any collision / near-dup


def test_generator_writes_loadable_hard_snapshot(tmp_path: Path) -> None:
    rng = random.Random(2024)
    cad = gh._gen_cad(rng, 12, "hard")
    inv = gh._gen_invention(rng, 6)
    gh._write(tmp_path, 2024, cad, inv, "hard")
    n_inv, n_cad = gh._verify_loadable(tmp_path)  # both bench loaders + MANIFEST hash
    assert n_cad == 12 and n_inv == 6
    import json

    doc = json.loads((tmp_path / "arbitrary_holdout_v1.json").read_text(encoding="utf-8"))
    assert doc["difficulty"] == "hard"


def test_unknown_difficulty_falls_back_to_easy() -> None:
    # _gen_cad is permissive (the CLI validates choices); an unexpected value must not crash
    assert {p["class"] for p in gh._gen_cad(random.Random(1), 4, "bogus")} == {
        "bracket",
        "plate",
        "mount",
        "standoff",
    }
