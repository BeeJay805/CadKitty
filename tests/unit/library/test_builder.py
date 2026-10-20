"""Library builder: deterministic enumeration + the NO-FALSE-PASS invariant.

The library's whole value is that a false-PASS poisons it, so the central test re-derives and
re-validates every candidate independently and asserts manifest membership EQUALS report.passed
(strict: overall == PASS). Determinism + full-catalog enumeration are also pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.library import FAMILIES, build_library
from forge_mind.library.builder import _validate_clean
from forge_mind.scaling.enumerate import enumerate_shapes

# This whole module compiles + multi-seed-validates many real parts (~50s) -- the heaviest unit
# file by far. Marked slow so the `-m "not slow and not llm"` inner loop stays quick; it still
# runs in the full gate (`-m "not llm"`) and CI, so the no-false-PASS guard is never skipped.
pytestmark = pytest.mark.slow


def test_enumerators_are_deterministic_and_unique() -> None:
    for _fam, enum in FAMILIES.items():
        first = [c.cid for c in enum()]
        second = [c.cid for c in enum()]
        assert first == second  # order-stable
        assert len(first) == len(set(first))  # unique ids within the family


def test_composable_enumerates_the_full_catalog() -> None:
    # The composable family is the existing 102-set's source: every enumerated shape (137).
    assert len(list(FAMILIES["composable_plate"]())) == len(enumerate_shapes())


def test_build_is_deterministic(tmp_path: Path) -> None:
    # The robust seeded keep gate makes the manifest byte-stable across runs.
    a = build_library(tmp_path / "a", families=["electronics_tray", "computed_body"])
    b = build_library(tmp_path / "b", families=["electronics_tray", "computed_body"])
    assert [(p.id, p.mass_g, p.dims_mm, p.tiers_passed) for p in a.parts] == [
        (p.id, p.mass_g, p.dims_mm, p.tiers_passed) for p in b.parts
    ]
    assert [(s.family, s.passed, s.attempted) for s in a.families] == [
        (s.family, s.passed, s.attempted) for s in b.families
    ]


def test_no_false_pass_membership_equals_clean_validation(tmp_path: Path) -> None:
    # The invariant: a candidate is in the manifest IFF it re-derives + passes the SAME robust
    # seeded keep gate (_validate_clean: every seed clean). No part the validator does not
    # robustly clean-pass can enter -- a false-PASS would poison the library.
    manifest = build_library(tmp_path / "lib", families=["computed_body"])
    kept = {p.id for p in manifest.parts}
    rejected = {r.id for r in manifest.rejects}
    assert kept and not (kept & rejected)  # disjoint, non-empty
    seen = 0
    for cand in FAMILIES["computed_body"]():
        pid = f"{cand.family}::{cand.cid}"
        program = interpret_layout(cand.make())
        model = compile_program(program, tmp_path / "rev" / cand.cid)
        report, _failing = _validate_clean(model, program)
        if report is not None:
            assert pid in kept, f"{pid} validates clean but was not kept"
        else:
            assert pid in rejected, f"{pid} is not clean but was kept (FALSE-PASS)"
        seen += 1
    assert seen == sum(s.attempted for s in manifest.families)


def test_kept_parts_have_clean_tier_records(tmp_path: Path) -> None:
    manifest = build_library(tmp_path / "lib", families=["electronics_tray"])
    assert manifest.parts
    for part in manifest.parts:
        assert set(part.tiers_passed) <= {"mesh", "features", "printability", "interfaces"}
        assert "mesh" in part.tiers_passed and "printability" in part.tiers_passed
        assert part.mass_g > 0.0
        assert all(v > 0.0 for v in part.dims_mm.values())
