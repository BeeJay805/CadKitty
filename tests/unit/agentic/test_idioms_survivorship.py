"""Verified idiom pointers + the survivorship guard (ADR-0117): idioms ride only on
triggered findings; the guard names a detail attempt that died while the kept best lacks it
(measured mechanism: detail dies in the sandbox, blockiness ships)."""

from __future__ import annotations

from forge_mind.agentic.idioms import idiom_pointer
from forge_mind.agentic.loop import _survivorship_note


def test_idiom_pointer_selects_by_keyword() -> None:
    assert "PolarLocations" in (idiom_pointer("mounting_bosses") or "")
    assert "revolve" in (idiom_pointer("shallow V groove on the rim") or "")
    assert "sweep()" in (idiom_pointer("curved hook with rounded tip") or "")
    assert idiom_pointer("a plain rectangular plate") is None


def test_idioms_are_execution_verified_claims() -> None:
    # every idiom advertises its verification -- the contract that nothing unverified ships
    for text in ("boss", "groove", "hook"):
        assert "VERIFIED" in (idiom_pointer(text) or "")


def test_survivorship_note_names_the_dead_attempt() -> None:
    failed = "with BuildLine():\n    CenterArc((0,0), 12, 0, 180)\nsweep()"
    best = "Box(50, 40, 10)"
    note = _survivorship_note(failed + "\nsweep(", best)
    assert note is not None and "sweep" in note and "still required" in note


def test_survivorship_silent_when_best_already_has_the_op() -> None:
    failed = "revolve(axis=Axis.Z)"
    best = "revolve(axis=Axis.Z)  # working"
    assert _survivorship_note(failed, best) is None


def test_l70_tube_keywords_reach_the_swept_idiom() -> None:
    """Generation 3 crashed 4/6 iterations calling sweep() blind; runner/tube briefs must
    now reach the swept construction."""
    assert "sweep()" in (idiom_pointer("four curved runner tubes sweeping upwards") or "")
    assert "BuildLine" in (idiom_pointer("24 mm outer diameter tube") or "")


def test_junction_idiom_outranks_the_swept_idiom_on_manifold_text() -> None:
    """A manifold mentions both 'collector' and 'tube'; the merged-junction correction is
    the load-bearing one there and must win the first-match scan."""
    hit = idiom_pointer("runners converge into a single collector tube") or ""
    assert "MERGED-JUNCTION" in hit
