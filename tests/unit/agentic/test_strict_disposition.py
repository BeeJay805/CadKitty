"""Strict disposition: a caveat must not absorb PHYSICAL IMPOSSIBILITY.

H3 shipped at `shipped_with_caveats` while the engine's own report recorded it not-seated, with a
floating part and a pin fit its own note called "will bind / not assemble". The numbers below are
that run's, verbatim, from benchmarks/2026-07-23_humanoid_probe/H3/functional_report.json.

Half of these tests pin the EXCLUSIONS. Every one of them is a flag that looks like it means
"impossible" and does not; each cost a review round, and promoting any of them would introduce a
false-FAIL, which this programme forbids as firmly as a false-PASS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.agentic.functional import apply_strict_disposition, unassemblable_reasons
from forge_mind.materials import resolve_material

_PLA = resolve_material("PLA")


def _fit(seated=True, floating=None, **kw):
    return {"seated": seated, "floating_parts": floating or [], "interference_mm3": 0.0, **kw}


def _motion(gap=0.30, **kw):
    return {
        "material_pin_fit": {"material": "PLA", "measured_clearance_mm": gap, "ok": True},
        **kw,
    }


def test_the_real_h3_assembly_is_refused() -> None:
    # THE CASE THIS EXISTS FOR: H3's own measured values -> three independent blockers.
    r = unassemblable_reasons(_fit(seated=False, floating=[6]), _motion(gap=0.10), _PLA)
    assert len(r) == 3, r
    assert any("does not seat" in x for x in r)
    assert any("float free" in x for x in r)
    assert any("bind / not assemble" in x for x in r)


def test_a_sound_assembly_has_no_reasons() -> None:
    assert unassemblable_reasons(_fit(), _motion(), _PLA) == []


def test_each_blocker_fires_alone() -> None:
    assert len(unassemblable_reasons(_fit(seated=False), _motion(), _PLA)) == 1
    assert len(unassemblable_reasons(_fit(floating=[2]), _motion(), _PLA)) == 1
    assert len(unassemblable_reasons(_fit(), _motion(gap=0.10), _PLA)) == 1


def test_an_over_wide_pin_fit_is_not_blocking() -> None:
    # EXCLUSION. materials.clearance_fit fails a gap above the band too ("loose / rattly"), but a
    # loose joint still assembles. Gating on the `ok` boolean would have blocked it.
    loose = unassemblable_reasons(_fit(), _motion(gap=5.0), _PLA)
    assert loose == [], loose


def test_the_pin_floor_matches_the_materials_module_threshold() -> None:
    # The bind branch is `gap < lo - 0.05`, not `gap < lo`: a gap marginally under the band is a
    # caveat, not an impossibility. Pinned so the two cannot drift apart.
    lo = _PLA.clearance_mm[0]
    assert unassemblable_reasons(_fit(), _motion(gap=lo - 0.04), _PLA) == []
    assert len(unassemblable_reasons(_fit(), _motion(gap=lo - 0.06), _PLA)) == 1


def test_excluded_flags_do_not_block() -> None:
    # EXCLUSIONS, each with the reason it is not an impossibility:
    #  - assembly_order.stuck: samples straight extraction only; failure does not PROVE it.
    #  - worst_case issues: flags zero-gap FIXED mates, where zero gap is intentional.
    #  - assembly_fails aggregate: includes merely rattly print-in-place gaps.
    #  - articulates False: also covers a joint that was never validated.
    #  - interference: a legitimate PRESS FIT is interpenetration in nominal geometry.
    noisy = _motion(
        assembly_fails=[{"id": "assembly.print_in_place_gap", "message": "gap too wide"}],
        articulates=False,
    )
    assert unassemblable_reasons(_fit(interference_mm3=717.6), noisy, _PLA) == []


def test_missing_reports_fail_open_never_manufacture_a_blocker() -> None:
    # An unrun check must never invent an impossibility: this decides what SHIPS.
    assert unassemblable_reasons(None, None, _PLA) == []
    assert unassemblable_reasons({}, {}, _PLA) == []
    assert unassemblable_reasons(_fit(), {"material_pin_fit": {"material": "PLA"}}, _PLA) == []


def test_a_non_finite_pin_gap_does_not_block() -> None:
    # NaN < x is False, so this would not have blocked anyway -- pinned so a future refactor
    # that flips the comparison cannot start rejecting on unmeasurable gaps.
    for bad in (float("nan"), None, "0.1"):
        assert unassemblable_reasons(_fit(), _motion(gap=bad), _PLA) == []


# --- the wiring itself, driven by the REAL archived H3 report ------------------------------
# unassemblable_reasons() was unit-tested above, but the branch that CALLS it lives inside a
# 3000-line function that needs a full generation run to exercise. apply_strict_disposition() is
# that branch, extracted so both shipping doors share ONE implementation and it can be driven
# here by H3's actual functional_report.json rather than a hand-made fixture.

_H3 = Path("E:/ForgeMind/benchmarks/2026-07-23_humanoid_probe/H3/functional_report.json")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("FORGE_STRICT_DISPOSITION", "1")


def test_on_by_default_blocks_an_unassemblable_ship(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_STRICT_DISPOSITION", raising=False)
    doomed = _fit(seated=False, floating=[6])
    for d in ("shipped", "shipped_with_caveats"):
        new, reasons = apply_strict_disposition(d, doomed, _motion(gap=0.10), _PLA)
        assert new == "not_assemblable" and reasons


@pytest.mark.skipif(not _H3.exists(), reason="H3 archive not present on this machine")
def test_the_real_archived_h3_report_is_refused(strict) -> None:
    """END-TO-END on real data: H3's own report, through the real decision function."""
    r = json.loads(_H3.read_text(encoding="utf-8"))
    assert r["disposition"] == "shipped_with_caveats", "the archived run shipped, as recorded"
    new, reasons = apply_strict_disposition(r["disposition"], r.get("fit"), r.get("motion"), _PLA)
    assert new == "not_assemblable", reasons
    assert len(reasons) == 3, reasons


def test_an_honest_rejection_is_never_relabelled(strict) -> None:
    # Only a SHIPPING disposition may be downgraded; rejected/no_build carry their own meaning.
    doomed = _fit(seated=False)
    for d in ("rejected", "no_build"):
        assert apply_strict_disposition(d, doomed, _motion(), _PLA) == (d, [])


def test_a_sound_assembly_still_ships_under_the_flag(strict) -> None:
    for d in ("shipped", "shipped_with_caveats"):
        assert apply_strict_disposition(d, _fit(), _motion(), _PLA) == (d, [])


@pytest.mark.skipif(not _H3.exists(), reason="H3 archive not present on this machine")
def test_explicit_rollback_still_ships_the_real_h3(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_STRICT_DISPOSITION", "0")
    r = json.loads(_H3.read_text(encoding="utf-8"))
    assert apply_strict_disposition(r["disposition"], r.get("fit"), r.get("motion"), _PLA) == (
        "shipped_with_caveats",
        [],
    )


def test_explicit_rollback_is_inert(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_STRICT_DISPOSITION", "0")
    doomed = _fit(seated=False, floating=[6])
    for d in ("shipped", "shipped_with_caveats", "rejected", "no_build"):
        assert apply_strict_disposition(d, doomed, _motion(gap=0.10), _PLA) == (d, [])


# --- moving vs merely-validated joints (the L89 misreading) ---------------------------------


def test_moving_joints_are_counted_separately_from_validated_ones() -> None:
    """H3 was logged as "13/13 joints full-range PASS". The engine had said "All 2 moving
    joint(s)": joints_validated counts joints EXAMINED, and a FIXED joint passes a sweep
    trivially because it has no range. The two counts now travel together."""
    from types import SimpleNamespace

    from forge_mind.agentic.functional import joint_mobility_counts

    # H3's real joint mix: 9 fixed, 2 coaxial_insert, 2 revolute -> 2 moving, 11 rigid.
    h3 = SimpleNamespace(
        parts=[SimpleNamespace(name=f"p{i}") for i in range(14)],
        joints=[
            SimpleNamespace(type=t)
            for t in ["fixed"] * 9 + ["coaxial_insert"] * 2 + ["revolute"] * 2
        ],
    )
    assert joint_mobility_counts(h3) == (2, 11)

    # A real mechanism: every joint moves.
    mech = SimpleNamespace(
        parts=[SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        joints=[SimpleNamespace(type="revolute"), SimpleNamespace(type="prismatic")],
    )
    assert joint_mobility_counts(mech) == (2, 0)

    # A statue: nothing moves at all.
    statue = SimpleNamespace(
        parts=[SimpleNamespace(name="a")], joints=[SimpleNamespace(type="fixed")]
    )
    assert joint_mobility_counts(statue) == (0, 1)


def test_undriven_dof_separates_a_mechanism_from_a_mannequin() -> None:
    """A sweep proves a joint does not COLLIDE, never that anything can MOVE it. H3 reported
    articulates=true with ZERO hardware declared, so every DOF it had was undriven."""
    from types import SimpleNamespace

    from forge_mind.agentic.functional import undriven_dof_lower_bound

    h3 = SimpleNamespace(
        parts=[SimpleNamespace(name=f"p{i}") for i in range(14)],
        joints=[
            SimpleNamespace(type=t)
            for t in ["fixed"] * 9 + ["coaxial_insert"] * 2 + ["revolute"] * 2
        ],
    )
    assert undriven_dof_lower_bound(h3, None) == 2  # 2 moving, nothing to drive them
    assert undriven_dof_lower_bound(h3, []) == 2

    # The L85 pan platform: 2 moving joints, 2 mapped MG996Rs -> fully driven.
    pan = SimpleNamespace(
        parts=[SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        joints=[SimpleNamespace(type="revolute"), SimpleNamespace(type="revolute")],
    )
    assert undriven_dof_lower_bound(pan, [object(), object()]) == 0

    # A statue has no DOF at all, so nothing is undriven -- this measures actuation, not motion.
    statue = SimpleNamespace(
        parts=[SimpleNamespace(name="a")], joints=[SimpleNamespace(type="fixed")]
    )
    assert undriven_dof_lower_bound(statue, None) == 0

    # It is a LOWER bound: surplus hardware (bearings, screws) never drives the count negative.
    assert undriven_dof_lower_bound(pan, [object()] * 9) == 0
