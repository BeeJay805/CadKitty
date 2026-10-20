"""Strict disposition, part 2: a caveat must not absorb a mechanism that CANNOT MOVE.

Measured 2026-07-25 over all 84 archived runs: `certificate.motion` literally reads `"fail"` on
6 of the 18 runs this engine ever shipped -- every one a named pair colliding at a named pose with
a measured penetration depth, published in the certificate while the disposition read
`shipped_with_caveats`. The numbers below are those runs', verbatim.

This is a PEER of `not_assemblable`, not an extension of it: a gearbox whose planet gears
interpenetrate their own pins at -200 deg still goes together. It just cannot turn.

Half of these tests pin the EXCLUSIONS, same discipline as ADR-0126. The one that matters most is
`certificate.motion == "fail"` itself -- the field the archive query surfaced, and the WRONG field
to gate on, because `assembly.range_of_motion` also reads `fail` when no gear check ran and when
the plan declared no moving joint at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forge_mind.agentic.functional import apply_strict_disposition, motion_blocked_reasons
from forge_mind.assembly.joints import Joint, JointType
from forge_mind.materials import resolve_material
from forge_mind.validator.assembly_checks import _sweep_evidence
from forge_mind.validator.report import Evidence

_PLA = resolve_material("PLA")


def _sweep_fail(pair="pan_servo__pan_bracket", evidence="measured", msg="hits at 50 deg"):
    return {"id": f"assembly.sweep.{pair}", "message": msg, "evidence": evidence}


def _fit(seated=True, floating=None, **kw):
    return {"seated": seated, "floating_parts": floating or [], **kw}


def _sound_motion(**kw):
    return {
        "material_pin_fit": {"material": "PLA", "measured_clearance_mm": 0.30, "ok": True},
        **kw,
    }


# --- the reason function ------------------------------------------------------------------


def test_a_measured_collision_inside_the_declared_range_blocks() -> None:
    r = motion_blocked_reasons({"assembly_fails": [_sweep_fail()]})
    assert len(r) == 1, r
    assert "blocked inside its declared range" in r[0]
    assert "hits at 50 deg" in r[0], "the measured detail must survive into the reason"


def test_every_colliding_pair_is_reported() -> None:
    # The repair loop and the user both need the full set, not just the worst one.
    fails = [_sweep_fail("a__b"), _sweep_fail("c__d"), _sweep_fail("e__f")]
    assert len(motion_blocked_reasons({"assembly_fails": fails})) == 3


def test_an_assumed_sweep_does_not_block() -> None:
    # EXCLUSION, and the whole reason the evidence field exists: a sweep FAILs on a SUBSTITUTED
    # default range and on an unmeasurable depth. Neither proves the mechanism is blocked.
    assert motion_blocked_reasons({"assembly_fails": [_sweep_fail(evidence="assumed")]}) == []


def test_an_unclassified_sweep_does_not_block() -> None:
    # Fail-open by construction: a check must positively claim "measured" to reach a
    # non-shipping disposition. Every archived report predates the field and lands here.
    assert motion_blocked_reasons({"assembly_fails": [_sweep_fail(evidence=None)]}) == []
    assert (
        motion_blocked_reasons({"assembly_fails": [{"id": "assembly.sweep.a__b", "m": ""}]}) == []
    )


def test_the_certificate_motion_field_is_not_what_gates() -> None:
    # THE EXCLUSION THIS FILE EXISTS TO PIN. `certificate.motion == "fail"` is the measurement
    # that found this defect class, and gating on it directly would have been a false-FAIL:
    # `assembly.range_of_motion` emits FAIL from three places, and only one is a proven collision.
    #   (a) no gear check ran -> "under-specified"  = absence of proof
    #   (b) no revolute/prismatic joint declared    = it assembles fine, it just does not move
    for detail in ("0 gear-mesh DOF, gears_validated=False", "joint is FIXED (0 DOF)"):
        motion = {
            "motion_status": "fail",
            "motion_detail": detail,
            "assembly_fails": [{"id": "assembly.range_of_motion", "message": detail}],
        }
        assert motion_blocked_reasons(motion) == [], detail


def test_non_sweep_assembly_fails_do_not_block() -> None:
    # EXCLUSION (inherited from ADR-0126): the aggregate mixes in merely-rattly gaps, and a
    # static-clearance interpenetration may be a legitimate press fit.
    noisy = {
        "assembly_fails": [
            {
                "id": "assembly.print_in_place_gap",
                "message": "gap too wide",
                "evidence": "measured",
            },
            {
                "id": "assembly.static_clearance",
                "message": "parts interpenetrate",
                "evidence": "measured",
            },
            {
                "id": "assembly.mount_pattern.tilt_servo.horn",
                "message": "missing holes",
                "evidence": "measured",
            },
        ]
    }
    assert motion_blocked_reasons(noisy) == []


def test_a_corner_pose_sweep_does_not_block_yet() -> None:
    # EXCLUSION, and an honest residual gap: assembly.sweep_corner.* is a real measurement, but
    # its driver returns no depths, so it cannot claim MEASURED. It stays a caveat until it can.
    fails = [
        {
            "id": "assembly.sweep_corner.parent__child",
            "message": "collision",
            "evidence": "measured",
        }
    ]
    assert motion_blocked_reasons({"assembly_fails": fails}) == []


def test_missing_reports_fail_open() -> None:
    for empty in (None, {}, {"assembly_fails": None}, {"assembly_fails": []}):
        assert motion_blocked_reasons(empty) == []
    assert motion_blocked_reasons({"assembly_fails": ["not a dict", 7]}) == []


# --- the classifier that makes gating safe ------------------------------------------------


def _joint(range_deg=(0.0, 90.0), jtype=JointType.REVOLUTE, range_mm=None):
    return Joint(a="a.port", b="b.port", type=jtype, range_deg=range_deg, range_mm=range_mm)


def test_a_measured_depth_in_a_declared_range_is_measured() -> None:
    assert _sweep_evidence(_joint(), {("b", "a"): (1.35, 40.0)}) is Evidence.MEASURED


def test_an_undeclared_range_is_assumed() -> None:
    # `range_deg or (0.0, 180.0)` substitutes a default; failing a design on an engine-invented
    # range would be the engine inventing the acceptance criterion.
    assert _sweep_evidence(_joint(range_deg=None), {("b", "a"): (1.35, 40.0)}) is Evidence.ASSUMED
    prismatic = _joint(range_deg=None, jtype=JointType.PRISMATIC)
    assert _sweep_evidence(prismatic, {("b", "a"): (1.0, 5.0)}) is Evidence.ASSUMED
    assert (
        _sweep_evidence(
            _joint(range_deg=None, jtype=JointType.PRISMATIC, range_mm=(0.0, 20.0)),
            {("b", "a"): (1.0, 5.0)},
        )
        is Evidence.MEASURED
    )


def test_an_unmeasurable_depth_is_assumed() -> None:
    # _pair_overlap_depth returns inf when the collision backend fails, and the sweep treats it
    # as an overlap (fail-closed, deliberately). That is "could not check", not "does not fit".
    assert _sweep_evidence(_joint(), {("b", "a"): (math.inf, 40.0)}) is Evidence.ASSUMED


def test_one_finite_depth_among_unmeasurables_still_proves_it() -> None:
    cols = {("b", "a"): (math.inf, 40.0), ("c", "a"): (2.5, 30.0)}
    assert _sweep_evidence(_joint(), cols) is Evidence.MEASURED


# --- the wiring ----------------------------------------------------------------------------


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("FORGE_STRICT_DISPOSITION", "1")


def test_on_by_default_blocks_measured_motion(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_STRICT_DISPOSITION", raising=False)
    blocked = _sound_motion(assembly_fails=[_sweep_fail()])
    for d in ("shipped", "shipped_with_caveats"):
        assert apply_strict_disposition(d, _fit(), blocked, _PLA)[0] == "motion_blocked"


def test_explicit_rollback_is_inert(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_STRICT_DISPOSITION", "0")
    blocked = _sound_motion(assembly_fails=[_sweep_fail()])
    for d in ("shipped", "shipped_with_caveats", "rejected", "no_build"):
        assert apply_strict_disposition(d, _fit(), blocked, _PLA) == (d, [])


def test_a_blocked_mechanism_is_downgraded(strict) -> None:
    blocked = _sound_motion(assembly_fails=[_sweep_fail()])
    for d in ("shipped", "shipped_with_caveats"):
        new, reasons = apply_strict_disposition(d, _fit(), blocked, _PLA)
        assert new == "motion_blocked", (new, reasons)
        assert len(reasons) == 1


def test_cannot_build_outranks_cannot_move(strict) -> None:
    # If it never goes together, its motion is moot -- and `not_assemblable` is the truer word.
    blocked = _sound_motion(assembly_fails=[_sweep_fail()])
    new, reasons = apply_strict_disposition(
        "shipped_with_caveats", _fit(seated=False), blocked, _PLA
    )
    assert new == "not_assemblable", reasons


def test_an_honest_rejection_is_never_relabelled(strict) -> None:
    blocked = _sound_motion(assembly_fails=[_sweep_fail()])
    for d in ("rejected", "no_build"):
        assert apply_strict_disposition(d, _fit(), blocked, _PLA) == (d, [])


def test_a_moving_mechanism_still_ships(strict) -> None:
    for d in ("shipped", "shipped_with_caveats"):
        assert apply_strict_disposition(d, _fit(), _sound_motion(), _PLA) == (d, [])


# --- retroactive replay on the REAL archived run -------------------------------------------
# The archive predates the `evidence` field, so a raw replay is vacuous by construction (every
# entry is unclassified -> fails open -> no downgrade). This reconstructs the classification the
# way _sweep_evidence computes it, from the run's OWN plan.json (declared range) and its OWN
# recorded depths -- then drives the real decision function. Both inputs are measured facts of
# that run; nothing here is hand-tuned to make it fire.

_P1 = Path("E:/ForgeMind/benchmarks/2026-07-22_r3_pantilt/P1/pan_tilt_neck_s0")


def _replayed_motion(run_dir: Path) -> dict:
    """That run's motion verdict with `evidence` reconstructed the way _sweep_evidence would."""
    report = json.loads((run_dir / "functional_report.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    declared = {
        (str(j["a"]).split(".")[0], str(j["b"]).split(".")[0])
        for j in plan["joints"]
        if j.get("range_deg") is not None or j.get("range_mm") is not None
    }
    motion = dict(report["motion"])
    motion["assembly_fails"] = [
        {
            **f,
            "evidence": (
                "measured"
                if str(f["id"]).startswith("assembly.sweep.")
                and tuple(f["id"][len("assembly.sweep.") :].split("__")) in declared
                and "depth unmeasurable" not in f["message"]
                else None
            ),
        }
        for f in report["motion"]["assembly_fails"]
    ]
    return motion


@pytest.mark.skipif(not _P1.exists(), reason="pantilt archive not present on this machine")
def test_the_real_archived_pantilt_is_measurably_blocked() -> None:
    """END-TO-END on real data: both of the pan-tilt's declared joints jam, measurably."""
    report = json.loads((_P1 / "functional_report.json").read_text(encoding="utf-8"))
    assert report["disposition"] == "shipped_with_caveats", "the archived run shipped, as recorded"
    assert report["certificate"]["motion"] == "fail", "...while its certificate said motion failed"

    reasons = motion_blocked_reasons(_replayed_motion(_P1))
    assert len(reasons) == 2, reasons  # pan_servo<->pan_bracket and tilt_servo<->tilt_head
    assert any("4.45 mm" in r for r in reasons), "the measured depth rides the reason"


@pytest.mark.skipif(not _P1.exists(), reason="pantilt archive not present on this machine")
def test_on_the_real_archive_cannot_build_still_outranks(strict) -> None:
    """THE MEASURED MARGINAL VALUE OF THIS TIER ON THE ARCHIVE IS ZERO, and that is pinned here.

    Replayed 2026-07-25 over all 84 archived runs: `motion_blocked` fires on 9 of them and is
    outranked by `not_assemblable` on every single one -- a run bad enough to jam was also bad
    enough to not seat. This tier exists for the residual class (an assembly that DOES go
    together and still cannot move), which the archive does not yet contain.

    If this assertion ever flips to `motion_blocked`, that is not a regression: it means the
    engine finally produced a mechanism that assembles correctly and merely jams, and the tier
    earned its keep. Read docs/decision_log/0127-motion-blocked-disposition.md before changing it.
    """
    report = json.loads((_P1 / "functional_report.json").read_text(encoding="utf-8"))
    new, reasons = apply_strict_disposition(
        report["disposition"], report.get("fit"), _replayed_motion(_P1), _PLA
    )
    assert new == "not_assemblable", (new, reasons)
