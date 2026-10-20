"""Continuous-sweep certification (Sol P0.4): a discrete pose sweep may witness a FAIL but may
NEVER certify a PASS. These fixtures pin exactly that asymmetry, including Sol's named
regression: a thin mid-sweep collision invisible at both sampled endpoints must NOT pass.

Reference geometry: a 100 mm limb (R_max = 100) swept about its joint. Over a 10 deg step any
point travels at most 100 * radians(10) = 17.45 mm, so a 20 mm clearance certifies and an 8 mm
clearance cannot -- the arithmetic a reviewer can check by hand."""

from __future__ import annotations

import math

import pytest

from forge_mind.validator.sweep_certificate import (
    PoseSample,
    certify_revolute_sweep,
    max_travel_revolute_mm,
    required_sample_step_deg,
)

_R = 100.0  # mm, farthest point of the moving body from the joint axis


def test_travel_bound_is_conservative_vs_exact_chord() -> None:
    # The bound R*dtheta must never UNDER-estimate the true chord 2R sin(dtheta/2).
    for dq in (1.0, 5.0, 10.0, 45.0, 90.0, 179.0):
        chord = 2.0 * _R * math.sin(math.radians(dq) / 2.0)
        assert max_travel_revolute_mm(_R, dq) >= chord - 1e-9


def test_generous_clearance_certifies() -> None:
    # 10 deg steps (travel <= 17.45 mm) with 30 mm clearance everywhere -> certified.
    samples = [PoseSample(q, 30.0) for q in (0.0, 10.0, 20.0, 30.0)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.certified, cert.verdict.message
    assert cert.verdict.measured["n_intervals"] == 3.0
    assert cert.verdict.measured["worst_certification_margin_mm"] == pytest.approx(
        30.0 - max_travel_revolute_mm(_R, 10.0), rel=1e-6
    )


def test_thin_midsweep_collision_is_not_certified() -> None:
    # SOL'S REGRESSION: endpoints look clear (8 mm) but a 10 deg step can move 17.45 mm, so a
    # collision can hide entirely between samples. The old "sampled and saw nothing" logic
    # would PASS this; certification must REFUSE it (not_run), naming the interval.
    samples = [PoseSample(0.0, 8.0), PoseSample(10.0, 8.0)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.verdict.status == "not_run", cert.verdict.message
    assert "NOT certified" in cert.verdict.message
    assert cert.verdict.measured["uncertified_q_lo"] == 0.0
    assert cert.verdict.measured["uncertified_q_hi"] == 10.0


def test_subdividing_the_same_span_certifies_it() -> None:
    # The refusal is actionable: same 0..10 deg span, finer steps (1 deg -> travel 1.75 mm)
    # with the same 8 mm clearance now certifies. Proves the bound is not simply pessimistic.
    samples = [PoseSample(float(i), 8.0) for i in range(11)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.certified, cert.verdict.message


def test_measured_contact_is_a_fail_witness_not_a_refusal() -> None:
    # A pose actually at/inside the floor is a real FAIL (a witness), which outranks any
    # uncertifiable-interval refusal.
    samples = [PoseSample(0.0, 30.0), PoseSample(10.0, 0.0), PoseSample(20.0, 30.0)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.verdict.status == "fail", cert.verdict.message
    assert "collision witness" in cert.verdict.message
    assert cert.verdict.measured["q"] == 10.0


def test_penetration_is_a_fail_witness() -> None:
    samples = [
        PoseSample(0.0, 30.0),
        PoseSample(
            10.0,
            -2.5,
        ),
        PoseSample(20.0, 30.0),
    ]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.verdict.status == "fail" and cert.verdict.measured["min_separation_mm"] == -2.5


def test_clearance_floor_tightens_certification() -> None:
    # 20 mm clearance certifies at a 0 floor (travel 17.45) but not against a 5 mm floor,
    # because only 15 mm of usable slack remains.
    samples = [PoseSample(0.0, 20.0), PoseSample(10.0, 20.0)]
    assert certify_revolute_sweep(samples, radius_max_mm=_R).certified
    strict = certify_revolute_sweep(samples, radius_max_mm=_R, clearance_floor_mm=5.0)
    assert strict.verdict.status == "not_run", strict.verdict.message


def test_unmeasured_separation_is_not_run_never_pass() -> None:
    samples = [PoseSample(0.0, 30.0), PoseSample(10.0, None), PoseSample(20.0, 30.0)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.verdict.status == "not_run" and "unmeasured" in cert.verdict.message


def test_malformed_inputs_are_not_run() -> None:
    ok = [PoseSample(0.0, 30.0), PoseSample(10.0, 30.0)]
    assert certify_revolute_sweep(ok, radius_max_mm=0.0).verdict.status == "not_run"
    assert certify_revolute_sweep(ok, radius_max_mm=-5.0).verdict.status == "not_run"
    assert (
        certify_revolute_sweep(ok, radius_max_mm=_R, clearance_floor_mm=-1.0).verdict.status
        == "not_run"
    )
    assert (
        certify_revolute_sweep([PoseSample(0.0, 30.0)], radius_max_mm=_R).verdict.status
        == "not_run"
    )
    bad_q = [PoseSample(float("nan"), 30.0), PoseSample(10.0, 30.0)]
    assert certify_revolute_sweep(bad_q, radius_max_mm=_R).verdict.status == "not_run"


def test_unordered_samples_are_sorted_before_certifying() -> None:
    samples = [PoseSample(20.0, 30.0), PoseSample(0.0, 30.0), PoseSample(10.0, 30.0)]
    cert = certify_revolute_sweep(samples, radius_max_mm=_R)
    assert cert.certified, cert.verdict.message
    assert [(iv.q_lo, iv.q_hi) for iv in cert.intervals] == [(0.0, 10.0), (10.0, 20.0)]


def test_required_step_planning_aid_matches_the_bound() -> None:
    # With R=100 and 8 mm of slack, steps must be under degrees(8/100) ~ 4.58 deg.
    step = required_sample_step_deg(_R, 8.0)
    assert step == pytest.approx(math.degrees(0.08), rel=1e-9)
    just_under = [PoseSample(0.0, 8.0), PoseSample(step * 0.9, 8.0)]
    assert certify_revolute_sweep(just_under, radius_max_mm=_R).certified
    assert required_sample_step_deg(_R, 0.0) is None  # no slack -> uncertifiable at any step
    assert required_sample_step_deg(0.0, 8.0) is None


def test_non_finite_separation_never_certifies() -> None:
    # Cross-examiner catch (measured live before the fix): a NaN separation CERTIFIED because
    # min(30.0, nan) returns 30.0 -- and it was order-dependent (nan first -> not_run, nan
    # second -> pass). An infinite separation certified outright. A failed measurement is not
    # a clearance; both orders and both values must now be not_run.
    nan, inf = float("nan"), float("inf")
    assert (
        certify_revolute_sweep(
            [PoseSample(0.0, 30.0), PoseSample(10.0, nan)], radius_max_mm=_R
        ).verdict.status
        == "not_run"
    )
    assert (
        certify_revolute_sweep(
            [PoseSample(0.0, nan), PoseSample(10.0, 30.0)], radius_max_mm=_R
        ).verdict.status
        == "not_run"
    )
    assert (
        certify_revolute_sweep(
            [PoseSample(0.0, inf), PoseSample(10.0, inf)], radius_max_mm=_R
        ).verdict.status
        == "not_run"
    )
    assert (
        certify_revolute_sweep(
            [PoseSample(0.0, 30.0), PoseSample(10.0, -inf)], radius_max_mm=_R
        ).verdict.status
        == "not_run"
    )


def test_finite_collision_witness_outranks_a_non_finite_sample() -> None:
    # Cross-examiner catch: the non-finite refusal ran BEFORE the witness scan, so a NaN
    # elsewhere masked a PROVEN penetration as not_run. A witness must win.
    nan = float("nan")
    v = certify_revolute_sweep(
        [PoseSample(0.0, nan), PoseSample(10.0, -1.0)], radius_max_mm=_R
    ).verdict
    assert v.status == "fail", v.message
    assert v.measured["min_separation_mm"] == -1.0


# ---- multi-joint chain bound (Sol P0.4 generalization) ------------------------------------

from forge_mind.validator.sweep_certificate import (  # noqa: E402
    JointMotion,
    certify_pair_sweep,
    chain_travel_bound_mm,
)


def test_chain_bound_sums_moving_ancestors() -> None:
    # A shoulder (R=200 out to the wrist) and an elbow (R=100) each rotating 5 deg: the wrist's
    # travel bound is the SUM, not either one alone -- the whole point of Sol's generalization.
    shoulder = JointMotion("shoulder", 200.0, 5.0)
    elbow = JointMotion("elbow", 100.0, 5.0)
    both = chain_travel_bound_mm([shoulder, elbow])
    assert both == pytest.approx(
        max_travel_revolute_mm(200.0, 5.0) + max_travel_revolute_mm(100.0, 5.0), rel=1e-12
    )
    # Strictly larger than any single joint's bound: using one alone UNDER-estimates travel.
    assert both > chain_travel_bound_mm([shoulder])
    assert both > chain_travel_bound_mm([elbow])


def test_prismatic_contributes_its_stroke_directly() -> None:
    assert chain_travel_bound_mm([JointMotion("slide", 0.0, 7.5, kind="prismatic")]) == 7.5


def test_malformed_motion_is_refused() -> None:
    assert chain_travel_bound_mm([JointMotion("j", 100.0, float("nan"))]) is None
    assert chain_travel_bound_mm([JointMotion("j", 0.0, 5.0)]) is None  # zero radius, revolute
    assert chain_travel_bound_mm([JointMotion("j", 100.0, 5.0, kind="helical")]) is None


def test_chain_motion_that_single_joint_bound_would_falsely_certify() -> None:
    # THE FALSE PASS Sol's generalization prevents. Clearance 20 mm at both endpoints.
    # Single shoulder-only bound: 200*rad(5) = 17.45 mm -> would CERTIFY (20 > 17.45).
    # True chain bound (shoulder + elbow): 17.45 + 8.73 = 26.18 mm -> must REFUSE.
    samples = [PoseSample(0.0, 20.0), PoseSample(5.0, 20.0)]
    motions = [[JointMotion("shoulder", 200.0, 5.0), JointMotion("elbow", 100.0, 5.0)]]
    single_only = certify_revolute_sweep(samples, radius_max_mm=200.0)
    assert single_only.certified, "precondition: the single-joint bound does certify this"
    pair = certify_pair_sweep(samples, motions_a=motions)
    assert pair.verdict.status == "not_run", pair.verdict.message
    assert pair.intervals[0].max_travel_mm == pytest.approx(26.18, rel=1e-3)


def test_both_bodies_moving_sums_their_bounds() -> None:
    # Two moving limbs closing on each other: U_AB = U_A + U_B.
    samples = [PoseSample(0.0, 30.0), PoseSample(5.0, 30.0)]
    a = [[JointMotion("a", 100.0, 5.0)]]  # 8.73 mm
    b = [[JointMotion("b", 100.0, 5.0)]]  # 8.73 mm
    assert certify_pair_sweep(samples, motions_a=a).certified  # B static: 8.73 < 30
    both = certify_pair_sweep(samples, motions_a=a, motions_b=b)
    assert both.certified
    assert both.intervals[0].max_travel_mm == pytest.approx(17.45, rel=1e-3)
    tight = [PoseSample(0.0, 12.0), PoseSample(5.0, 12.0)]
    assert certify_pair_sweep(tight, motions_a=a).certified  # 8.73 < 12 alone
    assert certify_pair_sweep(tight, motions_a=a, motions_b=b).verdict.status == "not_run"


def test_pair_witness_and_epsilon_and_shape_guards() -> None:
    a = [[JointMotion("a", 100.0, 5.0)]]
    # A proven contact is a FAIL witness even with a NaN elsewhere.
    w = certify_pair_sweep(
        [PoseSample(0.0, float("nan")), PoseSample(5.0, -0.5)], motions_a=a
    ).verdict
    assert w.status == "fail" and w.measured["min_separation_mm"] == -0.5
    # Measurement uncertainty eats into the usable clearance.
    s = [PoseSample(0.0, 9.0), PoseSample(5.0, 9.0)]
    assert certify_pair_sweep(s, motions_a=a).certified  # 8.73 < 9.0
    assert certify_pair_sweep(s, motions_a=a, epsilon_mm=0.5).verdict.status == "not_run"
    # One motion entry per interval is mandatory (a missing interval cannot be certified).
    three = [PoseSample(0.0, 30.0), PoseSample(5.0, 30.0), PoseSample(10.0, 30.0)]
    assert certify_pair_sweep(three, motions_a=a).verdict.status == "not_run"
