"""Worst-case tolerance stackup (Sol P0.5): a fit verified only at nominal is not verified.

Fixtures use the real H3 hip interface (a Ø14.0 thigh tenon into a Ø14.3 pelvis bore, i.e. a
0.3 mm nominal running clearance) plus FDM-realistic tolerances, so the arithmetic is
hand-checkable: at +/-0.15 mm on each member the worst case is 14.15 - 14.15 = 0.00 mm --
exactly the "binds as printed" condition H3 shipped.

Four tests below are REGRESSION PINS on false-PASSes the first draft of this module actually
produced (each reproduced numerically against that code before it was rewritten):
``test_absent_requirement_is_not_run``, ``test_a_two_sided_extra_widens_both_ends``,
``test_correlated_contributors_are_no_longer_discounted`` and
``test_unbounded_acceptance_limits_are_refused``.
"""

from __future__ import annotations

import pytest

from forge_mind.validator.tolerance_stack import (
    Contributor,
    FitRequirement,
    cylindrical_fit_stack,
)

_SRC = "qualified print profile v1"
_SPEC = "hip ICD rev B"


def _hole(nominal=14.3, minus=0.15, plus=0.15, **kw) -> Contributor:
    return Contributor("pelvis_hip_bore", nominal, minus, plus, kw.pop("source", _SRC), **kw)


def _shaft(nominal=14.0, minus=0.15, plus=0.15, **kw) -> Contributor:
    return Contributor("thigh_tenon", nominal, minus, plus, kw.pop("source", _SRC), **kw)


def _running(lo=0.10, hi=0.70) -> FitRequirement:
    return FitRequirement(lo, hi, "clearance", _SPEC)


def test_nominal_looks_fine_but_worst_case_binds() -> None:
    # THE H3 CONDITION: 0.3 mm nominal clearance, +/-0.15 on each member -> worst case 0.00 mm.
    # A nominal-only check calls this a running fit; the worst-case check calls it binding.
    r = cylindrical_fit_stack(hole=_hole(), shaft=_shaft(), requirement=_running())
    assert r.verdict.status == "fail", r.verdict.message
    assert r.c_min_mm == pytest.approx(0.0, abs=1e-9)
    assert "BINDS" in r.verdict.message


def test_tighter_tolerances_make_the_same_fit_pass() -> None:
    # Same nominal geometry, +/-0.05 -> worst case [0.20, 0.40], inside a [0.10, 0.70] spec.
    r = cylindrical_fit_stack(
        hole=_hole(minus=0.05, plus=0.05),
        shaft=_shaft(minus=0.05, plus=0.05),
        requirement=_running(),
    )
    assert r.verdict.status == "pass", r.verdict.message
    assert r.c_min_mm == pytest.approx(0.20, abs=1e-9)
    assert r.c_max_mm == pytest.approx(0.40, abs=1e-9)
    assert "running fit" in r.verdict.message


def test_upper_bound_catches_a_sloppy_fit() -> None:
    # Worst-case C_max too large: the joint rattles even though it never binds.
    r = cylindrical_fit_stack(
        hole=_hole(nominal=15.0, minus=0.4, plus=0.4),
        shaft=_shaft(minus=0.4, plus=0.4),
        requirement=_running(0.10, 0.50),
    )
    assert r.verdict.status == "fail", r.verdict.message
    assert r.c_max_mm == pytest.approx(1.80, abs=1e-9)
    assert "TOO LOOSE" in r.verdict.message and "BINDS" not in r.verdict.message


def test_absent_requirement_is_not_run() -> None:
    # REGRESSION PIN. The first draft defaulted to required_min_clearance_mm=0.0, so the H3
    # condition it exists to catch (worst case exactly 0.000 mm) evaluated 0.0 >= 0.0 and PASSED.
    # A fit's allowed interval is frozen-spec input; its absence can never be a PASS.
    r = cylindrical_fit_stack(hole=_hole(), shaft=_shaft())
    assert r.verdict.status == "not_run" and "no FitRequirement" in r.verdict.message


def test_a_two_sided_extra_widens_both_ends() -> None:
    # REGRESSION PIN, the serious one: this false-PASS came from cited, realistic inputs.
    # The first draft sorted extras into a hole side applied only to C_max and a shaft side
    # applied only to C_min, so a +/-0.20 mm datum shift left one end untouched -- it reported
    # C_min 0.200 (true 0.000) when declared hole-side and C_max 0.400 (true 0.600) shaft-side,
    # and PASSED a [0.10, 0.50] spec both ways. A two-sided quantity widens BOTH ends, and the
    # sign it is declared with cannot change that.
    base = {"hole": _hole(minus=0.05, plus=0.05), "shaft": _shaft(minus=0.05, plus=0.05)}
    for coefficient in (1.0, -1.0):
        shift = Contributor("datum_shift", 0.0, 0.20, 0.20, _SRC, coefficient=coefficient)
        r = cylindrical_fit_stack(**base, extras=[shift], requirement=_running(0.10, 0.50))
        assert r.c_min_mm == pytest.approx(0.00, abs=1e-9), coefficient
        assert r.c_max_mm == pytest.approx(0.60, abs=1e-9), coefficient
        assert r.verdict.status == "fail", r.verdict.message


def test_correlated_contributors_are_no_longer_discounted() -> None:
    # REGRESSION PIN. The first draft let contributors share a correlation_group and counted only
    # the group's LARGEST deviation (0.20 instead of 0.35), reasoning that one physical cause
    # cannot take two values at once. Wrong: one cause taking ONE value still moves several
    # dimensions the same way, and those effects add. The discount was unconservative in exactly
    # the direction that mints a PASS, so contributors now stack independently.
    r = cylindrical_fit_stack(
        hole=_hole(minus=0.05, plus=0.05),
        shaft=_shaft(minus=0.05, plus=0.05),
        extras=[
            Contributor("datum_shift_a", 0.0, 0.20, 0.20, _SRC, coefficient=-1.0),
            Contributor("datum_shift_b", 0.0, 0.15, 0.15, _SRC, coefficient=-1.0),
        ],
        requirement=_running(0.0, 0.80),
    )
    assert r.c_min_mm == pytest.approx(-0.15, abs=1e-9)  # 0.20 - 0.20 - 0.15, not 0.20 - 0.20
    assert r.c_max_mm == pytest.approx(0.75, abs=1e-9)
    assert r.verdict.status == "fail", r.verdict.message


def test_unbounded_acceptance_limits_are_refused() -> None:
    # REGRESSION PIN. -inf as a floor made C_low >= floor vacuously true: the first draft passed
    # a nominal clearance fit carrying 9.7 mm of interference. +inf is refused too -- a clearance
    # fit can be too loose, so an unbounded ceiling cannot gate one either.
    wide = {"hole": _hole(minus=5.0, plus=5.0), "shaft": _shaft(minus=5.0, plus=5.0)}
    for lo, hi in ((float("-inf"), 0.70), (0.10, float("inf")), (float("nan"), 0.70)):
        r = cylindrical_fit_stack(**wide, requirement=FitRequirement(lo, hi, "clearance", _SPEC))
        assert r.verdict.status == "not_run", (lo, hi, r.verdict.message)
        assert "finite" in r.verdict.message


def test_an_extra_must_declare_its_signed_coefficient() -> None:
    # An unsigned extra has no role to derive a sign from, and a guessed sign is a guessed result.
    r = cylindrical_fit_stack(
        hole=_hole(),
        shaft=_shaft(),
        extras=[Contributor("datum_shift", 0.0, 0.2, 0.2, _SRC)],
        requirement=_running(),
    )
    assert r.verdict.status == "not_run" and "signed diametral coefficient" in r.verdict.message


def test_a_radial_quantity_enters_with_the_factor_two() -> None:
    # The equation is DIAMETRAL: a 0.10 +/-0.02 mm radial coating on the shaft eats clearance
    # twice, so it is declared with coefficient -2 and removes [0.16, 0.24] mm of diameter.
    r = cylindrical_fit_stack(
        hole=_hole(minus=0.05, plus=0.05),
        shaft=_shaft(minus=0.05, plus=0.05),
        extras=[Contributor("shaft_coating", 0.10, 0.02, 0.02, _SRC, coefficient=-2.0)],
        requirement=_running(0.0, 0.70),
    )
    assert r.c_min_mm == pytest.approx(0.20 - 0.24, abs=1e-9)
    assert r.c_max_mm == pytest.approx(0.40 - 0.16, abs=1e-9)
    assert r.verdict.status == "fail", r.verdict.message


def test_uncited_tolerance_is_not_run_never_pass() -> None:
    # Provenance: an uncited tolerance is a guess and cannot certify a fit -- on either side.
    r = cylindrical_fit_stack(hole=_hole(source="  "), shaft=_shaft(), requirement=_running())
    assert r.verdict.status == "not_run" and "no cited tolerance source" in r.verdict.message
    r = cylindrical_fit_stack(
        hole=_hole(), shaft=_shaft(), requirement=FitRequirement(0.10, 0.70, "clearance", " ")
    )
    assert r.verdict.status == "not_run" and "cites no source" in r.verdict.message


def test_malformed_contributors_are_not_run() -> None:
    nan = float("nan")
    req = _running()
    assert (
        cylindrical_fit_stack(hole=_hole(nominal=nan), shaft=_shaft(), requirement=req).verdict
    ).status == "not_run"
    assert (
        cylindrical_fit_stack(hole=_hole(minus=-0.1), shaft=_shaft(), requirement=req).verdict
    ).status == "not_run"
    # A diameter that can reach zero is not a diameter.
    assert (
        cylindrical_fit_stack(hole=_hole(minus=14.3), shaft=_shaft(), requirement=req).verdict
    ).status == "not_run"


def test_duplicate_contributor_names_are_not_run() -> None:
    # Two records sharing an id are either one quantity double-counted or two quantities with an
    # ambiguous multiplicity. Either way the stack cannot be trusted.
    r = cylindrical_fit_stack(
        hole=_hole(),
        shaft=_shaft(),
        extras=[
            Contributor("datum_shift", 0.0, 0.2, 0.2, _SRC, coefficient=-1.0),
            Contributor("datum_shift", 0.0, 0.1, 0.1, _SRC, coefficient=-1.0),
        ],
        requirement=_running(),
    )
    assert r.verdict.status == "not_run" and "duplicate contributor" in r.verdict.message


def test_a_primary_may_not_contradict_the_sign_its_role_forces() -> None:
    r = cylindrical_fit_stack(hole=_hole(coefficient=-1.0), shaft=_shaft(), requirement=_running())
    assert r.verdict.status == "not_run" and "conflicts with the +1" in r.verdict.message


def test_the_acceptance_interval_must_agree_with_its_fit_class() -> None:
    for req, marker in (
        (FitRequirement(-0.10, 0.50, "clearance", _SPEC), "may not permit interference"),
        (FitRequirement(-0.20, 0.10, "interference", _SPEC), "may not permit a gap"),
        (FitRequirement(0.10, 0.50, "transition", _SPEC), "must straddle zero"),
        (FitRequirement(0.50, 0.10, "clearance", _SPEC), "reversed"),
    ):
        r = cylindrical_fit_stack(hole=_hole(), shaft=_shaft(), requirement=req)
        assert r.verdict.status == "not_run", req
        assert marker in r.verdict.message, req


def test_a_zero_floor_certifies_only_that_nothing_interferes() -> None:
    # Sol's ruling: worst-case clearance >= 0 supports "no predicted interference", never
    # "running fit". The verdict must say which claim it earned.
    r = cylindrical_fit_stack(
        hole=_hole(minus=0.05, plus=0.05),
        shaft=_shaft(minus=0.05, plus=0.05),
        requirement=_running(0.0, 0.70),
    )
    assert r.verdict.status == "pass", r.verdict.message
    assert "NOT a running fit" in r.verdict.message


def test_interference_fit_must_always_interfere() -> None:
    # A press fit: shaft bigger than hole. It PASSES only if it interferes at EVERY extreme, and
    # by no more than the spec permits -- the signed interval expresses both bounds.
    req = FitRequirement(-0.20, -0.02, "interference", _SPEC)
    r = cylindrical_fit_stack(
        hole=Contributor("boss_bore", 6.00, 0.02, 0.02, _SRC),
        shaft=Contributor("pin", 6.10, 0.02, 0.02, _SRC),
        requirement=req,
    )
    assert r.verdict.status == "pass", r.verdict.message
    assert r.c_max_mm == pytest.approx(-0.06, abs=1e-9)  # still interfering at its loosest
    # Widen the tolerances until it can come out loose -> must fail.
    loose = cylindrical_fit_stack(
        hole=Contributor("boss_bore", 6.00, 0.10, 0.10, _SRC),
        shaft=Contributor("pin", 6.10, 0.10, 0.10, _SRC),
        requirement=req,
    )
    assert loose.verdict.status == "fail" and loose.c_max_mm == pytest.approx(0.10, abs=1e-9)
