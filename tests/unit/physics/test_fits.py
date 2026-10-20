"""Validation tests for the fit engine (forge_mind.physics.fits).

These assert the engine against PUBLISHED ground truth -- ISO 286 standard tolerances and
fundamental deviations, and Lamé thick-cylinder press-fit mechanics -- so the numbers are
decision-grade, not illustrative. A value here is the standard's value or it is a bug.
"""

from __future__ import annotations

import math

import pytest

from forge_mind.physics import fits
from forge_mind.physics.fits import (
    FitError,
    _it_from_formula,
    _lame_pressure_mpa,
    evaluate,
    parse_designation,
    press_fit,
    resolve_fit,
    shaft_deviation_um,
    standard_tolerance_um,
)

# --------------------------------------------------------------------------- ISO 286: IT grades


@pytest.mark.parametrize("band", [b for b in fits._SIZE_BANDS_MM if b[0] >= 18 and b[1] <= 315])
@pytest.mark.parametrize("grade", [5, 6, 7, 8])
def test_it_table_matches_formula_midrange(band: tuple[float, float], grade: int) -> None:
    # The published IT table must equal the ISO closed form where the form is exact (18..315 mm,
    # grades <= IT8). This auto-catches any transcription typo in the table.
    mid = (band[0] + band[1]) / 2
    assert standard_tolerance_um(mid, grade) == _it_from_formula(mid, grade)


@pytest.mark.parametrize(
    ("nominal", "grade", "expected_um"),
    [
        (50, 7, 25),
        (50, 6, 16),
        (10, 6, 9),
        (10, 7, 15),
        (2, 6, 6),
        (2, 7, 10),  # smallest band: the formula drifts, the table is right
        (450, 6, 40),
        (450, 7, 63),  # largest band: ditto
    ],
)
def test_it_published_anchors(nominal: float, grade: int, expected_um: int) -> None:
    assert standard_tolerance_um(nominal, grade) == expected_um


# --------------------------------------------------------------------------- ISO 286: deviations

# (es, ei) in micrometers, from published ISO 286-2 tables. u splits sub-bands at 24/40 mm.
_PUBLISHED_SHAFT_DEV = {
    10: {
        "g": (-5, -14),
        "h": (0, -9),
        "k": (10, 1),
        "n": (19, 10),
        "p": (24, 15),
        "s": (32, 23),
        "u": (37, 28),
    },
    30: {
        "g": (-7, -20),
        "h": (0, -13),
        "k": (15, 2),
        "n": (28, 15),
        "p": (35, 22),
        "s": (48, 35),
        "u": (61, 48),
    },
    50: {
        "g": (-9, -25),
        "h": (0, -16),
        "k": (18, 2),
        "n": (33, 17),
        "p": (42, 26),
        "s": (59, 43),
        "u": (86, 70),
    },
}


@pytest.mark.parametrize("nominal", sorted(_PUBLISHED_SHAFT_DEV))
def test_shaft_deviations_match_published(nominal: int) -> None:
    for letter, expected in _PUBLISHED_SHAFT_DEV[nominal].items():
        assert shaft_deviation_um(letter, 6, nominal) == expected, f"{nominal} {letter}6"


def test_hole_h_is_basis() -> None:
    assert fits.hole_deviation_um("H", 7, 50) == (25, 0)
    with pytest.raises(FitError):
        fits.hole_deviation_um("G", 7, 50)  # only hole-basis H in v1


# --------------------------------------------------------------------------- fit classification


def test_clearance_fit_50_h7_g6() -> None:
    r = resolve_fit(50, ("H", 7), ("g", 6))
    assert r.fit_type == "clearance"
    assert (r.min_clearance_um, r.max_clearance_um) == (
        9,
        50,
    )  # the classic 9..50 um free-running fit
    assert r.hole_limits_mm == pytest.approx((50.000, 50.025))
    assert r.shaft_limits_mm == pytest.approx((49.975, 49.991))


def test_transition_fit_50_h7_k6() -> None:
    assert resolve_fit(50, ("H", 7), ("k", 6)).fit_type == "transition"


def test_interference_fit_50_h7_p6() -> None:
    r = resolve_fit(50, ("H", 7), ("p", 6))
    assert r.fit_type == "interference"
    # interference 1..42 um  (clearance is negative)
    assert (-r.max_clearance_um, -r.min_clearance_um) == (1, 42)


# --------------------------------------------------------------------------- Lamé press-fit math


def test_lame_pressure_canonical() -> None:
    # d50, hub OD100, solid steel shaft, 0.025 mm RADIAL interference -> 75.0 MPa (closed-form check).
    p = _lame_pressure_mpa(50, 100, 0, 0.025, 200_000, 0.3, 200_000, 0.3)
    assert p == pytest.approx(75.0, abs=0.05)
    a, b = 25, 50
    hoop = p * (b * b + a * a) / (b * b - a * a)
    assert hoop == pytest.approx(125.0, abs=0.1)


def test_press_fit_steel_s6() -> None:
    r = press_fit("50H7/s6", hub_od_mm=90, engagement_mm=20)
    assert r.verdict == "ok"
    assert r.interference_um == (18, 59)
    assert r.safety_factor == pytest.approx(1.62, abs=0.03)
    lo, hi = r.insertion_force_kn
    assert 0 < lo < hi  # force is a range over interference + friction


def test_press_fit_torque_is_force_times_radius_in_newton_metres() -> None:
    # Regression (audit H3): torque capacity was 1000x too low -- kN*mm was divided by 1000 as if to
    # reach N.m, but kN*mm already IS N.m. T = F * r, so torque_nm must equal insertion_kn * radius_mm.
    r = press_fit("50H7/s6", hub_od_mm=90, engagement_mm=20)
    radius_mm = 50 / 2
    for f_kn, t_nm in zip(r.insertion_force_kn, r.torque_capacity_nm, strict=True):
        # T = F * r; f_kn[kN] * radius[mm] == N.m. The tolerance absorbs the engine's rounding but
        # is orders below the 1000x error it guards against.
        assert t_nm == pytest.approx(f_kn * radius_mm, rel=0.05)
    # a real 50 mm steel s6 press fit holds far more than 1 N.m (the bug reported < 1).
    assert r.torque_capacity_nm[0] > 1.0


def test_press_fit_refuses_clearance() -> None:
    with pytest.raises(FitError, match="not a press fit"):
        press_fit("50H7/g6", hub_od_mm=90, engagement_mm=20)


def test_press_fit_refuses_unknown_material() -> None:
    # PLA has no mechanical data -> refuse rather than emit a wrong-but-confident number.
    with pytest.raises(FitError, match="elastic data unknown"):
        press_fit("50H7/p6", hub_od_mm=90, engagement_mm=20, hub_material="pla")


def test_press_fit_refuses_hub_too_small() -> None:
    with pytest.raises(FitError, match="must exceed"):
        press_fit("50H7/p6", hub_od_mm=50, engagement_mm=20)


def test_interference_over_50mm_refuses() -> None:
    # interference letters are tabulated to 50 mm in v1; refuse beyond, don't guess.
    with pytest.raises(FitError, match="supply explicit"):
        resolve_fit(60, ("H", 7), ("p", 6))


# --------------------------------------------------------------------------- parsing + evaluate


def test_parse_designation() -> None:
    assert parse_designation("50 H7/g6") == (50.0, ("H", 7), ("g", 6))
    assert parse_designation("12H7/p6") == (12.0, ("H", 7), ("p", 6))
    with pytest.raises(FitError):
        parse_designation("nonsense")


def test_evaluate_clearance_has_note_no_press() -> None:
    out = evaluate("50 H7/g6")
    assert out.press is None and out.note is not None


def test_evaluate_interference_with_geometry() -> None:
    out = evaluate("50 H7/p6", hub_od_mm=90, engagement_mm=20)
    assert out.press is not None
    assert out.press.safety_factor > 0
    assert math.isfinite(out.press.hub_hoop_stress_mpa)
