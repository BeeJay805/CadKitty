"""Decision-grade fit engine: ISO 286 limits & fits + press-fit mechanics.

Every number here is *reproducible*, not illustrative: ISO 286 hole/shaft deviations come
from the published standard, and press-fit results from Lamé thick-cylinder elasticity. Results
are stated as engineering values for making a decision, with the assumptions named and -- where an
input is a range (e.g. the friction band) -- the result given as a range. The engine REFUSES with
guidance (raising :class:`FitError`, never a quiet wrong number and never a false "safe") only when a
result would be unreliable: an unknown material's elastic data, an unsupported designation, or a hub
driven past yield.

Validated in ``tests/unit/physics/test_fits.py`` against published values:
  * ISO 286 -- 50 H7/g6 -> hole +25/0 um, shaft -9/-25 um, clearance 9..50 um (a free-running fit).
  * Lamé   -- d=50, hub OD=100, 0.025 mm radial interference, steel: contact pressure 75 MPa,
    hub bore hoop stress 125 MPa, safety factor 2.0; insertion 88 kN, torque 2.2 kN.m (mu=0.15, L=50).

v1 scope: hole-basis ``H`` holes; shafts f, g, h (clearance), k, n (transition), p, s, u
(interference, supported to 50 mm -- larger interference fits raise, pending the full sub-band table).
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel

from forge_mind.materials import get_material

__all__ = [
    "FitError",
    "FitResult",
    "PressFitResult",
    "evaluate",
    "parse_designation",
    "press_fit",
    "resolve_fit",
    "shaft_deviation_um",
    "standard_tolerance_um",
]


class FitError(ValueError):
    """A fit cannot be resolved or computed reliably -- raised instead of returning a wrong number."""


# ---------------------------------------------------------------------------
# ISO 286-1: standard tolerance grades (IT) from the standard tolerance factor.
# ---------------------------------------------------------------------------
_SIZE_BANDS_MM: tuple[tuple[float, float], ...] = (
    (0, 3),
    (3, 6),
    (6, 10),
    (10, 18),
    (18, 30),
    (30, 50),
    (50, 80),
    (80, 120),
    (120, 180),
    (180, 250),
    (250, 315),
    (315, 400),
    (400, 500),
)
# ISO 286-1 standard tolerances IT5..IT11 (micrometers), tabulated per size band. The closed form
# i = 0.45*cbrt(D)+0.001*D reproduces these exactly for 18..315 mm but drifts +/-1 um at the extremes
# (ISO rounds IT to R-series steps), so the published TABLE is the source of truth; the formula is kept
# only to cross-check the table in tests (see test_fits.py::test_it_table_matches_formula_midrange).
_IT_GRADES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)
_IT_TABLE: dict[tuple[float, float], tuple[int, ...]] = {
    (0, 3): (4, 6, 10, 14, 25, 40, 60),
    (3, 6): (5, 8, 12, 18, 30, 48, 75),
    (6, 10): (6, 9, 15, 22, 36, 58, 90),
    (10, 18): (8, 11, 18, 27, 43, 70, 110),
    (18, 30): (9, 13, 21, 33, 52, 84, 130),
    (30, 50): (11, 16, 25, 39, 62, 100, 160),
    (50, 80): (13, 19, 30, 46, 74, 120, 190),
    (80, 120): (15, 22, 35, 54, 87, 140, 220),
    (120, 180): (18, 25, 40, 63, 100, 160, 250),
    (180, 250): (20, 29, 46, 72, 115, 185, 290),
    (250, 315): (23, 32, 52, 81, 130, 210, 320),
    (315, 400): (25, 36, 57, 89, 140, 230, 360),
    (400, 500): (27, 40, 63, 97, 155, 250, 400),
}


def _band(nominal_mm: float) -> tuple[float, float]:
    for lo, hi in _SIZE_BANDS_MM:
        if lo < nominal_mm <= hi:
            return lo, hi
    raise FitError(f"nominal size {nominal_mm} mm is outside the ISO 286 range (0, 500] mm")


def _geomean(lo: float, hi: float) -> float:
    # ISO uses the geometric mean of the band; the first band (0,3] uses sqrt(1*3).
    return math.sqrt(max(lo, 1.0) * hi)


def standard_tolerance_um(nominal_mm: float, grade: int) -> int:
    """ISO 286 standard tolerance (IT grade width) in micrometers, from the published table."""
    if grade not in _IT_GRADES:
        raise FitError(f"IT grade {grade} not supported (v1 has IT5..IT11)")
    return _IT_TABLE[_band(nominal_mm)][grade - 5]


def _it_from_formula(nominal_mm: float, grade: int) -> int:
    """ISO closed form (i = 0.45*cbrt(D)+0.001*D). Exact for 18..315 mm; used only to cross-check
    :data:`_IT_TABLE` in tests, never in a production result."""
    mult = {5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100}[grade]
    d = _geomean(*_band(nominal_mm))
    i_um: float = 0.45 * d ** (1 / 3) + 0.001 * d
    return round(mult * i_um)


# ---------------------------------------------------------------------------
# ISO 286-2: fundamental deviations. Clearance/transition letters are computed from
# the standard's formulas (validated); interference letters carry the standard's
# tabulated lower deviation ei (um), keyed by sub-band (u splits at 24 and 40 mm).
# ---------------------------------------------------------------------------
_SHAFT_EI_INTERFERENCE: dict[str, tuple[tuple[float, float, int], ...]] = {
    "p": ((0, 3, 6), (3, 6, 12), (6, 10, 15), (10, 18, 18), (18, 30, 22), (30, 50, 26)),
    "s": ((0, 3, 14), (3, 6, 19), (6, 10, 23), (10, 18, 28), (18, 30, 35), (30, 50, 43)),
    "u": (
        (0, 3, 18),
        (3, 6, 23),
        (6, 10, 28),
        (10, 18, 33),
        (18, 24, 41),
        (24, 30, 48),
        (30, 40, 60),
        (40, 50, 70),
    ),
}
_INTERFERENCE_MAX_MM = 50.0


def _lookup_interference_ei(letter: str, nominal_mm: float) -> int:
    for lo, hi, ei in _SHAFT_EI_INTERFERENCE[letter]:
        if lo < nominal_mm <= hi:
            return ei
    raise FitError(
        f"interference fit '{letter}' is tabulated to {_INTERFERENCE_MAX_MM:.0f} mm in v1; "
        f"for {nominal_mm} mm, supply explicit hole/shaft limits instead"
    )


def shaft_deviation_um(letter: str, grade: int, nominal_mm: float) -> tuple[int, int]:
    """(upper ``es``, lower ``ei``) deviation in micrometers for a shaft letter+grade."""
    it = standard_tolerance_um(nominal_mm, grade)
    d = _geomean(*_band(nominal_mm))
    if letter == "h":
        return 0, -it
    if letter == "g":
        es = round(-2.5 * d**0.34)
        return es, es - it
    if letter == "f":
        es = round(-5.5 * d**0.41)
        return es, es - it
    if letter == "k":
        if not 5 <= grade <= 7:
            raise FitError("shaft 'k' deviation in v1 is defined for grades IT5..IT7")
        ei = round(0.6 * d ** (1 / 3))
        return ei + it, ei
    if letter == "n":
        ei = round(5.0 * d**0.34)
        return ei + it, ei
    if letter in _SHAFT_EI_INTERFERENCE:
        ei = _lookup_interference_ei(letter, nominal_mm)
        return ei + it, ei
    raise FitError(f"shaft letter '{letter}' not supported in v1 (have f, g, h, k, n, p, s, u)")


def hole_deviation_um(letter: str, grade: int, nominal_mm: float) -> tuple[int, int]:
    """(upper ``ES``, lower ``EI``) deviation in micrometers for a hole. v1 = hole-basis ``H``."""
    if letter != "H":
        raise FitError(
            "v1 supports hole-basis 'H' only (e.g. H7); supply explicit limits otherwise"
        )
    return standard_tolerance_um(nominal_mm, grade), 0


# ---------------------------------------------------------------------------
# Designation parsing + fit resolution
# ---------------------------------------------------------------------------
_DESIGNATION = re.compile(r"^\s*([\d.]+)\s*([A-Za-z])\s*(\d+)\s*/\s*([A-Za-z])\s*(\d+)\s*$")


def parse_designation(text: str) -> tuple[float, tuple[str, int], tuple[str, int]]:
    """Parse e.g. ``"50H7/g6"`` or ``"50 H7/g6"`` into (nominal_mm, (hole, grade), (shaft, grade))."""
    m = _DESIGNATION.match(text)
    if not m:
        raise FitError(f"cannot parse fit designation {text!r}; expected e.g. '50 H7/g6'")
    nominal = float(m.group(1))
    if nominal <= 0:
        raise FitError("nominal size must be positive")
    return nominal, (m.group(2).upper(), int(m.group(3))), (m.group(4).lower(), int(m.group(5)))


def _fmt_mm(x: float) -> str:
    return f"{x:g}"


class FitResult(BaseModel):
    """An ISO 286 fit, fully resolved. Deviations are exact integers from the standard."""

    designation: str
    nominal_mm: float
    hole_dev_um: tuple[int, int]  # ES, EI
    shaft_dev_um: tuple[int, int]  # es, ei
    hole_limits_mm: tuple[float, float]  # min, max
    shaft_limits_mm: tuple[float, float]  # min, max
    fit_type: Literal["clearance", "transition", "interference"]
    min_clearance_um: int  # negative value = interference
    max_clearance_um: int
    summary: str
    source: str = "ISO 286-1/2 (computed; validated against published tables)"


def _fit_summary(ftype: str, min_clear: int, max_clear: int, desig: str) -> str:
    if ftype == "clearance":
        return f"{desig}: clearance fit, {min_clear}-{max_clear} um gap."
    if ftype == "interference":
        return f"{desig}: interference fit, {-max_clear}-{-min_clear} um interference (press required)."
    return (
        f"{desig}: transition fit, from {-min_clear} um interference (tightest) "
        f"to {max_clear} um clearance (loosest)."
    )


def resolve_fit(
    nominal_mm: float, hole_code: tuple[str, int], shaft_code: tuple[str, int]
) -> FitResult:
    """Resolve a hole/shaft fit to limits + clearance/interference range. Pure, exact."""
    es_h, ei_h = hole_deviation_um(*hole_code, nominal_mm)
    es_s, ei_s = shaft_deviation_um(*shaft_code, nominal_mm)
    min_clear = ei_h - es_s  # smallest hole minus largest shaft
    max_clear = es_h - ei_s  # largest hole minus smallest shaft
    if min_clear >= 0:
        ftype: Literal["clearance", "transition", "interference"] = "clearance"
    elif max_clear <= 0:
        ftype = "interference"
    else:
        ftype = "transition"
    designation = (
        f"{_fmt_mm(nominal_mm)}{hole_code[0]}{hole_code[1]}/{shaft_code[0]}{shaft_code[1]}"
    )
    return FitResult(
        designation=designation,
        nominal_mm=nominal_mm,
        hole_dev_um=(es_h, ei_h),
        shaft_dev_um=(es_s, ei_s),
        hole_limits_mm=(nominal_mm + ei_h / 1000, nominal_mm + es_h / 1000),
        shaft_limits_mm=(nominal_mm + ei_s / 1000, nominal_mm + es_s / 1000),
        fit_type=ftype,
        min_clearance_um=min_clear,
        max_clearance_um=max_clear,
        summary=_fit_summary(ftype, min_clear, max_clear, designation),
    )


# ---------------------------------------------------------------------------
# Press-fit mechanics (Lamé thick-cylinder)
# ---------------------------------------------------------------------------
class PressFitResult(BaseModel):
    """A press/interference fit assessed by Lamé thick-cylinder elasticity. Decision-grade."""

    designation: str
    nominal_mm: float
    hub_od_mm: float
    engagement_mm: float
    hub_material: str
    shaft_material: str
    interference_um: tuple[int, int]  # diametral, min..max
    contact_pressure_mpa: tuple[float, float]  # at min..max interference
    insertion_force_kn: tuple[float, float]  # loosest+low mu .. tightest+high mu
    torque_capacity_nm: tuple[float, float]
    hub_hoop_stress_mpa: float  # at MAX interference (worst case)
    yield_mpa: float
    safety_factor: float  # yield / hoop stress at max interference
    mu_range: tuple[float, float]
    verdict: Literal["ok", "caution", "fail"]
    recommendation: str
    assumptions: list[str]
    source: str = "Lame thick-cylinder elasticity (computed)"


def _lame_pressure_mpa(
    d: float,
    hub_od: float,
    shaft_bore: float,
    delta_r_mm: float,
    eh_mpa: float,
    nu_h: float,
    es_mpa: float,
    nu_s: float,
) -> float:
    a, b, c = d / 2, hub_od / 2, shaft_bore / 2
    c_hub = (b * b + a * a) / (b * b - a * a) + nu_h
    c_shaft = (1 - nu_s) if c == 0 else ((a * a + c * c) / (a * a - c * c) - nu_s)
    return delta_r_mm / (a * (c_hub / eh_mpa + c_shaft / es_mpa))


def press_fit(
    designation: str,
    hub_od_mm: float,
    engagement_mm: float,
    *,
    hub_material: str = "steel",
    shaft_material: str = "steel",
    shaft_bore_mm: float = 0.0,
    mu_range: tuple[float, float] = (0.10, 0.15),
) -> PressFitResult:
    """Assess the press/interference fit named by ``designation``. Raises :class:`FitError`.

    Worst-case hub stress is taken at the MAXIMUM interference; the insertion force / torque are
    given as a range over the interference band and the friction band. Refuses (does not guess)
    when the fit is not an interference fit or a material's elastic data is unknown.
    """
    nominal, hole_code, shaft_code = parse_designation(designation)
    fit = resolve_fit(nominal, hole_code, shaft_code)
    es_s, ei_s = fit.shaft_dev_um
    es_h, ei_h = fit.hole_dev_um
    max_interf = es_s - ei_h  # diametral, um
    min_interf = max(ei_s - es_h, 0)  # transition's loose end engages at zero interference
    if max_interf <= 0:
        raise FitError(
            f"{fit.designation} is a {fit.fit_type} fit with no interference -- not a press fit"
        )
    if hub_od_mm <= nominal:
        raise FitError(
            f"hub outer diameter ({hub_od_mm} mm) must exceed the interface diameter ({nominal} mm)"
        )
    if shaft_bore_mm >= nominal:
        raise FitError("shaft bore must be smaller than the interface diameter")

    hub = get_material(hub_material)
    shaft = get_material(shaft_material)
    if hub.mechanical is None:
        raise FitError(
            f"elastic data unknown for hub material '{hub.name}' (print-dependent); "
            "use a metal (steel/aluminum/brass) or supply E, nu, yield"
        )
    if shaft.mechanical is None:
        raise FitError(
            f"elastic data unknown for shaft material '{shaft.name}' (print-dependent); "
            "use a metal or supply E, nu, yield"
        )

    eh, nu_h = hub.mechanical.youngs_modulus_gpa * 1000.0, hub.mechanical.poisson_ratio
    e_s, nu_s = shaft.mechanical.youngs_modulus_gpa * 1000.0, shaft.mechanical.poisson_ratio

    def at(interf_um: float) -> float:  # contact pressure (MPa) at a diametral interference
        return _lame_pressure_mpa(
            nominal, hub_od_mm, shaft_bore_mm, interf_um / 1000 / 2, eh, nu_h, e_s, nu_s
        )

    p_min, p_max = at(min_interf), at(max_interf)
    mu_lo, mu_hi = mu_range
    # insertion force F = mu * p * pi * d * L ; loosest/low-mu .. tightest/high-mu
    f_lo = mu_lo * p_min * math.pi * nominal * engagement_mm / 1000.0  # kN
    f_hi = mu_hi * p_max * math.pi * nominal * engagement_mm / 1000.0
    # Torque capacity = F * r. f_*[kN] * (nominal/2)[mm] = kN*mm, and kN*mm == N*m exactly
    # (1000 N * 0.001 m), so there is NO further /1000: the old `/ 1000.0` made the reported
    # torque 1000x too low on a decision-grade "computed" value (audit H3).
    t_lo = f_lo * (nominal / 2)  # kN*mm == N.m
    t_hi = f_hi * (nominal / 2)

    a, b = nominal / 2, hub_od_mm / 2
    hoop_max = p_max * (b * b + a * a) / (b * b - a * a)  # hub bore, at max interference
    yield_mpa = hub.mechanical.yield_strength_mpa
    sf = yield_mpa / hoop_max
    if sf < 1.0:
        verdict: Literal["ok", "caution", "fail"] = "fail"
        rec = (
            f"Hub yields at max interference (hoop {hoop_max:.0f} MPa > {yield_mpa:.0f} MPa yield, "
            f"SF {sf:.2f}). Reduce interference (looser fit) or increase hub OD."
        )
    elif sf < 1.5:
        verdict = "caution"
        rec = (
            f"Hub safety factor {sf:.2f} is thin (hoop {hoop_max:.0f} vs {yield_mpa:.0f} MPa yield). "
            "Consider a looser fit or a larger hub OD for margin."
        )
    else:
        verdict = "ok"
        rec = (
            f"Press fit holds with hub SF {sf:.2f}. Insertion force {f_lo:.1f}-{f_hi:.1f} kN; "
            f"torque capacity {t_lo:.1f}-{t_hi:.1f} N.m. Use a press rated above {f_hi * 1.3:.0f} kN."
        )

    return PressFitResult(
        designation=fit.designation,
        nominal_mm=nominal,
        hub_od_mm=hub_od_mm,
        engagement_mm=engagement_mm,
        hub_material=hub.name,
        shaft_material=shaft.name,
        interference_um=(round(min_interf), round(max_interf)),
        contact_pressure_mpa=(round(p_min, 1), round(p_max, 1)),
        insertion_force_kn=(round(f_lo, 1), round(f_hi, 1)),
        torque_capacity_nm=(round(t_lo, 1), round(t_hi, 1)),
        hub_hoop_stress_mpa=round(hoop_max, 1),
        yield_mpa=yield_mpa,
        safety_factor=round(sf, 2),
        mu_range=mu_range,
        verdict=verdict,
        recommendation=rec,
        assumptions=[
            f"Hub: {hub.name} ({hub.mechanical.basis}); shaft: {shaft.name} ({shaft.mechanical.basis}). "
            "Supply your alloy's yield for an exact safety factor.",
            f"Friction coefficient {mu_lo}-{mu_hi} (dry, clean steel); lubricant lowers insertion force.",
            "Solid shaft" if shaft_bore_mm == 0 else f"Hollow shaft, bore {shaft_bore_mm} mm",
            "Lame thick-cylinder: elastic, smooth concentric contact, uniform temperature; "
            "surface-finish flattening (~few um) and chamfers not modeled.",
            "Hub hoop stress evaluated at MAX interference (worst case) at the bore.",
        ],
    )


class EvaluatedFit(BaseModel):
    """Combined output: the resolved fit plus the press-fit assessment when applicable."""

    fit: FitResult
    press: PressFitResult | None = None
    note: str | None = None


def evaluate(
    designation: str,
    *,
    hub_od_mm: float | None = None,
    engagement_mm: float | None = None,
    hub_material: str = "steel",
    shaft_material: str = "steel",
    shaft_bore_mm: float = 0.0,
    mu_range: tuple[float, float] = (0.10, 0.15),
) -> EvaluatedFit:
    """Resolve a fit, and for an interference fit with hub geometry given, add the press-fit math."""
    nominal, hole_code, shaft_code = parse_designation(designation)
    fit = resolve_fit(nominal, hole_code, shaft_code)
    if fit.fit_type == "clearance":
        return EvaluatedFit(
            fit=fit, note="Clearance fit -- slides freely; no press-fit assessment."
        )
    if hub_od_mm is None or engagement_mm is None:
        return EvaluatedFit(
            fit=fit,
            note="Interference present -- pass hub_od_mm and engagement_mm for the press-fit assessment.",
        )
    press = press_fit(
        designation,
        hub_od_mm,
        engagement_mm,
        hub_material=hub_material,
        shaft_material=shaft_material,
        shaft_bore_mm=shaft_bore_mm,
        mu_range=mu_range,
    )
    return EvaluatedFit(fit=fit, press=press)
