"""Checker localization (ADR-0015): the per-placement region must INCLUDE an interface's
own bolts yet EXCLUDE a same-radius neighbour's bolts at the minimum feasible separation.

Step 0 de-risk: prove a clean radial window exists for the realistic design space before
localizing the checkers. Pure geometry, deterministic, no compile.
"""

from __future__ import annotations

import math

from forge_mind.dsl.interfaces.fan import (
    FAN_40MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_40MM_BOLT_SQUARE_MM,
    FAN_60MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_60MM_BOLT_SQUARE_MM,
)
from forge_mind.dsl.interfaces.nema17 import BOLT_PITCH_MM, M3_CLEARANCE_RADIUS_MM
from forge_mind.validator.interfaces.base import region_radius_for_square

# Each M3 bolt-pattern interface: (square side, axis-aligned half-extent used by the
# composable no-overlap feasibility). nema's half-extent is the M3 cluster (15.5+1.6);
# a fan's is max(bolt half + M3 r, airflow bore r) -- matching _interface_half_extent.
_M3R = M3_CLEARANCE_RADIUS_MM
_INTERFACES = {
    "nema17": (BOLT_PITCH_MM, max(BOLT_PITCH_MM / 2 + _M3R, 0.0)),  # 31, 17.1
    "fan_40": (
        FAN_40MM_BOLT_SQUARE_MM,
        max(FAN_40MM_BOLT_SQUARE_MM / 2 + _M3R, FAN_40MM_AIRFLOW_BORE_RADIUS_MM),
    ),  # 32, 19
    "fan_60": (
        FAN_60MM_BOLT_SQUARE_MM,
        max(FAN_60MM_BOLT_SQUARE_MM / 2 + _M3R, FAN_60MM_AIRFLOW_BORE_RADIUS_MM),
    ),  # 50, 28
}


def _corner_radial(square: float) -> float:
    return (square / 2.0) * math.sqrt(2.0)


def _neighbour_nearest_bolt(neighbour_square: float, center_sep: float) -> float:
    """Distance from THIS interface's centre (origin) to the nearest bolt of a neighbour
    whose centre sits ``center_sep`` away on +X. Neighbour bolts are at
    (sep +/- sq/2, +/- sq/2); the nearest is (sep - sq/2, +/- sq/2)."""
    half = neighbour_square / 2.0
    return math.hypot(center_sep - half, half)


def _window(own: str, neighbour: str, min_wall: float) -> tuple[float, float, float]:
    """(region_radius, own_corner, neighbour_nearest) at the minimum feasible separation."""
    own_sq, own_half = _INTERFACES[own]
    nbr_sq, nbr_half = _INTERFACES[neighbour]
    sep = own_half + nbr_half + min_wall  # the no-overlap feasibility floor (ADR-0014)
    return (
        region_radius_for_square(own_sq),
        _corner_radial(own_sq),
        _neighbour_nearest_bolt(nbr_sq, sep),
    )


def test_region_separates_all_pairs_at_bench_min_wall() -> None:
    # At the bench / payoff wall thickness (4 mm), the region cleanly INCLUDES own bolts
    # and EXCLUDES every same-radius neighbour's nearest bolt -- for ALL M3 pairs,
    # including the binding fan_60-next-to-nema17 case.
    names = list(_INTERFACES)
    for own in names:
        for nbr in names:
            radius, own_corner, nbr_nearest = _window(own, nbr, min_wall=4.0)
            assert radius >= own_corner, f"{own}: region drops its own bolts"
            assert radius < nbr_nearest, (
                f"{own} next to {nbr} @ mw=4: region {radius:.2f} catches the neighbour's "
                f"bolt at {nbr_nearest:.2f} (own corner {own_corner:.2f})"
            )


def test_binding_case_is_fan60_next_to_nema17() -> None:
    # The tightest window: a 60 mm fan's large square reaches out toward a small NEMA 17.
    radius, own_corner, nbr_nearest = _window("fan_60", "nema17", min_wall=4.0)
    assert math.isclose(own_corner, 35.36, abs_tol=0.05)
    assert nbr_nearest < 37.5  # ~37.0
    assert own_corner < radius < nbr_nearest  # clean window exists, margin fits inside it


def test_dual_nema_window_is_comfortable() -> None:
    # The user's stated de-risk case (own ~21.9, neighbour ~27.5): comfortably clean,
    # even at a thin 2 mm wall.
    radius, own_corner, nbr_nearest = _window("nema17", "nema17", min_wall=2.0)
    assert own_corner < radius < nbr_nearest
    assert nbr_nearest - own_corner > 3.0  # wide window


def test_known_bound_fan60_nema_interleaves_below_thin_wall() -> None:
    # HONEST BOUND (documented, not shipped-around): below ~2.2 mm wall a 60 mm fan's
    # corner bolts interleave RADIALLY with an adjacent NEMA 17's bolts -- no radial
    # window can separate them (the neighbour's nearest bolt falls inside the fan's own
    # corner radial). This is a conservative FALSE-FAIL (never a false-pass) on an
    # extreme corner no bench/payoff hits; the robust fix is pattern-fit selection,
    # deferred (ADR-0015 Q5). This test PINS the bound so it is explicit, not a surprise.
    _radius, own_corner, nbr_nearest = _window("fan_60", "nema17", min_wall=2.0)
    assert nbr_nearest < own_corner  # window has closed: radial cannot separate here


# --- ADR-0017: extrusion-pair localization de-risk. The extrusion's 2 T-nut holes sit on a
# LINE at +/- spacing/2 from the centre (not a square), so its region radial is spacing/2.
# Prove a clean window exists for the same-radius combos BEFORE localizing the M5 checker.
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM as _M5  # noqa: E402
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM as _M6  # noqa: E402
from forge_mind.validator.interfaces.base import region_radius_for_pair  # noqa: E402
from forge_mind.validator.interfaces.extrusion_2020_slot import SPACING_TOL as _STOL  # noqa: E402

# NEMA 23 is the only M5-bolt MOTOR (its bolts would pollute a same-M5 extrusion's count).
_NEMA23_SQUARE = 47.14
_NEMA23_HALF = max(_NEMA23_SQUARE / 2.0 + _M5, 19.05)  # 26.17 (matches _interface_half_extent)


def _ext_region(spacing: float) -> tuple[float, float, float]:
    """(region, own corner, loosest in-tolerance corner). The region must INCLUDE both own
    distances (so an in-tolerance loose-fit pair is not wrongly rejected)."""
    return region_radius_for_pair(spacing, _STOL), spacing / 2.0, (spacing + _STOL) / 2.0


def _ext_vs_motor_nearest(
    spacing: float, ext_clear: float, motor_sq: float, motor_half: float, mw: float
):
    """Nearest same-radius MOTOR bolt at the min feasible separation (no-overlap floor)."""
    sep = ext_clear + motor_half + mw
    return math.hypot(sep - motor_sq / 2.0, motor_sq / 2.0)


def _ext_vs_ext_nearest(nbr_spacing: float, clear: float, mw: float):
    """Nearest same-radius EXTRUSION hole: neighbour holes at (sep, +/- Sn/2)."""
    sep = clear + clear + mw
    return math.hypot(sep, nbr_spacing / 2.0)


def test_extrusion_region_separates_nema23_plus_2020() -> None:
    # NEMA 23 (M5) + 2020 (M5): the region includes the extrusion's 2 own M5 holes (incl. the
    # loose-fit tolerance) and excludes the NEMA 23's 4 M5 bolts at the min feasible separation.
    region, _own, own_tol = _ext_region(20.0)
    nearest = _ext_vs_motor_nearest(20.0, _M5, _NEMA23_SQUARE, _NEMA23_HALF, 4.0)
    assert region >= own_tol, "region drops an in-tolerance T-nut pair"
    assert region < nearest, f"region {region:.2f} catches a NEMA 23 bolt at {nearest:.2f}"


def test_extrusion_region_separates_dual_2020_and_3030() -> None:
    for spacing, clear in ((20.0, _M5), (30.0, _M6)):
        region, _own, own_tol = _ext_region(spacing)
        nearest = _ext_vs_ext_nearest(spacing, clear, 4.0)
        assert region >= own_tol, (
            f"dual S={spacing}: region {region:.2f} drops in-tol own {own_tol:.2f}"
        )
        assert region < nearest, (
            f"dual extrusion S={spacing}: region {region:.2f} catches neighbour at {nearest:.2f}"
        )


def test_extrusion_mixed_tiny_spacing_is_a_known_false_fail() -> None:
    # HONEST BOUND (pinned, not shipped-around): a default-spacing extrusion next to a
    # same-radius TINY-spacing one closes the radial window. The registry default spacings
    # (20/30) never hit this -- it needs an explicit tiny spacing -- and it is a conservative
    # FALSE-FAIL (over-reject), never a false-pass. Mirrors the ADR-0015 fan60/nema bound.
    region, _own, _own_tol = _ext_region(20.0)
    nearest = _ext_vs_ext_nearest(6.0, _M5, 4.0)
    assert nearest < region  # window has closed for this exotic mix
