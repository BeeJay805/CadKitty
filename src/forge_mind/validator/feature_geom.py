"""Geometry-level feature-presence verification (Sol consultation 2026-07-24, P0.3 / task 3,
decision-independent): is a declared ICD feature actually REALIZED in the built geometry, at
its declared station and size — measured on the B-rep, not asserted in a card?

The H3 gap: a plan card can declare "Ø14.3 hip bore, axis Z through (0,26,167)" and the
author can emit a part with no such bore, or one 14 mm off station, and nothing measured it.
``interface_audit`` checks stated OD *magnitudes* but not *placement*; this closes that: it
finds a bore of the declared diameter whose axis line passes through the declared station.

Standalone by design (like ``symmetry``): returns its own fail-closed verdict, NOT wired into
the live acceptance/disposition path (that is the held intent-gate reclassification). Reuses
the shared ``GeomVerdict`` result type (origin: ``symmetry``; consolidate to a geom_verify
module once the family grows). Builds on the existing ``extract_cylinders`` / ``cluster_coaxial``
B-rep feature extractor — no new topology code.

Discipline (cad-validator-designer): returns the measured value + limit + evidence, and
``not_run`` — never PASS — when the geometry cannot be extracted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from forge_mind.validator.symmetry import GeomVerdict

Vec3 = tuple[float, float, float]


def finite_positive(x: Any) -> bool:
    """A usable positive scalar (rejects None, non-numbers, nan/inf, <= 0). Shared fail-closed
    input guard for the geometry validators (mechanics_geom imports this)."""
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0


def finite_nonneg(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x >= 0.0


def _three_finite(v: Any) -> tuple[float, float, float] | None:
    """``v`` as exactly three finite floats, or None. Rejects None, wrong-length sequences
    (a 2- or 4-vector that would silently broadcast or truncate later), and nan/inf."""
    try:
        seq = list(v)
    except Exception:
        return None
    if len(seq) != 3:
        return None
    try:
        a, b, c = (float(seq[0]), float(seq[1]), float(seq[2]))
    except Exception:
        return None
    return (a, b, c) if all(math.isfinite(x) for x in (a, b, c)) else None


def finite_point(p: Any) -> bool:
    """A usable 3D point: exactly three finite components (any values)."""
    return _three_finite(p) is not None


def valid_axis(direction: Any) -> bool:
    """Exactly three finite components with a nonzero length. A zero/nan/wrong-length axis
    silently collapses a lever/parallelism test, so it must fail closed rather than compute a
    defensible-looking result."""
    t = _three_finite(direction)
    if t is None:
        return False
    a, b, c = t
    return math.sqrt(a * a + b * b + c * c) > 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _point_to_line_distance(p: Vec3, line_point: Vec3, line_dir: Vec3) -> float:
    """Perpendicular distance from point ``p`` to the infinite line through ``line_point``
    along ``line_dir`` — i.e. how far the declared station sits off the measured bore axis."""
    pa = np.asarray(p, dtype=float) - np.asarray(line_point, dtype=float)
    d = _unit(np.asarray(line_dir, dtype=float))
    return float(np.linalg.norm(pa - d * float(np.dot(pa, d))))


def _axes_parallel(d1: Vec3, d2: Vec3, tol_deg: float) -> bool:
    a = _unit(np.asarray(d1, dtype=float))
    b = _unit(np.asarray(d2, dtype=float))
    cos = abs(float(np.dot(a, b)))
    ang = float(np.degrees(np.arccos(min(1.0, max(0.0, cos)))))
    return ang <= tol_deg


def _axis_projection(p: Vec3, axis_dir: Vec3) -> float:
    """Signed coordinate of point ``p`` along ``axis_dir`` (the same basis a CylinderStack's
    ``axial_extent`` is measured in) -- so the declared station can be tested for containment
    inside the bore's measured axial span."""
    d = _unit(np.asarray(axis_dir, dtype=float))
    return float(np.dot(np.asarray(p, dtype=float), d))


def bore_at_station_in_stacks(
    stacks: list[Any],
    *,
    diameter_mm: float,
    axis_point: Vec3,
    axis_dir: Vec3,
    dia_tol_mm: float = 0.4,
    station_tol_mm: float = 1.0,
    angle_tol_deg: float = 5.0,
) -> GeomVerdict:
    """Pure check over already-extracted coaxial cylinder stacks (the unit-testable core):
    is there a HOLE (concave) stack of diameter ``diameter_mm`` whose axis is parallel to
    ``axis_dir`` and passes through ``axis_point``? PASS with the measured match, else FAIL
    naming the nearest parallel bore (present-but-misplaced vs absent are distinguished)."""
    check_id = "feature.bore_at_station"
    cat = "features"
    limit = {
        "diameter_mm": diameter_mm,
        "dia_tol_mm": dia_tol_mm,
        "station_tol_mm": station_tol_mm,
    }
    # Fail closed on a malformed query: a nonpositive diameter or a degenerate axis makes the
    # match meaningless (a zero axis reads as parallel to nothing). not_run, never a verdict.
    if not finite_positive(diameter_mm):
        return GeomVerdict(check_id, cat, "not_run", f"invalid diameter {diameter_mm!r}", {}, limit)
    if not valid_axis(axis_dir):
        return GeomVerdict(
            check_id, cat, "not_run", f"degenerate declared axis {axis_dir!r}", {}, limit
        )
    holes = [s for s in stacks if getattr(s, "concave", None) is True]
    if not holes:
        return GeomVerdict(
            check_id,
            cat,
            "fail",
            "no bore (concave cylinder) found in the part",
            {"n_holes": 0.0},
            limit,
        )
    best: tuple[float, float, float] | None = None  # (dia, radial_off, dia_err) nearest-by-dia
    misplaced_axial = False  # a dia+axis+radial match exists but the station is off its axial span
    unmeasurable_extent = False  # a match exists but its axial span could not be measured
    for s in holes:
        if not _axes_parallel(s.axis_dir, axis_dir, angle_tol_deg):
            continue
        d = 2.0 * float(s.radius)
        dia_err = abs(d - diameter_mm)
        radial_off = _point_to_line_distance(axis_point, s.axis_point, s.axis_dir)
        if dia_err <= dia_tol_mm and radial_off <= station_tol_mm:
            # Right diameter, right axis LINE, right radial offset. But a bore anywhere ALONG
            # that infinite line would pass so far -- require the station to fall inside the
            # bore's measured TRIMMED axial span (Sol P0.3 false-PASS catch).
            ext = getattr(s, "axial_extent", None)
            if ext is None:
                unmeasurable_extent = True  # cannot confirm the station is inside this bore
            else:
                station_axial = _axis_projection(axis_point, s.axis_dir)
                if ext[0] - station_tol_mm <= station_axial <= ext[1] + station_tol_mm:
                    measured = {
                        "measured_dia_mm": d,
                        "dia_err_mm": dia_err,
                        "radial_offset_mm": radial_off,
                        "axial_station_mm": station_axial,
                        "axial_span_min_mm": float(ext[0]),
                        "axial_span_max_mm": float(ext[1]),
                    }
                    msg = (
                        f"declared Ø{diameter_mm} bore realized: measured Ø{d:.2f} "
                        f"(err {dia_err:.2f} mm), radial {radial_off:.2f} mm, station within "
                        f"axial span [{ext[0]:.1f}, {ext[1]:.1f}]"
                    )
                    return GeomVerdict(check_id, cat, "pass", msg, measured, limit)
                misplaced_axial = True  # on the axis, wrong axial position
        if best is None or dia_err < best[2]:
            best = (d, radial_off, dia_err)
    # No fully-verified match. Fail closed when a candidate's extent was unmeasurable.
    if unmeasurable_extent:
        return GeomVerdict(
            check_id,
            cat,
            "not_run",
            "a bore matches the declared diameter/axis/radial station but its axial span is "
            "unmeasurable -- cannot confirm the station lies inside the bore",
            {"n_holes": float(len(holes))},
            limit,
        )
    if best is None:
        return GeomVerdict(
            check_id,
            cat,
            "fail",
            f"no bore parallel to the declared axis {tuple(float(x) for x in axis_dir)} found",
            {"n_holes": float(len(holes))},
            limit,
        )
    d, radial_off, dia_err = best
    measured = {"nearest_dia_mm": d, "dia_err_mm": dia_err, "radial_offset_mm": radial_off}
    reason = (
        "a bore of that diameter is on the axis but the station is outside its axial span"
        if misplaced_axial
        else "nearest parallel bore is off diameter and/or radial station"
    )
    msg = (
        f"declared Ø{diameter_mm} bore at station NOT realized ({reason}): nearest parallel bore "
        f"Ø{d:.2f} (dia err {dia_err:.2f} mm, limit {dia_tol_mm}), radial {radial_off:.2f} mm "
        f"(limit {station_tol_mm})"
    )
    return GeomVerdict(check_id, cat, "fail", msg, measured, limit)


def bore_at_station(
    part: Any,
    *,
    diameter_mm: float,
    axis_point: Vec3,
    axis_dir: Vec3,
    dia_tol_mm: float = 0.4,
    station_tol_mm: float = 1.0,
    angle_tol_deg: float = 5.0,
) -> GeomVerdict:
    """Full check over a built build123d ``part``: extract coaxial cylinder stacks, then run
    :func:`bore_at_station_in_stacks`. Fail-closed ``not_run`` when extraction fails (a part
    with no measurable topology is never a silent PASS)."""
    check_id = "feature.bore_at_station"
    limit = {"diameter_mm": diameter_mm, "dia_tol_mm": dia_tol_mm, "station_tol_mm": station_tol_mm}
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = cluster_coaxial(extract_cylinders(part))
    except Exception as exc:
        return GeomVerdict(
            check_id,
            "features",
            "not_run",
            f"cylinder extraction failed: {type(exc).__name__}: {exc}",
            {},
            limit,
        )
    return bore_at_station_in_stacks(
        stacks,
        diameter_mm=diameter_mm,
        axis_point=axis_point,
        axis_dir=axis_dir,
        dia_tol_mm=dia_tol_mm,
        station_tol_mm=station_tol_mm,
        angle_tol_deg=angle_tol_deg,
    )


__all__ = ["bore_at_station", "bore_at_station_in_stacks"]
