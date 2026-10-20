"""Metric fasteners (M2-M16): the most-placed hardware. Clearance holes per ISO 273, coarse-pitch
tap drills (ISO 261 / general machinist tables), and ISO 4762 socket-head-cap-screw head dims for
counterbored recesses. All standard (not provisional). Values web-verified against ISO 273 /
ISO 4032 references this session; SHCS heads per ISO 4762. M10-M16 added 2026-07-01 (ADR-0067):
the brief legitimately calls large fasteners (a crankshaft flywheel bolt circle is M10/M12), and a
declared size the registry could not supply was a build-crashing poison-pill (the helper raised).

Sources:
  - ISO 273:1979 clearance holes (fine/medium/coarse): https://www.iso.org/standard/4183.html
  - ISO 4762 socket-head cap screw head diameter/height.
  - Coarse-pitch tap drill = nominal - pitch (ISO 261 coarse pitches).
"""

from __future__ import annotations

from forge_mind.dsl.standards.framework import Fit, Registry, StandardSpec

_SRC = "ISO 273 (clearance), ISO 261 coarse-pitch tap drill, ISO 4762 (SHCS head); counterbore = head + 0.5mm recess"


class MetricFastener(StandardSpec):
    """One metric thread size: clearance triple (ISO 273), coarse tap drill, SHCS head + counterbore."""

    nominal_diameter_mm: float
    clearance_close_mm: float  # ISO 273 fine
    clearance_normal_mm: float  # ISO 273 medium
    clearance_coarse_mm: float  # ISO 273 coarse
    tap_drill_coarse_mm: float
    socket_head_diameter_mm: float  # ISO 4762
    socket_head_height_mm: float
    counterbore_diameter_mm: float  # SHCS head + recess clearance


def _f(
    key: str, nom: float, clr: tuple[float, float, float], tap: float, head_d: float, head_h: float
) -> MetricFastener:
    return MetricFastener(
        key=key,
        source=_SRC,
        nominal_diameter_mm=nom,
        clearance_close_mm=clr[0],
        clearance_normal_mm=clr[1],
        clearance_coarse_mm=clr[2],
        tap_drill_coarse_mm=tap,
        socket_head_diameter_mm=head_d,
        socket_head_height_mm=head_h,
        counterbore_diameter_mm=round(head_d + 0.5, 2),
    )


FASTENERS: Registry[MetricFastener] = Registry(
    "metric_fastener",
    (
        _f("M2", 2.0, (2.2, 2.4, 2.6), 1.6, 3.8, 2.0),
        _f("M2.5", 2.5, (2.7, 2.9, 3.1), 2.05, 4.5, 2.5),
        _f("M3", 3.0, (3.2, 3.4, 3.6), 2.5, 5.5, 3.0),
        _f("M4", 4.0, (4.3, 4.5, 4.8), 3.3, 7.0, 4.0),
        _f("M5", 5.0, (5.3, 5.5, 5.8), 4.2, 8.5, 5.0),
        _f("M6", 6.0, (6.4, 6.6, 7.0), 5.0, 10.0, 6.0),
        _f("M8", 8.0, (8.4, 9.0, 10.0), 6.8, 13.0, 8.0),
        # M10-M16: ISO 273 medium clearance (close/medium/coarse), ISO 4762 SHCS head dia, coarse
        # tap drill = nominal - coarse pitch (1.5/1.75/2.0/2.0). Web-verified 2026-07-01.
        _f("M10", 10.0, (10.5, 11.0, 12.0), 8.5, 16.0, 10.0),
        _f("M12", 12.0, (13.0, 13.5, 14.5), 10.2, 18.0, 12.0),
        _f("M14", 14.0, (15.0, 15.5, 16.5), 12.0, 21.0, 14.0),
        _f("M16", 16.0, (17.0, 17.5, 18.5), 14.0, 24.0, 16.0),
    ),
)

_FIT_FIELD = {
    Fit.CLOSE: "clearance_close_mm",
    Fit.NORMAL: "clearance_normal_mm",
    Fit.COARSE: "clearance_coarse_mm",
}


def clearance_hole(size: str, fit: Fit | str = Fit.NORMAL) -> float:
    """Clearance-hole DIAMETER (mm) for a bolt to pass through freely, per ISO 273 fit class."""
    return float(getattr(FASTENERS.get(size), _FIT_FIELD[Fit(fit)]))


def tap_drill(size: str) -> float:
    """Tapping-drill DIAMETER (mm) for cutting a coarse-pitch thread into the part."""
    return FASTENERS.get(size).tap_drill_coarse_mm


def thread_nominal(size: str) -> float:
    """Nominal (major) DIAMETER (mm) of the thread -- the size that IDENTIFIES it (M8 -> 8.0).
    The modeling convention for a THREADED hole in a single-piece part (ADR-0081): the bore
    represents the thread's nominal envelope, not the tap drill and not the bolt clearance."""
    return float(FASTENERS.get(size).nominal_diameter_mm)


def counterbore(size: str) -> tuple[float, float]:
    """(diameter, depth) mm to recess a socket-head cap screw flush (ISO 4762 head + clearance)."""
    f = FASTENERS.get(size)
    return f.counterbore_diameter_mm, f.socket_head_height_mm


__all__ = [
    "FASTENERS",
    "MetricFastener",
    "clearance_hole",
    "counterbore",
    "tap_drill",
    "thread_nominal",
]
