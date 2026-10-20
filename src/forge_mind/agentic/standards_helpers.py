"""Standards-backed hardware DIMENSIONS for the agentic author (importable in the sandbox).

The author hand-models hole diameters today ("an M3 clearance hole" -> it guesses 3.0 or 3.5).
This wrapper hands it the EXACT ISO/DIN value instead, so a standard part's mating geometry is
never invented. Each function is a thin, friendly re-export of the canonical
``forge_mind.dsl.standards`` registry (pure data, no geometry, no I/O) -- the SAME source of truth
the validator's conformance gate checks against, so what the author builds and what the validator
expects can never drift.

Pattern: these return plain numbers (mm); feed them into the geometry helpers in
``forge_mind.agentic.helpers`` (``bolt_holes`` / ``counterbore_tool`` / ``bearing_seat_tool``).

    plate - (Pos(0, 0, 3) * bolt_holes(pcd=40, hole_d=clearance_hole_d("M5"), count=4, depth=10))

PROVENANCE: dimensions marked PROVISIONAL below are NOT from a published standard (a manufacturer
recommendation or an FDM rule-of-thumb) -- prefer a published-standard size where one exists, and
verify before production. An unknown size/designation raises ``ValueError`` listing the valid keys,
so a hallucinated designation fails loudly rather than silently.
"""

from __future__ import annotations

from forge_mind.dsl.standards import (
    bearing,
    bearing_housing_bore,
    captive_nut_pocket,
    clearance_hole,
    counterbore,
    dowel_hole,
    heat_set_insert,
    magnet_pocket,
    tap_drill,
    thread_nominal,
)


def clearance_hole_d(size: str, fit: str = "normal") -> float:
    """Through-hole DIAMETER (mm) for a bolt to pass freely (ISO 273). ``fit``: close|normal|coarse.

    Sizes: M2, M2.5, M3, M4, M5, M6, M8. E.g. clearance_hole_d("M5") -> 5.5.
    """
    return clearance_hole(size, fit)


def tap_hole_d(size: str) -> float:
    """Tapping-drill DIAMETER (mm) to cut a coarse-pitch thread INTO the part (ISO 261). E.g. M3 -> 2.5."""
    return tap_drill(size)


def thread_hole_d(size: str) -> float:
    """Nominal (major) DIAMETER (mm) for modeling a THREADED / TAPPED / set-screw hole (ADR-0081):
    the bore represents the thread's nominal envelope. E.g. thread_hole_d("M8") -> 8.0. Use this
    for 'M8 threaded bore', 'M4 tapped', 'M3 set-screw hole'; use ``clearance_hole_d`` only when a
    bolt PASSES THROUGH, and drill a raw STATED diameter exactly."""
    return thread_nominal(size)


def counterbore_dims(size: str) -> tuple[float, float]:
    """(diameter, depth) mm to recess a socket-head cap screw flush (ISO 4762 head + clearance)."""
    return counterbore(size)


def nut_pocket_dims(size: str, clearance_mm: float = 0.2) -> tuple[float, float, float]:
    """(across_flats, across_corners, depth) mm for a side pocket that captures a hex nut (ISO 4032)."""
    return captive_nut_pocket(size, clearance_mm)


def bearing_seat_dims(designation: str, fit: str = "press") -> tuple[float, float, float]:
    """(shaft_bore_d, seat_d, width) mm for a deep-groove ball bearing (ISO 15 boundary dims).

    ``shaft_bore_d`` = the bearing bore (shaft passes through), ``seat_d`` = the housing bore the
    bearing OD seats into (the press/slip FIT offset is an FDM convention -- PROVISIONAL), ``width``
    = the seat depth. Feed into bearing_seat_tool(bore_d=shaft_bore_d, seat_d=seat_d,
    seat_depth=width, thru_depth=...). Designations: 623, 625, 626, 608, 688, 6800, 6900, 6000.
    """
    b = bearing(designation)
    return (
        b.bore_id_mm,
        bearing_housing_bore(designation, "press" if fit == "press" else "slip"),
        b.width_mm,
    )


def heat_set_hole_d(size: str) -> float:
    """Printed pilot-hole DIAMETER (mm) a heat-set threaded insert melts into. PROVISIONAL
    (manufacturer guidance, not ISO) -- verify against the insert datasheet. Sizes M2-M5."""
    return heat_set_insert(size).hole_diameter_mm


def dowel_hole_d(size: str, fit: str = "press") -> float:
    """Hole DIAMETER (mm) for a dowel pin (ISO 2338 nominal; the press/slip FIT offset is an FDM
    convention -- PROVISIONAL). Sizes (mm): 2, 3, 4, 5, 6, 8."""
    return dowel_hole(size, "press" if fit == "press" else "slip")


def magnet_pocket_dims(size: str, clearance_mm: float = 0.1) -> tuple[float, float]:
    """(diameter, depth) mm for a slip-fit disc-magnet pocket. PROVISIONAL (no single standard).
    Sizes (dia x thk): 5x2, 6x3, 8x3, 10x3."""
    return magnet_pocket(size, clearance_mm)


__all__ = [
    "bearing_seat_dims",
    "clearance_hole_d",
    "counterbore_dims",
    "dowel_hole_d",
    "heat_set_hole_d",
    "magnet_pocket_dims",
    "nut_pocket_dims",
    "tap_hole_d",
    "thread_hole_d",
]
