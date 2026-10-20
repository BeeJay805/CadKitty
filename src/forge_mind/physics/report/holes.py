"""Detected hole features + fastener-size classification for BoltBrain.

``DetectedHole`` is a PLAIN measurement (no build123d) so the projectors/composer stay light — the seam
runs ``extract_cylinders`` on the STEP B-rep and hands these over, exactly like the validator hands over
finished ``Check`` objects. ``classify_hole`` matches a measured bore diameter to a standard metric
fastener (ISO 273 clearance / ISO 261 coarse tap drill) — a deterministic registry lookup, decision-grade.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.dsl.standards.fasteners import FASTENERS
from forge_mind.types import Vec3

# Metric sizes the registry carries (M2..M8). Iterated for nearest-match classification.
_SIZES = ("M2", "M2.5", "M3", "M4", "M5", "M6", "M8")
# A bore must be within this of a standard value to be CALLED that size — tight enough that a
# slightly-off hole reads as "non-standard" rather than being snapped to the wrong fastener.
_MATCH_TOL_MM = 0.3

# A fastener hole's CENTER should sit at least this multiple of its diameter from the part edge, or it
# can blow out under load/torque (a common machinist/FDM rule of thumb).
MIN_EDGE_DISTANCE_RATIO = 1.5
# A hole smaller than this prints unreliably on a typical 0.4 mm FDM nozzle (closes up / needs drilling).
MIN_PRINTABLE_HOLE_MM = 2.0


@dataclass(frozen=True)
class DetectedHole:
    """A measured cylindrical hole wall (concave) from the B-rep, in the STL/B-rep frame (stl_z_up).

    ``wall_mm`` is the measured minimum material thickness from the hole wall to the nearest part edge
    (None if it could not be measured). center-to-edge distance = ``diameter_mm/2 + wall_mm``."""

    diameter_mm: float
    center_mm: Vec3
    axis: Vec3
    depth_mm: float | None = None
    segments: int = 1
    wall_mm: float | None = None


@dataclass(frozen=True)
class HoleClass:
    """The nearest standard fastener a bore matches: size (e.g. 'M3'), kind, and the matched value."""

    size: str
    kind: str  # "clearance" | "tap drill"
    nominal_value_mm: float


def classify_hole(diameter_mm: float) -> HoleClass | None:
    """Nearest standard fastener for a measured bore, or None if nothing is within _MATCH_TOL_MM.

    Prefers the closest match across every fastener's clearance triple (ISO 273) and coarse tap drill
    (ISO 261). Returns None for a non-standard diameter rather than snapping to a wrong size."""
    best: HoleClass | None = None
    best_err = _MATCH_TOL_MM
    for size in _SIZES:
        f = FASTENERS.get(size)
        candidates = (
            ("clearance", f.clearance_close_mm),
            ("clearance", f.clearance_normal_mm),
            ("clearance", f.clearance_coarse_mm),
            ("tap drill", f.tap_drill_coarse_mm),
        )
        for kind, value in candidates:
            err = abs(diameter_mm - value)
            if err < best_err:
                best_err = err
                best = HoleClass(size=size, kind=kind, nominal_value_mm=value)
    return best


__all__ = [
    "MIN_EDGE_DISTANCE_RATIO",
    "MIN_PRINTABLE_HOLE_MM",
    "_MATCH_TOL_MM",
    "DetectedHole",
    "HoleClass",
    "classify_hole",
]
