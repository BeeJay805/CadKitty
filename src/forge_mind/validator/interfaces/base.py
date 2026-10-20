"""BaseInterface — contract for an interface checker plus shared hole-matching
helpers used by every concrete interface.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration
from forge_mind.types import Vec3
from forge_mind.validator.geometry import CylinderFeature, planar_distance
from forge_mind.validator.report import Check, Fix

Vec3T = tuple[float, float, float]


@dataclass(frozen=True)
class InterfaceContext:
    """Geometry a checker may need beyond the cylindrical features.

    Cylinder-based checkers (nema17, extrusion, m5) ignore this; checkers that
    reason about non-cylindrical geometry (e.g. pi4_ports' rectangular wall
    openings) use ``stl_path``, and lands are measured against ``min_wall_mm``.
    """

    part: Any  # build123d.Part
    stl_path: Path | None
    min_wall_mm: float | None


class BaseInterface(abc.ABC):
    """A checkable mating interface.

    ``check`` receives the declaration (where the part claims the interface is),
    all cylindrical features of the body, and an InterfaceContext, and returns
    granular Checks plus Fix hints.
    """

    #: interface_type value this checker handles (matches InterfaceType).
    type_name: str

    @abc.abstractmethod
    def check(
        self,
        declaration: InterfaceDeclaration,
        cylinders: list[CylinderFeature],
        ctx: InterfaceContext,
    ) -> tuple[list[Check], list[Fix]]:
        raise NotImplementedError


def holes_matching(
    cylinders: list[CylinderFeature],
    normal: Vec3,
    radius: float,
    radius_tol: float,
    axis_tol_deg: float = 5.0,
) -> list[CylinderFeature]:
    """Return HOLE walls whose axis is (anti)parallel to ``normal`` and whose radius is within
    ``radius_tol`` of ``radius``.

    Only a measured-concave cylinder (``concave is True``) counts: a boss/pin (``False``) or an
    unmeasurable face (``None``) must NOT satisfy a required clearance hole. Without this guard a
    solid pin of the right radius and axis passed the interface tier -- a false PASS on a part that
    cannot actually be bolted (audit H1; lands the deferred ADR-0035 B1). Fail closed: an unmeasured
    cylinder never certifies a hole.
    """
    n: Vec3T = (normal.x, normal.y, normal.z)
    out: list[CylinderFeature] = []
    for c in cylinders:
        if c.concave is not True:
            continue
        if not c.axis_is_parallel_to(n, axis_tol_deg):
            continue
        if abs(c.radius - radius) <= radius_tol:
            out.append(c)
    return out


def is_square_pattern(
    bolts: list[CylinderFeature],
    axis: Vec3T,
    side_mm: float,
    side_tol_mm: float,
) -> bool:
    """True if four bolt centers form a square of side ``side_mm`` (within tol).

    A square has, among its six pairwise center distances, four equal to the side
    and two equal to the diagonal (side*sqrt(2)). Shared by the NEMA17 face and the
    fan face -- both are "4 bolts on a square of side S"; only S and the radius
    differ. (NEMA17's checker predates this helper and keeps its inline copy; new
    square-pattern interfaces call this so the math is defined once.)
    """
    if len(bolts) != 4:
        return False
    pts = [b.axis_point for b in bolts]
    dists = sorted(planar_distance(pts[i], pts[j], axis) for i in range(4) for j in range(i + 1, 4))
    sides, diagonals = dists[:4], dists[4:]
    expected_diag = side_mm * math.sqrt(2)
    sides_ok = all(abs(s - side_mm) <= side_tol_mm for s in sides)
    diag_ok = all(abs(d - expected_diag) <= side_tol_mm * math.sqrt(2) for d in diagonals)
    return sides_ok and diag_ok


LOCALIZATION_MARGIN_MM = 0.5  # ADR-0015 Q1: per-placement region = corner radial + this.


def region_radius_for_square(bolt_square_mm: float) -> float:
    """In-plane radius that scopes a square bolt pattern's hole count to its declared
    centre (ADR-0015): the pattern's corner radial (half-diagonal) + a small margin.

    The margin is small ON PURPOSE: the region must INCLUDE the pattern's own 4 corner
    bolts (exactly at the corner radial) yet EXCLUDE a same-radius neighbour's bolts at
    the minimum feasible separation. The binding case is a 60 mm fan next to a NEMA 17
    (the fan's large square reaches out toward the small motor); the clean-window proof +
    its bound are in tests/unit/validator/test_interface_localization.py.
    """
    return (bolt_square_mm / 2.0) * math.sqrt(2.0) + LOCALIZATION_MARGIN_MM


def region_radius_for_pair(spacing_mm: float, spacing_tol_mm: float = 0.0) -> float:
    """In-plane radius that scopes a 2-hole T-nut PAIR's hole count to its declared centre
    (ADR-0017): the pair's half-spacing (holes sit at +/- spacing/2) + the slot's spacing
    tolerance + a small margin. Including the spacing tolerance is load-bearing: T-nuts slide
    in the channel, so an in-tolerance pair's measured spacing may legitimately differ by up
    to spacing_tol -- its holes must stay INSIDE the region, else a valid loose-fit part would
    be wrongly count-rejected (and the spacing check it should reach would be dead). The
    line-pair analogue of ``region_radius_for_square``; the clean-window proof (own + the slot
    tolerance INCLUDED, a same-radius neighbour EXCLUDED) is in test_interface_localization.py."""
    return spacing_mm / 2.0 + spacing_tol_mm + LOCALIZATION_MARGIN_MM


def holes_in_region(
    holes: list[CylinderFeature], center: Vec3, normal: Vec3, radius: float
) -> list[CylinderFeature]:
    """Keep only holes whose axis is within ``radius`` (in-plane) of ``center`` -- the
    per-placement localization that scopes a global hole list to ONE declared interface
    (ADR-0015). With a single interface (or no same-radius peer) every own hole is inside
    its region, so the localized count is byte-identical to the global count.
    """
    c: Vec3T = (center.x, center.y, center.z)
    ax: Vec3T = (normal.x, normal.y, normal.z)
    return [h for h in holes if planar_distance(h.axis_point, c, ax) <= radius]


__all__ = [
    "LOCALIZATION_MARGIN_MM",
    "BaseInterface",
    "InterfaceContext",
    "holes_in_region",
    "holes_matching",
    "is_square_pattern",
    "region_radius_for_pair",
    "region_radius_for_square",
]
