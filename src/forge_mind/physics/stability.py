"""Stand-or-tip: static stability from the center of mass vs. the base footprint.

MEASURED geometry, not a simulation. Drop a plumb line from the center of mass; if it
lands inside the convex hull of the resting footprint (the lowest face's contact points),
the part stands. The signed distance from that point to the nearest footprint edge is the
tip-over MARGIN (mm); the tilt to topple about the nearest edge is ``atan(margin / com_height)``.

Honest scope (carry this into the UI caption): STATIC, uniform density, a RIGID part on a
FLAT surface, resting on its lowest face along +Z. NEVER a guarantee against a bump,
vibration, or being carried. Refuses (returns ``None``) when there is no clear flat base --
refuse, don't guess (the no-false-PASS doctrine applied to physics, mirroring
``physics/static_load.py``). No SciPy dependency: a self-contained monotone-chain hull.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

# Vertices within this of the lowest point along +Z form the resting footprint.
BASE_CONTACT_TOL_MM = 0.5


@dataclass(frozen=True)
class StabilityResult:
    """Where the center of mass sits relative to the base the part would rest on."""

    stable: bool
    margin_mm: float  # signed: > 0 = inside the footprint (stands); < 0 = outside (tips)
    tip_angle_deg: float  # tilt to topple about the nearest edge; 0.0 when already tipping
    com: tuple[float, float, float]
    footprint: tuple[tuple[float, float], ...]  # support-polygon hull (xy), CCW
    com_height_mm: float


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain -- the CCW 2D convex hull, no SciPy. ``points`` is (N, 2);
    returns hull vertices (M, 2). Fewer than 3 returned points means a degenerate base."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _signed_margin(poly: np.ndarray, pt: np.ndarray) -> float:
    """Distance from ``pt`` to the polygon boundary: positive inside, negative outside.
    ``poly`` is a CCW (M, 2) hull; ``pt`` is (2,). Crossing-number inside test + nearest
    edge-segment distance."""
    n = len(poly)
    inside = False
    min_d = math.inf
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if (a[1] > pt[1]) != (b[1] > pt[1]):
            x_cross = a[0] + (pt[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            if pt[0] < x_cross:
                inside = not inside
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom == 0.0 else float(np.clip(float(np.dot(pt - a, ab)) / denom, 0.0, 1.0))
        proj = a + t * ab
        min_d = min(min_d, float(np.linalg.norm(pt - proj)))
    return min_d if inside else -min_d


def compute_stability(mesh: trimesh.Trimesh) -> StabilityResult | None:
    """Stand-or-tip for a part resting on its lowest face (+Z up). ``None`` when there is no
    clear flat base (fewer than 3 contact points after hulling) -- refuse, don't guess."""
    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) < 4:
        return None
    com = np.asarray(mesh.center_mass, dtype=float)
    base_z = float(verts[:, 2].min())
    contact = verts[verts[:, 2] <= base_z + BASE_CONTACT_TOL_MM][:, :2]
    hull = _convex_hull_2d(contact)
    if len(hull) < 3:
        return None
    margin = _signed_margin(hull, com[:2])
    height = max(float(com[2] - base_z), 1e-6)
    tip_angle = math.degrees(math.atan2(margin, height)) if margin > 0.0 else 0.0
    return StabilityResult(
        stable=margin > 0.0,
        margin_mm=round(margin, 2),
        tip_angle_deg=round(tip_angle, 1),
        com=(round(float(com[0]), 2), round(float(com[1]), 2), round(float(com[2]), 2)),
        footprint=tuple((round(float(x), 2), round(float(y), 2)) for x, y in hull),
        com_height_mm=round(height, 2),
    )


__all__ = ["BASE_CONTACT_TOL_MM", "StabilityResult", "compute_stability"]
