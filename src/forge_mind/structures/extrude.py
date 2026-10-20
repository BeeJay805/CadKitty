"""Density field -> B-rep part (Fable's ruling: vectorize the contour, do NOT remesh voxels).

The SIMP output is a 2-D density grid. Marching-cubes voxel extrusion produces a stair-stepped
surface whose triangle count defeats pytetwild (ADR-0096 measured that tet count is driven by
surface detail, not edge_length_fac) -- so the honest 3-D artifact comes from VECTORIZING the
thresholded density contour (skimage ``find_contours`` + Douglas-Peucker ``approximate_polygon``)
and extruding the polygon-with-holes in build123d. That yields a real B-rep: STEP + a clean STL
the tet FEA eats -- the same artifact pair every other author emits (the C2 contract).

Outer boundary vs holes is decided by polygon CONTAINMENT (not winding, which find_contours does
not guarantee); islands nested inside holes are dropped (unreachable material). Fail-closed: no
valid loop, a degenerate contour, or a failed face/extrude returns None -- no artifact, no claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import binary_opening, disk


@dataclass(frozen=True)
class ExtrudeResult:
    stl: Path
    step: Path
    n_outer: int  # outer loops (disjoint solids in the plane)
    n_holes: int


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace area (abs). ``pts`` is (N, 2)."""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-cast containment for a representative point (dependency-free)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi:
            inside = not inside
        j = i
    return inside


def density_to_loops(
    density: np.ndarray,
    *,
    thresh: float = 0.4,
    simplify_tol: float = 0.75,
    min_area: float = 4.0,
    open_radius_cells: int | None = None,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Vectorize a (nely, nelx) density field into [(outer_loop, [hole_loops...]), ...] in grid
    units. Contours are closed by zero-padding (material touching the domain edge still closes),
    simplified by Douglas-Peucker, and classified outer-vs-hole by containment (area-ordered).

    ``open_radius_cells``: morphological OPENING of the thresholded field with a disk of that
    radius -- the min-member guarantee the SIMP filter alone cannot give (post-threshold necks
    survive ``rmin``; measured: 0.36-0.51 mm slivers failed the min-wall gate). Opening deletes
    every material feature thinner than ~2x the radius; the FEA then re-verifies the FINAL
    geometry, so a load path lost to opening is caught, never claimed."""
    field = np.asarray(density, dtype=float)
    if open_radius_cells is not None and open_radius_cells > 0:
        opened = binary_opening(field >= thresh, disk(open_radius_cells))
        field = opened.astype(float)
        thresh = 0.5  # the field is binary now
    padded = np.pad(field, 1, constant_values=0.0)
    raw = find_contours(padded, thresh)
    loops: list[np.ndarray] = []
    for c in raw:
        approx = approximate_polygon(c, tolerance=simplify_tol)
        if len(approx) < 4:
            continue
        pts = approx[:, ::-1] - 1.0  # (row, col) -> (x, y), unpad
        if _polygon_area(pts) < min_area:
            continue
        loops.append(pts)
    loops.sort(key=_polygon_area, reverse=True)

    out: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for lp in loops:
        probe = lp.mean(axis=0)
        placed = False
        for outer, holes in out:
            if _point_in_polygon(probe, outer):
                # First nesting level = a hole; deeper nesting (an island inside a hole) is
                # dropped: unreachable material, fail-closed.
                if not any(_point_in_polygon(probe, h) for h in holes):
                    holes.append(lp)
                placed = True
                break
        if not placed:
            out.append((lp, []))
    return out


def _extrude_loop_set(loop_set: list[np.ndarray], *, sx: float, sy: float, height_mm: float) -> Any:
    """One solid from a set of disjoint loops: sketch faces -> extrude."""
    from build123d import BuildLine, BuildPart, BuildSketch, Polyline, extrude, make_face

    with BuildPart() as bp:
        with BuildSketch():
            for lp in loop_set:
                with BuildLine():
                    Polyline(*[(float(p[0]) * sx, float(p[1]) * sy) for p in lp], close=True)
                make_face()
        extrude(amount=height_mm)
    return bp.part


def extrude_density(
    density: np.ndarray,
    *,
    out_dir: Path,
    scale_xy: tuple[float, float],
    height_mm: float,
    thresh: float = 0.4,
    simplify_tol: float = 0.75,
    open_radius_cells: int | None = None,
    name: str = "generative_part",
) -> ExtrudeResult | None:
    """Extrude a density field to a real B-rep (STL + STEP). ``scale_xy`` maps grid units to mm.
    Returns None (fail-closed) when vectorization or the build fails -- no artifact, no claim."""
    from build123d import export_step, export_stl

    loops = density_to_loops(
        density, thresh=thresh, simplify_tol=simplify_tol, open_radius_cells=open_radius_cells
    )
    if not loops:
        return None
    sx, sy = scale_xy
    outers = [outer for outer, _ in loops]
    holes = [h for _, hs in loops for h in hs]
    try:
        part = _extrude_loop_set(outers, sx=sx, sy=sy, height_mm=height_mm)
        if holes:
            part = part - _extrude_loop_set(holes, sx=sx, sy=sy, height_mm=height_mm)
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / f"{name}.stl"
    step = out_dir / f"{name}.step"
    try:
        export_stl(part, str(stl))
        export_step(part, str(step))
    except Exception:
        return None
    return ExtrudeResult(stl=stl, step=step, n_outer=len(outers), n_holes=len(holes))


__all__ = ["ExtrudeResult", "density_to_loops", "extrude_density"]
