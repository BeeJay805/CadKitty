"""Signed-distance-field (SDF) primitives, boolean ops, and the marching-cubes mesher.

The implicit / voxel geometry lane (ADR-0026, Phase 7) for parts B-rep can't express -- lattices,
gyroid/TPMS infills, organic blends. A solid is a scalar field ``f(p) <= 0``; booleans are min/max,
an offset is ``f - r``, a shell is ``abs(f) - t/2``. We author the field math in numpy (deterministic,
ADR-0009) and mesh it with scikit-image marching cubes.

The mesher recipe is load-bearing and was pinned by the ADR-0026 de-risk spike (5/5 watertight):
  pad the sampled volume's border POSITIVE so every surface closes -> ``skimage.measure.marching_cubes``
  -> ``trimesh.Trimesh(process=False)``. process=True / trimesh.repair.* REMERGE vertices and turn the
  already-manifold MC output non-watertight, so we keep the raw mesh.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# An SDF field: (X, Y, Z) meshgrids -> signed distances (negative inside). Typed loosely (Any) -- this
# is numpy-array math whose correctness is unit-tested (watertight assertions), and strict numpy
# return-typing fights every ufunc; consistent with how the repo treats trimesh/build123d.
Field = Callable[[Any, Any, Any], Any]

# Largest grid (per axis) we will sample -- a hard guard against a runaway voxel count.
_MAX_SAMPLES_PER_AXIS = 260


# -- primitives -------------------------------------------------------------------------------------


def box(hx: float, hy: float, hz: float) -> Field:
    """An axis-aligned box centred at the origin with the given half-extents (mm)."""

    def f(x: Any, y: Any, z: Any) -> Any:
        return np.maximum(np.maximum(np.abs(x) - hx, np.abs(y) - hy), np.abs(z) - hz)

    return f


def sphere(r: float) -> Field:
    def f(x: Any, y: Any, z: Any) -> Any:
        return np.sqrt(x**2 + y**2 + z**2) - r

    return f


def cylinder_z(r: float, half_h: float) -> Field:
    """A finite cylinder along Z, radius ``r``, half-height ``half_h``, centred at the origin."""

    def f(x: Any, y: Any, z: Any) -> Any:
        radial = np.sqrt(x**2 + y**2) - r
        return np.maximum(radial, np.abs(z) - half_h)

    return f


def gyroid(cell: float, thickness: float) -> Field:
    """A gyroid TPMS WALL of the given thickness (mm), periodic with ``cell`` (mm). This has no B-rep
    formula -- the canonical implicit win. ``|sin·cos sum| - t`` is a thresholded periodic field, not a
    true Euclidean distance, but it is sign-correct for meshing a wall of controlled thickness."""

    def f(x: Any, y: Any, z: Any) -> Any:
        q = 2.0 * np.pi / cell
        g = (
            np.sin(q * x) * np.cos(q * y)
            + np.sin(q * y) * np.cos(q * z)
            + np.sin(q * z) * np.cos(q * x)
        )
        return np.abs(g) - thickness

    return f


def schwarz_p(cell: float, thickness: float) -> Field:
    """A Schwarz-P TPMS wall -- a second minimal-surface family (``cos+cos+cos``)."""

    def f(x: Any, y: Any, z: Any) -> Any:
        q = 2.0 * np.pi / cell
        p = np.cos(q * x) + np.cos(q * y) + np.cos(q * z)
        return np.abs(p) - thickness

    return f


def strut_lattice(cell: float, strut_d: float) -> Field:
    """A cubic strut lattice: the union of three axis-aligned cylinder grids (struts meet at nodes)."""
    r = strut_d / 2.0

    def f(x: Any, y: Any, z: Any) -> Any:
        xm = (x + cell / 2) % cell - cell / 2
        ym = (y + cell / 2) % cell - cell / 2
        zm = (z + cell / 2) % cell - cell / 2
        sx = np.sqrt(ym**2 + zm**2) - r
        sy = np.sqrt(xm**2 + zm**2) - r
        sz = np.sqrt(xm**2 + ym**2) - r
        return np.minimum(np.minimum(sx, sy), sz)

    return f


# -- boolean ops ------------------------------------------------------------------------------------


def op_union(a: Field, b: Field) -> Field:
    def f(x: Any, y: Any, z: Any) -> Any:
        return np.minimum(a(x, y, z), b(x, y, z))

    return f


def op_intersect(a: Field, b: Field) -> Field:
    def f(x: Any, y: Any, z: Any) -> Any:
        return np.maximum(a(x, y, z), b(x, y, z))

    return f


def op_difference(a: Field, b: Field) -> Field:
    """``a`` minus ``b`` -- ``max(a, -b)``."""

    def f(x: Any, y: Any, z: Any) -> Any:
        return np.maximum(a(x, y, z), -b(x, y, z))

    return f


def op_shell(a: Field, thickness: float) -> Field:
    """Hollow ``a`` to a shell of the given wall thickness -- ``abs(f) - t/2``."""

    def f(x: Any, y: Any, z: Any) -> Any:
        return np.abs(a(x, y, z)) - thickness / 2.0

    return f


# -- mesher -----------------------------------------------------------------------------------------


def estimate_axis_samples(
    lo: tuple[float, float, float], hi: tuple[float, float, float], step: float
) -> int:
    """The largest per-axis sample count a (lo, hi, step) grid would use (for the guard / cost sense)."""
    return max(1, *(int(np.ceil((hi[i] - lo[i]) / step)) + 1 for i in range(3)))


def sdf_to_mesh(
    field: Field,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
    step: float,
) -> Any:
    """Sample ``field`` on a grid spanning [lo, hi] at ``step`` mm and mesh the zero level set.

    Returns a watertight ``trimesh.Trimesh`` in real mm coordinates (vertices offset back into world
    space). ``trimesh`` + ``scikit-image`` are imported lazily so importing this module is cheap and
    only the implicit lane pulls them. Raises ``ValueError`` if the grid would exceed the sample guard.
    """
    import trimesh
    from skimage import measure

    n = estimate_axis_samples(lo, hi, step)
    if n > _MAX_SAMPLES_PER_AXIS:
        raise ValueError(
            f"implicit grid too fine: {n} samples/axis > {_MAX_SAMPLES_PER_AXIS} "
            f"(coarsen the step or shrink the part)"
        )

    axes = [np.arange(lo[i], hi[i] + step, step, dtype=np.float64) for i in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    vol = np.asarray(field(grid_x, grid_y, grid_z), dtype=np.float64)
    # Pad the border with a positive (outside) value so every surface closes -> watertight.
    fill = float(np.abs(vol).max()) + 1.0
    vol = np.pad(vol, 1, mode="constant", constant_values=fill)

    verts, faces, _normals, _values = measure.marching_cubes(
        vol, level=0.0, spacing=(step, step, step)
    )
    # Padded index 0 maps to world (lo - step); shift the vertices into real mm coordinates.
    origin = np.array([lo[0] - step, lo[1] - step, lo[2] - step], dtype=np.float64)
    verts = verts + origin
    # process=False is load-bearing: MC output is already manifold; trimesh's remerge breaks it.
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


__all__ = [
    "Field",
    "box",
    "cylinder_z",
    "estimate_axis_samples",
    "gyroid",
    "op_difference",
    "op_intersect",
    "op_shell",
    "op_union",
    "schwarz_p",
    "sdf_to_mesh",
    "sphere",
    "strut_lattice",
]
