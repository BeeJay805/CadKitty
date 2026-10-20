"""3-D density field -> printable artifact (the 3-D counterpart of ``extrude``).

A 3-D SIMP field has no extrusion shortcut: the honest path is marching cubes on the
GAUSSIAN-SMOOTHED scalar field (smoothing BEFORE meshing is what kills the voxel stairstep at
the source -- the same lesson as vectorize-before-extrude in 2-D), then mesh hygiene (largest
component, Taubin smooth, quadric decimation, hole fill) to a watertight STL the tet FEA eats.

STEP honesty: an organic marched surface has no native B-rep. A tessellated STEP is attempted
via build123d's Mesher import (best-effort); when it fails the artifact ships STL-only and the
uniform gate's ``step_present`` FAILS -- 3-D output is then measurable but NOT claimable, which
is the correct fail-closed posture until tessellated-STEP conversion proves itself. Never
weaken the gate to admit the new author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class Mesh3DResult:
    stl: Path
    step: Path | None  # None = tessellated-STEP conversion unavailable (gate will fail it)
    faces: int
    watertight: bool


def density3d_to_mesh(
    density: np.ndarray,
    *,
    out_dir: Path,
    cell_mm: tuple[float, float, float],
    smooth_sigma: float = 1.0,
    level: float = 0.5,
    target_faces: int = 12000,
    name: str = "generative3d_part",
) -> Mesh3DResult | None:
    """March the smoothed field to a cleaned, watertight STL (+ best-effort STEP).

    Fail-closed: no surface at ``level``, or a mesh that cannot be made watertight, returns
    None -- no artifact, no claim."""
    import trimesh

    field = gaussian_filter(np.asarray(density, dtype=float), smooth_sigma, mode="constant")
    padded = np.pad(field, 1, constant_values=0.0)  # closes surfaces at the domain boundary
    if float(padded.max()) <= level or float(padded.min()) >= level:
        return None
    verts, faces, _normals, _vals = marching_cubes(padded, level=level, spacing=cell_mm)
    offset = np.asarray(cell_mm, dtype=float)  # un-pad: one cell in every axis
    mesh = trimesh.Trimesh(vertices=verts - offset, faces=faces, process=True)

    # Hygiene: the largest body IS the part (disconnected slivers are optimizer noise).
    if mesh.body_count > 1:
        mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
    trimesh.smoothing.filter_taubin(mesh, iterations=6)  # type: ignore[no-untyped-call]
    if len(mesh.faces) > target_faces:
        import contextlib

        with contextlib.suppress(Exception):  # decimation is an optimization, not a gate
            mesh = mesh.simplify_quadric_decimation(target_faces)
    mesh.process(validate=True)
    mesh.fill_holes()
    if not bool(mesh.is_watertight):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / f"{name}.stl"
    mesh.export(str(stl))
    step = _try_tessellated_step(stl, out_dir / f"{name}.step")
    return Mesh3DResult(stl=stl, step=step, faces=len(mesh.faces), watertight=True)


def _try_tessellated_step(stl: Path, step: Path) -> Path | None:
    """Best-effort STL -> tessellated STEP via build123d's Mesher import. Failure -> None
    (the artifact stays STL-only and the uniform gate fails ``step_present`` -- fail-closed)."""
    try:
        from build123d import Mesher, export_step

        shapes = Mesher().read(str(stl))
        shape: Any = shapes[0] if isinstance(shapes, (list, tuple)) else shapes
        export_step(shape, str(step))
        return step if step.exists() and step.stat().st_size > 0 else None
    except Exception:
        return None


__all__ = ["Mesh3DResult", "density3d_to_mesh"]
