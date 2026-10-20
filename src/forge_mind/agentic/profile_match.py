"""Axial profile: the reference as GEOMETRY, not pixels (ADR-0073 Stage 2).

The reference mesh contains the full answer to "what should this part's form be", but the engine
was reducing it to a render and two scalar descriptors. This module extracts the structure the
optimizer needs as a TARGET: slice the mesh into ``n_slices`` bins along its own principal axis
and record, per slice,

  * ``r_max``  -- the silhouette envelope (how far material reaches from the axis),
  * ``r_mean`` -- the fill (a solid disc slice reads high; a crescent/bladed slice reads low --
    the per-slice concavity signal), and
  * ``offset`` -- how far the slice's material centroid sits OFF the axis (zero for anything
    axisymmetric; large exactly where a crank throw or eccentric lobe lives).

All three channels are normalized by the global ``r_max`` and the axis position by its extent, so
profiles are pose- AND scale-invariant: the user's stated dimensions stay sovereign; only SHAPE is
compared. ``profile_distance`` is flip-invariant (a principal axis has no preferred direction).

Deterministic, surface-sample based (works on non-watertight artist references), no LLM, no
watertight requirement -- the same contract as ``shape_signature``, whose sampler it reuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

N_SLICES = 32
_N_POINTS = 16384  # denser than shape_signature's 4096: the per-slice OFFSET mean needs ~500
# samples/slice to push its noise floor to ~0.04 normalized (128/slice measured ~0.18 peaks)


def _surface_points(mesh: Any) -> np.ndarray | None:
    """Deterministic area-weighted surface sample (same RNG discipline as shape_signature,
    denser budget -- see _N_POINTS)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        pts = np.asarray(mesh.sample(_N_POINTS), dtype=np.float64)
    except Exception:
        return None
    finally:
        np.random.set_state(rng_state)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return None
    return pts


@dataclass(frozen=True)
class AxialProfile:
    """Per-slice normalized channels along the principal axis (each length ``n_slices``)."""

    r_max: tuple[float, ...]
    r_mean: tuple[float, ...]
    offset: tuple[float, ...]

    @property
    def n_slices(self) -> int:
        return len(self.r_max)


def axial_profile(mesh_path: Path, *, n_slices: int = N_SLICES) -> AxialProfile | None:
    """The mesh's axial profile, or None if it cannot be loaded/sampled (fail-soft)."""
    try:
        mesh: Any = trimesh.load(str(mesh_path), force="mesh")
    except Exception:
        return None
    points = _surface_points(mesh)
    if points is None or n_slices < 2:
        return None

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (3, 3):
        return None
    _vals, vecs = np.linalg.eigh(cov)  # ascending
    axis = vecs[:, -1]  # the principal (longest-spread) axis

    t = centered @ axis  # axial coordinate of each sample
    span = float(t.max() - t.min())
    if span <= 1e-9:
        return None
    perp = centered - np.outer(t, axis)  # perpendicular component
    r = np.linalg.norm(perp, axis=1)
    r_global = float(r.max())
    if r_global <= 1e-9:
        return None

    bins = np.clip(((t - t.min()) / span * n_slices).astype(int), 0, n_slices - 1)
    r_max = np.zeros(n_slices)
    r_mean = np.zeros(n_slices)
    offset = np.zeros(n_slices)
    for i in range(n_slices):
        mask = bins == i
        if not mask.any():
            continue  # an empty slice (no surface there) reads as 0 material
        r_max[i] = r[mask].max()
        r_mean[i] = r[mask].mean()
        offset[i] = float(np.linalg.norm(perp[mask].mean(axis=0)))

    return AxialProfile(
        r_max=tuple((r_max / r_global).tolist()),
        r_mean=tuple((r_mean / r_global).tolist()),
        offset=tuple((offset / r_global).tolist()),
    )


def profile_distance(a: AxialProfile, b: AxialProfile) -> float | None:
    """Mean absolute per-slice difference across the three channels, minimized over the axis-flip
    (a principal axis has no preferred direction). 0 = identical form; None if incomparable."""
    if a.n_slices != b.n_slices or a.n_slices == 0:
        return None

    def stack(p: AxialProfile, flipped: bool) -> np.ndarray:
        arr = np.array([p.r_max, p.r_mean, p.offset])
        return arr[:, ::-1] if flipped else arr

    base = stack(a, False)
    d_fwd = float(np.mean(np.abs(base - stack(b, False))))
    d_rev = float(np.mean(np.abs(base - stack(b, True))))
    return min(d_fwd, d_rev)


__all__ = ["N_SLICES", "AxialProfile", "axial_profile", "profile_distance"]
