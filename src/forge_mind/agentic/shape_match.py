"""Shape similarity (Step 4 of the reference spine): a coarse, POSE- and SCALE-INVARIANT
proximity score between the generated build and the real reference model.

It uses the D2 shape distribution (Osada et al.): the normalized histogram of Euclidean
distances between random pairs of surface-sampled points. That is rotation- and
translation-invariant (distances don't change with pose), scale-normalized (divided by the
mean distance), and needs NO fragile mesh alignment -- which is exactly what we need for the
non-watertight, arbitrarily-oriented artist meshes Sketchfab returns.

HONEST SCOPE: this is COARSE. It captures overall proportion and elongation well (a long
crankshaft vs a cube scores low) but NOT fine features (a box and a sphere score similar). It
is ADVISORY -- a "is the overall form in the right ballpark" number that should rise as the
build gets closer to the reference; it never gates the verdict, and the vision critic remains
the source of specific, feature-level fixes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

_N_POINTS = 4096  # surface samples per mesh
_N_PAIRS = 12288  # random point pairs for the distance histogram
_BINS = 64


def _d2_histogram(mesh: Any) -> np.ndarray | None:
    """Scale-normalized D2 distance histogram of a mesh's surface, or None if unsampleable."""
    # Seed trimesh's area-weighted surface sampling (it uses numpy's global RNG) deterministically,
    # restoring the global state after -- so a fixed program reproduces its score and identical
    # meshes sample identically (self-similarity = 1.0), without mutating global randomness.
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        points = np.asarray(mesh.sample(_N_POINTS), dtype=np.float64)
    except Exception:
        return None
    finally:
        np.random.set_state(rng_state)
    if points.ndim != 2 or len(points) < 2:
        return None
    rng = np.random.default_rng(0)  # deterministic -- a fixed program reproduces its score
    i = rng.integers(0, len(points), _N_PAIRS)
    j = rng.integers(0, len(points), _N_PAIRS)
    dist = np.linalg.norm(points[i] - points[j], axis=1)
    mean = float(dist.mean())
    if mean <= 1e-9:
        return None
    dist = dist / mean  # scale-invariant
    hist, _ = np.histogram(dist, bins=_BINS, range=(0.0, 4.0))
    total = int(hist.sum())
    if total <= 0:
        return None
    return hist.astype(np.float64) / total


def shape_similarity(build: Path, reference: Path) -> float | None:
    """A coarse [0, 1] shape-proximity score between the build and reference meshes (1 =
    identical shape distribution), pose- and scale-invariant. None if either mesh cannot be
    loaded or sampled -- fail-closed, never raises."""
    build_mesh: Any
    ref_mesh: Any
    try:
        build_mesh = trimesh.load(str(build), force="mesh")
        ref_mesh = trimesh.load(str(reference), force="mesh")
    except Exception:
        return None
    hist_a = _d2_histogram(build_mesh)
    hist_b = _d2_histogram(ref_mesh)
    if hist_a is None or hist_b is None:
        return None
    # 1 - half the L1 distance between two probability histograms -> [0, 1].
    return float(1.0 - 0.5 * float(np.abs(hist_a - hist_b).sum()))


__all__ = ["shape_similarity"]
