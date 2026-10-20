"""Shape-similarity metrics — graded "how close to intended shape", not pass/fail.

The existing validator answers "is this part valid?" (watertight, single body,
bbox in range, features present). It does not answer "is this part the SAME
SHAPE as the reference?" A part can pass every validator check and still be the
wrong shape — half the size, shifted, or geometrically different in a way the
feature-based checks don't notice.

This module measures that gap. It compares a generated mesh against a reference
mesh and returns three complementary scores in absolute mm space (deliberately
NOT normalized — we want to know about dimensional accuracy, not just shape
agreement after a hidden scale-invariant fit):

    chamfer_distance_mm  — bidirectional mean surface-to-surface distance
    volumetric_iou       — voxel-grid intersection-over-union
    f1_at_threshold      — surface coverage F1 within a millimetre threshold

Deterministic (a single ``seed`` controls all sampling), self-contained, no LLM,
no scipy — trimesh-native proximity and numpy only. Not wired into any pipeline
yet; intended to be called from a benchmark harness once that wiring lands on
main.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
from trimesh.sample import sample_surface

MeshOrPath = trimesh.Trimesh | str | Path

DEFAULT_SAMPLES = 10_000
DEFAULT_VOXEL_PITCH_MM = 1.0
DEFAULT_THRESHOLD_MM = 1.0
DEFAULT_SEED = 0


@dataclass(frozen=True)
class ShapeSimilarity:
    """Triple of shape-similarity scores between a generated and reference mesh.

    All distance quantities are in millimetres in world space (the same frame
    the meshes were loaded in — no centering, no scale normalization).

    * ``chamfer_distance_mm`` — lower is better, 0 means surfaces coincide.
    * ``volumetric_iou`` — in [0, 1], 1 means identical occupancy.
    * ``f1_at_threshold`` — in [0, 1], harmonic mean of the fraction of
      generated surface samples within ``threshold_mm`` of the reference
      surface (precision) and the reverse (recall).
    """

    chamfer_distance_mm: float
    volumetric_iou: float
    f1_at_threshold: float
    threshold_mm: float
    voxel_pitch_mm: float
    samples: int


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _as_trimesh(mesh_or_path: MeshOrPath) -> trimesh.Trimesh:
    """Accept either a trimesh.Trimesh or a path to an STL/mesh file and
    return a single-body Trimesh. Scenes are concatenated into one mesh so
    the comparison still works for assemblies exported as multi-body STL.
    """
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path
    loaded = trimesh.load(str(mesh_or_path), force="mesh")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    # ``force="mesh"`` returns a Trimesh in almost every case; this branch is a
    # belt-and-braces fallback for exotic loaders that still hand back a Scene.
    if isinstance(loaded, trimesh.Scene):
        concatenated = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
        if isinstance(concatenated, trimesh.Trimesh):
            return concatenated
    raise TypeError(
        f"Could not coerce {mesh_or_path!r} to a Trimesh (got {type(loaded).__name__})."
    )


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def chamfer_distance(
    generated: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """Bidirectional mean nearest-neighbour surface distance in millimetres.

    ``samples`` points are drawn from each mesh's surface (seeded). For each
    point on ``generated`` we measure the distance to the nearest point on the
    closed surface of ``reference``, average those, then do the reverse, then
    average the two means. Symmetric by construction.
    """
    gen_pts, _ = sample_surface(generated, samples, seed=seed)
    # Different stream for the reference so the two samplings are independent
    # but still reproducible from the single user-supplied seed.
    ref_pts, _ = sample_surface(reference, samples, seed=seed + 1)

    pq_ref = ProximityQuery(reference)  # type: ignore[no-untyped-call]
    pq_gen = ProximityQuery(generated)  # type: ignore[no-untyped-call]
    _, d_gen_to_ref, _ = pq_ref.on_surface(gen_pts)
    _, d_ref_to_gen, _ = pq_gen.on_surface(ref_pts)
    return float(0.5 * (np.mean(d_gen_to_ref) + np.mean(d_ref_to_gen)))


def volumetric_iou(
    generated: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    *,
    pitch_mm: float = DEFAULT_VOXEL_PITCH_MM,
) -> float:
    """Voxelized intersection-over-union of the two solids' occupancy.

    Both meshes are voxelized onto the SAME world-space grid (a regular lattice
    spanning the union of both bounding boxes, with one cell of margin) using
    ``contains()`` rather than trimesh's per-mesh ``voxelized()`` — the latter
    gives each mesh its own grid origin and the two are not directly
    comparable. IoU is the count of cells inside both solids divided by the
    count of cells inside either.

    Returns 1.0 if both meshes are empty (degenerate but well-defined), 0.0 if
    one is empty and the other is not.
    """
    if pitch_mm <= 0.0:
        raise ValueError(f"pitch_mm must be > 0 (got {pitch_mm}).")

    # Union of the two bounding boxes, padded by one cell so surface points
    # don't fall exactly on the boundary and get classified inconsistently.
    lo = np.minimum(generated.bounds[0], reference.bounds[0]) - pitch_mm
    hi = np.maximum(generated.bounds[1], reference.bounds[1]) + pitch_mm

    # Cell centres along each axis. ``+ 0.5 * pitch`` makes samples land in the
    # middle of each voxel, which gives a more consistent inside/outside answer
    # near the surface than sampling the corners.
    axes = [np.arange(lo[i] + 0.5 * pitch_mm, hi[i], pitch_mm, dtype=np.float64) for i in range(3)]
    if any(a.size == 0 for a in axes):
        # Both meshes degenerate to a point/line — fall back to "identical iff
        # bounding boxes coincide".
        return 1.0 if np.allclose(generated.bounds, reference.bounds) else 0.0

    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")
    query_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    in_gen = generated.contains(query_points)
    in_ref = reference.contains(query_points)
    intersection = int(np.count_nonzero(in_gen & in_ref))
    union = int(np.count_nonzero(in_gen | in_ref))
    if union == 0:
        # Neither mesh contains any sample point: a non-watertight / degenerate input where
        # trimesh.contains() classified nothing -- NOT "identical". Returning 1.0 here was a false
        # PASS (two disjoint single-triangle meshes scored a perfect 1.0; audit IoU finding). Report
        # no measurable overlap instead.
        return 0.0
    return intersection / union


def f1_at_threshold(
    generated: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    *,
    threshold_mm: float = DEFAULT_THRESHOLD_MM,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> float:
    """F1 of surface coverage at a millimetre tolerance.

    Precision is the fraction of generated-surface samples that lie within
    ``threshold_mm`` of the reference surface (i.e. "how much of what we
    produced is actually on the reference"). Recall is the reverse. F1 is the
    harmonic mean. A perfectly-matching pair scores 1.0; non-overlapping shapes
    score 0.0.
    """
    if threshold_mm <= 0.0:
        raise ValueError(f"threshold_mm must be > 0 (got {threshold_mm}).")

    gen_pts, _ = sample_surface(generated, samples, seed=seed)
    ref_pts, _ = sample_surface(reference, samples, seed=seed + 1)

    pq_ref = ProximityQuery(reference)  # type: ignore[no-untyped-call]
    pq_gen = ProximityQuery(generated)  # type: ignore[no-untyped-call]
    _, d_gen_to_ref, _ = pq_ref.on_surface(gen_pts)
    _, d_ref_to_gen, _ = pq_gen.on_surface(ref_pts)

    precision = float(np.mean(d_gen_to_ref <= threshold_mm))
    recall = float(np.mean(d_ref_to_gen <= threshold_mm))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def compare_meshes(
    generated: MeshOrPath,
    reference: MeshOrPath,
    *,
    samples: int = DEFAULT_SAMPLES,
    voxel_pitch_mm: float = DEFAULT_VOXEL_PITCH_MM,
    threshold_mm: float = DEFAULT_THRESHOLD_MM,
    seed: int = DEFAULT_SEED,
) -> ShapeSimilarity:
    """Compute all three similarity metrics in one pass.

    Accepts either ``trimesh.Trimesh`` instances or paths to STL files. The
    surface samplings for ``chamfer_distance`` and ``f1_at_threshold`` are
    independent (they could share samples, but recomputing keeps each metric
    callable in isolation without surprising coupling).
    """
    gen = _as_trimesh(generated)
    ref = _as_trimesh(reference)

    return ShapeSimilarity(
        chamfer_distance_mm=chamfer_distance(gen, ref, samples=samples, seed=seed),
        volumetric_iou=volumetric_iou(gen, ref, pitch_mm=voxel_pitch_mm),
        f1_at_threshold=f1_at_threshold(
            gen, ref, threshold_mm=threshold_mm, samples=samples, seed=seed
        ),
        threshold_mm=threshold_mm,
        voxel_pitch_mm=voxel_pitch_mm,
        samples=samples,
    )


__all__ = [
    "ShapeSimilarity",
    "chamfer_distance",
    "compare_meshes",
    "f1_at_threshold",
    "volumetric_iou",
]
