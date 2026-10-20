"""Unit tests for shape-similarity metrics.

Uses primitive trimesh shapes (box / sphere / scaled box) as fixtures rather
than driving the full compiler — these tests need to be fast and self-contained
so they can run on every commit without pulling in build123d. The compiler is
already exercised by the validator tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.analysis.shape_similarity import (
    ShapeSimilarity,
    chamfer_distance,
    compare_meshes,
    f1_at_threshold,
    volumetric_iou,
)

# Chamfer / IoU sample thousands of surface points per assert (2-3.5s each, above the 1s `slow`
# threshold). Marked slow so the inner loop stays fast; still in the full `-m "not llm"` gate.
pytestmark = pytest.mark.slow

SAMPLES = 4_000  # smaller than the production default (10k) to keep tests snappy


def _box(size: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh: trimesh.Trimesh = trimesh.creation.box(extents=size)
    return mesh


# ---------------------------------------------------------------------------
# Identical meshes — the "perfect score" baseline
# ---------------------------------------------------------------------------


def test_identical_meshes_score_perfectly() -> None:
    a = _box((20.0, 20.0, 20.0))
    b = a.copy()
    sim = compare_meshes(a, b, samples=SAMPLES, voxel_pitch_mm=1.0, threshold_mm=1.0)

    assert isinstance(sim, ShapeSimilarity)
    assert sim.chamfer_distance_mm == pytest.approx(0.0, abs=1e-6)
    assert sim.volumetric_iou == pytest.approx(1.0, abs=1e-6)
    assert sim.f1_at_threshold == pytest.approx(1.0, abs=1e-6)


def test_metrics_are_deterministic_for_a_given_seed() -> None:
    # Two independent runs at the same seed must produce identical numbers,
    # not just close ones — sample_surface() is seeded end-to-end.
    a = _box((20.0, 20.0, 20.0))
    b = a.copy()
    b.apply_translation([0.4, 0.0, 0.0])
    one = compare_meshes(a, b, samples=SAMPLES, seed=7)
    two = compare_meshes(a, b, samples=SAMPLES, seed=7)
    assert one == two


# ---------------------------------------------------------------------------
# Scaled / translated copies — metrics must degrade in the expected direction
# ---------------------------------------------------------------------------


def test_translation_degrades_metrics() -> None:
    a = _box((20.0, 20.0, 20.0))
    b = a.copy()
    b.apply_translation([5.0, 0.0, 0.0])  # 5mm shift

    sim = compare_meshes(a, b, samples=SAMPLES, voxel_pitch_mm=1.0, threshold_mm=1.0)

    # Chamfer should pick up the ~5mm shift. The exact value depends on which
    # surfaces line up, but it must be clearly nonzero and well above the
    # tolerance threshold.
    assert sim.chamfer_distance_mm > 0.5
    # IoU of a 20mm cube shifted by 5mm: overlap volume = 15*20*20 = 6000,
    # union = 8000 - 6000 + 8000 = 10000  -> IoU = 0.6
    assert sim.volumetric_iou == pytest.approx(0.6, abs=0.05)
    # F1 must be strictly less than 1 — most surface points are now > 1mm from
    # the other mesh's surface — but the two end faces still match perfectly.
    assert 0.0 < sim.f1_at_threshold < 1.0


def test_uniform_scale_degrades_metrics_monotonically() -> None:
    # A part validator might happily accept a part that is the right shape but
    # uniformly half-size. Chamfer must catch that.
    a = _box((20.0, 20.0, 20.0))
    small = _box((10.0, 10.0, 10.0))
    smaller = _box((4.0, 4.0, 4.0))

    sim_small = compare_meshes(a, small, samples=SAMPLES, voxel_pitch_mm=1.0)
    sim_smaller = compare_meshes(a, smaller, samples=SAMPLES, voxel_pitch_mm=1.0)

    # More wrong = larger chamfer, smaller IoU. (F1 at a 1mm threshold can't
    # discriminate between two nested boxes that are both > 1mm away from the
    # reference everywhere — both saturate to zero. That's a feature of F1, not
    # a bug; chamfer and IoU pick up the dimensional gap.)
    assert sim_smaller.chamfer_distance_mm > sim_small.chamfer_distance_mm
    assert sim_smaller.volumetric_iou < sim_small.volumetric_iou
    assert sim_smaller.f1_at_threshold <= sim_small.f1_at_threshold

    # And the half-scale box should not be confused with the original.
    assert sim_small.volumetric_iou < 0.2  # 1000 / 8000 = 0.125
    assert sim_small.chamfer_distance_mm > 2.0


# ---------------------------------------------------------------------------
# Valid-but-wrong-shaped — the case the existing validator can't catch
# ---------------------------------------------------------------------------


def test_valid_but_wrong_shape_scores_low() -> None:
    # Both meshes are watertight, single-bodied, positive-volume — i.e. they
    # both pass every Tier-1 mesh check. But they are geometrically different
    # shapes (a 20mm cube vs. a 10mm-radius sphere). Shape similarity has to
    # flag this; the validator alone wouldn't.
    cube = _box((20.0, 20.0, 20.0))
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    assert cube.is_watertight and sphere.is_watertight

    sim = compare_meshes(cube, sphere, samples=SAMPLES, voxel_pitch_mm=1.0, threshold_mm=1.0)

    # Cube circumscribes the sphere; the corners and edges are millimetres away
    # from the sphere surface. Chamfer must reflect that.
    assert sim.chamfer_distance_mm > 1.0
    # IoU upper-bounded by sphere_vol / cube_vol = (4/3*pi*1000) / 8000 ~ 0.52.
    assert sim.volumetric_iou < 0.6
    assert sim.f1_at_threshold < 0.9


def test_disjoint_shapes_score_zero_iou() -> None:
    a = _box((10.0, 10.0, 10.0))
    b = _box((10.0, 10.0, 10.0))
    b.apply_translation([100.0, 0.0, 0.0])  # nowhere near each other

    sim = compare_meshes(a, b, samples=SAMPLES, voxel_pitch_mm=1.0, threshold_mm=1.0)

    assert sim.volumetric_iou == 0.0
    assert sim.f1_at_threshold == 0.0
    # Chamfer should be roughly the centre-to-centre distance, ~100mm.
    assert sim.chamfer_distance_mm > 50.0


def test_non_watertight_disjoint_meshes_do_not_score_identical() -> None:
    # Regression (audit IoU finding): when both meshes are non-watertight, trimesh.contains()
    # classifies no occupancy -> union == 0. That used to return 1.0 ("identical") -- a false PASS:
    # two disjoint single-triangle meshes scored a perfect IoU. It must read as no measurable overlap.
    tri_a = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], faces=[[0, 1, 2]]
    )
    tri_b = tri_a.copy()
    tri_b.apply_translation([173.0, 0.0, 0.0])
    assert not tri_a.is_watertight  # the path under test: contains() cannot classify occupancy
    assert volumetric_iou(tri_a, tri_b, pitch_mm=1.0) == 0.0


# ---------------------------------------------------------------------------
# Sanity checks on individual entry points
# ---------------------------------------------------------------------------


def test_individual_metrics_are_independently_callable() -> None:
    a = _box((10.0, 10.0, 10.0))
    b = a.copy()

    assert chamfer_distance(a, b, samples=1_000) == pytest.approx(0.0, abs=1e-6)
    assert volumetric_iou(a, b, pitch_mm=1.0) == pytest.approx(1.0, abs=1e-6)
    assert f1_at_threshold(a, b, samples=1_000) == pytest.approx(1.0, abs=1e-6)


def test_compare_meshes_accepts_stl_paths(tmp_path: Path) -> None:
    a = _box((15.0, 15.0, 15.0))
    b = a.copy()
    b.apply_translation([1.0, 0.0, 0.0])
    path_a = tmp_path / "a.stl"
    path_b = tmp_path / "b.stl"
    a.export(path_a)
    b.export(path_b)

    sim = compare_meshes(path_a, path_b, samples=SAMPLES, voxel_pitch_mm=1.0)

    # Closed-form for a cube of side L shifted by d on one axis:
    #   intersection = (L-d) * L^2 ;  union = L^2 * (L+d) ;  IoU = (L-d)/(L+d)
    # For L=15, d=1: 14/16 = 0.875.
    assert sim.volumetric_iou == pytest.approx(14.0 / 16.0, abs=0.05)
    assert sim.f1_at_threshold > 0.8
    # Chamfer is roughly the shift on the displaced faces, averaged with the
    # untouched faces — modest but nonzero.
    assert 0.05 < sim.chamfer_distance_mm < 2.0


def test_invalid_voxel_pitch_raises() -> None:
    a = _box((10.0, 10.0, 10.0))
    with pytest.raises(ValueError):
        volumetric_iou(a, a, pitch_mm=0.0)


def test_invalid_threshold_raises() -> None:
    a = _box((10.0, 10.0, 10.0))
    with pytest.raises(ValueError):
        f1_at_threshold(a, a, threshold_mm=0.0)
