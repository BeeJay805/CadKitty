"""Geometry mirror-congruence check (Sol P0.2 / task 3): a mirrored pair must be a real
mirror in GEOMETRY, not just name-paired. Hand-authored known-good / known-bad fixtures,
per the cad-validator-designer discipline: known-bad must FAIL, known-good must PASS, and
a missing input is not_run (never a silent PASS).

Fixture shape: an asymmetric right-triangular prism (NOT symmetric under Y-reflection), so
a true mirror is non-trivial and a non-mirror is genuinely caught (the H3 shin_foot_L=58 /
shin_foot_R=32 defect class)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from forge_mind.validator.symmetry import GeomVerdict, mirror_congruence

trimesh = pytest.importorskip("trimesh")


def _tri_prism(a: float = 20.0, b: float = 10.0, h: float = 8.0):
    """A watertight right-triangular prism: triangle (0,0)-(a,0)-(0,b) extruded in Z.
    Asymmetric under Y-reflection by construction."""
    v = np.array(
        [[0, 0, 0], [a, 0, 0], [0, b, 0], [0, 0, h], [a, 0, h], [0, b, h]],
        dtype=float,
    )
    f = np.array(
        [
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ]
    )
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    m.fix_normals()
    return m


def _reflect_y(mesh):
    m = mesh.copy()
    t = np.eye(4)
    t[1, 1] = -1.0
    m.apply_transform(t)
    m.fix_normals()
    return m


def test_true_mirror_pair_passes() -> None:
    left = _tri_prism()
    right = _reflect_y(left)  # the genuine Y-mirror
    v = mirror_congruence(left, right, mirror_axis="y", n_samples=400)
    assert v.status == "pass", v.message
    assert v.ok
    assert v.measured["max_surface_dev_mm"] < 0.5


def test_non_mirrored_pair_fails() -> None:
    # Two IDENTICAL asymmetric parts (both "left"): a pair that should mirror but does not.
    left = _tri_prism()
    right = _tri_prism()  # NOT reflected -- the H3 defect (present but not a mirror)
    v = mirror_congruence(left, right, mirror_axis="y", n_samples=400)
    assert v.status == "fail", v.message
    assert v.measured["max_surface_dev_mm"] > 0.5


def test_scaled_mirror_fails_on_volume_and_surface() -> None:
    left = _tri_prism()
    right = _reflect_y(left)
    right.apply_scale(1.15)  # right side 15% larger -- the shin_foot_L=58 vs _R=32 class
    v = mirror_congruence(left, right, mirror_axis="y", n_samples=400)
    assert v.status == "fail", v.message
    assert v.measured["volume_delta_frac"] > 0.02


def test_missing_mesh_is_not_run_never_pass() -> None:
    left = _tri_prism()
    assert mirror_congruence(left, None, mirror_axis="y").status == "not_run"
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    assert mirror_congruence(left, empty, mirror_axis="y").status == "not_run"


def test_unknown_axis_is_not_run() -> None:
    left = _tri_prism()
    v = mirror_congruence(left, left, mirror_axis="w")
    assert v.status == "not_run" and "unknown mirror_axis" in v.message


def test_verdict_is_frozen_and_carries_evidence() -> None:
    v = mirror_congruence(_tri_prism(), _reflect_y(_tri_prism()), n_samples=400)
    assert isinstance(v, GeomVerdict)
    with pytest.raises(FrozenInstanceError):
        v.status = "fail"  # type: ignore[misc]  # frozen dataclass
    # evidence is present and self-describing
    assert set(v.measured) == {
        "max_surface_dev_mm",
        "mean_surface_dev_mm",
        "bbox_extent_delta_mm",
        "volume_delta_frac",
    }
    assert v.limit["tol_mm"] == 0.5


def test_verdict_is_reproducible_across_runs() -> None:
    # Cross-examiner catch (measured live before the fix): unseeded surface sampling made the
    # same input produce max deviations from 0.299 to 0.479 mm across identical runs, so a
    # defect near the 0.5 mm threshold alternated pass/fail. Evidence backing a frozen
    # acceptance manifest must be reproducible.
    left = _tri_prism()
    right = _reflect_y(left)
    v = right.vertices.copy()
    v[0] = v[0] + np.array([0.0, 0.0, 0.55])  # localized near-threshold defect
    right.vertices = v
    runs = [mirror_congruence(left, right, n_samples=300) for _ in range(5)]
    devs = {round(r.measured["max_surface_dev_mm"], 9) for r in runs}
    assert len(devs) == 1, f"non-reproducible deviations: {devs}"
    assert len({r.status for r in runs}) == 1


def test_vertex_defect_is_always_measured_not_missed_by_sampling() -> None:
    # Vertices are probed unconditionally, so a localized vertex defect is caught every time
    # rather than depending on random face samples landing on it.
    left = _tri_prism()
    right = _reflect_y(left)
    v = right.vertices.copy()
    v[0] = v[0] + np.array([0.0, 0.0, 3.0])  # unmistakable 3 mm vertex defect
    right.vertices = v
    for n in (0, 50, 400):  # even with ZERO random samples, the vertex probe catches it
        r = mirror_congruence(left, right, n_samples=n)
        assert r.status == "fail", f"n_samples={n}: {r.message}"


def test_orphan_vertices_do_not_fail_identical_surfaces() -> None:
    # Cross-examiner catch: probing ALL vertices (including ones no face references, possible
    # with process=False) compares points that are not on the surface, failing two identical
    # surfaces. Only face-referenced vertices are probed.
    left = _tri_prism()
    right = _reflect_y(left)
    stray = np.vstack([right.vertices, np.array([[0.0, 0.0, 99.0]])])  # unreferenced point
    right_with_orphan = trimesh.Trimesh(vertices=stray, faces=right.faces, process=False)
    v = mirror_congruence(left, right_with_orphan, n_samples=200)
    assert v.status == "pass", v.message


def test_negative_sample_count_is_not_run() -> None:
    # Fail closed on a malformed measurement request rather than silently degrading to a
    # vertex-only check that could PASS. (n_samples=0 IS the explicit vertex-only mode.)
    left = _tri_prism()
    assert mirror_congruence(left, _reflect_y(left), n_samples=-5).status == "not_run"
    assert mirror_congruence(left, _reflect_y(left), n_samples=0).status == "pass"
