"""Shape similarity (Step 4): a pose- and scale-invariant D2 proximity score, robust to the
non-watertight artist meshes the reference retrieval returns. Coarse but honest: identical and
pose-varied shapes score high, differently-proportioned shapes score lower, bad input -> None."""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic.shape_match import shape_similarity


def _stl(tmp_path: Path, name: str, mesh: trimesh.Trimesh) -> Path:
    p = tmp_path / name
    mesh.export(str(p))
    return p


def test_identical_shape_scores_near_one(tmp_path: Path) -> None:
    box = _stl(tmp_path, "a.stl", trimesh.creation.box([40, 40, 40]))
    assert shape_similarity(box, box) > 0.98


def test_score_is_pose_invariant(tmp_path: Path) -> None:
    box = _stl(tmp_path, "box.stl", trimesh.creation.box([40, 40, 40]))
    rotated = trimesh.creation.box([40, 40, 40])
    rotated.apply_transform(trimesh.transformations.rotation_matrix(0.7, [1.0, 1.0, 0.0]))
    rot = _stl(tmp_path, "rot.stl", rotated)
    assert shape_similarity(box, rot) > 0.9  # same shape, different pose -> still high


def test_different_proportions_score_lower_than_identical(tmp_path: Path) -> None:
    box = _stl(tmp_path, "box.stl", trimesh.creation.box([40, 40, 40]))
    long_bar = _stl(tmp_path, "bar.stl", trimesh.creation.box([160, 20, 20]))
    compact = shape_similarity(box, box)
    elongated = shape_similarity(box, long_bar)
    assert compact is not None and elongated is not None
    assert elongated < compact - 0.1  # a long bar is measurably farther from a cube


def test_fails_closed_on_a_bad_path(tmp_path: Path) -> None:
    box = _stl(tmp_path, "box.stl", trimesh.creation.box([40, 40, 40]))
    assert shape_similarity(box, tmp_path / "nope.stl") is None
