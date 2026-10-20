"""CAD mesher identities, trimmed-boundary checks and native hollow geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import build123d as b
import numpy as np
import pytest

from forge_mind.physics.cad_meshing import (
    _face_samples,
    _mapped_elements,
    _mesh_facts,
    mesh_cad_solid,
)


def test_noncontiguous_gmsh_tags_preserve_actual_connectivity() -> None:
    points, cells = _mapped_elements(
        [40, 2, 99, 7], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [4], [np.array([99, 40, 7, 2])]
    )
    assert cells.tolist() == [[2, 0, 3, 1]]
    facts, boundary = _mesh_facts(points, cells)
    assert facts["mesh_volume_mm3"] == pytest.approx(1 / 6)
    assert facts["p2_displacement_dofs"] == 30
    assert len(boundary) == 4


@pytest.mark.parametrize(
    "types,nodes,match",
    [
        ([11], [[1] * 10], "first-order"),
        ([4], [[1, 2, 3, 70]], "unknown"),
        ([4], [[1, 2, 3, 3]], "repeats"),
        ([4], [[1, 2, 3]], "Malformed"),
    ],
)
def test_bad_element_types_and_tags_are_rejected(types: Any, nodes: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _mapped_elements([1, 2, 3, 4], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], types, nodes)


def test_disconnected_or_degenerate_tets_rejected() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError, match="disconnected"):
        _mesh_facts(np.concatenate([points, points + 3]), np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    points[3] = [1, 1, 0]
    with pytest.raises(ValueError, match="degenerate"):
        _mesh_facts(points, np.array([[0, 1, 2, 3]]))


def test_filled_bore_rejected_by_trimmed_face_samples_after_rotation() -> None:
    plate = (b.Box(10, 10, 2) - b.Cylinder(2, 5)).rotate(b.Axis.Y, 31).translate((12, -7, 9))
    axis = np.asarray(tuple(b.Vector(0, 0, 1).rotate(b.Axis.Y, 31)))
    face = next(
        f
        for f in plate.faces()
        if f.geom_type == b.GeomType.PLANE and np.dot(tuple(f.normal_at()), axis) > 0.99
    )
    # All three corners lie on real face material; their centroid lies in the bore.
    local = [(3, 0, 1), (-1.5, 2.598076211, 1), (-1.5, -2.598076211, 1)]
    points = np.array(
        [tuple(b.Vector(p).rotate(b.Axis.Y, 31) + b.Vector(12, -7, 9)) for p in local]
    )
    with pytest.raises(ValueError, match="trimmed CAD face"):
        _face_samples(face, points, np.array([[0, 1, 2]]), 0.01)


def test_optional_dependency_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import forge_mind.physics.cad_meshing as meshing

    def missing(name: str) -> Any:
        raise ImportError("not installed")

    monkeypatch.setattr(meshing.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="gmsh is unavailable"):
        mesh_cad_solid(b.Box(2, 2, 2), 0.16, tmp_path)


def test_point_contact_is_not_connected_material() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError, match="disconnected"):
        _mesh_facts(np.concatenate([points, -points[1:]]), np.array([[0, 1, 2, 3], [0, 4, 5, 6]]))


def test_native_session_finalized_when_geometry_validation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gmsh = pytest.importorskip("gmsh")
    import forge_mind.physics.cad_meshing as meshing

    def rejected(*args: Any) -> Any:
        raise ValueError("Deliberately missing boundary coverage")

    monkeypatch.setattr(meshing, "_boundary_receipts", rejected)
    with pytest.raises(ValueError, match="missing boundary coverage"):
        mesh_cad_solid(b.Box(2, 2, 2), 0.16, tmp_path)
    assert not gmsh.isInitialized()


def test_missing_cad_face_inventory_rejected() -> None:
    from types import SimpleNamespace

    from forge_mind.physics.cad_meshing import _boundary_receipts

    gmsh = SimpleNamespace(model=SimpleNamespace(getEntities=lambda dimension: [(2, 1)]))
    with pytest.raises(ValueError, match="face count"):
        _boundary_receipts(gmsh, b.Box(2, 2, 2), np.zeros((4, 3)), [1, 2, 3, 4], set(), 0.01, 3.0)


def test_native_rotated_bored_plate_preserves_faces_and_limits(tmp_path: Path) -> None:
    gmsh = pytest.importorskip("gmsh")
    solid = (b.Box(20, 12, 4) - b.Cylinder(2, 6)).rotate(b.Axis.Y, 37).translate((22, -10, 40))
    points, cells, receipt = mesh_cad_solid(solid, 0.16, tmp_path)
    assert not gmsh.isInitialized()
    assert len(cells) < 12000 and receipt["p2_displacement_dofs"] < 30000
    assert receipt["relative_volume_error"] < 0.005
    assert len(receipt["faces"]) == len(solid.faces()) == 7
    assert all(face["triangle_count"] > 0 for face in receipt["faces"])
    assert sum(face["triangle_count"] for face in receipt["faces"]) == receipt["boundary_triangles"]
    assert points.min(axis=0) == pytest.approx(list(solid.bounding_box().min), abs=0.03)
    assert len(receipt["selected_step_sha256"]) == 64
    assert (tmp_path / "selected.step").is_file()


def test_budget_guard_reports_actual_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_TET_LIMIT", "1")
    with pytest.raises(ValueError, match="CAD tetrahedra requires 2, limit is 1"):
        _mapped_elements(
            [1, 2, 3, 4], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [4], [[1, 2, 3, 4, 1, 2, 3, 4]]
        )
