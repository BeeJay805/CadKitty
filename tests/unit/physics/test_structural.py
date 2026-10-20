"""Quantitative field, equilibrium, recovery and fail-closed structural checks."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

pytest.importorskip("skfem")
from skfem import MeshTet

from forge_mind.physics.structural import _mesh_resolution, _solve_field, structural_field


def test_volume_mesh_digest_ignores_node_element_and_winding_order() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -3]], dtype=float)
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    original = _mesh_resolution(points, cells)
    permutation = np.array([4, 2, 0, 3, 1])
    inverse = np.argsort(permutation)
    reordered_points = points[permutation].copy()
    reordered_points[reordered_points == 0] = -0.0
    reordered = _mesh_resolution(reordered_points, inverse[cells[::-1, ::-1]])
    assert reordered["volume_mesh_sha256"] == original["volume_mesh_sha256"]
    assert reordered["volume_weighted_rms_tet_diameter_mm"] == pytest.approx(
        original["volume_weighted_rms_tet_diameter_mm"]
    )
    changed = points.copy()
    changed[4, 2] = -2
    assert _mesh_resolution(changed, cells)["volume_mesh_sha256"] != original["volume_mesh_sha256"]


def test_resolution_weights_actual_tet_volumes_not_element_counts() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -3]], dtype=float)
    stats = _mesh_resolution(points, np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))
    # Tetra volumes are 1/6 and 1/2; squared diameters are 2 and 10.
    assert stats["volume_weighted_rms_tet_diameter_mm"] == pytest.approx(np.sqrt(8))
    assert stats["maximum_tet_diameter_mm"] == pytest.approx(np.sqrt(10))


def _bar(
    *,
    force: float = 1,
    elastic: float = 1000,
    direction: tuple[float, float, float] = (0, 0, 1),
    shift: float = 0,
    fixed_end: str = "minimum",
) -> dict[str, Any]:
    mesh = MeshTet().refined(2).scaled((2, 2, 20)).translated((shift, shift, shift))
    return _solve_field(
        mesh,
        force_n=force,
        direction=direction,
        e_mpa=elastic,
        nu=0.0,
        fixed_end=fixed_end,
        planes=(shift, shift + 20),
        tolerance=1e-8,
        original_areas=(4, 4),
    )


def test_axial_actual_tensors_and_full_quadratic_surface() -> None:
    result = _bar()
    json.dumps(result, allow_nan=False)
    summary, field, diagnostics = result["summary"], result["field"], result["diagnostics"]
    assert summary["max_displacement_mm"] == pytest.approx(1 * 20 / (1000 * 4), rel=1e-8)
    assert summary["peak_von_mises_mpa"] == pytest.approx(0.25, rel=1e-8)
    tensors = np.asarray(field["stress_tensors_mpa"])
    strains = np.asarray(field["strain_tensors"])
    assert tensors[:, 2, 2] == pytest.approx(np.full(len(tensors), 0.25), abs=1e-8)
    assert strains[:, 2, 2] == pytest.approx(np.full(len(strains), 0.00025), abs=1e-10)
    assert diagnostics["reaction_force_n"] == pytest.approx([0, 0, -1], abs=1e-8)
    assert diagnostics["status"] == "pass"
    assert diagnostics["relative_residual"] < 1e-8
    assert diagnostics["moment_balance_relative_error"] < 1e-8
    positions = np.asarray(field["positions_mm"])
    triangles = np.asarray(field["triangles"])
    assert triangles.max() < len(positions)
    assert np.any(np.isclose(positions[:, 2], 2.5))  # P2 edge nodes between 5 mm layers
    assert len(field["fixed_triangles"]) > 0
    assert np.allclose(positions[triangles[field["fixed_triangles"]], 2], 0)
    assert np.allclose(positions[triangles[field["loaded_triangles"]], 2], 20)
    facets = positions[triangles]
    normals = np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0])
    assert np.all(np.sum(normals * (facets.mean(axis=1) - [1, 1, 10]), axis=1) > 0)
    volume_points = np.asarray(field["volume_positions_mm"])
    tetrahedra = np.asarray(field["tetrahedra"])
    element_ids = np.asarray(field["quadrature_element_ids"])
    quadrature = np.asarray(field["quadrature_positions_mm"])
    assert tetrahedra.shape == (diagnostics["tetrahedra"], 4)
    assert tetrahedra.min() >= 0 and tetrahedra.max() < len(volume_points)
    assert element_ids.shape == (len(tensors),)
    assert np.array_equal(element_ids, np.repeat(np.arange(len(tetrahedra)), 14))
    # Every tensor sample belongs inside its identified actual volume element.
    corners = volume_points[tetrahedra[element_ids]]
    barycentric = np.linalg.solve(
        (corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1),
        (quadrature - corners[:, 0])[..., None],
    )[..., 0]
    assert barycentric.min() >= -1e-10
    assert np.max(barycentric.sum(axis=1)) <= 1 + 1e-10


def test_force_modulus_scaling_and_translated_frame() -> None:
    baseline = _bar()
    doubled = _bar(force=2)
    stiffer = _bar(elastic=2000)
    translated = _bar(shift=17)
    for key in ("max_displacement_mm", "peak_von_mises_mpa"):
        assert doubled["summary"][key] == pytest.approx(2 * baseline["summary"][key], rel=1e-8)
        assert translated["summary"][key] == pytest.approx(baseline["summary"][key], rel=1e-8)
    assert stiffer["summary"]["max_displacement_mm"] == pytest.approx(
        baseline["summary"]["max_displacement_mm"] / 2
    )
    assert stiffer["summary"]["peak_von_mises_mpa"] == pytest.approx(
        baseline["summary"]["peak_von_mises_mpa"]
    )


def test_bending_peak_is_quadrature_statistic_not_display_maximum() -> None:
    # Near-isotropic tets resolve bending; simply stretching a coarse cube mesh does not.
    mesh = MeshTet.init_tensor(np.linspace(0, 2, 5), np.linspace(0, 2, 5), np.linspace(0, 20, 21))
    result = _solve_field(
        mesh,
        force_n=1,
        direction=(1, 0, 0),
        e_mpa=1000,
        nu=0,
        fixed_end="minimum",
        planes=(0, 20),
        tolerance=1e-8,
        original_areas=(4, 4),
    )
    inertia = 2 * 2**3 / 12
    expected = 20**3 / (3 * 1000 * inertia)
    assert result["summary"]["max_displacement_mm"] == pytest.approx(expected, rel=0.06)
    tensors = np.asarray(result["field"]["stress_tensors_mpa"])
    dev = tensors - np.trace(tensors, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(dev**2, axis=(1, 2)))
    assert result["summary"]["peak_von_mises_mpa"] == pytest.approx(vm.max())
    assert result["summary"]["display_max_von_mises_mpa"] < vm.max()


def test_load_rotation_about_analysis_axis_and_reversed_restraint() -> None:
    x = _bar(direction=(1, 0, 0))
    y = _bar(direction=(0, 1, 0))
    reverse = _bar(direction=(0, 0, -1), fixed_end="maximum")
    assert x["summary"]["max_displacement_mm"] == pytest.approx(
        y["summary"]["max_displacement_mm"], rel=0.01
    )
    assert reverse["summary"]["max_displacement_mm"] == pytest.approx(0.005, rel=1e-8)
    assert reverse["diagnostics"]["reaction_force_n"] == pytest.approx([0, 0, 1], abs=1e-8)


def test_mesh_budget_and_boundary_mismatch_fail_before_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_DOF_LIMIT", "1")
    with pytest.raises(ValueError, match="DOFs"):
        _bar()
    monkeypatch.delenv("FORGE_WORKSPACE_STRUCTURAL_DOF_LIMIT")
    mesh = MeshTet().scaled((2, 2, 20))
    with pytest.raises(ValueError, match="area"):
        _solve_field(
            mesh,
            force_n=1,
            direction=(0, 0, 1),
            e_mpa=1000,
            nu=0,
            fixed_end="minimum",
            planes=(0, 20),
            tolerance=1e-8,
            original_areas=(5, 4),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force_n": -1},
        {"nu": 0.5},
        {"direction": (0, 0, 0)},
        {"mesh_factor": 0.01},
        {"fixed_end": "side"},
    ],
)
def test_invalid_setup_rejected_before_native_mesh(kwargs: dict[str, Any]) -> None:
    settings = {
        "force_n": 1,
        "direction": (0, 0, 1),
        "e_mpa": 1000,
        "nu": 0.3,
        "mesh_factor": 0.2,
        "fixed_end": "minimum",
    }
    settings.update(kwargs)
    with pytest.raises(ValueError):
        structural_field(None, **settings)


def test_native_planar_component_field() -> None:
    pytest.importorskip("pytetwild")
    import trimesh

    result = structural_field(
        trimesh.creation.box(extents=(5, 5, 20)),
        force_n=1,
        direction=(0, 0, 1),
        e_mpa=70000,
        nu=0,
        mesh_factor=0.2,
        fixed_end="minimum",
    )
    assert result["summary"]["max_displacement_mm"] == pytest.approx(20 / (70000 * 25), rel=0.01)
    assert result["summary"]["peak_von_mises_mpa"] == pytest.approx(1 / 25, rel=0.01)
    assert result["diagnostics"]["loaded_area_mm2"] == pytest.approx(25, rel=0.005)
    assert result["diagnostics"]["status"] == "pass"
    assert all(case["passed"] for case in result["solver"]["validation_cases"])


def test_finite_but_wrong_displacement_fails_equilibrium(monkeypatch: pytest.MonkeyPatch) -> None:
    import skfem

    monkeypatch.setattr(skfem, "solve", lambda matrix, rhs, initial, free: np.zeros_like(initial))
    with pytest.raises(RuntimeError, match="equilibrium"):
        _bar()


def test_failed_new_solver_validation_prevents_meshing(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge_mind.physics.structural as structural

    monkeypatch.setattr(structural, "_validation_cases", lambda: ({"passed": False},))
    with pytest.raises(RuntimeError, match="validation gate"):
        structural_field(
            None,
            force_n=1,
            direction=(0, 0, 1),
            e_mpa=1000,
            nu=0,
            mesh_factor=0.2,
            fixed_end="minimum",
        )


def test_curved_extremes_cannot_substitute_a_patch() -> None:
    import trimesh

    with pytest.raises(ValueError, match="planar faces"):
        structural_field(
            trimesh.creation.icosphere(subdivisions=1),
            force_n=1,
            direction=(0, 0, 1),
            e_mpa=1000,
            nu=0,
            mesh_factor=0.2,
            fixed_end="minimum",
        )


def test_supplied_cad_mesh_uses_same_solver_without_surface_remeshing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    import trimesh

    # This path must work without TetWild installed, retaining the same tensor,
    # support-area and equilibrium checks used for a surface-generated mesh.
    monkeypatch.setitem(sys.modules, "pytetwild", None)
    volume = MeshTet().refined(2).scaled((2, 2, 20)).translated((-1, -1, -10))
    result = structural_field(
        trimesh.creation.box(extents=(2, 2, 20)),
        force_n=1,
        direction=(0, 0, 1),
        e_mpa=1000,
        nu=0,
        mesh_factor=0.1,
        fixed_end="minimum",
        tetra_mesh=volume,
        mesher_metadata={"name": "Gmsh", "version": "test-recorded-version"},
    )
    assert result["summary"]["max_displacement_mm"] == pytest.approx(0.005, rel=1e-8)
    assert result["summary"]["peak_von_mises_mpa"] == pytest.approx(0.25, rel=1e-8)
    assert result["diagnostics"]["status"] == "pass"
    assert result["solver"]["mesher"]["name"] == "Gmsh"
    assert result["solver"]["mesher_version"] == "test-recorded-version"
    assert result["solver"]["mesh_factor"] == 0.1
    # A supplied volume mesh does not bypass comparison to original CAD caps.
    with pytest.raises(ValueError, match="area"):
        structural_field(
            trimesh.creation.box(extents=(4, 2, 20)),
            force_n=1,
            direction=(0, 0, 1),
            e_mpa=1000,
            nu=0,
            mesh_factor=0.1,
            fixed_end="minimum",
            tetra_mesh=volume,
        )
