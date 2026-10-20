"""Actual STEP import preserves dimensions and selects one closed body."""

from itertools import pairwise

import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.flow_meshing import _layer_heights, mesh_flow
from forge_mind.workspace.flow_models import FlowMeshSettings, FlowSettings


def test_actual_cad_mesh_has_si_dimensions_and_bound_hashes(tmp_path):
    pytest.importorskip("gmsh")
    from build123d import Sphere, export_step

    source = tmp_path / "source.step"
    export_step(Sphere(10), source)
    settings = FlowSettings(
        density_kg_m3=1,
        dynamic_viscosity_pa_s=1,
        velocity_m_s=(1, 0, 0),
        reference_area_m2=0.001,
        reference_length_m=0.02,
        mesh={"surface_size_factor": 0.4, "domain_padding_factor": 3, "farfield_size_factor": 3},
    )
    result = mesh_flow(source, tmp_path, settings)
    assert result["body_length_m"] == pytest.approx(0.02)
    assert result["body_volume_m3"] == pytest.approx(4.188790204786391e-6)
    assert result["reynolds_body_length"] == pytest.approx(0.02)
    assert result["declared_regime"] == "laminar"
    assert result["regime_applicability"] == "unverified"
    assert result["minimum_scaled_jacobian"] > 0
    assert result["nodes"] > 10
    assert result["mesh_sha256"] == file_sha256(tmp_path / "mesh.su2")
    assert result["source_step_sha256"] == file_sha256(source)


@pytest.mark.parametrize("scale", [0.8, 0.65, 0.2])
@pytest.mark.parametrize("layers", [1, 6, 12])
def test_wall_refinement_preserves_coverage_without_final_sliver(scale, layers):
    coarse = FlowMeshSettings(wall_layers=layers, first_layer_size_factor=0.001)
    fine = coarse.model_copy(update={"refinement": "fine", "fine_scale": scale})
    original = _layer_heights(coarse, 0.04, 0.04)
    refined = _layer_heights(fine, 0.04, 0.04)
    assert refined[-1] == pytest.approx(original[-1])
    assert refined[0] <= original[0] * scale * (1 + 1e-12)
    assert len(refined) > len(original)
    thicknesses = [refined[0], *(b - a for a, b in pairwise(refined))]
    assert all(b / a == pytest.approx(coarse.layer_growth) for a, b in pairwise(thicknesses))


def test_shared_reference_does_not_override_body_safety_limit():
    mesh = FlowMeshSettings(wall_layers=6, first_layer_size_factor=0.01)
    with pytest.raises(ValueError, match="half the body length"):
        _layer_heights(mesh, 10, 0.02)


def test_actual_reference_grading_and_extent_controls(tmp_path):
    pytest.importorskip("gmsh")
    from build123d import Sphere, export_step

    source = tmp_path / "source.step"
    export_step(Sphere(10), source)
    settings = FlowSettings(
        density_kg_m3=1,
        dynamic_viscosity_pa_s=1,
        velocity_m_s=(1, 0, 0),
        reference_area_m2=0.001,
        reference_length_m=0.02,
        mesh={
            "reference_length_m": 0.03,
            "surface_size_factor": 0.3,
            "farfield_size_factor": 0.8,
            "domain_padding_factor": 6,
            "wall_layers": 3,
            "first_layer_size_factor": 0.001,
        },
    )
    receipts = {}
    for name, changes in {
        "uniform": {},
        "graded": {"grading_rate": 0.4},
        "expanded": {"grading_rate": 0.4, "domain_padding_factor": 9},
        "fine": {"grading_rate": 0.4, "refinement": "fine", "fine_scale": 0.8},
    }.items():
        directory = tmp_path / name
        directory.mkdir()
        profile = settings.model_copy(update={"mesh": settings.mesh.model_copy(update=changes)})
        receipts[name] = mesh_flow(source, directory, profile)
    uniform, graded, expanded, fine = (
        receipts[k] for k in ("uniform", "graded", "expanded", "fine")
    )
    for receipt in receipts.values():
        assert receipt["minimum_scaled_jacobian"] > 0
        assert receipt["body_length_m"] == pytest.approx(0.02)
        assert receipt["mesh_reference_length_m"] == pytest.approx(0.03)
        assert receipt["reynolds_body_length"] == pytest.approx(0.02)
        assert receipt["prisms"] > 0
    assert uniform["grading_expression"] is None
    assert graded["nodes"] < uniform["nodes"]
    assert graded["domain_bounds_m"] == uniform["domain_bounds_m"]
    assert graded["domain_bounds_m"][0] == pytest.approx([-0.19] * 3)
    assert graded["surface_size_m"] == pytest.approx(0.009)
    assert graded["refinement_distance_m"] == pytest.approx(0.09)
    assert expanded["domain_bounds_m"][0] == pytest.approx([-0.28] * 3)
    assert expanded["grading_expression"] == graded["grading_expression"]
    assert expanded["surface_size_m"] == graded["surface_size_m"]
    assert expanded["farfield_size_m"] == graded["farfield_size_m"]
    assert expanded["layer_heights_m"] == graded["layer_heights_m"]
    assert fine["grading_rate_scaled"] == pytest.approx(0.4 * 0.8)
    assert fine["surface_size_m"] == pytest.approx(graded["surface_size_m"] * 0.8)
    assert fine["wall_layer_coverage_m"] == pytest.approx(graded["wall_layer_coverage_m"])
    assert fine["actual_wall_layers"] > graded["actual_wall_layers"]
    assert fine["actual_first_layer_height_m"] <= graded["actual_first_layer_height_m"] * 0.8


def test_closed_spline_surface_has_real_prismatic_wall_elements(tmp_path):
    pytest.importorskip("gmsh")
    from build123d import Matrix, Sphere, export_step

    source = tmp_path / "spline.step"
    body = Sphere(10).transform_geometry(
        Matrix([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )
    export_step(body, source)
    settings = FlowSettings(
        density_kg_m3=1,
        dynamic_viscosity_pa_s=1e-5,
        velocity_m_s=(1, 0, 0),
        reference_area_m2=0.001,
        reference_length_m=0.04,
        mesh={
            "surface_size_factor": 0.3,
            "domain_padding_factor": 3,
            "wall_layers": 6,
            "first_layer_size_factor": 0.001,
        },
    )
    result = mesh_flow(source, tmp_path, settings)
    assert result["body_length_m"] == pytest.approx(0.04)
    assert result["reynolds_body_length"] == pytest.approx(4000)
    assert result["regime_applicability"] == "unverified"
    assert result["prisms"] > 0
    assert result["tetrahedra"] > 0
    assert result["surface_algorithm"] == "MeshAdapt"
    assert len(result["layer_heights_m"]) == 6
    assert result["minimum_scaled_jacobian"] > 0
