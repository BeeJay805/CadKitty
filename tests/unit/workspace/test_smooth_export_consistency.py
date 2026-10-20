"""Real smooth STEP exports retain strict geometry checks without treating sampled extrema as CAD."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh
from build123d import Box, Compound, Pos, Sphere, export_step, export_stl, import_step, scale

from forge_mind.workspace.measure import measure_revision
from forge_mind.workspace.models import Contract
from forge_mind.workspace.tessellation import (
    STL_ANGULAR_TOLERANCE,
    STL_LINEAR_TOLERANCE,
    export_verified_tessellation,
    read_tessellation_settings,
)


@pytest.fixture
def sphere_export(tmp_path: Path):
    export_step(Sphere(1000), tmp_path / "part.step")
    export_stl(
        import_step(tmp_path / "part.step"),
        tmp_path / "part.stl",
        tolerance=STL_LINEAR_TOLERANCE,
        angular_tolerance=STL_ANGULAR_TOLERANCE,
    )
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")
    faces = mesh.faces
    mesh.update_faces(
        (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(tmp_path / "part.stl")
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert np.max(np.abs(mesh.bounds - [[-1000] * 3, [1000] * 3])) > 0.01
    return tmp_path, mesh


@pytest.mark.parametrize("structured", [False, True])
def test_real_smooth_export_has_independent_tessellation_identity(sphere_export, structured):
    directory, _ = sphere_export
    report = measure_revision(directory, "sphere", Contract(), structured_assembly=structured)
    assert report.geometry_valid
    consistency = report.facts["export_consistency"]
    assert consistency["volume_within_relative_0_001"]
    assert not consistency["bounds_within_0_01_mm"]
    assert consistency["reference_tessellation"]["triangle_inventory_matches"]
    assert not report.requirements_satisfied
    assert "manufacturing precision" in consistency["scope"]


@pytest.mark.parametrize("change", ["translate", "vertex", "reverse", "volume"])
def test_mismatched_smooth_mesh_is_not_excused_by_approximate_bounds(sphere_export, change):
    directory, mesh = sphere_export
    if change == "translate":
        mesh.apply_translation([0.005, 0, 0])
    elif change == "vertex":
        mesh.vertices[100, 0] += 0.001
    elif change == "reverse":
        mesh.faces = mesh.faces[:, ::-1]
    else:
        mesh.apply_scale(1.01)
    assert mesh.is_watertight and mesh.is_winding_consistent
    mesh.export(directory / "part.stl")
    report = measure_revision(directory, "sphere", Contract())
    assert not report.geometry_valid
    assert not report.facts["export_agrees"]
    if change != "volume":
        assert report.facts["export_consistency"]["volume_within_relative_0_001"]
        assert not report.facts["export_consistency"]["reference_tessellation"][
            "triangle_inventory_matches"
        ]


def ellipsoid(radius):
    base = Sphere(radius)
    result = scale(base, by=(1.5, 1.5**-0.5, 1.5**-0.5))
    return scale(result, by=(base.volume / result.volume) ** (1 / 3))


@pytest.mark.parametrize("radius", [100, 1000, 10000])
def test_real_bspline_ellipsoid_refines_only_until_volume_agrees(tmp_path, radius):
    export_step(ellipsoid(radius), tmp_path / "part.step")
    export_stl(
        import_step(tmp_path / "part.step"),
        tmp_path / "part.stl",
        tolerance=STL_LINEAR_TOLERANCE,
        angular_tolerance=STL_ANGULAR_TOLERANCE,
    )
    initial = trimesh.load(tmp_path / "part.stl", force="mesh")
    volume = float(import_step(tmp_path / "part.step").volume)
    assert not math.isclose(initial.volume, volume, rel_tol=0.001, abs_tol=0.001)
    linear, angular, _ = export_verified_tessellation(tmp_path)
    assert (linear, angular) == (0.0025, 0.025)
    final = trimesh.load(tmp_path / "part.stl", force="mesh")
    assert len(initial.faces) < len(final.faces) < len(initial.faces) * 5
    report = measure_revision(tmp_path, "ellipsoid", Contract())
    assert report.geometry_valid
    assert report.facts["body_mesh_evidence"]["triangle_inventory_matches"]
    assert "tessellation.json" in report.artifact_hashes


def test_large_body_does_not_mask_small_body_volume_error(tmp_path):
    # The box dominates aggregate volume, but the small ellipsoid still needs refinement.
    shape = Compound(children=[Box(10000, 10000, 10000), Pos(20000, 0, 0) * ellipsoid(100)])
    export_step(shape, tmp_path / "part.step")
    assert export_verified_tessellation(tmp_path)[:2] == (0.0025, 0.025)
    report = measure_revision(tmp_path, "assembly", Contract(), structured_assembly=True)
    assert report.geometry_valid
    assert all(body["valid"] for body in report.facts["body_mesh_evidence"]["bodies"])


@pytest.fixture(scope="module")
def adaptive_sphere(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adaptive-sphere")
    export_step(Sphere(1000), directory / "part.step")
    assert export_verified_tessellation(directory)[:2] == (0.005, 0.05)
    return directory


@pytest.mark.parametrize(
    "change", ["step_hash", "coarser", "zero", "nan", "boolean", "extra", "different_finer", "mesh"]
)
def test_adaptive_metadata_and_mesh_tampering_fail_closed(adaptive_sphere, tmp_path, change):
    for name in ("part.step", "part.stl", "tessellation.json"):
        shutil.copyfile(adaptive_sphere / name, tmp_path / name)
    path = tmp_path / "tessellation.json"
    metadata = json.loads(path.read_text())
    if change == "step_hash":
        metadata["step_sha256"] = "0" * 64
    elif change == "coarser":
        metadata["linear_deflection"] = 0.01
    elif change == "zero":
        metadata["angular_deflection"] = 0
    elif change == "nan":
        metadata["linear_deflection"] = float("nan")
    elif change == "boolean":
        metadata["version"] = True
    elif change == "extra":
        metadata["trusted"] = True
    elif change == "different_finer":
        metadata.update(linear_deflection=0.0025, angular_deflection=0.025)
    else:
        mesh = trimesh.load(tmp_path / "part.stl", force="mesh")
        mesh.apply_translation([0.005, 0, 0])
        mesh.export(tmp_path / "part.stl")
    path.write_text(json.dumps(metadata))
    assert not measure_revision(tmp_path, "tampered", Contract()).geometry_valid


def test_legacy_export_defaults_remain_compatible(tmp_path):
    assert read_tessellation_settings(tmp_path) == (0.005, 0.05)


@pytest.mark.parametrize("invalid", ["duplicate", "oversized"])
def test_metadata_is_validated_before_hashing_or_meshing(tmp_path, invalid):
    path = tmp_path / "tessellation.json"
    path.write_text('{"version":1,"version":1}' if invalid == "duplicate" else " " * 4097)
    with pytest.raises(ValueError, match=r"Duplicate|metadata file"):
        read_tessellation_settings(tmp_path)


def test_real_postprocessor_publishes_refined_stl_glb_and_bound_metadata(tmp_path):
    from forge_mind.workspace.processing import process_revision

    export_step(ellipsoid(1000), tmp_path / "part.step")
    result = process_revision(
        tmp_path,
        {"revision_id": "ellipsoid", "contract": Contract().model_dump()},
        60,
        lambda: False,
    )
    assert result["geometry_valid"]
    assert result["facts"]["tessellation"]["linear_deflection"] == 0.0025
    assert "tessellation.json" in result["artifact_hashes"]
    assert (tmp_path / "part.glb").stat().st_size > 0
    assert "0.0025" in result["facts"]["render_source"]


def test_failed_refinement_stagnation_stops_without_metadata(tmp_path, monkeypatch):
    import build123d

    export_step(ellipsoid(1000), tmp_path / "part.step")
    real_export = build123d.export_stl
    calls = []

    def stuck(shape, path, **kwargs):
        calls.append(kwargs)
        return real_export(shape, path, tolerance=0.005, angular_tolerance=0.05)

    monkeypatch.setattr(build123d, "export_stl", stuck)
    with pytest.raises(ValueError, match="stagnated"):
        export_verified_tessellation(tmp_path)
    assert len(calls) == 2
    assert not (tmp_path / "tessellation.json").exists()
