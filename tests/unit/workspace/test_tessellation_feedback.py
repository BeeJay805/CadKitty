from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from forge_mind.workspace import tessellation


def record(mesh: Any, *, identity=(None, "not_named"), index=0, volume=None):
    message = tessellation._mesh_failure_message(
        mesh, index, float(mesh.volume) if volume is None else volume, identity
    )
    prefix = "STEP tessellation has invalid body topology or volume: "
    assert message.startswith(prefix)
    assert len(message) <= 1400
    return json.loads(message[len(prefix) :])


def edge_contact_mesh():
    # Two closed tetrahedra meet along one edge, producing incidence four.
    vertices = [[0, 0, 0], [0, 0, 2], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    faces += [[0, 4, 1], [0, 1, 5], [0, 5, 4], [1, 4, 5]]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_nonmanifold_edge_control_has_exact_incidence_and_cad_coordinates():
    mesh = edge_contact_mesh()
    result = record(mesh, index=3, identity=("ck_occ_4_base", "bound_exported_occurrence_label"))
    assert result["body_index"] == 3
    assert result["occurrence_label"] == "ck_occ_4_base"
    assert "not_watertight" in result["failed_conditions"]
    assert result["boundary_edges"] == {"count": 0, "sample": [], "omitted": 0}
    assert result["nonmanifold_edges"] == {
        "count": 1,
        "sample": [{"incidence": 4, "endpoints_mm": [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]}],
        "omitted": 0,
    }
    assert result["frame"] == "input_step_mm"


def test_open_mesh_reports_boundary_edges_without_inventing_nonmanifold_edges():
    mesh = trimesh.creation.box(extents=[2, 3, 4])
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    result = record(mesh)
    assert result["boundary_edges"]["count"] == 3
    assert len(result["boundary_edges"]["sample"]) == 2
    assert result["boundary_edges"]["omitted"] == 1
    assert result["nonmanifold_edges"]["count"] == 0


def test_ordinary_closed_mesh_has_no_failed_conditions_or_bad_edges():
    mesh = trimesh.creation.box(extents=[2, 3, 4])
    result = record(mesh)
    assert result["failed_conditions"] == []
    assert result["mesh_volume_mm3"] == 24
    assert result["boundary_edges"]["count"] == result["nonmanifold_edges"]["count"] == 0


@pytest.mark.parametrize(
    "volume,condition",
    [
        (math.nan, "mesh_volume_nonfinite"),
        (-2.0, "mesh_volume_nonpositive"),
        (0.0, "mesh_volume_nonpositive"),
    ],
)
def test_existing_volume_conditions_are_explicit_and_json_finite(volume, condition):
    result = record(trimesh.creation.box(), volume=volume)
    assert condition in result["failed_conditions"]
    assert result["mesh_volume_mm3"] == (None if math.isnan(volume) else volume)


def test_winding_failure_is_distinct_from_edge_incidence_failure():
    mesh = trimesh.creation.box()
    faces = mesh.faces.copy()
    faces[0] = faces[0][::-1]
    mesh.faces = faces
    result = record(mesh)
    assert mesh.is_watertight
    assert result["failed_conditions"] == ["winding_inconsistent"]
    assert result["boundary_edges"]["count"] == result["nonmanifold_edges"]["count"] == 0


def test_samples_and_message_are_bounded_without_changing_mesh():
    vertices = np.arange(18000, dtype=float).reshape(-1, 3) * 1e100
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(6000).reshape(-1, 3), process=False)
    before_faces, before_vertices = mesh.faces.copy(), mesh.vertices.copy()
    result = record(mesh, volume=1.0, identity=("é" * 64, "bound_exported_occurrence_label"))
    assert result["boundary_edges"]["count"] == 6000
    assert len(result["boundary_edges"]["sample"]) <= 2
    assert result["boundary_edges"]["omitted"] == 6000 - len(result["boundary_edges"]["sample"])
    np.testing.assert_array_equal(mesh.faces, before_faces)
    np.testing.assert_array_equal(mesh.vertices, before_vertices)


def test_unavailable_edge_inventory_does_not_become_zero_edges():
    mesh = types.SimpleNamespace(is_watertight=False, is_winding_consistent=True)
    result = record(mesh, volume=1.0)
    assert result["edge_inventory_status"] == "unavailable:AttributeError"
    assert "boundary_edges" not in result and "nonmanifold_edges" not in result


def test_export_identity_requires_unique_label_and_unique_exported_body():
    a, b = object(), object()

    def same(left, right):
        return left is right

    assert tessellation._unique_occurrence_label({"ck_occ_4_base": a}, [a, b], 0, same) == (
        "ck_occ_4_base",
        "bound_exported_occurrence_label",
    )
    assert tessellation._unique_occurrence_label({"a": a, "b": a}, [a], 0, same) == (
        None,
        "ambiguous",
    )
    assert tessellation._unique_occurrence_label({"a": a}, [a, a], 0, same) == (None, "ambiguous")
    assert tessellation._unique_occurrence_label({}, [a], 0, same) == (None, "not_named")
    assert tessellation._unique_occurrence_label({"x" * 129: a}, [a], 0, same) == (
        None,
        "label_exceeds_diagnostic_bound",
    )


def mocked_export(tmp_path: Path, monkeypatch, meshes):
    solids = [types.SimpleNamespace(is_valid=True, volume=float(mesh.volume)) for mesh in meshes]
    part = types.SimpleNamespace(is_valid=True, solids=lambda: solids)
    calls = []
    fake = types.ModuleType("build123d")
    fake.import_step = lambda path: part

    def export(shape, path, **settings):
        calls.append((shape, Path(path).name, settings))
        Path(path).write_bytes(b"unchanged-exporter-output")

    fake.export_stl = export
    monkeypatch.setitem(sys.modules, "build123d", fake)
    monkeypatch.setattr(
        trimesh, "load", lambda *args, **kwargs: meshes[min(len(calls) - 1, len(meshes) - 1)]
    )
    (tmp_path / "part.step").write_bytes(b"saved STEP fixture")
    return calls


def test_successful_export_never_calls_failure_identity_and_keeps_outputs(tmp_path, monkeypatch):
    calls = mocked_export(tmp_path, monkeypatch, [trimesh.creation.box()])
    monkeypatch.setattr(
        tessellation,
        "_exported_body_identity",
        lambda *args: pytest.fail("Success called diagnostic identity"),
    )
    assert tessellation.export_verified_tessellation(tmp_path) == (0.005, 0.05, 0)
    assert [call[1] for call in calls] == ["body.stl", "part.stl"]
    assert all(call[2] == {"tolerance": 0.005, "angular_tolerance": 0.05} for call in calls)
    assert (tmp_path / "part.stl").read_bytes() == b"unchanged-exporter-output"
    assert json.loads((tmp_path / "tessellation.json").read_text())["linear_deflection"] == 0.005


def test_failed_body_is_identified_without_publishing_success_artifacts(tmp_path, monkeypatch):
    calls = mocked_export(tmp_path, monkeypatch, [trimesh.creation.box(), edge_contact_mesh()])
    monkeypatch.setattr(
        tessellation,
        "_exported_body_identity",
        lambda part, solids, index: ("ck_occ_4_base", "bound_exported_occurrence_label"),
    )
    with pytest.raises(ValueError) as caught:
        tessellation.export_verified_tessellation(tmp_path)
    result = json.loads(str(caught.value).split(": ", 1)[1])
    assert result["body_index"] == 1
    assert result["nonmanifold_edges"]["count"] == 1
    assert len(calls) == 2
    assert not (tmp_path / "part.stl").exists()
    assert not (tmp_path / "tessellation.json").exists()


def test_native_tangent_boss_rib_has_localized_refusal(tmp_path):
    # A valid native solid can still have nonmanifold mesh edges at line contacts.
    import build123d as b

    plate = b.Box(20, 20, 1, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    rib = b.Pos(0, -4, 1) * b.Box(12, 2, 5, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    boss = b.Pos(0, 0, 1) * b.Cylinder(3, 5, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    shape = plate + rib + boss
    shape.label = "ck_occ_4_base"
    assert shape.is_valid and shape.volume > 0 and len(shape.solids()) == 1
    b.export_step(shape, tmp_path / "part.step")
    with pytest.raises(ValueError, match="invalid body topology or volume") as caught:
        tessellation.export_verified_tessellation(tmp_path)
    result = json.loads(str(caught.value).split(": ", 1)[1])
    assert result["body_index"] == 0
    assert result["occurrence_label"] == "ck_occ_4_base"
    assert result["nonmanifold_edges"]["count"] > 0
    assert result["boundary_edges"]["count"] == 0
    assert all(edge["incidence"] > 2 for edge in result["nonmanifold_edges"]["sample"])


def test_native_ordinary_box_export_remains_valid(tmp_path):
    import build123d as b

    b.export_step(b.Box(2, 3, 4), tmp_path / "part.step")
    assert tessellation.export_verified_tessellation(tmp_path) == (0.005, 0.05, 0)
    mesh = trimesh.load(tmp_path / "part.stl", force="mesh")
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(24)
