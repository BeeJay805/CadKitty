"""Capacity policies are optional; geometry and numerical validity remain mandatory."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from forge_mind.physics import structural as physics
from forge_mind.workspace import structural_service
from forge_mind.workspace.processing import resource_policy
from forge_mind.workspace.store import Store
from forge_mind.workspace.structural import write_field
from forge_mind.workspace.structural_service import validate_field


@pytest.fixture(autouse=True)
def no_policy(monkeypatch):
    for name in (
        "TET_LIMIT",
        "DOF_LIMIT",
        "SURFACE_FACE_LIMIT",
        "SURFACE_VERTEX_LIMIT",
        "FIELD_POINT_LIMIT",
        "FIELD_TRIANGLE_LIMIT",
        "QUADRATURE_LIMIT",
        "FIELD_BYTES",
        "RECEIPT_BYTES",
    ):
        monkeypatch.delenv("FORGE_WORKSPACE_STRUCTURAL_" + name, raising=False)


def field():
    return {
        "schema_version": 1,
        "frame": "cad_z_up_mm",
        "positions_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "triangles": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        "displacements_mm": [[0, 0, 0]] * 4,
        "von_mises_mpa": [0] * 4,
        "strain_equivalent": [0] * 4,
        "fixed_triangles": [0],
        "loaded_triangles": [3],
    }


def solve(mesh):
    return physics._solve_field(
        mesh,
        force_n=1,
        direction=(0, 0, 1),
        e_mpa=1000,
        nu=0,
        fixed_end="minimum",
        planes=(0, 1),
        tolerance=1e-8,
        original_areas=(1, 1),
    )


def test_structural_policies_are_discoverable_and_dynamic(monkeypatch):
    before = resource_policy()
    assert len(before["structural"]) == 9
    assert all(value["value"] is None for value in before["structural"].values())
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_TET_LIMIT", "12345")
    after = resource_policy()
    assert after["structural"]["structural_tet_limit"]["value"] == 12345
    assert after["hosted"] == before["hosted"]


@pytest.mark.parametrize("policy", ["TET_LIMIT", "DOF_LIMIT"])
def test_explicit_solver_policy_rejects_before_stiffness_or_quadrature_allocation(
    monkeypatch, policy
):
    import skfem

    mesh = skfem.MeshTet().refined(1)
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_" + policy, "1")
    monkeypatch.setattr(
        skfem, "Basis", lambda *a, **kw: pytest.fail("Solver allocated beyond policy")
    )
    with pytest.raises(ValueError, match="Operator resource policy"):
        solve(mesh)


def test_mesh_above_previous_tet_and_dof_limits_reaches_physical_checks_without_solving(
    monkeypatch,
):
    from skfem import MeshTet

    mesh = MeshTet.init_tensor(*[np.linspace(0, 1, 14)] * 3)
    assert mesh.t.shape[1] > 12000
    assert 3 * (mesh.p.shape[1] + mesh.edges.shape[1]) > 30000

    def stop_at_boundary(*args):
        raise RuntimeError("Reached physical boundary checks")

    monkeypatch.setattr(physics, "_boundary_facets", stop_at_boundary)
    with pytest.raises(RuntimeError, match="Reached physical boundary checks"):
        solve(mesh)


def test_real_exported_surface_above_previous_face_and_vertex_limits(monkeypatch, tmp_path):
    import trimesh

    mesh = trimesh.creation.box(extents=(2, 2, 20))
    for _ in range(6):
        mesh = mesh.subdivide()
    assert len(mesh.vertices) > 20000 and len(mesh.faces) > 40000
    path = tmp_path / "large-surface.stl"
    mesh.export(path)
    assert path.stat().st_size > 0
    reached = []
    monkeypatch.setattr(physics, "_validation_cases", lambda: ({"passed": True},))
    monkeypatch.setattr(
        physics, "_solve_field", lambda *a, **kw: reached.append(True) or {"solver": {}}
    )
    physics.structural_field(
        mesh,
        force_n=1,
        direction=(0, 0, 1),
        e_mpa=1000,
        nu=0,
        mesh_factor=0.1,
        fixed_end="minimum",
        tetra_mesh=object(),
    )
    assert reached == [True]
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_SURFACE_FACE_LIMIT", "40000")
    with pytest.raises(ValueError, match="structural surface faces"):
        physics.structural_field(
            mesh,
            force_n=1,
            direction=(0, 0, 1),
            e_mpa=1000,
            nu=0,
            mesh_factor=0.1,
            fixed_end="minimum",
            tetra_mesh=object(),
        )
    assert reached == [True]


def test_field_above_previous_vertex_triangle_and_tet_limits_is_validated():
    value = field()
    count = 200001
    value.update(
        positions_mm=np.broadcast_to([0.0, 0.0, 0.0], (count, 3)),
        displacements_mm=np.broadcast_to([0.0, 0.0, 0.0], (count, 3)),
        von_mises_mpa=np.zeros(count),
        strain_equivalent=np.zeros(count),
        triangles=np.broadcast_to([0, 1, 2], (400001, 3)),
        volume_positions_mm=np.broadcast_to([0.0, 0.0, 0.0], (count, 3)),
        tetrahedra=np.broadcast_to([0, 1, 2, 3], (12001, 4)),
        quadrature_positions_mm=[[0.0, 0.0, 0.0]],
        stress_tensors_mpa=[np.eye(3)],
        strain_tensors=[np.eye(3)],
        quadrature_element_ids=[12000],
    )
    validate_field(value)
    value["quadrature_element_ids"] = [12001]
    with pytest.raises(ValueError, match="volume mesh connectivity"):
        validate_field(value)


def test_quadrature_above_previous_sample_limit_keeps_tensor_checks():
    value = field()
    count = 1000001
    value.update(
        quadrature_positions_mm=np.broadcast_to([0.0, 0.0, 0.0], (count, 3)),
        stress_tensors_mpa=np.broadcast_to(np.eye(3), (count, 3, 3)),
        strain_tensors=np.broadcast_to(np.eye(3), (count, 3, 3)),
    )
    validate_field(value)
    value["strain_tensors"] = np.broadcast_to(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], (count, 3, 3)
    )
    with pytest.raises(ValueError, match="tensor association"):
        validate_field(value)


@pytest.mark.parametrize(
    "policy,key,count",
    [
        ("FIELD_POINT_LIMIT", "positions_mm", 5),
        ("FIELD_TRIANGLE_LIMIT", "triangles", 5),
        ("QUADRATURE_LIMIT", "quadrature_positions_mm", 5),
        ("TET_LIMIT", "tetrahedra", 5),
    ],
)
def test_field_policy_rejects_before_numpy_copy(monkeypatch, policy, key, count):
    value = field()
    value[key] = [[0, 1, 2]] * count
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_" + policy, "4")
    monkeypatch.setattr(
        structural_service.np,
        "asarray",
        lambda *a, **kw: pytest.fail("Array allocated beyond policy"),
    )
    with pytest.raises(ValueError, match="Operator resource policy"):
        validate_field(value)


def test_field_export_above_previous_byte_limit_and_explicit_rejection(tmp_path, monkeypatch):
    value = field()
    value["note"] = "x" * (64 * 1024**2 + 1)
    path = tmp_path / "field.json"
    write_field(path, value)
    assert path.stat().st_size > 64 * 1024**2
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_FIELD_BYTES", "100")
    with pytest.raises(ValueError, match="structural field bytes"):
        write_field(tmp_path / "rejected.json", field())
    assert not (tmp_path / "rejected.json").exists()


def test_receipt_above_previous_byte_limit_keeps_digest_validation(tmp_path, monkeypatch):
    service = SimpleNamespace(store=Store(tmp_path))
    sid = service.store.create("owner", "study", None)["id"]
    structural_service._table(service)
    record = {"notes": "x" * (2 * 1024**2 + 1)}
    encoded = json.dumps(record)
    with service.store.connect() as con:
        con.execute(
            "INSERT INTO structural_studies VALUES (?,?,?,?)",
            ("study", sid, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    assert structural_service._read_record(service, sid, "owner", "study") == record
    monkeypatch.setenv("FORGE_WORKSPACE_STRUCTURAL_RECEIPT_BYTES", "100")
    with pytest.raises(ValueError, match="structural receipt bytes"):
        structural_service._read_record(service, sid, "owner", "study")
    monkeypatch.delenv("FORGE_WORKSPACE_STRUCTURAL_RECEIPT_BYTES")
    with service.store.connect() as con:
        con.execute("UPDATE structural_studies SET digest='wrong'")
    with pytest.raises(ValueError, match="integrity"):
        structural_service._read_record(service, sid, "owner", "study")
