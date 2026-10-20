"""Structural studies preserve the selected CAD and never launder qualification."""

import copy
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.structural import analysis_frame, compare_study, world_field
from forge_mind.workspace.structural_models import StructuralStudy
from forge_mind.workspace.structural_service import list_studies, read_study, validate_field


def study():
    return {
        "name": "Declared axial study",
        "target": {"body_index": 0},
        "material": {
            "name": "Analytic test material",
            "source": "Independent numerical test fixture",
            "density_kg_m3": 2700,
            "elastic_modulus_mpa": 70000,
            "poisson_ratio": 0.3,
            "yield_strength_mpa": 200,
        },
        "axis": [0, 0, 1],
        "direction": [0, 0, 1],
        "force_n": 100,
        "max_deflection_mm": 0.1,
        "min_safety_factor": 2,
        "basis": "Test an isolated component, not a real assembly load assumption",
    }


def field():
    return {
        "schema_version": 1,
        "frame": "cad_z_up_mm",
        "positions_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "triangles": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        "displacements_mm": [[0, 0, 0]] * 3 + [[0, 0, 0.01]],
        "von_mises_mpa": [1, 1, 1, 1],
        "strain_equivalent": [0.001] * 4,
        "fixed_triangles": [0],
        "loaded_triangles": [3],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("force_n", 0),
        ("force_n", float("nan")),
        ("direction", [0, 0, 0]),
        ("axis", [0, 0, 0]),
        ("basis", ""),
        ("mesh_factors", [0.1, 0.16]),
        ("target", {"body_index": 0, "occurrence_path": "other"}),
    ],
)
def test_study_rejects_undefined_physical_setup(key, value):
    with pytest.raises(ValidationError):
        StructuralStudy.model_validate({**study(), key: value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("frame", "gltf_y_up"),
        ("positions_mm", [[0, 0, 0]]),
        ("von_mises_mpa", [0] * 3),
        ("von_mises_mpa", [0, 0, 0, float("nan")]),
        ("triangles", [[0, 1, 9]] * 4),
        ("triangles", [[0.0, 1.0, 2.0]] * 4),
        ("fixed_triangles", []),
        ("loaded_triangles", [0]),
        ("loaded_triangles", [3, 3]),
    ],
)
def test_missing_or_corrupt_field_cannot_be_displayed_as_zero(key, value):
    with pytest.raises(ValueError):
        validate_field({**field(), key: value})


def test_field_and_analysis_frame_preserve_vector_semantics():
    validate_field(field())
    frame = analysis_frame([1, 2, -3])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1)
    vector = np.array([1, -2, 4])
    np.testing.assert_allclose((vector @ frame) @ frame.T, vector, atol=1e-12)


def test_world_transform_preserves_tensor_invariants_and_translates_only_points():
    value = field()
    value.update(
        quadrature_positions_mm=[[0.25, 0.25, 0.25]],
        stress_tensors_mpa=[np.diag([1, 2, 7]).tolist()],
        strain_tensors=[np.diag([0.1, 0.2, 0.7]).tolist()],
        volume_positions_mm=copy.deepcopy(value["positions_mm"]),
        tetrahedra=[[0, 1, 2, 3]],
        quadrature_element_ids=[0],
    )
    frame = analysis_frame([2, -1, 3])
    origin = np.array([30, 40, -60])
    world_field(value, frame, origin)
    validate_field(value)
    np.testing.assert_allclose(np.linalg.eigvalsh(value["stress_tensors_mpa"][0]), [1, 2, 7])
    np.testing.assert_allclose(value["displacements_mm"][0], [0, 0, 0])
    np.testing.assert_allclose(value["positions_mm"][0], origin)
    np.testing.assert_allclose(value["volume_positions_mm"], value["positions_mm"])
    np.testing.assert_allclose(
        value["quadrature_positions_mm"][0], np.mean(value["volume_positions_mm"], axis=0)
    )
    value["tetrahedra"] = [[0, 1, 2, 4]]
    with pytest.raises(ValueError, match="volume mesh connectivity"):
        validate_field(value)
    value["tetrahedra"] = [[0, 1, 2, 3]]
    value["strain_tensors"] = []
    with pytest.raises(ValueError, match="tensor association"):
        validate_field(value)


def result(displacement=0.01, stress=20, valid=True):
    return {
        "summary": {
            "max_displacement_mm": displacement,
            "peak_von_mises_mpa": stress,
            "display_max_von_mises_mpa": stress / 2,
        },
        "diagnostics": {"status": "pass" if valid else "unverified"},
        "solver": {},
    }


def mesh_result(points, cells, **kwargs):
    from forge_mind.physics.structural import _mesh_resolution

    points, cells = np.asarray(points, dtype=float), np.asarray(cells, dtype=np.int64)
    edges = np.unique(
        np.sort(
            np.concatenate(
                [cells[:, pair] for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
            ),
            axis=1,
        ),
        axis=0,
    )
    value = result(**kwargs)
    value["diagnostics"].update(
        _mesh_resolution(points, cells),
        displacement_dofs=3 * (len(np.unique(cells)) + len(edges)),
        tetrahedra=len(cells),
    )
    return value


def refined_results(**kwargs):
    # Subdivide the same tetrahedral domain using edge midpoints, retaining volume.
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ]
    )
    cells = [
        [0, 4, 5, 6],
        [4, 1, 7, 8],
        [5, 7, 2, 9],
        [6, 8, 9, 3],
        [4, 5, 6, 9],
        [4, 5, 7, 9],
        [4, 6, 8, 9],
        [4, 7, 8, 9],
    ]
    return mesh_result(points[:4], [[0, 1, 2, 3]], **kwargs), mesh_result(points, cells, **kwargs)


def test_actual_refinement_can_support_stable_response_comparison():
    coarse, fine = refined_results()
    response = compare_study(StructuralStudy.model_validate(study()), coarse, fine)
    assert response["numerical_quality"]["status"] == "pass"
    assert 0 < response["numerical_quality"]["refinement"]["diameter_ratio"] < 0.7
    assert all(c["status"] == "pass" for c in response["comparisons"])


def test_identical_mesh_and_missing_mesh_never_prove_refinement():
    coarse, _ = refined_results()
    for a, b in ((coarse, copy.deepcopy(coarse)), (result(), result())):
        response = compare_study(StructuralStudy.model_validate(study()), a, b)
        assert response["numerical_quality"]["status"] == "unverified"
        assert response["numerical_quality"]["refinement"]["status"] == "unverified"
        assert all(c["status"] == "unverified" for c in response["comparisons"])
        assert response["summary"]["max_displacement_mm"] == 0.01


def test_equally_coarse_different_triangulation_is_not_refinement():
    # Two diagonal choices partition exactly the same square pyramid.
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]
    coarse = mesh_result(points, [[0, 1, 2, 4], [0, 2, 3, 4]])
    fine = mesh_result(points, [[0, 1, 3, 4], [1, 2, 3, 4]])
    assert coarse["diagnostics"]["volume_mesh_sha256"] != fine["diagnostics"]["volume_mesh_sha256"]
    response = compare_study(StructuralStudy.model_validate(study()), coarse, fine)
    assert response["numerical_quality"]["refinement"]["diameter_ratio"] == pytest.approx(1)
    assert response["numerical_quality"]["status"] == "unverified"


def test_more_dofs_without_smaller_actual_diameter_is_not_refinement():
    # Inserting a centroid increases DOFs/elements but leaves every boundary edge intact.
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]]
    coarse = mesh_result(points[:4], [[0, 1, 2, 3]])
    fine = mesh_result(points, [[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
    assert fine["diagnostics"]["displacement_dofs"] > coarse["diagnostics"]["displacement_dofs"]
    assert fine["diagnostics"]["tetrahedra"] > coarse["diagnostics"]["tetrahedra"]
    response = compare_study(StructuralStudy.model_validate(study()), coarse, fine)
    assert response["numerical_quality"]["refinement"]["diameter_ratio"] == pytest.approx(1)
    assert response["numerical_quality"]["status"] == "unverified"


def test_numerical_criteria_use_unsmoothed_sampled_peak_not_display_peak():
    response = compare_study(StructuralStudy.model_validate(study()), *refined_results(stress=120))
    assert response["comparisons"][1]["status"] == "fail"
    assert response["summary"]["yield_to_sampled_stress_ratio"] == pytest.approx(200 / 120)


def test_nonconverged_stress_or_invalid_solver_cannot_pass():
    config = StructuralStudy.model_validate(study())
    coarse, fine = refined_results()
    bad_stress, bad_solver = copy.deepcopy(fine), copy.deepcopy(fine)
    bad_stress["summary"]["peak_von_mises_mpa"] = 2
    bad_solver["diagnostics"]["status"] = "unverified"
    for fine in (bad_stress, bad_solver):
        response = compare_study(config, coarse, fine)
        assert response["numerical_quality"]["status"] == "unverified"
        assert all(c["status"] == "unverified" for c in response["comparisons"])


def test_missing_yield_or_limits_preserves_computed_response_without_approval():
    config = study()
    config["material"].pop("yield_strength_mpa")
    config.pop("max_deflection_mm")
    response = compare_study(StructuralStudy.model_validate(config), *refined_results())
    assert response["summary"]["max_displacement_mm"] == 0.01
    assert all(c["status"] == "unverified" for c in response["comparisons"])


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from forge_mind.workspace import structural_service

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    sid = service.store.create("local-owner", "Test structure", {"requirements": []})["id"]
    rid = "a" * 32
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "contract": {"requirements": []},
            "measurement": {
                "geometry_valid": True,
                "engine_fingerprint": "engine",
                "artifact_hashes": {"part.step": "step", "topology.json": "topology"},
            },
        },
    )
    proposal = {"revision_id": rid, "based_on": None, "eligible": True}
    service.store.update(sid, proposal=proposal)
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    monkeypatch.setattr(structural_service, "engine_fingerprint", lambda: "engine")

    def computed(directory, payload, timeout, cancelled):
        assert timeout is None
        (directory / "field.json").write_text(json.dumps(field()), encoding="utf-8")
        return {
            "status": "computed",
            "summary": result()["summary"],
            "comparisons": [],
            "target": {"body_index": 0},
            "numerical_quality": {"status": "pass"},
        }

    monkeypatch.setattr(structural_service, "process_revision", computed)
    return TestClient(create_app(service=service, local=True)), service, sid, rid, proposal


def enqueue(workspace, **overrides):
    client, _, sid, rid, _ = workspace
    return client.post(
        f"/workspace/sessions/{sid}/structural-studies",
        json={"revision_id": rid, "input_version": 0, "study": study(), **overrides},
    )


@pytest.fixture
def cad_workspace(workspace, monkeypatch):
    from forge_mind.workspace import structural_service

    original = structural_service.process_revision

    def computed_with_meshes(directory, *args):
        result = original(directory, *args)
        for stage in ("coarse", "fine"):
            target = directory / stage
            target.mkdir()
            (target / "selected.step").write_bytes(b"Synthetic selected STEP for integrity test")
            np.savez_compressed(
                target / "mesh.npz",
                positions_mm=np.asarray(field()["positions_mm"]),
                tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int64),
            )
        return result

    monkeypatch.setattr(structural_service, "process_revision", computed_with_meshes)
    return workspace


def test_cad_mesh_receipt_roundtrip_hashes_and_source_staleness(cad_workspace):
    from forge_mind.workspace.product_sources import _append
    from forge_mind.workspace.structural_service import STUDY_ARTIFACTS, study_path

    _, service, sid, _, proposal = cad_workspace
    queued = enqueue(cad_workspace, study={**study(), "mesher": "cad"}).json()
    service.run_job(service.store.claim())
    receipt = read_study(service, sid, "local-owner", queued["study_id"])
    assert set(receipt["hashes"]) == STUDY_ARTIFACTS
    directory = study_path(service, sid, receipt["id"])
    for name, expected in receipt["hashes"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    for stage in ("coarse", "fine"):
        with np.load(directory / stage / "mesh.npz", allow_pickle=False) as mesh:
            np.testing.assert_array_equal(mesh["tetrahedra"], [[0, 1, 2, 3]])
            np.testing.assert_array_equal(mesh["positions_mm"], field()["positions_mm"])
    _append(service.store, sid, "local-owner", {"kind": "artifact", "data": {"id": "new_source"}})
    assert read_study(service, sid, "local-owner", receipt["id"])["stale"]
    assert service.store.get(sid, "local-owner")["proposal"] == proposal


@pytest.mark.parametrize(
    "name", ["coarse/selected.step", "fine/selected.step", "coarse/mesh.npz", "fine/mesh.npz"]
)
@pytest.mark.parametrize("mutation", ["tamper", "remove"])
def test_cad_mesh_receipt_rejects_changed_or_missing_artifact(cad_workspace, name, mutation):
    from forge_mind.workspace.structural_service import study_path

    _, service, sid, _, _ = cad_workspace
    queued = enqueue(cad_workspace, study={**study(), "mesher": "cad"}).json()
    service.run_job(service.store.claim())
    assert read_study(service, sid, "local-owner", queued["study_id"])["status"] == "computed"
    path = study_path(service, sid, queued["study_id"]) / name
    if mutation == "tamper":
        path.write_bytes(b"altered")
    else:
        path.unlink()
    with pytest.raises(WorkspaceError, match="artifact changed"):
        read_study(service, sid, "local-owner", queued["study_id"])


@pytest.mark.parametrize(
    "name", ["coarse/selected.step", "fine/selected.step", "coarse/mesh.npz", "fine/mesh.npz"]
)
def test_computed_cad_worker_must_supply_every_mesh_artifact(cad_workspace, monkeypatch, name):
    from forge_mind.workspace import structural_service

    _, service, sid, _, _ = cad_workspace
    original = structural_service.process_revision

    def incomplete(directory, *args):
        result = original(directory, *args)
        (directory / name).unlink()
        return result

    monkeypatch.setattr(structural_service, "process_revision", incomplete)
    assert enqueue(cad_workspace, study={**study(), "mesher": "cad"}).status_code == 202
    service.run_job(service.store.claim())
    assert list_studies(service, sid, "local-owner") == []


def test_real_queue_preserves_candidate_and_publishes_hash_checked_fields(workspace):
    client, service, sid, rid, proposal = workspace
    queued = enqueue(workspace)
    assert queued.status_code == 202, queued.text
    assert len(queued.content) < 1024
    job = service.store.claim()
    service.run_job(job)
    after = service.store.get(sid, "local-owner")
    assert after["proposal"] == proposal
    assert after["selected_revision"] is None
    receipt = read_study(service, sid, "local-owner", queued.json()["study_id"])
    assert receipt["qualification_status"] == "unverified"
    assert not receipt["stale"]
    assert receipt["matches_current_design"]
    downloaded = client.get(
        f"/workspace/sessions/{sid}/structural-studies/{receipt['id']}/field.json"
    )
    assert hashlib.sha256(downloaded.content).hexdigest() == receipt["hashes"]["field.json"]
    assert downloaded.json() == field()
    assert (
        client.get(f"/workspace/sessions/{sid}/structural-studies?revision_id={rid}").json()[
            "studies"
        ][0]["id"]
        == receipt["id"]
    )


def test_input_revision_and_owner_guards_before_work(workspace):
    client, service, sid, rid, _ = workspace
    assert enqueue(workspace, input_version=4).status_code == 409
    service.store.update(sid, proposal={"revision_id": "b" * 32})
    assert enqueue(workspace).status_code == 409
    assert service.store.claim() is None
    assert (
        client.post(
            f"/workspace/sessions/{'c' * 32}/structural-studies",
            json={"revision_id": rid, "input_version": 0, "study": study()},
        ).status_code
        == 409
    )


@pytest.mark.parametrize("mutation", ["cancel", "input", "source", "revision"])
def test_late_mutation_cannot_publish_a_success(workspace, monkeypatch, mutation):
    from forge_mind.workspace import structural_service

    _, service, sid, _, _ = workspace
    original = structural_service.process_revision

    def modified(*args):
        result = original(*args)
        with service.store.connect() as con:
            if mutation == "cancel":
                con.execute("UPDATE jobs SET cancel=1 WHERE session=?", (sid,))
            elif mutation == "input":
                con.execute(
                    "INSERT INTO engineering_inputs VALUES (?,?,?)", (sid, 1, json.dumps({}))
                )
            elif mutation == "revision":
                record = json.loads(
                    con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
                )
                record["proposal"] = {"revision_id": "b" * 32}
                con.execute(
                    "UPDATE sessions SET document=? WHERE id=?",
                    (json.dumps(record), sid),
                )
            else:
                # A changed source fingerprint must fail the final transaction.
                monkeypatch.setattr(
                    "forge_mind.workspace.product_sources.source_fingerprint", lambda *a: "changed"
                )
        return result

    monkeypatch.setattr(structural_service, "process_revision", modified)
    assert enqueue(workspace).status_code == 202
    service.run_job(service.store.claim())
    assert list_studies(service, sid, "local-owner") == []


def test_tampered_fields_and_changed_inputs_invalidate_results(workspace):
    from forge_mind.workspace.structural_service import study_path

    _, service, sid, _, _ = workspace
    queued = enqueue(workspace).json()
    service.run_job(service.store.claim())
    study_id = queued["study_id"]
    original = service.store.engineering(sid)
    service.store.save_engineering(sid, "local-owner", 0, original["inputs"])
    assert read_study(service, sid, "local-owner", study_id)["stale"]
    (study_path(service, sid, study_id) / "field.json").write_text("{}")
    with pytest.raises(WorkspaceError, match="artifact changed"):
        read_study(service, sid, "local-owner", study_id)


def test_historical_filter_precedes_limit_and_lists_do_not_expand_history(workspace, monkeypatch):
    from forge_mind.workspace.structural_service import _table

    _, service, sid, rid, _ = workspace
    queued = enqueue(workspace).json()
    service.run_job(service.store.claim())
    original = read_study(service, sid, "local-owner", queued["study_id"])
    _table(service)
    with service.store.connect() as con:
        for i in range(31):
            item = copy.deepcopy(original)
            item.update(id=f"{i + 1:032x}", revision_id="b" * 32)
            encoded = json.dumps(item)
            con.execute(
                "INSERT INTO structural_studies VALUES (?,?,?,?)",
                (item["id"], sid, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
            )
    monkeypatch.setattr(service.store, "get", lambda *a: pytest.fail("List expanded full history"))
    records = list_studies(service, sid, "local-owner", rid)
    assert [r["id"] for r in records] == [queued["study_id"]]
    assert records[0]["request_hash"] == digest(
        StructuralStudy.model_validate(study()).model_dump()
    )
