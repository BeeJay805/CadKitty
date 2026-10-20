"""Shared private-engineering eligibility refuses guaranteed failures before queueing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import Header
from tests.unit.workspace.test_motion_check_admission import RID, assert_unallocated, fixture_case

from forge_mind.workspace import engineering_service, structural_api, structural_service
from forge_mind.workspace.store import WorkspaceError
from forge_mind.workspace.structural_api import structural_router

REASONS = {
    "review": "Engineering parser/solver worker is private-local only until container verification",
    "structure": "Structural worker is private-local until container verification",
}


def case_for(tmp_path, monkeypatch, *, mode="docker", valid=True, qualified=False):
    case = fixture_case(tmp_path, monkeypatch, mode=mode, valid=valid)
    if qualified:
        case.service.executor._typed_container_verified = True

    def owner(authorization: str = Header()):
        return authorization

    case.client.app.include_router(structural_router(case.store, case.service, owner))
    case.study_ids = []

    def identifier():
        case.study_ids.append("id")
        return SimpleNamespace(hex="e" * 32)

    monkeypatch.setattr(structural_api, "uuid", SimpleNamespace(uuid4=identifier))

    def forbidden(*args, **kwargs):
        pytest.fail("Admission tests cannot run parsers, native workers or solvers")

    monkeypatch.setattr(engineering_service, "process_revision", forbidden)
    monkeypatch.setattr(structural_service, "process_revision", forbidden)
    return case


def submit(case, operation, *, owner="owner", revision_id=RID):
    body = {"revision_id": revision_id}
    if operation == "structure":
        body.update(
            input_version=0,
            study={
                "name": "Saved admission fixture",
                "target": {"body_index": 0},
                "material": {
                    "name": "Declared fixture",
                    "source": "Synthetic declared inputs",
                    "density_kg_m3": 1000,
                    "elastic_modulus_mpa": 100,
                    "poisson_ratio": 0.3,
                },
                "force_n": 1,
                "direction": [0, 0, 1],
                "basis": "No engineering result inferred",
            },
        )
    path = "engineering" if operation == "review" else "structural-studies"
    return case.client.post(
        f"/workspace/sessions/{case.sid}/{path}", headers={"Authorization": owner}, json=body
    )


def untouched(case):
    assert_unallocated(case)
    assert case.study_ids == []
    assert not (case.store.root / "reports").exists()
    assert not (case.store.root / "structural-studies").exists()


@pytest.mark.parametrize("operation", ["review", "structure"])
@pytest.mark.parametrize("qualified", [False, True], ids=["unqualified", "typed_cad_qualified"])
def test_docker_solver_routes_refuse_before_any_job_or_study_id(
    tmp_path, monkeypatch, operation, qualified
):
    case = case_for(tmp_path, monkeypatch, qualified=qualified)
    response = submit(case, operation)
    assert response.status_code == 409
    assert response.json() == {"detail": REASONS[operation]}
    assert len(case.loaded) == 1
    untouched(case)


@pytest.mark.parametrize("operation", ["review", "structure"])
def test_valid_trusted_sources_keep_original_job_payload_and_queue(
    tmp_path, monkeypatch, operation
):
    case = case_for(tmp_path, monkeypatch, mode="trusted_local")
    response = submit(case, operation)
    assert response.status_code == 202
    assert response.json()["state"] == "queued"
    assert len(case.admitted) == 1 and case.started == ["start"]
    with case.store.connect() as con:
        rows = con.execute("SELECT state,payload FROM jobs").fetchall()
    assert len(rows) == 1 and rows[0]["state"] == "queued"
    payload = json.loads(rows[0]["payload"])
    assert payload["engineering_operation"] == (
        "review" if operation == "review" else "structural_study"
    )
    assert payload["revision_id"] == RID
    if operation == "structure":
        assert case.study_ids == ["id"]
        assert response.json()["study_id"] == payload["study_id"] == "e" * 32
    else:
        assert case.study_ids == []
        assert payload["operation_binding"]["version"] == 1
    assert not (case.store.root / "reports").exists()
    assert not (case.store.root / "structural-studies").exists()
    assert case.store.revision(case.sid, RID) == case.revision
    assert case.store.get(case.sid, "owner").get("accepted_revision") is None


@pytest.mark.parametrize("operation,status", [("review", 404), ("structure", 409)])
def test_existing_owner_error_precedes_executor_and_artifact_eligibility(
    tmp_path, monkeypatch, operation, status
):
    case = case_for(tmp_path, monkeypatch)
    (case.directory / "part.step").write_bytes(b"changed")
    response = submit(case, operation, owner="foreign")
    assert response.status_code == status
    assert response.json()["detail"] != REASONS[operation]
    assert case.loaded == []
    untouched(case)


@pytest.mark.parametrize("operation", ["review", "structure"])
@pytest.mark.parametrize("changed", [True, False], ids=["changed_artifact", "missing_revision"])
def test_existing_integrity_and_revision_errors_precede_executor_eligibility(
    tmp_path, monkeypatch, operation, changed
):
    case = case_for(tmp_path, monkeypatch)
    if changed:
        (case.directory / "part.step").write_bytes(b"changed")
    response = submit(case, operation, revision_id=RID if changed else "c" * 32)
    assert response.status_code == 409
    assert response.json()["detail"] != REASONS[operation]
    assert (
        "artifacts changed" in response.json()["detail"]
        if changed
        else "revision not found" in response.json()["detail"]
    )
    untouched(case)


def test_structural_existing_geometry_refusal_stays_before_new_mode_check(tmp_path, monkeypatch):
    case = case_for(tmp_path, monkeypatch, valid=False)
    response = submit(case, "structure")
    assert response.status_code == 409
    assert response.json()["detail"] == "Structural study requires valid exported geometry"
    untouched(case)


@pytest.mark.parametrize("operation", ["review", "structure"])
def test_existing_workers_still_refuse_before_any_parser_solver_or_files(
    tmp_path, monkeypatch, operation
):
    case = case_for(tmp_path, monkeypatch)
    job = {"session": case.sid, "owner": "owner", "revision_id": RID}
    function = (
        engineering_service.run_report_job
        if operation == "review"
        else structural_service.run_structural_job
    )
    with pytest.raises(WorkspaceError, match=REASONS[operation]):
        function(case.service, job, lambda: False)
    assert case.loaded == []
    untouched(case)
