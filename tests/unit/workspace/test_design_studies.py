"""Adversarial design-study lifecycle controls, using actual exported geometry."""

from __future__ import annotations

import json

import build123d as b
import pytest
from fastapi.testclient import TestClient

from forge_mind.server.workspace_app import create_app
from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_study_models import (
    AddStudyCandidate,
    CreateDesignStudy,
    DesignStudySpec,
    SearchDesignStudy,
)
from forge_mind.workspace.design_study_service import (
    _save_candidate,
    add_candidate,
    candidate_directory,
    compare_candidates,
    create_study,
    enqueue_study,
    read_study,
    run_study_job,
)
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store, WorkspaceError


def study_spec():
    return DesignStudySpec.model_validate(
        {
            "name": "Reduce material while preserving useful span",
            "objectives": [{"metric": "geometry.volume_mm3", "direction": "minimize"}],
            "constraints": [
                {"metric": "geometry.volume_mm3", "operator": "ge", "value": 3000},
                {"metric": "geometry.extent_x_mm", "operator": "ge", "value": 100},
            ],
            "variables": {"width": [2, 8], "thickness": [2, 8]},
            "evaluators": [{"kind": "geometry"}],
            "comparison_tolerance": 0.001,
        }
    )


@pytest.fixture
def studies(tmp_path, monkeypatch):
    service = WorkspaceService(
        Store(tmp_path),
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Unexpected model call"),
    )
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    owner = "local-owner"
    contract = Contract()
    sid = service.store.create(owner, "Preserve span", contract.model_dump())["id"]
    document = DesignDocument.model_validate(
        {
            "name": "Test part",
            "variables": {"width": 8, "thickness": 8},
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "kind": "box",
                    "params": {"x": 100},
                    "bindings": {"y": "width", "z": "thickness"},
                }
            ],
            "outputs": ["body"],
        }
    )
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    shape, _ = compile_document(document, {})
    b.export_step(shape, str(directory / "part.step"))
    fingerprint = engine_fingerprint()
    revision = {
        "code": "",
        "id": rid,
        "document": document.model_dump(),
        "contract": contract.model_dump(),
        "measurement": {
            "geometry_valid": True,
            "engine_fingerprint": fingerprint,
            "artifact_hashes": {"part.step": file_sha256(directory / "part.step")},
            "facts": {"solid_count": 1},
            "findings": [],
        },
    }
    service.store.add_revision(sid, revision)
    service.store.update(sid, selected_revision=rid)
    # Tests freeze instruments while independent implementation agents may edit unrelated source.
    monkeypatch.setattr(
        "forge_mind.workspace.design_study_service.engine_fingerprint", lambda: fingerprint
    )
    monkeypatch.setattr("forge_mind.workspace.lab_records.engine_fingerprint", lambda: fingerprint)
    from forge_mind.workspace.design_evaluators import geometry_worker

    def evaluate(
        _service, _sid, revision, evaluator, directory, cancelled, sensitivity_tolerance=0.02
    ):
        if cancelled():
            raise ValueError("cancelled")
        directory.mkdir(parents=True, exist_ok=True)
        result = geometry_worker(service.store.directory(sid, revision["id"]) / "part.step")
        result["metrics"] = {"geometry." + k: v for k, v in result["metrics"].items()}
        (directory / "measurement.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("forge_mind.workspace.design_study_service.evaluate_design", evaluate)
    return service, owner, sid, rid, document


def create(studies):
    service, owner, sid, rid, _ = studies
    return create_study(
        service, sid, owner, CreateDesignStudy(revision_id=rid, input_version=0, spec=study_spec())
    )


def claim_and_run(studies, study, *, search=None, retry_failed=False, cancelled=lambda: False):
    service, owner, sid, _, _ = studies
    queued = enqueue_study(
        service, sid, owner, study["id"], search=search, retry_failed=retry_failed
    )
    job = service.store.claim()
    assert job is not None and job["id"] == queued["job_id"]
    run_study_job(service, job, cancelled)
    service.store.finish(job["id"], sid, "cancelled" if cancelled() else "needs_review")
    return read_study(service, sid, owner, study["id"])


def test_fixed_protocol_and_actual_geometry_score(studies):
    study = create(studies)
    after = claim_and_run(studies, study)
    assert after["candidates"][0]["status"] == "complete", after["candidates"][0].get("error")
    assert after["candidates"][0]["metrics"]["geometry.volume_mm3"] == pytest.approx(6400)
    assert after["comparison"]["conclusion"] == "baseline_best"
    assert after["spec_hash"] == digest(study["spec"])


def test_candidate_dedup_and_bounds(studies):
    service, owner, sid, _, _ = studies
    study = create(studies)
    baseline = add_candidate(
        service, sid, owner, study["id"], AddStudyCandidate(parameters={"width": 8})
    )
    assert baseline["id"] == study["candidates"][0]["id"]
    a = add_candidate(service, sid, owner, study["id"], AddStudyCandidate(parameters={"width": 5}))
    b_candidate = add_candidate(
        service, sid, owner, study["id"], AddStudyCandidate(parameters={"width": 5})
    )
    assert a["id"] == b_candidate["id"]
    with pytest.raises(WorkspaceError, match="bounds"):
        add_candidate(service, sid, owner, study["id"], AddStudyCandidate(parameters={"width": 1}))
    with pytest.raises(WorkspaceError, match="outside"):
        add_candidate(
            service, sid, owner, study["id"], AddStudyCandidate(parameters={"invented": 5})
        )


def test_failure_missing_metric_and_unconverged_cannot_win():
    spec = study_spec()
    base = {
        "id": "base",
        "status": "complete",
        "eligible": True,
        "feasible": True,
        "metrics": {"geometry.volume_mm3": 6000},
    }
    for status, eligible, feasible in [
        ("failed", False, None),
        ("incomplete", False, True),
        ("complete", True, False),
    ]:
        bad = {
            "id": "bad",
            "status": status,
            "eligible": eligible,
            "feasible": feasible,
            "metrics": {"geometry.volume_mm3": 1},
        }
        assert compare_candidates(spec, [base, bad], "base")["best_candidate_id"] == "base"
    better = {
        "id": "better",
        "status": "complete",
        "eligible": True,
        "feasible": True,
        "metrics": {"geometry.volume_mm3": 5999},
        "metric_uncertainty": {"geometry.volume_mm3": 5},
    }
    assert compare_candidates(spec, [base, better], "base")["conclusion"] == "inconclusive"


def test_corrupt_receipt_and_artifact_rejected(studies):
    service, owner, sid, _, _ = studies
    study = claim_and_run(studies, create(studies))
    candidate = study["candidates"][0]
    receipt = candidate["evaluations"][0]
    path = (
        candidate_directory(service, sid, study["id"], candidate["id"])
        / receipt["attempt"]
        / "geometry"
        / "measurement.json"
    )
    path.write_text("{}")
    with pytest.raises(WorkspaceError, match="artifact changed"):
        read_study(service, sid, owner, study["id"])
    with service.store.connect() as con:
        con.execute("UPDATE design_studies SET document='{}' WHERE id=?", (study["id"],))
    with pytest.raises(WorkspaceError, match="integrity"):
        read_study(service, sid, owner, study["id"])


def test_owner_boundary_and_input_change(studies):
    service, owner, sid, _, _ = studies
    study = create(studies)
    with pytest.raises(WorkspaceError):
        read_study(service, sid, "another-owner", study["id"])
    from forge_mind.workspace.engineering_models import EngineeringInputs

    service.store.save_engineering(sid, owner, 0, EngineeringInputs().model_dump())
    assert read_study(service, sid, owner, study["id"])["current"] is False
    with pytest.raises(WorkspaceError, match="changed"):
        enqueue_study(service, sid, owner, study["id"])


def test_cancel_resume_retains_completed_evaluation(studies):
    service, owner, sid, _, _ = studies
    study = claim_and_run(studies, create(studies))
    baseline = study["candidates"][0]
    candidate = add_candidate(
        service, sid, owner, study["id"], AddStudyCandidate(parameters={"width": 6})
    )
    cancelled = claim_and_run(studies, study, cancelled=lambda: True)
    assert cancelled["status"] == "cancelled"
    assert cancelled["candidates"][0]["evaluations"] == baseline["evaluations"]
    # Mark an interrupted candidate explicitly; completed evaluations remain immutable on resume.
    candidate["status"] = "evaluating"
    _save_candidate(service, candidate)
    result = claim_and_run(studies, study)
    assert result["candidates"][0]["evaluations"] == baseline["evaluations"]
    assert result["candidates"][1]["status"] == "complete", result["candidates"][1].get("error")
    assert result["comparison"]["conclusion"] == "improved"


def test_multivariable_search_with_real_rebuild_and_failing_constraint(studies):
    service, owner, sid, _, _ = studies
    study = claim_and_run(
        studies, create(studies), search=SearchDesignStudy(rounds=2, step_fraction=0.5)
    )
    assert len(study["candidates"]) >= 5
    assert any(c["feasible"] is False for c in study["candidates"])
    best = next(
        c for c in study["candidates"] if c["id"] == study["comparison"]["best_candidate_id"]
    )
    assert 3000 <= best["metrics"]["geometry.volume_mm3"] < 6400
    assert best["metrics"]["geometry.extent_x_mm"] == pytest.approx(100)
    assert study["comparison"]["conclusion"] == "improved"
    assert service.store.session_record(sid, owner)["selected_revision"] == "a" * 32


def test_api_discovery_creation_and_stale_proposal(studies):
    service, _owner, sid, rid, _ = studies
    client = TestClient(create_app(local=True, service=service))
    schema = client.get("/workspace/design-study-schema")
    assert schema.status_code == 200
    assert {e["kind"] for e in schema.json()["evaluators"]} == {
        "geometry",
        "structure",
        "flow",
        "interface_geometry",
    }
    response = client.post(
        f"/workspace/sessions/{sid}/design-studies",
        json={"revision_id": rid, "input_version": 0, "spec": study_spec().model_dump()},
    )
    assert response.status_code == 201, response.text
    study = claim_and_run(studies, response.json())
    response = client.post(
        f"/workspace/sessions/{sid}/design-studies/{study['id']}/accept",
        json={"candidate_id": study["candidates"][0]["id"], "expected_current_revision": "f" * 32},
    )
    assert response.status_code == 409


def test_large_scoped_api_pages_are_revision_bound(studies):
    service, _, sid, rid, _ = studies
    client = TestClient(create_app(local=True, service=service))
    path = f"/workspace/sessions/{sid}/design/{rid}/page"
    first = client.get(path, params={"collection": "variables", "limit": 1}).json()
    second = client.get(
        path, params={"collection": "variables", "limit": 1, "cursor": first["next_cursor"]}
    ).json()
    assert first["items"][0]["name"] != second["items"][0]["name"]
    assert second["next_cursor"] is None
    rejected = client.get(path, params={"collection": "parts", "cursor": first["next_cursor"]})
    assert rejected.status_code == 409
