"""Real motion admission refuses known-ineligible jobs before queue allocation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from forge_mind.workspace import motion_service, operation_recovery
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.engineering_api import engineering_router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.store import Store, WorkspaceError

RID = "a" * 32
ENGINE = "b" * 64


def fixture_case(tmp_path, monkeypatch, *, mode="trusted_local", kind="rig", valid=True):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Saved motion admission fixture", contract.model_dump())["id"]
    service = WorkspaceService(store, Executor(mode))
    started, admitted, loaded = [], [], []
    monkeypatch.setattr(service, "start", lambda: started.append("start"))
    monkeypatch.setattr(operation_recovery, "engine_fingerprint", lambda: ENGINE)
    monkeypatch.setattr(motion_service, "engine_fingerprint", lambda: ENGINE)
    directory = store.directory(sid, RID)
    directory.mkdir(parents=True)
    artifacts = {
        "program.py": b"# Saved fixture, never execute\n",
        "part.step": b"Saved bytes, no actual geometry proof",
        "part.stl": b"Saved bytes, never render",
    }
    for name, content in artifacts.items():
        (directory / name).write_bytes(content)
    facts = {}
    if kind == "rig":
        facts["motion_rig"] = {
            "engine_fingerprint": ENGINE,
            "revision_id": RID,
            "drivers": [{"id": "move", "minimum": 0, "maximum": 1}],
            "scope": "Synthetic saved metadata, no kinematic proof",
        }
    elif kind == "mechanism":
        facts["mechanism"] = {"status": "valid", "scope": "A mechanism is not motion_rig"}
    measurement = Measurement(
        revision_id=RID,
        contract_hash=contract.fingerprint(),
        engine_fingerprint=ENGINE,
        geometry_valid=valid,
        requirements_satisfied=False,
        facts=facts,
        artifact_hashes={name: hashlib.sha256(raw).hexdigest() for name, raw in artifacts.items()},
    )
    revision = {
        "id": RID,
        "parent_id": None,
        "code": artifacts["program.py"].decode(),
        "contract": contract.model_dump(),
        "measurement": measurement.model_dump(),
    }
    store.add_revision(sid, revision)
    store.update(sid, selected_revision=RID)
    original_enqueue = store.enqueue

    def enqueue(*args, **kwargs):
        admitted.append((args, kwargs))
        return original_enqueue(*args, **kwargs)

    monkeypatch.setattr(store, "enqueue", enqueue)
    original_load = service._load_revision

    def load(*args, **kwargs):
        loaded.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(service, "_load_revision", load)

    def forbidden(*args, **kwargs):
        pytest.fail("Admission tests must never start a native worker or model")

    monkeypatch.setattr(motion_service, "process_revision", forbidden)
    app = FastAPI()

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request: Request, exc: WorkspaceError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def owner(authorization: str = Header()):
        return authorization

    app.include_router(engineering_router(store, service, owner))
    return SimpleNamespace(
        store=store,
        service=service,
        sid=sid,
        client=TestClient(app),
        directory=directory,
        revision=revision,
        started=started,
        admitted=admitted,
        loaded=loaded,
        before=deepcopy(store.session_record(sid, "owner")),
    )


def request(case, *, owner="owner", revision_id=RID):
    return case.client.post(
        f"/workspace/sessions/{case.sid}/motion-check",
        headers={"Authorization": owner},
        json={"revision_id": revision_id},
    )


def assert_unallocated(case):
    assert case.admitted == []
    assert case.started == []
    with case.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert not (case.store.root / "motion-checks").exists()
    assert case.store.session_record(case.sid, "owner") == case.before
    assert case.store.revision(case.sid, RID) == case.revision


@pytest.mark.parametrize(
    "mode,kind",
    [
        ("docker", "rig"),
        ("docker", "missing"),
        ("trusted_local", "missing"),
        ("trusted_local", "mechanism"),
    ],
)
def test_known_ineligible_motion_requests_are_refused_without_allocations(
    tmp_path, monkeypatch, mode, kind
):
    case = fixture_case(tmp_path, monkeypatch, mode=mode, kind=kind)
    response = request(case)
    assert response.status_code == 409
    assert response.json() == {
        "detail": "Motion checking requires a private-local revision with a motion rig"
    }
    assert len(case.loaded) == 1
    assert_unallocated(case)


def test_trusted_saved_rig_keeps_normal_queue_semantics(tmp_path, monkeypatch):
    case = fixture_case(tmp_path, monkeypatch)
    response = request(case)
    assert response.status_code == 202
    body = response.json()
    assert body["id"] == case.sid and body["state"] == "queued"
    assert len(body["job_id"]) == 32
    assert len(case.admitted) == 1 and case.started == ["start"]
    with case.store.connect() as con:
        rows = con.execute("SELECT state,payload FROM jobs").fetchall()
    assert len(rows) == 1 and rows[0]["state"] == "queued"
    payload = json.loads(rows[0]["payload"])
    assert payload["engineering_operation"] == "motion_check"
    assert payload["revision_id"] == RID
    assert payload["settings"] == {
        "samples_per_driver": 7,
        "include_corners": True,
        "max_configurations": None,
    }
    assert payload["operation_binding"]["version"] == 1
    assert not (case.store.root / "motion-checks").exists()
    assert case.store.get(case.sid, "owner").get("accepted_revision") is None
    assert case.store.revision(case.sid, RID) == case.revision


def test_owner_refusal_precedes_revision_and_eligibility_reads(tmp_path, monkeypatch):
    case = fixture_case(tmp_path, monkeypatch, mode="docker", kind="missing")
    (case.directory / "part.step").write_bytes(b"tampered")
    response = request(case, owner="foreign")
    assert response.status_code == 404 and response.json() == {"detail": "Workspace not found"}
    assert case.loaded == []
    assert_unallocated(case)


@pytest.mark.parametrize("revision_id", [RID, "c" * 32])
def test_integrity_or_missing_revision_refusal_precedes_eligibility(
    tmp_path, monkeypatch, revision_id
):
    case = fixture_case(tmp_path, monkeypatch, mode="docker", kind="missing")
    if revision_id == RID:
        (case.directory / "part.step").write_bytes(b"tampered")
    response = request(case, revision_id=revision_id)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "private-local" not in detail
    assert (
        ("artifacts changed" in detail) if revision_id == RID else ("revision not found" in detail)
    )
    assert_unallocated(case)


def test_foreign_revision_is_not_an_eligible_local_source(tmp_path, monkeypatch):
    case = fixture_case(tmp_path, monkeypatch, mode="docker")
    foreign = case.store.create("other", "Foreign saved bytes", Contract().model_dump())["id"]
    other = {**case.revision, "id": "c" * 32}
    case.store.add_revision(foreign, other)
    response = request(case, revision_id="c" * 32)
    assert response.status_code == 409 and "revision not found" in response.json()["detail"]
    assert_unallocated(case)


def test_geometry_invalid_error_keeps_worker_precedence(tmp_path, monkeypatch):
    case = fixture_case(tmp_path, monkeypatch, mode="docker", kind="missing", valid=False)
    response = request(case)
    assert response.status_code == 409
    assert response.json()["detail"] == "Motion checking requires valid exported geometry"
    assert_unallocated(case)


@pytest.mark.parametrize(
    "mode,kind", [("docker", "rig"), ("trusted_local", "missing"), ("trusted_local", "mechanism")]
)
def test_worker_defense_remains_before_check_directory_or_native_dispatch(
    tmp_path, monkeypatch, mode, kind
):
    case = fixture_case(tmp_path, monkeypatch, mode=mode, kind=kind)
    job = {"session": case.sid, "owner": "owner", "revision_id": RID, "settings": {}}
    with pytest.raises(WorkspaceError, match="private-local revision with a motion rig"):
        motion_service.run_motion_job(case.service, job, lambda: False)
    assert_unallocated(case)
