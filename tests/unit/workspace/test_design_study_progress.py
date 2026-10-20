from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from forge_mind.workspace import design_study_progress as progress
from forge_mind.workspace.design_study_api import design_study_router
from forge_mind.workspace.design_study_service import _tables
from forge_mind.workspace.document import digest
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def records(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("owner", "study", None)["id"]
    rid, study_id, candidate_id = "a" * 32, "b" * 32, "c" * 32
    service = SimpleNamespace(
        store=store,
        _load_revision=lambda *_: pytest.fail("Expensive revision verification during progress"),
    )
    _tables(service)
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    (directory / "part.step").write_bytes(b"step")
    store.add_revision(
        sid,
        {"id": rid, "document": {}, "measurement": {"artifact_hashes": {"part.step": "expected"}}},
    )
    store.update(sid, selected_revision=rid)
    study = {
        "id": study_id,
        "baseline_revision_id": rid,
        "spec": {"evaluators": [{"kind": "geometry"}]},
        "binding": {},
    }
    candidate = {
        "id": candidate_id,
        "revision_id": rid,
        "evaluations": [
            {"attempt": "attempt1", "kind": "geometry", "hashes": {"result.json": "expected"}}
        ],
    }
    artifact = store.root / "studies" / candidate_id / "attempt1" / "geometry" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"measurement")
    with store.connect() as con:
        con.execute(
            "INSERT INTO design_studies VALUES (?,?,?,?)",
            (study_id, sid, json.dumps(study), digest(study)),
        )
        con.execute(
            "INSERT INTO design_candidates VALUES (?,?,?,?,?)",
            (candidate_id, study_id, "key", json.dumps(candidate), digest(candidate)),
        )
    monkeypatch.setattr(progress, "engine_fingerprint", lambda: "engine")
    monkeypatch.setattr(
        "forge_mind.workspace.design_study_service.file_sha256",
        lambda *_: pytest.fail("Expensive evidence hash during progress"),
    )
    return service, sid, study_id, artifact


def test_unchanged_progress_reads_no_artifact_bytes_and_makes_no_evidence_claim(
    records, monkeypatch
):
    service, sid, study_id, _ = records
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Artifact byte read"))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("Artifact text read"))
    first = progress.study_progress(service, sid, "owner", study_id)
    assert progress.study_progress(service, sid, "owner", study_id) == first
    assert len(first["token"]) == 64 and first["status"] == "draft"
    assert "current" not in first and "eligible" not in first and "metrics" not in first
    assert "does not verify" in first["scope"]


@pytest.mark.parametrize("change", ["mtime", "size", "delete", "hardlink", "directory"])
def test_artifact_identity_changes_invalidate_poll_token(records, change):
    service, sid, study_id, artifact = records
    before = progress.study_progress(service, sid, "owner", study_id)["token"]
    if change == "mtime":
        info = artifact.stat()
        os.utime(artifact, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    elif change == "size":
        with artifact.open("r+b") as writer:
            writer.truncate(6)
    elif change == "hardlink":
        os.link(artifact, artifact.with_suffix(".linked"))
    else:
        artifact.unlink()
        if change == "directory":
            artifact.mkdir()
    assert progress.study_progress(service, sid, "owner", study_id)["token"] != before


@pytest.mark.parametrize(
    "change",
    ["baseline_record", "baseline_artifact", "input_content", "input_version", "source", "engine"],
)
def test_baseline_and_freshness_changes_invalidate_poll_token(records, monkeypatch, change):
    service, sid, study_id, _ = records
    before = progress.study_progress(service, sid, "owner", study_id)["token"]
    if change == "baseline_record":
        with service.store.connect() as con:
            con.execute(
                "UPDATE revisions SET document=? WHERE session=?",
                (json.dumps({"id": "a" * 32, "document": {"changed": True}}), sid),
            )
    elif change == "baseline_artifact":
        (service.store.directory(sid, "a" * 32) / "part.step").unlink()
    elif change.startswith("input_"):
        original = service.store.engineering(sid)
        if change == "input_content":
            original["inputs"] = {"changed": True}
        else:
            original["version"] += 1
        monkeypatch.setattr(service.store, "engineering", lambda _: original)
    elif change == "source":
        monkeypatch.setattr(progress, "source_fingerprint", lambda *_: "changed")
    else:
        monkeypatch.setattr(progress, "engine_fingerprint", lambda: "changed")
    assert progress.study_progress(service, sid, "owner", study_id)["token"] != before


@pytest.mark.parametrize("table", ["design_studies", "design_candidates", "design_study_runs"])
def test_untrusted_record_digests_are_verified(records, table):
    service, sid, study_id, _ = records
    with service.store.connect() as con:
        if table == "design_study_runs":
            con.execute(
                "INSERT INTO design_study_runs VALUES (?,?,?,?)",
                ("run", study_id, '{"state":"running","job_id":"missing"}', "wrong"),
            )
        else:
            con.execute(f"UPDATE {table} SET sha256='wrong'")
    with pytest.raises(WorkspaceError, match="integrity"):
        progress.study_progress(service, sid, "owner", study_id)


def test_progress_route_is_owner_scoped(records):
    service, sid, study_id, _ = records
    app = FastAPI()

    def owner(authorization: str = Header()) -> str:
        return authorization

    app.include_router(design_study_router(service, owner))
    client = TestClient(app)
    path = f"/workspace/sessions/{sid}/design-studies/{study_id}/progress"
    denied = client.get(path, headers={"Authorization": "other"})
    assert denied.status_code == 409 and "workspace not found" in denied.text
    result = client.get(path, headers={"Authorization": "owner"})
    assert result.status_code == 200 and result.json()["id"] == study_id


def diagnostic_path(records):
    service, sid, study_id, _ = records
    root = progress.candidate_directory(service, sid, study_id, "c" * 32)
    path = root / "flow-cache" / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return root, path


def test_flow_iteration_updates_do_not_change_evidence_token(records):
    root, path = diagnostic_path(records)
    stage = root / "flow-cache" / "attempt" / "solve"
    stage.mkdir(parents=True)
    history = stage / "history.csv"
    history.write_text('"Inner_Iter", "rms[P]"\n1,-3\n2,-4\n')
    snapshot = {
        "stage": "mesh-2-domain-1",
        "phase": "solving",
        "mesh_level": 2,
        "domain_level": 1,
        "completed_stages": 3,
        "reused_stages": 2,
        "elapsed_seconds": 12.5,
        "stage_directory": "flow-cache/attempt/solve",
        "eligible": True,
        "sensitivity": {"drag_n": {"relative_change": 0.046}, "secret": "ignore"},
    }
    path.write_text(json.dumps(snapshot))
    service, sid, study_id, _ = records
    first = progress.study_progress(service, sid, "owner", study_id)
    history.write_text('"Inner_Iter", "rms[P]"\n1,-3\n2,-4\n3,-5\n4,')
    path.write_text(json.dumps({**snapshot, "elapsed_seconds": 15.5}))
    second = progress.study_progress(service, sid, "owner", study_id)
    assert first["token"] == second["token"]
    assert first["flow_progress"][0]["iteration"] == 2
    result = second["flow_progress"][0]
    assert result["iteration"] == 3 and result["elapsed_seconds"] == 15.5
    assert result["completed_stages"] == 3 and result["reused_stages"] == 2
    assert result["sensitivity"] == {"drag_n": {"relative_change": 0.046}}
    assert "eligible" not in result and "stage_directory" not in result
    assert str(root) not in json.dumps(result)


def test_snapshot_age_changes_without_rehashing_or_advancing_reported_runtime(records, monkeypatch):
    _, path = diagnostic_path(records)
    path.write_text(json.dumps({"phase": "solving", "elapsed_seconds": 12.5}))
    os.utime(path, (95, 95))
    now = [100.0]
    monkeypatch.setattr(progress.time, "time", lambda: now[0])
    service, sid, study_id, _ = records
    first = progress.study_progress(service, sid, "owner", study_id)
    now[0] = 120
    delayed = progress.study_progress(service, sid, "owner", study_id)
    assert first["flow_progress"][0]["snapshot_age_seconds"] == 5
    assert delayed["flow_progress"][0]["snapshot_age_seconds"] == 25
    assert delayed["flow_progress"][0]["elapsed_seconds"] == 12.5
    assert delayed["token"] == first["token"]
    os.utime(path, (119, 119))
    fresh = progress.study_progress(service, sid, "owner", study_id)
    assert fresh["flow_progress"][0]["snapshot_age_seconds"] == 1
    assert fresh["token"] == first["token"]


@pytest.mark.parametrize("invalid", ["malformed", "oversized", "hardlink", "directory"])
def test_invalid_diagnostic_snapshot_is_unavailable_without_affecting_evidence(records, invalid):
    _, path = diagnostic_path(records)
    service, sid, study_id, _ = records
    before = progress.study_progress(service, sid, "owner", study_id)
    if invalid == "directory":
        path.mkdir()
    else:
        path.write_text(
            "{"
            if invalid == "malformed"
            else " " * 16385
            if invalid == "oversized"
            else '{"phase":"solving"}'
        )
        if invalid == "hardlink":
            os.link(path, path.with_suffix(".linked"))
    after = progress.study_progress(service, sid, "owner", study_id)
    assert after["token"] == before["token"]
    assert after["flow_progress"] == []


def test_diagnostics_cannot_read_history_outside_candidate_or_expose_paths(records, monkeypatch):
    _, path = diagnostic_path(records)
    outside = path.parent.parent.parent / "outside"
    outside.mkdir()
    (outside / "history.csv").write_text('"Inner_Iter"\n999\n')
    path.write_text(
        json.dumps(
            {
                "phase": "solving",
                "stage_directory": str(outside),
                "stage": str(outside),
                "stop_reason": str(outside),
            }
        )
    )
    original = Path.open

    def guarded(file, *args, **kwargs):
        assert file != outside / "history.csv"
        return original(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    service, sid, study_id, _ = records
    result = progress.study_progress(service, sid, "owner", study_id)["flow_progress"][0]
    assert "iteration" not in result and "stage" not in result and "stop_reason" not in result


def test_solver_history_read_is_bounded_to_header_and_tail(records, monkeypatch):
    root, path = diagnostic_path(records)
    stage = root / "flow-cache" / "solve"
    stage.mkdir()
    history = stage / "history.csv"
    history.write_text('"Inner_Iter", "rms[P]"\n' + "1,-3\n" * 20000 + "25000,-6\n")
    path.write_text(json.dumps({"phase": "solving", "stage_directory": "flow-cache/solve"}))
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("Unbounded text read"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **kw: pytest.fail("Unbounded byte read"))
    service, sid, study_id, _ = records
    assert (
        progress.study_progress(service, sid, "owner", study_id)["flow_progress"][0]["iteration"]
        == 25000
    )
