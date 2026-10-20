from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.inspection import variable_impact
from forge_mind.workspace.project_access import (
    assembly_structure,
    design_relationships,
    project_overview,
)
from forge_mind.workspace.store import Store, WorkspaceError


def assembly() -> dict:
    return {
        "variables": {"width": 20, "gap": 5},
        "relations": [
            {"name": "pitch", "operation": "add", "a": "width", "b": "gap"},
            {"name": "offset", "operation": "multiply", "a": "pitch", "b": 2},
        ],
        "assembly": {
            "parts": [
                {
                    "id": key,
                    "name": key,
                    "features": [
                        {
                            "id": "block",
                            "name": "block",
                            "kind": "box",
                            "params": {"x": 20, "y": 10, "z": 5},
                            "bindings": {"x": "width"},
                        }
                    ],
                    "output": "block",
                    "production": {"role": "make", "process": "cnc"},
                    "ports": [{"name": "axis", "kind": "bore", "diameter": 3}],
                }
                for key in ("a", "b")
            ],
            "roots": [
                {
                    "id": f"instance_{i}",
                    "definition": "a" if i % 2 else "b",
                    "position": ["offset", 0, 0],
                }
                for i in range(57)
            ],
        },
        "motion_rig": {
            "drivers": [
                {"id": "travel", "name": "Travel", "unit": "mm", "minimum": 0, "maximum": 20}
            ],
            "groups": [
                {
                    "id": "slide",
                    "name": "Slide",
                    "driver": "travel",
                    "kind": "slide",
                    "pivot": ["pitch", 0, 0],
                    "occurrences": ["instance_0"],
                }
            ],
        },
    }


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store = Store(tmp_path)
    service = WorkspaceService(
        store,
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Unexpected provider call"),
    )
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "_load_revision", store.revision)
    client = TestClient(create_app(local=True, service=service))
    sid = store.create("local-owner", "Complex fixture", {})["id"]
    document = DesignDocument.model_validate(assembly()).model_dump(mode="json")
    rid = "a" * 32
    store.add_revision(sid, {"id": rid, "document": document, "measurement": {}})
    store.update(sid, selected_revision=rid)
    return client, store, sid, rid, document


def test_compact_polling_stays_small_as_history_grows(fixture, monkeypatch):
    client, store, sid, rid, _ = fixture
    for i in range(40):
        store.add_revision(
            sid, {"id": f"{i:032x}", "code": "x" * 40000, "document": assembly(), "measurement": {}}
        )
    monkeypatch.setattr(store, "revisions", lambda *_: pytest.fail("Expanded history during poll"))
    result = client.get(f"/workspace/sessions/{sid}/overview")
    assert result.status_code == 200
    assert len(result.content) < 16384
    assert result.json()["revision_count"] == 41
    assert result.json()["selected_revision"] == rid
    assert "features" not in result.text and "code" not in result.json()
    page = client.get(f"/workspace/sessions/{sid}/history?limit=3").json()
    assert len(page["items"]) == 3 and page["next_offset"] == 3
    assert "code" not in json.dumps(page)


def test_compact_polling_exposes_unavailable_operation_recovery(fixture, monkeypatch):
    client, store, sid, rid, _ = fixture
    descriptor = {
        "job_id": "b" * 32,
        "operation": "legacy_operation",
        "revision_id": rid,
        "available": False,
        "reason": "Inspect saved evidence; this operation cannot be rerun.",
        "action_label": "",
        "scope": "No authoring or acceptance changes.",
    }
    original = store.session_record
    monkeypatch.setattr(
        store, "session_record", lambda *args: {**original(*args), "operation_recovery": descriptor}
    )
    monkeypatch.setattr(store, "revisions", lambda *_: pytest.fail("Expanded history during poll"))
    result = client.get(f"/workspace/sessions/{sid}/overview")
    assert result.status_code == 200
    assert result.json()["operation_recovery"] == descriptor


def test_focused_read_and_edit_preserve_unrelated_structure(fixture):
    client, store, sid, rid, document = fixture
    read = client.get(f"/workspace/sessions/{sid}/design/{rid}/parts/a").json()
    assert read["part"]["id"] == "a" and "assembly" not in read
    part = read["part"]
    part["features"][0]["params"]["z"] = 8
    part.pop("production")
    result = client.post(
        f"/workspace/sessions/{sid}/assembly-edit",
        json={
            "revision_id": rid,
            "input_version": 0,
            "edit": {"parts": [part]},
        },
    )
    assert result.status_code == 202, result.text
    job = store.claim()
    merged = job["document"]
    assert merged["assembly"]["parts"][0]["features"][0]["params"]["z"] == 8
    assert merged["assembly"]["parts"][1] == document["assembly"]["parts"][1]
    assert (
        merged["assembly"]["parts"][0]["production"]
        == document["assembly"]["parts"][0]["production"]
    )
    for field in ("roots", "mates", "subassemblies"):
        assert merged["assembly"][field] == document["assembly"][field]
    assert merged["motion_rig"] == document["motion_rig"]
    assert job["expected_current_revision"] == rid and job["preview_only"]


def test_stale_design_or_engineering_version_cannot_queue(fixture):
    client, store, sid, rid, _ = fixture
    body = {"revision_id": rid, "input_version": 1, "edit": {"variables": {"gap": 8}}}
    assert client.post(f"/workspace/sessions/{sid}/assembly-edit", json=body).status_code == 409
    store.update(sid, selected_revision="b" * 32)
    body["input_version"] = 0
    assert client.post(f"/workspace/sessions/{sid}/assembly-edit", json=body).status_code == 409
    assert store.claim() is None


def test_identity_and_owner_failures_preserve_project(fixture):
    client, store, sid, rid, document = fixture
    part = document["assembly"]["parts"][0]
    part["ports"] = []
    result = client.post(
        f"/workspace/sessions/{sid}/assembly-edit",
        json={
            "revision_id": rid,
            "input_version": 0,
            "edit": {"parts": [part]},
        },
    )
    assert result.status_code == 409
    assert store.claim() is None
    with pytest.raises(WorkspaceError):
        project_overview(store, sid, "someone-else")
    other = store.create("someone-else", "private", None)["id"]
    assert client.get(f"/workspace/sessions/{other}/overview").status_code == 409
    assert client.get(f"/workspace/sessions/{other}/history").status_code == 409
    assert client.get(f"/workspace/sessions/{other}/design/{rid}/parts/a").status_code == 409


def test_candidate_is_editable_without_accepting_and_old_baseline_is_stale(fixture):
    client, store, sid, rid, document = fixture
    candidate = "b" * 32
    store.add_revision(sid, {"id": candidate, "document": document, "measurement": {}})
    store.update(
        sid,
        accepted_revision=rid,
        selected_revision=rid,
        proposal={"revision_id": candidate, "input_version": 0},
    )
    body = {"revision_id": rid, "input_version": 0, "edit": {"variables": {"gap": 8}}}
    assert client.post(f"/workspace/sessions/{sid}/assembly-edit", json=body).status_code == 409
    assert store.claim() is None
    assert client.get(f"/workspace/sessions/{sid}/overview").json()["design_revision"] == candidate
    body["revision_id"] = candidate
    assert client.post(f"/workspace/sessions/{sid}/assembly-edit", json=body).status_code == 202
    job = store.claim()
    assert job["revision_id"] == candidate and job["based_on"] == rid


def test_ai_edit_guards_reject_stale_source_without_consuming_a_job(fixture):
    client, store, sid, rid, _ = fixture
    body = {
        "instruction": "Widen the gap",
        "revision_id": rid,
        "expected_current_revision": "b" * 32,
        "expected_input_version": 0,
    }
    assert client.post(f"/workspace/sessions/{sid}/edit", json=body).status_code == 409
    assert store.claim() is None
    body["expected_current_revision"] = rid
    body["expected_input_version"] = 3
    assert client.post(f"/workspace/sessions/{sid}/edit", json=body).status_code == 409
    assert store.claim() is None
    body["expected_input_version"] = 0
    assert client.post(f"/workspace/sessions/{sid}/edit", json=body).status_code == 202
    assert store.claim()["revision_id"] == rid


def test_relationship_graph_retains_transitive_dimensional_and_motion_impact():
    result = design_relationships(DesignDocument.model_validate(assembly()))
    assert result["relations"][1]["value"] == 50
    assert result["relations"][1]["depends_on"] == ["pitch"]
    assert result["graph_semantics"] == "direct_v2"
    assert variable_impact(result, ["width"])["parts"] == ["a", "b"]
    assert len(variable_impact(result, ["gap"])["placements"]) == 57
    assert variable_impact(result, ["gap"])["motion"] == ["slide"]
    assert variable_impact(result, ["gap"])["parts"] == []


def test_flat_features_report_transitive_impact():
    document = DesignDocument.model_validate(
        {
            "variables": {"width": 20},
            "features": [
                {
                    "id": "plate",
                    "name": "plate",
                    "kind": "box",
                    "params": {"x": 20, "y": 10, "z": 3},
                    "bindings": {"x": "width"},
                },
                {
                    "id": "moved",
                    "name": "moved",
                    "kind": "transform",
                    "inputs": ["plate"],
                    "params": {"x": 3},
                },
            ],
            "outputs": ["moved"],
        }
    )
    impact = variable_impact(design_relationships(document), ["width"])
    assert impact["features"] == ["moved", "plate"]


def test_structure_retains_literal_and_nested_placements_without_feature_bodies():
    raw = assembly()
    raw["assembly"]["subassemblies"] = [
        {
            "id": "nested",
            "name": "Nested pair",
            "children": [
                {
                    "id": "literal",
                    "definition": "a",
                    "position": [12, 3, 4],
                    "rotation": [0, 0, 90],
                },
                {"id": "derived", "definition": "b", "position": ["offset", 0, 0]},
            ],
        }
    ]
    raw["assembly"]["roots"].append(
        {"id": "nested_root", "definition": "nested", "position": [0, 0, 30]}
    )
    document = DesignDocument.model_validate(raw)
    result = assembly_structure(document)
    assert result["roots"] == [r.model_dump() for r in document.assembly.roots]
    assert result["subassemblies"] == [s.model_dump() for s in document.assembly.subassemblies]
    assert result["motion_rig"] == document.motion_rig.model_dump()
    assert result["values"]["offset"] == 50
    assert result["variables"] == document.variables
    assert result["relations"] == [r.model_dump() for r in document.relations]
    assert all("features" not in p and p["feature_count"] == 1 for p in result["parts"])
    assert '"params"' not in json.dumps(result)
    assert len(json.dumps(result)) < 20000


def test_structure_route_uses_verified_revision_and_owner_boundary(fixture, monkeypatch):
    client, store, sid, rid, document = fixture
    response = client.get(f"/workspace/sessions/{sid}/design/{rid}/structure")
    assert response.status_code == 200
    assert response.json()["revision_id"] == rid
    assert response.json()["roots"] == document["assembly"]["roots"]
    other = store.create("someone-else", "private", None)["id"]
    assert client.get(f"/workspace/sessions/{other}/design/{rid}/structure").status_code == 409


def test_structure_does_not_bypass_artifact_verification(tmp_path, monkeypatch):
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    sid = store.create("local-owner", "Bound structure", {})["id"]
    seen = []

    def fail_verification(session_id, revision_id):
        seen.append((session_id, revision_id))
        raise WorkspaceError("Artifact content changed; evidence is stale")

    monkeypatch.setattr(service, "_load_revision", fail_verification)
    client = TestClient(create_app(local=True, service=service))
    response = client.get(f"/workspace/sessions/{sid}/design/{'a' * 32}/structure")
    assert response.status_code == 409
    assert "evidence is stale" in response.text
    assert seen == [(sid, "a" * 32)]


@pytest.mark.parametrize("compact", [False, True])
def test_acceptance_can_return_compact_state_without_expanding_history(
    tmp_path, monkeypatch, compact
):
    from forge_mind.workspace.measure import engine_fingerprint

    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "_load_revision", store.revision)
    sid = store.create("local-owner", "Large history", {})["id"]
    rid = "a" * 32
    store.add_revision(
        sid, {"id": rid, "measurement": {"engine_fingerprint": engine_fingerprint()}}
    )
    for index in range(40):
        store.add_revision(sid, {"id": f"{index:032x}", "code": "x" * 40000, "measurement": {}})
    store.update(
        sid,
        proposal={
            "revision_id": rid,
            "input_version": 0,
            "based_on": None,
            "eligible": True,
            "engine_fingerprint": engine_fingerprint(),
        },
    )
    if compact:
        monkeypatch.setattr(
            store, "revisions", lambda *_: pytest.fail("Expanded history during compact acceptance")
        )
    client = TestClient(create_app(local=True, service=service))
    response = client.post(
        f"/workspace/sessions/{sid}/accept" + ("?compact=true" if compact else ""),
        json={"revision_id": rid, "input_version": 0, "expected_revision": None},
    )
    assert response.status_code == 200, response.text
    assert response.json()["accepted_revision"] == rid
    assert response.json()["selected_revision"] == rid
    if compact:
        assert len(response.content) < 16384
        assert "revisions" not in response.json()
        assert response.json()["candidate"] is None
    else:
        assert len(response.json()["revisions"]) == 41
        assert response.json()["proposal"] is None
