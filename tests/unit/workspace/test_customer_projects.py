from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store, WorkspaceError


def request(**changes: object) -> dict:
    return {"request_id": "a" * 32, "prompt": "A mounting plate", **changes}


def test_customer_creation_is_durable_and_exactly_once_under_concurrent_replay(tmp_path: Path):
    store = Store(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(lambda _: store.create_customer_project("alice", request()), range(8))
        )
    assert len({value["id"] for value in results}) == 1
    assert len({value["initial_job_id"] for value in results}) == 1
    restored = Store(tmp_path).customer_request("alice", "a" * 32)
    assert restored["id"] == results[0]["id"]
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        job = json.loads(con.execute("SELECT payload FROM jobs").fetchone()[0])
    assert restored["authoring_mode"] == "native_operations"
    assert restored["creation_request_id"] == "a" * 32
    assert restored["project_kind"] == "workspace"
    assert job["preview_only"] is False
    assert job["delivery_requested"] is True


def test_customer_request_identity_rejects_changed_payload_and_is_owner_scoped(tmp_path: Path):
    store = Store(tmp_path)
    first = store.create_customer_project("alice", request())
    with pytest.raises(WorkspaceError, match="different request"):
        store.create_customer_project("alice", request(prompt="Another plate"))
    with pytest.raises(WorkspaceError, match="not found"):
        store.customer_request("bob", "a" * 32)
    second = store.create_customer_project("bob", request())
    assert second["id"] != first["id"]


def test_full_queue_rolls_back_customer_identity_and_session(tmp_path: Path):
    store = Store(tmp_path)
    for index in range(16):
        sid = store.create("other", f"Plate {index}", None)["id"]
        store.enqueue(sid, "other", {})
    with pytest.raises(WorkspaceError, match="queue is full"):
        store.create_customer_project("alice", request())
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM sessions").fetchone()[0] == 16
        assert con.execute("SELECT count(*) FROM customer_requests").fetchone()[0] == 0


@pytest.fixture
def customer_api(tmp_path: Path, monkeypatch):
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    app = FastAPI()

    def owner(authorization: str = Header()) -> str:
        return authorization

    app.include_router(router(store, service, owner))
    return TestClient(app), store, service


def test_customer_api_recovers_lost_response_without_another_job(customer_api):
    client, store, _ = customer_api
    headers = {"Authorization": "alice"}
    first = client.post("/workspace/customer-projects", headers=headers, json=request())
    assert first.status_code == 202
    replay = client.post("/workspace/customer-projects", headers=headers, json=request())
    assert replay.status_code == 202
    recovered = client.get("/workspace/customer-requests/" + "a" * 32, headers=headers)
    assert recovered.json()["id"] == replay.json()["id"] == first.json()["id"]
    assert (
        client.get(
            "/workspace/customer-requests/" + "a" * 32, headers={"Authorization": "bob"}
        ).status_code
        == 404
    )
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_unqualified_docker_customer_creation_is_disabled_but_schema_is_public(customer_api):
    client, store, service = customer_api
    service.executor = Executor("docker")
    result = client.get("/workspace/schema")
    assert result.status_code == 200
    assert result.json()["customer_projects"]["enabled"] is False
    assert (
        client.post(
            "/workspace/customer-projects", headers={"Authorization": "alice"}, json=request()
        ).status_code
        == 503
    )
    assert store.list_sessions("alice") == []


def test_legacy_native_create_keeps_its_preview_policy(customer_api):
    client, store, _ = customer_api
    response = client.post(
        "/workspace/sessions",
        headers={"Authorization": "alice"},
        json={"prompt": "A plate", "authoring_mode": "native_operations", "preview_only": False},
    )
    assert response.status_code == 202
    with store.connect() as con:
        payload = json.loads(con.execute("SELECT payload FROM jobs").fetchone()[0])
    assert payload["preview_only"] is True
    assert "creation_request_id" not in response.json()


def test_new_customer_print_choice_is_durable_and_part_of_request_identity(tmp_path):
    store = Store(tmp_path)
    initial = store.create_customer_project("alice", request())
    assert initial["customer_request_version"] == 2
    assert initial["customer_printable"] is True
    assert store.create_customer_project("alice", request(printable=True))["id"] == initial["id"]
    with pytest.raises(WorkspaceError, match="different request"):
        store.create_customer_project("alice", request(printable=False))
    digital = store.create_customer_project("alice", request(request_id="b" * 32, printable=False))
    assert digital["customer_printable"] is False


def legacy_customer_identity(store):
    from forge_mind.workspace.customer_projects import CustomerProjectRequest

    initial = store.create_customer_project("alice", request())
    values = CustomerProjectRequest.model_validate(request()).model_dump(mode="json")
    values.pop("request_id")
    values.pop("printable")
    legacy_hash = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with store.connect() as con:
        document = json.loads(
            con.execute("SELECT document FROM sessions WHERE id=?", (initial["id"],)).fetchone()[0]
        )
        document.pop("customer_request_version")
        document.pop("customer_printable")
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?", (json.dumps(document), initial["id"])
        )
        con.execute(
            "UPDATE customer_requests SET payload_hash=? WHERE owner=? AND request_id=?",
            (legacy_hash, "alice", "a" * 32),
        )
    return initial["id"]


def test_legacy_customer_hash_replay_preserves_absent_print_policy(customer_api):
    client, store, _ = customer_api
    sid = legacy_customer_identity(store)
    response = client.post(
        "/workspace/customer-projects", headers={"Authorization": "alice"}, json=request()
    )
    assert response.status_code == 202 and response.json()["id"] == sid
    assert "customer_printable" not in response.json()
    assert "customer_request_version" not in response.json()
    for printable in (True, False):
        response = client.post(
            "/workspace/customer-projects",
            headers={"Authorization": "alice"},
            json=request(printable=printable),
        )
        assert response.status_code == 409
    with store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_version_two_identity_cannot_fall_back_to_legacy_hash(tmp_path):
    store = Store(tmp_path)
    sid = legacy_customer_identity(store)
    store.update(sid, customer_request_version=2, customer_printable=True)
    with pytest.raises(WorkspaceError, match="different request"):
        store.create_customer_project("alice", request())
