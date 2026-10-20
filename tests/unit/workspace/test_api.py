from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import EditRequest, WorkspaceRequest
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Store]:
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor())
    monkeypatch.setattr(service, "start", lambda: None)

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(router(store, service, owner))
    return TestClient(app), store


def test_other_owner_cannot_read_edit_cancel_or_fetch(api: tuple[TestClient, Store]) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]
    headers = {"Authorization": "bob"}
    assert client.get(f"/workspace/sessions/{sid}", headers=headers).status_code == 404
    assert (
        client.post(
            f"/workspace/sessions/{sid}/edit", json={"instruction": "change"}, headers=headers
        ).status_code
        == 404
    )
    assert client.post(f"/workspace/sessions/{sid}/cancel", headers=headers).status_code == 404
    assert (
        client.get(
            f"/workspace/sessions/{sid}/revisions/abc/part.step", headers=headers
        ).status_code
        == 404
    )


def test_create_queues_and_rejects_parallel_edit(api: tuple[TestClient, Store]) -> None:
    client, _ = api
    headers = {"Authorization": "alice"}
    r = client.post("/workspace/sessions", json={"prompt": "a plate"}, headers=headers)
    assert r.status_code == 202
    sid = r.json()["id"]
    assert r.json()["state"] == "queued"
    r = client.post(
        f"/workspace/sessions/{sid}/edit", json={"instruction": "change"}, headers=headers
    )
    assert r.status_code == 409


def test_capabilities_report_override_as_configuration(api, monkeypatch):
    client, _ = api
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-6-sol")
    value = client.get("/workspace/capabilities", headers={"Authorization": "alice"}).json()
    assert value["provider"] == "codex"
    assert value["model"] == "gpt-6-sol"
    assert value["model_identity"] == "configured_only"


def test_operation_retry_has_explicit_owner_scoped_admission(api, monkeypatch) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]
    calls = []

    def enqueue(session_id, owner, payload):
        calls.append((session_id, owner, payload))
        return "b" * 32

    monkeypatch.setattr(store, "enqueue", enqueue)
    url = f"/workspace/sessions/{sid}/retry-operation"
    assert (
        client.post(url, headers={"Authorization": "bob"}, json={"job_id": "a" * 32}).status_code
        == 404
    )
    assert calls == []
    result = client.post(url, headers={"Authorization": "alice"}, json={"job_id": "a" * 32})
    assert result.status_code == 202
    assert result.json() == {"id": sid, "job_id": "b" * 32}
    assert calls == [(sid, "alice", {"retry_operation": "a" * 32})]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"job_id": "bad"},
        {"job_id": "a" * 32, "resume": True},
        {"job_id": "a" * 32, "max_seconds": 60},
        {"job_id": "a" * 32, "operation": "author"},
    ],
)
def test_operation_retry_rejects_arbitrary_payload(api, monkeypatch, body) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]
    monkeypatch.setattr(store, "enqueue", lambda *_: pytest.fail("Invalid retry admitted"))
    result = client.post(
        f"/workspace/sessions/{sid}/retry-operation", headers={"Authorization": "alice"}, json=body
    )
    assert result.status_code == 422


def test_operation_retry_reports_unavailable_without_starting(api, monkeypatch) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]

    def enqueue(*_):
        raise WorkspaceError("Source inputs changed")

    monkeypatch.setattr(store, "enqueue", enqueue)
    result = client.post(
        f"/workspace/sessions/{sid}/retry-operation",
        headers={"Authorization": "alice"},
        json={"job_id": "a" * 32},
    )
    assert result.status_code == 409
    assert result.json()["detail"] == "Source inputs changed"


@pytest.mark.parametrize("operation", ["create", "edit", "resume"])
@pytest.mark.parametrize(
    "limits",
    [{}, {"max_actions": None, "max_seconds": None}, {"max_actions": 2}, {"max_seconds": 60}],
)
def test_authoring_limits_are_optional_and_forwarded(
    api: tuple[TestClient, Store],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    limits: dict[str, int | None],
) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]
    queued: list[dict] = []

    def enqueue(sid: str, owner: str, payload: dict) -> str:
        queued.append(payload)
        return "test-job"

    monkeypatch.setattr(store, "enqueue", enqueue)
    path = (
        "/workspace/sessions" if operation == "create" else f"/workspace/sessions/{sid}/{operation}"
    )
    body = (
        {"prompt": "plate"}
        if operation == "create"
        else {"instruction": "widen"}
        if operation == "edit"
        else {}
    )
    response = client.post(path, json={**body, **limits}, headers={"Authorization": "alice"})
    assert response.status_code == 202
    assert len(queued) == 1
    assert queued[0].get("max_actions") == limits.get("max_actions")
    assert queued[0].get("max_seconds") == limits.get("max_seconds")


@pytest.mark.parametrize("operation", ["create", "edit", "resume"])
@pytest.mark.parametrize(
    "limits", [{"max_actions": 0}, {"max_actions": -1}, {"max_seconds": 0}, {"max_seconds": -1}]
)
def test_explicit_authoring_limits_remain_validated(
    api: tuple[TestClient, Store], operation: str, limits: dict[str, int]
) -> None:
    client, store = api
    sid = store.create("alice", "plate", None)["id"]
    path = (
        "/workspace/sessions" if operation == "create" else f"/workspace/sessions/{sid}/{operation}"
    )
    body = (
        {"prompt": "plate"}
        if operation == "create"
        else {"instruction": "widen"}
        if operation == "edit"
        else {}
    )
    assert (
        client.post(path, json={**body, **limits}, headers={"Authorization": "alice"}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "limits", [{"max_actions": 10000, "max_seconds": 100000}, {"max_seconds": 1}]
)
def test_explicit_authoring_budget_schemas_accept_positive_values_without_hidden_ceilings(
    limits: dict[str, int],
) -> None:
    for request in (WorkspaceRequest(prompt="plate", **limits), EditRequest(**limits)):
        assert request.max_actions == limits.get("max_actions")
        assert request.max_seconds == limits.get("max_seconds")


def test_project_name_is_owned_metadata_only(api: tuple[TestClient, Store]) -> None:
    client, store = api
    original = store.create("alice", "Create a precise mounting plate", None)
    original = store.get(original["id"], "alice")
    path = f"/workspace/sessions/{original['id']}/name"
    assert (
        client.post(path, json={"title": "Stolen"}, headers={"Authorization": "bob"}).status_code
        == 404
    )
    result = client.post(
        path, json={"title": "  Motor mount  "}, headers={"Authorization": "alice"}
    )
    assert result.status_code == 200
    renamed = result.json()
    assert renamed["title"] == "Motor mount"
    assert {key: value for key, value in renamed.items() if key not in {"title", "updated_at"}} == {
        key: value for key, value in original.items() if key != "updated_at"
    }
    assert Store(store.root).get(original["id"], "alice")["title"] == "Motor mount"
    assert (
        client.get("/workspace/sessions", headers={"Authorization": "alice"}).json()[0]["title"]
        == "Motor mount"
    )


@pytest.mark.parametrize("name", ["", "   ", "a" * 81, None, 123])
def test_project_name_rejects_invalid_values(api: tuple[TestClient, Store], name: object) -> None:
    client, store = api
    original = store.create("alice", "plate", None)
    result = client.post(
        f"/workspace/sessions/{original['id']}/name",
        json={"title": name},
        headers={"Authorization": "alice"},
    )
    assert result.status_code == 422
    assert "title" not in store.get(original["id"], "alice")


def test_no_silent_fallback_from_container_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("forge_mind.workspace.executor.shutil.which", lambda _: None)
    result = Executor().build("raise RuntimeError('must not run')", tmp_path, 1, lambda: False)
    assert not result.ok
    assert result.error_type == "isolation_unavailable"


def test_private_mode_rejects_foreign_origin_and_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_mind.server.workspace_app import create_app

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    app = create_app(local=True, service=service)
    client = TestClient(app)
    assert (
        client.get(
            "/workspace/sessions", headers={"Origin": "https://attacker.example"}
        ).status_code
        == 403
    )
    assert (
        client.get("/workspace/sessions", headers={"Host": "attacker.example"}).status_code == 403
    )


def test_public_mode_requires_auth(api: tuple[TestClient, Store]) -> None:
    from forge_mind.server.workspace_app import create_app

    _, store = api
    service = WorkspaceService(store, Executor())
    client = TestClient(create_app(local=False, service=service))
    assert client.get("/workspace/sessions").status_code == 401
