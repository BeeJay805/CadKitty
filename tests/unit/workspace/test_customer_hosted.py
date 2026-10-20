from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from tests.unit.workspace.test_operation_plan import contract, document, plan
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.container_probe import path_observation
from forge_mind.workspace.customer_projects import customer_capabilities
from forge_mind.workspace.customer_routes import customer_delivery_router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.store import Store


def test_probe_distinguishes_denied_unknown_and_absent_paths(monkeypatch):
    target = Path("guarded")
    for exception, expected in (
        (PermissionError("denied"), "denied"),
        (FileNotFoundError("missing"), "absent"),
        (OSError("unknown failure"), "error"),
    ):

        def stat(self, *args, failure=exception, **kwargs):
            raise failure

        monkeypatch.setattr(Path, "stat", stat)
        assert path_observation(target)["state"] == expected


def test_docker_availability_requires_runtime_qualification_and_explicit_budgets(monkeypatch):
    executor = Executor("docker")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "3600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "80")
    assert customer_capabilities(executor)["enabled"] is False
    executor._typed_container_verified = True
    assert customer_capabilities(executor)["enabled"] is False
    executor._customer_container_verified = True
    assert customer_capabilities(executor)["enabled"] is True
    monkeypatch.delenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS")
    assert customer_capabilities(executor)["enabled"] is False


def test_customer_server_caps_bound_all_jobs_without_capping_total_to_child_limit(tmp_path):
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {"request_id": "a" * 32, "prompt": "A part", "max_seconds": 7200},
        hosted_limits={"max_seconds": 3600, "max_actions": 80},
    )
    job = store.claim()
    assert job["max_seconds"] == 3600
    assert job["max_actions"] == 80
    store.finish(job["id"], session["id"], "budget_exhausted")
    jid = store.enqueue(
        session["id"], "owner", {"resume": True, "max_seconds": 60, "max_actions": 1000}
    )
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
    assert payload["max_seconds"] == 60
    assert payload["max_actions"] == 80


def test_saved_observer_stages_owned_bytes_and_never_exposes_original_path(tmp_path, monkeypatch):
    executor = Executor("docker")
    executor._typed_container_verified = True
    monkeypatch.setattr("forge_mind.workspace.executor.shutil.which", lambda _: "docker")
    source = tmp_path / "original"
    source.mkdir()
    (source / "part.step").write_bytes(b"original saved STEP")
    output = tmp_path / "output"
    output.mkdir()
    observed = []

    def run(docker, directory, module, request, timeout, cancelled, *, image):
        payload = json.loads((directory / request).read_text())
        assert payload["source"] == "/work/source"
        assert str(source) not in (directory / request).read_text()
        assert (directory / "source" / "part.step").read_bytes() == b"original saved STEP"
        assert timeout <= 600
        observed.append(directory)
        (directory / "processed.json").write_text('{"status":"unverified"}')

    monkeypatch.setattr(executor, "_run_container", run)
    result = executor.process_observation(
        output,
        {"operation": "product_tool", "source": str(source), "request": {}},
        3600,
        lambda: False,
    )
    assert result["status"] == "unverified"
    assert (source / "part.step").read_bytes() == b"original saved STEP"
    assert not observed[0].exists()


def test_saved_observer_rejects_unknown_operation_and_changed_original(tmp_path, monkeypatch):
    executor = Executor("docker")
    with pytest.raises(ValueError, match="Unsupported"):
        executor.process_observation(tmp_path, {"operation": "engineering"}, 20, lambda: False)
    with pytest.raises(ValueError, match="qualified"):
        executor.process_observation(
            tmp_path, {"operation": "nominal_through_pattern"}, 20, lambda: False
        )
    executor._typed_container_verified = True
    monkeypatch.setattr("forge_mind.workspace.executor.shutil.which", lambda _: "docker")
    (tmp_path / "part.step").write_bytes(b"original")

    def run(docker, directory, *args, **kwargs):
        (tmp_path / "part.step").write_bytes(b"changed")
        (directory / "processed.json").write_text('{"status":"pass"}')

    monkeypatch.setattr(executor, "_run_container", run)
    with pytest.raises(ValueError, match="original input changed"):
        executor.process_observation(
            tmp_path, {"operation": "nominal_through_pattern"}, 20, lambda: False
        )


def test_hosted_delivery_routes_preserve_auth_ownership_and_exclude_advanced_writes(
    tmp_path, monkeypatch
):
    executor = Executor("docker")
    executor._typed_container_verified = executor._customer_container_verified = True
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "3600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "80")
    store = Store(tmp_path)
    sid = store.create("alice", "A project", None)["id"]
    service = WorkspaceService(store, executor)

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(customer_delivery_router(store, service, owner))
    client = TestClient(app)
    assert client.get(f"/workspace/sessions/{sid}/overview").status_code == 422
    assert (
        client.get(
            f"/workspace/sessions/{sid}/overview", headers={"Authorization": "alice"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/workspace/sessions/{sid}/overview", headers={"Authorization": "bob"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/workspace/sessions/{sid}/assembly-edit", headers={"Authorization": "alice"}, json={}
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/workspace/sessions/{sid}/guidance", headers={"Authorization": "alice"}, json={}
        ).status_code
        == 405
    )


def test_hosted_topology_is_owned_qualified_and_reads_current_registered_bytes(
    tmp_path, monkeypatch
):
    executor = Executor("docker")
    executor._typed_container_verified = executor._customer_container_verified = True
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "3600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "80")
    store = Store(tmp_path)
    sid = store.create("alice", "Width 20 mm", contract().model_dump())["id"]
    service = WorkspaceService(store, executor)
    rid = "b" * 32
    revision = saved_revision(
        store, sid, rid, document(), contract(), snapshot(plan(), 1, contract())
    )
    path = store.directory(sid, rid) / "topology.json"
    path.write_text('{"bodies":[],"scope":"synthetic viewer lifecycle fixture"}', encoding="utf-8")
    revision["measurement"]["artifact_hashes"][path.name] = file_sha256(path)
    with store.connect() as con:
        con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(customer_delivery_router(store, service, owner))
    client = TestClient(app)
    url = f"/workspace/sessions/{sid}/topology/{rid}"
    assert client.get(url, headers={"Authorization": "alice"}).json()["bodies"] == []
    assert client.get(url, headers={"Authorization": "bob"}).status_code == 404
    path.write_text('{"bodies":["changed"]}', encoding="utf-8")
    assert client.get(url, headers={"Authorization": "alice"}).status_code == 409
    executor._customer_container_verified = False
    assert client.get(url, headers={"Authorization": "alice"}).status_code == 503
