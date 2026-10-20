from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Finding, Measurement
from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        await asyncio.sleep(0.001)
        if token not in {
            "alice",
            "bob",
            "readonly",
            "wrong-resource",
            "wrong-issuer",
            "expired",
            "no-subject",
        }:
            return None
        return AccessToken(
            token=token,
            client_id="test-client",
            subject=None if token == "no-subject" else token,
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            resource="https://other.example/mcp" if token == "wrong-resource" else RESOURCE,
            expires_at=int(time.time()) + (-10 if token == "expired" else 600),
            claims={"iss": "https://other.example" if token == "wrong-issuer" else ISSUER},
        )


@pytest.fixture
def remote(tmp_path: Path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor())
    monkeypatch.setattr(service, "start", lambda: None)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service


def call(client, token, name, arguments=None):
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    return client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
    )


def result(response):
    assert response.status_code == 200, response.text
    return response.json()["result"]


def test_advertised_construction_selectors_work_over_remote_transport(remote):
    client, _ = remote
    index = result(call(client, "alice", "get_construction_reference"))["structuredContent"]
    for entry in index["recipes"]:
        response = result(
            call(client, "alice", "get_construction_reference", {"topic": entry["query"]})
        )
        assert not response.get("isError")
        assert response["structuredContent"]["name"] == entry["id"]
        assert "unknown_topic" not in response["structuredContent"]
    explicit = result(
        call(client, "alice", "get_construction_reference", {"topic": "construction"})
    )
    assert explicit["structuredContent"] == index


def test_remote_transport_delivers_shared_advice_in_initialize_and_index(remote):
    client, service = remote
    reply = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer alice",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "parameter-guidance-test", "version": "1"},
            },
        },
    )
    initialized = result(reply)
    assert initialized["instructions"].count(PARAMETRIC_DESIGN_GUIDANCE) == 1
    assert "qualified isolated worker" in initialized["instructions"]
    index = result(call(client, "alice", "get_construction_reference"))["structuredContent"]
    assert index["parametric_design"] == PARAMETRIC_DESIGN_GUIDANCE
    assert service.store.list_sessions("alice") == []


def test_compact_evidence_retains_remote_owner_and_artifact_boundary(remote):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    data = b"original owned bytes"
    (directory / "part.step").write_bytes(data)
    evidence = Measurement(
        revision_id=rid,
        contract_hash="contract",
        engine_fingerprint="engine",
        artifact_hashes={"part.step": hashlib.sha256(data).hexdigest()},
        geometry_valid=True,
        findings=[
            Finding(
                requirement_id="fit",
                status="unverified",
                method="unsupported",
                reason="Unmeasured fit",
            )
        ],
        caveats=["Not physical proof"],
    ).model_dump()
    service.store.add_revision(sid, {"id": rid, "measurement": evidence})
    args = {"project_id": sid, "revision_id": rid}
    assert result(call(client, "bob", "get_evidence_summary", args))["isError"]
    response = result(call(client, "alice", "get_evidence_summary", args))
    assert not response.get("isError")
    assert response["structuredContent"]["findings"] == evidence["findings"]
    assert response["structuredContent"]["integrity_scope"] == "revision_artifact_hashes"
    assert "Unmeasured fit" in response["content"][0]["text"]
    (directory / "part.step").write_bytes(b"tampered")
    assert result(call(client, "alice", "get_evidence_summary", args))["isError"]


def test_wait_remains_owner_scoped_and_distinguishes_replaced_jobs(remote):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    jid, rid = "b" * 32, "c" * 32
    service.store.update(sid, job_id=jid, state="needs_review", proposal={"revision_id": rid})
    args = {"project_id": sid, "job_id": jid, "timeout_seconds": 0.05}
    assert result(call(client, "bob", "wait_for_job", args))["isError"]
    reply = result(call(client, "alice", "wait_for_job", args))
    assert not reply.get("isError")
    assert reply["structuredContent"]["outcome"] == "terminal"
    assert reply["structuredContent"]["snapshot"]["design_revision"] == rid
    assert reply["structuredContent"]["snapshot"]["accepted_revision"] is None
    service.store.update(sid, job_id="d" * 32, state="needs_review")
    changed = result(call(client, "alice", "wait_for_job", args))["structuredContent"]
    assert changed["outcome"] == "superseded"
    assert changed["target_job_id"] == jid and changed["current_job_id"] == "d" * 32


def test_wait_checks_remote_ownership_on_every_poll(remote, monkeypatch):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    jid = "b" * 32
    service.store.update(sid, job_id=jid, state="running")
    original = service.store.session_record
    owners = []

    def read(project_id, owner):
        owners.append(owner)
        if len(owners) == 2:
            from forge_mind.workspace.store import WorkspaceError

            raise WorkspaceError("Ownership revoked")
        return original(project_id, owner)

    monkeypatch.setattr(service.store, "session_record", read)
    reply = result(
        call(
            client,
            "alice",
            "wait_for_job",
            {
                "project_id": sid,
                "job_id": jid,
                "timeout_seconds": 2,
            },
        )
    )
    assert reply["isError"] and owners == ["alice", "alice"]


def test_wait_deadline_includes_slow_remote_store_read(remote, monkeypatch):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    original = service.store.session_record

    def delayed(project_id, owner):
        time.sleep(0.7)
        return original(project_id, owner)

    monkeypatch.setattr(service.store, "session_record", delayed)
    started = time.monotonic()
    reply = result(
        call(
            client,
            "alice",
            "wait_for_job",
            {
                "project_id": sid,
                "job_id": "b" * 32,
                "timeout_seconds": 0.05,
            },
        )
    )
    assert time.monotonic() - started < 0.5
    assert reply["isError"]
    assert "No owned project snapshot" in reply["content"][0]["text"]


@pytest.mark.parametrize("token", [None, "invalid", "wrong-resource", "expired"])
def test_native_transport_requires_current_valid_resource_token(remote, token):
    client, _ = remote
    assert call(client, token, "get_status").status_code == 401


@pytest.mark.parametrize("token", ["wrong-issuer", "no-subject"])
def test_tool_requires_exact_issuer_and_resource_owner(remote, token):
    client, _ = remote
    assert result(call(client, token, "get_status"))["isError"]


def test_scope_does_not_allow_readonly_owner_to_write(remote):
    client, service = remote
    denied = result(call(client, "readonly", "create_project", {"title": "plate"}))
    assert denied["isError"]
    assert service.store.list_sessions("readonly") == []


def test_parallel_callers_do_not_share_owner_or_token(remote):
    client, service = remote
    with ThreadPoolExecutor(max_workers=4) as pool:
        calls = list(
            pool.map(
                lambda owner: result(call(client, owner, "create_project", {"title": owner})),
                ["alice", "bob"] * 4,
            )
        )
    assert all(not response.get("isError") for response in calls)
    assert len(service.store.list_sessions("alice")) == 4
    assert len(service.store.list_sessions("bob")) == 4
    alice = service.store.list_sessions("alice")[0]["id"]
    for tool in ("get_project", "cancel_job"):
        assert result(call(client, "bob", tool, {"project_id": alice}))["isError"]


def test_unqualified_typed_generation_never_admits_job(remote):
    client, service = remote
    project = service.store.create("alice", "plate", None)
    response = result(
        call(
            client,
            "alice",
            "submit_document",
            {
                "project_id": project["id"],
                "document": {"name": "plate"},
                "contract": {},
            },
        )
    )
    assert response["isError"]
    assert service.store.get(project["id"], "alice")["job_id"] is None


def test_qualified_submission_uses_bounded_owner_scoped_job(remote):
    client, service = remote
    service.executor._typed_container_verified = True
    project = service.store.create("alice", "plate", None)
    response = result(
        call(
            client,
            "alice",
            "submit_document",
            {
                "project_id": project["id"],
                "document": {"name": "plate"},
                "contract": {},
            },
        )
    )
    assert not response.get("isError")
    job = service.store.claim()
    assert job["owner"] == "alice"
    assert job["max_seconds"] == 300
    assert job["preview_only"] is True
    assert job["document"]["name"] == "plate"


def test_polling_exposes_preview_candidate_without_accepting_it(remote):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    service.store.update(
        sid, proposal={"revision_id": "a" * 32, "eligible": True}, selected_revision=None
    )
    response = result(call(client, "alice", "get_project", {"project_id": sid}))[
        "structuredContent"
    ]
    assert response["selected_revision"] is None
    assert response["proposal"] == {"revision_id": "a" * 32, "eligible": True}


def test_repair_admits_current_candidate_and_keeps_accepted_baseline(remote):
    client, service = remote
    service.executor._typed_container_verified = True
    sid = service.store.create("alice", "plate", None)["id"]
    current, stale = "a" * 32, "b" * 32
    for rid in (current, stale):
        service.store.add_revision(sid, {"id": rid, "measurement": {"artifact_hashes": {}}})
    service.store.update(
        sid, proposal={"revision_id": current}, selected_revision=None, accepted_revision=None
    )
    arguments = {
        "project_id": sid,
        "document": {"name": "Repair"},
        "contract": {},
        "based_on": None,
    }
    assert result(
        call(client, "alice", "submit_document", {**arguments, "source_revision": stale})
    )["isError"]
    assert service.store.get(sid, "alice")["job_id"] is None
    admitted = result(
        call(client, "alice", "submit_document", {**arguments, "source_revision": current})
    )
    assert not admitted.get("isError")
    job = service.store.claim()
    assert job["revision_id"] == current
    assert job["expected_current_revision"] == current
    assert job["based_on"] is None


@pytest.mark.parametrize("seconds", [0, -1, 601, None])
def test_remote_generation_rejects_unbounded_budget_before_queue(remote, seconds):
    client, service = remote
    service.executor._typed_container_verified = True
    project = service.store.create("alice", "plate", None)
    response = result(
        call(
            client,
            "alice",
            "submit_document",
            {
                "project_id": project["id"],
                "document": {},
                "contract": {},
                "max_seconds": seconds,
            },
        )
    )
    assert response["isError"]
    assert service.store.get(project["id"], "alice")["job_id"] is None


def test_owned_export_refuses_tampered_or_foreign_artifacts(remote):
    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    data = b"immutable CAD bytes"
    (directory / "part.step").write_bytes(data)
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "contract": Contract().model_dump(),
            "measurement": {
                "revision_id": rid,
                "artifact_hashes": {"part.step": hashlib.sha256(data).hexdigest()},
            },
        },
    )
    arguments = {"project_id": sid, "revision_id": rid, "format": "step"}
    exported = result(call(client, "alice", "export_artifact", arguments))
    assert exported["structuredContent"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert "path" not in exported["structuredContent"]
    assert result(call(client, "bob", "export_artifact", arguments))["isError"]
    (directory / "part.step").write_bytes(b"changed")
    assert result(call(client, "alice", "export_artifact", arguments))["isError"]
    assert result(call(client, "alice", "get_evidence", {"project_id": sid, "revision_id": rid}))[
        "isError"
    ]


def test_remote_factory_never_exposes_trusted_local_executor(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    with pytest.raises(ValueError, match="trusted_local"):
        create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)


def test_remote_rejects_foreign_host_and_browser_origin(remote):
    client, _ = remote
    headers = {"Authorization": "Bearer alice", "Accept": "application/json, text/event-stream"}
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert (
        client.post(
            "/mcp", headers={**headers, "Host": "attacker.example"}, json=request
        ).status_code
        == 421
    )
    assert (
        client.post(
            "/mcp", headers={**headers, "Origin": "https://attacker.example"}, json=request
        ).status_code
        == 403
    )


def test_candidate_preview_document_and_topology_are_owned_and_hash_bound(remote):
    import base64
    import json

    client, service = remote
    sid = service.store.create("alice", "plate", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6qkAAAAASUVORK5CYII="
    )
    files = {
        "render.png": png,
        "document.json": json.dumps({"name": "Editable plate"}).encode(),
        "topology.json": json.dumps({"entities": [{"id": "face_a"}, {"id": "edge_b"}]}).encode(),
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "measurement": {
                "revision_id": rid,
                "artifact_hashes": {
                    name: hashlib.sha256(data).hexdigest() for name, data in files.items()
                },
            },
        },
    )
    arguments = {"project_id": sid, "revision_id": rid}
    preview = result(call(client, "alice", "get_preview", arguments))
    assert preview["content"][0]["mimeType"] == "image/png"
    document = result(call(client, "alice", "get_document", arguments))
    assert document["structuredContent"]["document"]["name"] == "Editable plate"
    topology = result(call(client, "alice", "get_topology", {**arguments, "limit": 1}))
    assert topology["structuredContent"]["entities"] == [{"id": "face_a"}]
    assert topology["structuredContent"]["next_offset"] == 1
    for tool in ("get_preview", "get_document", "get_topology"):
        assert result(call(client, "bob", tool, arguments))["isError"]
    (directory / "render.png").write_bytes(b"tampered")
    assert result(call(client, "alice", "get_preview", arguments))["isError"]


def test_shutdown_stops_owned_worker_service(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor())
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert not result(call(client, "alice", "get_status")).get("isError")
        assert not service.stop.is_set()
    assert service.stop.is_set()


def test_evidence_accepts_host_asset_metadata_without_relaxing_worker_outputs(remote):
    import json
    import shutil

    from build123d import Box, export_step

    from forge_mind.workspace.executor import typed_artifact_name
    from forge_mind.workspace.remote_mcp import revision_artifact_name

    client, service = remote
    sid = service.store.create("alice", "Imported box", None)["id"]
    rid, aid = "a" * 32, "b" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    export_step(Box(10, 20, 30), directory / "part.step")
    imported = "input_" + aid + ".step"
    shutil.copyfile(directory / "part.step", directory / imported)
    (directory / "assets.json").write_text(json.dumps([{"id": aid}]), encoding="utf-8")
    hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ["part.step", imported, "assets.json"]
    }
    service.store.add_revision(
        sid, {"id": rid, "measurement": {"revision_id": rid, "artifact_hashes": hashes}}
    )
    evidence = result(
        call(client, "alice", "get_evidence", {"project_id": sid, "revision_id": rid})
    )
    assert not evidence.get("isError")
    assert evidence["structuredContent"]["artifact_hashes"] == hashes
    assert revision_artifact_name("assets.json") and revision_artifact_name(imported)
    assert not typed_artifact_name("assets.json") and not typed_artifact_name(imported)
    assert not revision_artifact_name("input_../../other.step")
    (directory / "assets.json").write_text("changed", encoding="utf-8")
    assert result(call(client, "alice", "get_evidence", {"project_id": sid, "revision_id": rid}))[
        "isError"
    ]


def _seed_named_export(service, owner="alice", role="make"):
    import json

    document = {
        "name": "Definition export fixture",
        "assembly": {
            "parts": [
                {
                    "id": "tray",
                    "name": "Tray",
                    "production": {"role": role},
                    "features": [
                        {
                            "id": "shape",
                            "name": "Shape",
                            "kind": "box",
                            "params": {"x": 10, "y": 8, "z": 3},
                        }
                    ],
                    "output": "shape",
                }
            ],
            "roots": [
                {"id": "tray_a", "definition": "tray"},
                {"id": "tray_b", "definition": "tray", "position": [20, 0, 0]},
            ],
        },
    }
    sid = service.store.create(owner, "named export", None)["id"]
    rid = "d" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    data = {
        "document.json": json.dumps(document).encode(),
        "definition_tray.step": b"immutable definition",
        "part.step": b"whole assembly",
    }
    for name, content in data.items():
        (directory / name).write_bytes(content)
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "measurement": {
                "revision_id": rid,
                "artifact_hashes": {
                    name: hashlib.sha256(content).hexdigest() for name, content in data.items()
                },
            },
        },
    )
    return sid, rid, directory, data


def _read_named_resource(client, token, uri):
    return result(
        client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer " + token,
            },
            json={"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": uri}},
        )
    )


@pytest.mark.parametrize("owner", ["alice", "readonly"])
@pytest.mark.parametrize("role", ["make", "buy", "reference", "unspecified"])
def test_remote_named_definition_export_is_owned_local_frame_resource(remote, owner, role):
    import base64

    client, service = remote
    sid, rid, directory, data = _seed_named_export(service, owner, role)
    args = {"project_id": sid, "revision_id": rid, "part_id": "tray"}
    exported = result(call(client, owner, "export_part", args))
    assert not exported.get("isError")
    content = exported["structuredContent"]
    assert "path" not in content
    assert content["part_id"] == "tray" and content["production_role"] == role
    assert content["frame"] == "definition_local" and content["status"] == "draft_candidate"
    assert content["sha256"] == hashlib.sha256(data["definition_tray.step"]).hexdigest()
    assert content["document_sha256"] == hashlib.sha256(data["document.json"]).hexdigest()
    assert content["uri"].endswith("/definition_tray.step")
    resource = _read_named_resource(client, owner, content["uri"])
    assert base64.b64decode(resource["contents"][0]["blob"]) == data["definition_tray.step"]
    assert result(call(client, "bob", "export_part", args))["isError"]
    assert result(call(client, owner, "export_part", {**args, "part_id": "tray_a"}))["isError"]
    assert result(call(client, owner, "export_part", {**args, "part_id": "../tray"}))["isError"]
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == data


@pytest.mark.parametrize("mutation", ["step", "document", "missing", "hardlink", "size"])
def test_remote_named_export_refuses_stale_or_linked_definitions(remote, mutation, monkeypatch):
    import os

    import forge_mind.workspace.remote_mcp as module

    client, service = remote
    sid, rid, directory, _ = _seed_named_export(service)
    args = {"project_id": sid, "revision_id": rid, "part_id": "tray"}
    exported = result(call(client, "alice", "export_part", args))["structuredContent"]
    if mutation == "step":
        (directory / "definition_tray.step").write_bytes(b"changed")
    elif mutation == "document":
        (directory / "document.json").write_bytes(b"changed")
    elif mutation == "missing":
        (directory / "definition_tray.step").unlink()
    elif mutation == "hardlink":
        os.link(directory / "definition_tray.step", directory / "linked.step")
    else:
        monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 1)
    assert result(call(client, "alice", "export_part", args))["isError"]
    denied = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Authorization": "Bearer alice"},
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": exported["uri"]},
        },
    )
    assert "error" in denied.json() or denied.json().get("result", {}).get("isError")


def test_remote_mechanism_reference_and_document_binding_are_read_only(remote):
    from forge_mind.workspace.authoring import authoring_schema
    from forge_mind.workspace.document import DesignDocument

    client, service = remote
    before = service.store.list_sessions("alice")
    for topic in ("mechanism", "construction:mechanism"):
        response = result(call(client, "alice", "get_construction_reference", {"topic": topic}))
        assert not response.get("isError")
        assert response["structuredContent"] == authoring_schema("construction:mechanism")
    schema = result(call(client, "alice", "get_schema", {"query": "document"}))
    assert not schema.get("isError")
    assert schema["structuredContent"]["document"] == DesignDocument.model_json_schema()
    assert schema["structuredContent"]["document"]["$defs"]["Component"]["properties"]["feature"][
        "description"
    ]
    assert service.store.list_sessions("alice") == before
