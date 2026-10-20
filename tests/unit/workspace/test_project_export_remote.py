"""SDK transport controls with narrow saved-package helper fixtures, no CAD worker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace import remote_mcp
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store, WorkspaceError

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "a" * 32
PACKAGE = "b" * 32
ZIP = b"PK\x03\x04saved package fixture bytes"
SHA = hashlib.sha256(ZIP).hexdigest()


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token not in {"alice", "bob", "readonly", "no-read", "wrong-issuer"}:
            return None
        return AccessToken(
            token=token,
            client_id="project-export-test",
            subject="alice" if token == "readonly" else token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": "https://wrong.example" if token == "wrong-issuer" else ISSUER},
        )


def rpc(client, token, method, params):
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )


def call(client, token, sid, **options):
    return rpc(
        client,
        token,
        "tools/call",
        {
            "name": "export_project",
            "arguments": {"project_id": sid, "revision_id": RID, "preview": "none", **options},
        },
    )


def result(response):
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.fixture
def packages(tmp_path: Path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor())
    sid = service.store.create("alice", "saved project", None)["id"]
    service.store.add_revision(sid, {"id": RID})
    monkeypatch.setattr(service, "start", lambda: None)
    module = ModuleType("forge_mind.workspace.project_delivery")
    calls = []
    discarded = threading.Event()
    directory = tmp_path / "saved-package-fixture"
    directory.mkdir()
    receipt = {
        "project_id": sid,
        "revision_id": RID,
        "package_id": PACKAGE,
        "include_stl": False,
        "sha256": SHA,
        "bytes": len(ZIP),
        "status": "draft_candidate",
        "manufacturing_ready": False,
        "physical_readiness": "unverified",
        "snapshot_sha256": "c" * 64,
        "context": {"selected_revision": None, "revision_id": RID},
        "source_scope": {"independent_source_recovery": "incomplete"},
        "made_parts": [{"part_id": "body", "quantity": 1}],
        "assembly": {"file": "assembly/part.step", "frame": "world"},
        "retention": "Durable captured resource; host retention policy applies.",
        "path": "PRIVATE-HOST-PATH",
    }

    def create(service, project_id, principal, revision_id, *, include_stl=False):
        service.store.get(project_id, principal)
        service.store.revision(project_id, revision_id)
        calls.append(("create", principal, include_stl))
        (directory / "package.zip").write_bytes(ZIP)
        return {**receipt, "include_stl": include_stl}

    def read(service, project_id, principal, package_id, sha256):
        service.store.get(project_id, principal)
        calls.append(("read", principal))
        if package_id != PACKAGE or sha256 != SHA:
            raise WorkspaceError("Package not found")
        data = (directory / "package.zip").read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Package bytes changed")
        return data, receipt.copy()

    def discard(service, project_id, principal, package_id):
        service.store.get(project_id, principal)
        assert package_id == PACKAGE
        calls.append(("discard", principal))
        (directory / "package.zip").unlink(missing_ok=True)
        discarded.set()

    module.create_project_export = create
    module.read_project_export = read
    module.discard_project_export = discard
    monkeypatch.setitem(sys.modules, module.__name__, module)
    yield service, sid, module, receipt, directory, calls, discarded


def app(service):
    return remote_mcp.create_app(
        service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )


def test_project_export_readonly_auth_and_durable_restart_readback(packages):
    service, sid, _, receipt, _, calls, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        response = result(call(client, "readonly", sid))
        assert not response.get("isError")
        exported = response["structuredContent"]
        assert exported == {
            **{key: value for key, value in receipt.items() if key != "path"},
            "uri": f"cadkitty://projects/{sid}/exports/{PACKAGE}/{SHA}.zip",
        }
        assert "PRIVATE-HOST-PATH" not in json.dumps(response)
        assert not service.executor.typed_container_verified
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as restarted:
        fetched = result(rpc(restarted, "readonly", "resources/read", {"uri": exported["uri"]}))
        assert fetched["contents"][0]["mimeType"] == "application/zip"
        assert base64.b64decode(fetched["contents"][0]["blob"], validate=True) == ZIP
        foreign = rpc(restarted, "bob", "resources/read", {"uri": exported["uri"]})
        assert "error" in foreign.json()
    assert calls == [("create", "alice", False), ("read", "alice")]
    assert [row["id"] for row in service.store.get(sid, "alice")["revisions"]] == [RID]


@pytest.mark.parametrize("token", [None, "invalid", "no-read", "wrong-issuer", "bob"])
def test_creation_denies_unauthorized_owner_without_packaging(packages, token):
    service, sid, _, _, directory, calls, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        response = call(client, token, sid)
        assert response.status_code in {401, 403} or result(response)["isError"]
    assert calls == [] and not (directory / "package.zip").exists()


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_include_stl_is_strict_before_dispatch(packages, value):
    service, sid, _, _, _, calls, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        assert result(call(client, "alice", sid, include_stl=value))["isError"]
    assert calls == []


def test_discovery_preserves_default_and_read_annotation(packages):
    service, _, _, _, _, _, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        initialized = result(
            rpc(
                client,
                "alice",
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "export-test", "version": "1"},
                },
            )
        )
        assert initialized["serverInfo"]["version"] == "0.0.20"
        assert "export_project" in initialized["instructions"]
        tools = result(rpc(client, "alice", "tools/list", {}))["tools"]
        tool = next(tool for tool in tools if tool["name"] == "export_project")
        assert tool["annotations"]["readOnlyHint"] is True
        assert tool["inputSchema"]["properties"]["include_stl"]["default"] is False
        assert tool["inputSchema"]["properties"]["include_stl"]["type"] == "boolean"


def test_stl_requested_and_resource_never_regenerates(packages):
    service, sid, _, _, directory, calls, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        exported = result(call(client, "alice", sid, include_stl=True))["structuredContent"]
        assert exported["include_stl"] is True
        (directory / "package.zip").unlink()
        missing = rpc(client, "alice", "resources/read", {"uri": exported["uri"]})
        assert "error" in missing.json()
        assert not (directory / "package.zip").exists()
    assert [call[0] for call in calls] == ["create", "read"]


def test_resource_tamper_swapped_hash_and_invalid_identifiers(packages):
    service, sid, _, _, directory, _, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        exported = result(call(client, "alice", sid))["structuredContent"]
        for uri in (
            exported["uri"].replace(SHA, "d" * 64),
            exported["uri"].replace(PACKAGE, "e" * 32),
            exported["uri"].replace(PACKAGE, "invalid"),
            exported["uri"].replace(sid, "invalid"),
        ):
            assert "error" in rpc(client, "alice", "resources/read", {"uri": uri}).json()
        (directory / "package.zip").write_bytes(b"tamper")
        assert "error" in rpc(client, "alice", "resources/read", {"uri": exported["uri"]}).json()


@pytest.mark.parametrize(
    "error,expected",
    [
        (WorkspaceError("Stale source"), "Stale source"),
        (OSError("PRIVATE-HOST-PATH"), "Project package files are unavailable"),
    ],
)
def test_helper_errors_are_explicit_tool_errors(packages, error, expected):
    service, sid, module, _, _, _, _ = packages

    def fail(*args, **kwargs):
        raise error

    module.create_project_export = fail
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        failed = result(call(client, "alice", sid))
        assert failed["isError"] and expected in failed["content"][0]["text"]
        assert "PRIVATE-HOST-PATH" not in json.dumps(failed)


def test_oversize_or_invalid_metadata_discards_unpublished_package(packages, monkeypatch):
    service, sid, _, receipt, directory, _, discarded = packages
    monkeypatch.setattr(remote_mcp, "MAX_ARTIFACT_BYTES", len(ZIP) - 1)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        assert result(call(client, "alice", sid))["isError"]
    assert discarded.is_set() and not (directory / "package.zip").exists()
    receipt["context"] = {"oversized": "x" * (remote_mcp.MAX_RESPONSE_BYTES + 1)}
    monkeypatch.setattr(remote_mcp, "MAX_ARTIFACT_BYTES", 64 * 1024 * 1024)
    discarded.clear()
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        assert result(call(client, "alice", sid))["isError"]
    assert discarded.is_set() and not (directory / "package.zip").exists()


def test_transport_rechecks_returned_resource_bytes_and_receipt(packages):
    service, sid, module, receipt, _, _, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        exported = result(call(client, "alice", sid))["structuredContent"]
        for changed in (b"tampered helper return", ZIP):
            bad_receipt = {**receipt, "package_id": "f" * 32} if changed == ZIP else receipt
            module.read_project_export = lambda *args, data=changed, record=bad_receipt: (
                data,
                record,
            )
            assert (
                "error" in rpc(client, "alice", "resources/read", {"uri": exported["uri"]}).json()
            )


def test_cancel_preserves_producer_until_unpublished_cleanup(packages, monkeypatch):
    service, sid, module, _, directory, calls, discarded = packages
    entered, finish = threading.Event(), threading.Event()
    create = module.create_project_export

    def delayed(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return create(*args, **kwargs)

    module.create_project_export = delayed

    async def control():
        token = await Verifier().verify_token("readonly")
        monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
        server = remote_mcp.create_server(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )

        request = asyncio.create_task(
            server.call_tool(
                "export_project", {"project_id": sid, "revision_id": RID, "preview": "none"}
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert not discarded.is_set()
        finish.set()
        assert await asyncio.to_thread(discarded.wait, 5)
        await asyncio.sleep(0.01)

    asyncio.run(control())
    assert calls == [("create", "alice", False), ("discard", "alice")]
    assert not (directory / "package.zip").exists()


@pytest.mark.parametrize("token", [None, "no-read", "wrong-issuer"])
def test_resource_requires_fresh_read_auth(packages, token):
    service, sid, _, _, _, calls, _ = packages
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        exported = result(call(client, "alice", sid))["structuredContent"]
        denied = rpc(client, token, "resources/read", {"uri": exported["uri"]})
        assert denied.status_code in {401, 403} or "error" in denied.json()
    assert calls == [("create", "alice", False)]


@pytest.mark.parametrize("field,value", [("project_id", "f" * 32), ("revision_id", "e" * 32)])
def test_mismatched_receipt_is_not_published(packages, field, value):
    service, sid, _, receipt, directory, _, discarded = packages
    receipt[field] = value
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        assert result(call(client, "alice", sid))["isError"]
    assert discarded.is_set() and not (directory / "package.zip").exists()


def test_cancel_after_production_before_publication_also_discards(packages, monkeypatch):
    service, sid, _, _, directory, calls, discarded = packages
    entered, release = threading.Event(), threading.Event()
    real_shield = asyncio.shield

    async def control():
        token = await Verifier().verify_token("alice")
        monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
        server = remote_mcp.create_server(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )

        async def delayed_shield(task):
            value = await real_shield(task)
            entered.set()
            await asyncio.to_thread(release.wait, 5)
            return value

        monkeypatch.setattr(remote_mcp.asyncio, "shield", delayed_shield)
        request = asyncio.create_task(
            server.call_tool(
                "export_project", {"project_id": sid, "revision_id": RID, "preview": "none"}
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        assert discarded.is_set()

    asyncio.run(control())
    assert calls == [("create", "alice", False), ("discard", "alice")]
    assert not (directory / "package.zip").exists()
