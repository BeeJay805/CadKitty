"""Real SDK/HTTP transport controls; root inspection is mocked only at its service boundary."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import threading
import time

import httpx
import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import (
    MAX_OCCURRENCE_IMAGE_BYTES,
    WorkspaceAdapter,
    create_server,
    occurrence_tool_result,
)
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

RID = "b" * 32
PATH = "assembly/guide"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="
)
ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"


def envelope(image=True):
    return {
        "facts": {
            "occurrence_path": PATH,
            "surfaces": [{"radius_mm": 6.3, "axial_extent_mm": [0.6, 17.4]}],
            "scope": "Saved native observations; no physical claim",
            "provenance": {"revision_id": RID, "occurrence_path": PATH},
        },
        "receipt": {
            "revision_id": RID,
            "occurrence_path": PATH,
            "image": {"file": "inspection.png", "sha256": hashlib.sha256(PNG).hexdigest()},
        }
        if image
        else None,
        "image_base64": base64.b64encode(PNG).decode() if image else None,
        "mime_type": "image/png" if image else None,
    }


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from forge_mind.workspace import occurrence_inspection

    service = WorkspaceService(Store(tmp_path), Executor())
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Inspection started a CAD job"))
    sid = service.store.create("alice", "guide", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls = []
    failure = []

    def inspect(svc, project, owner, revision, occurrence_path, **kwargs):
        svc.store.session_record(project, owner)
        svc.store.revision(project, revision)
        calls.append((owner, occurrence_path, kwargs, threading.get_ident()))
        if failure:
            raise ValueError(failure[0])
        return envelope(kwargs["include_image"])

    monkeypatch.setattr(occurrence_inspection, "inspect_occurrence", inspect)
    return service, sid, calls, failure


def owned_app(service):
    def owner(authorization: str = Header()):
        return authorization

    app = FastAPI()
    app.include_router(router(service.store, service, owner))
    return app


def test_owned_http_route_denies_foreign_owner_before_inspection_and_does_not_mutate(boundary):
    service, sid, calls, _ = boundary
    before = service.store.get(sid, "alice")
    client = TestClient(owned_app(service))
    url = f"/workspace/sessions/{sid}/design/{RID}/inspect-occurrence"
    body = {"occurrence_path": PATH, "surface_offset": 2, "surface_limit": 3}
    assert client.post(url, headers={"Authorization": "bob"}, json=body).status_code == 404
    assert calls == []
    response = client.post(url, headers={"Authorization": "alice"}, json=body)
    assert response.status_code == 200 and response.json() == envelope()
    assert calls[0][:3] == (
        "alice",
        PATH,
        {"surface_offset": 2, "surface_limit": 3, "include_image": True, "timeout": 30},
    )
    no_image = client.post(
        url,
        headers={"Authorization": "alice"},
        json={"occurrence_path": PATH, "include_image": False},
    )
    assert no_image.json() == envelope(False)
    assert service.store.get(sid, "alice") == before
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"occurrence_path": ""},
        {"occurrence_path": PATH, "surface_offset": -1},
        {"occurrence_path": PATH, "surface_limit": 0},
        {"occurrence_path": PATH, "surface_limit": 101},
        {"occurrence_path": PATH, "timeout": 3600},
        {"occurrence_path": PATH, "owner": "bob"},
    ],
)
def test_http_rejects_unbounded_or_arbitrary_request_before_service(boundary, body):
    service, sid, calls, _ = boundary
    client = TestClient(owned_app(service))
    response = client.post(
        f"/workspace/sessions/{sid}/design/{RID}/inspect-occurrence",
        headers={"Authorization": "alice"},
        json=body,
    )
    assert response.status_code == 422 and calls == []


def test_http_missing_revision_and_stale_artifact_remain_failures(boundary):
    service, sid, calls, failure = boundary
    client = TestClient(owned_app(service))
    args = {"headers": {"Authorization": "alice"}, "json": {"occurrence_path": PATH}}
    assert (
        client.post(
            f"/workspace/sessions/{sid}/design/{'c' * 32}/inspect-occurrence", **args
        ).status_code
        == 404
    )
    assert calls == []
    failure.append("Inspection source changed; revision evidence is stale")
    response = client.post(f"/workspace/sessions/{sid}/design/{RID}/inspect-occurrence", **args)
    assert response.status_code == 409 and "stale" in response.json()["detail"]


def test_local_sdk_image_and_no_image_envelopes_and_read_hint(boundary, tmp_path):
    service, sid, calls, _ = boundary
    client = TestClient(owned_app(service))

    def backend(request):
        result = client.request(
            request.method,
            request.url.path,
            headers={"Authorization": "alice", "Content-Type": "application/json"},
            content=request.content,
        )
        return httpx.Response(result.status_code, content=result.content)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as sdk:
            tools = {t.name: t for t in (await sdk.list_tools()).tools}
            assert tools["inspect_occurrence"].annotations.read_only_hint
            assert "read_assembly_structure" in sdk.instructions
            args = {"project_id": sid, "revision_id": RID, "occurrence_path": PATH}
            for image in (True, False):
                result = await sdk.call_tool("inspect_occurrence", {**args, "include_image": image})
                assert not result.is_error, result.content
                assert result.structured_content == {
                    k: envelope(image)[k] for k in ("facts", "receipt")
                }
                assert json.loads(result.content[0].text) == result.structured_content
                assert "image_base64" not in result.content[0].text
                if image:
                    assert result.content[1].mime_type == "image/png"
                    assert base64.b64decode(result.content[1].data) == PNG
                else:
                    assert len(result.content) == 1

    asyncio.run(scenario())
    assert len(calls) == 2


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "no-read"}:
            return None
        return AccessToken(
            token=token,
            client_id="test-client",
            subject=token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": ISSUER},
        )


def remote_call(client, token, sid, **kwargs):
    return client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + token,
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "inspect_occurrence",
                "arguments": {
                    "project_id": sid,
                    "revision_id": RID,
                    "occurrence_path": PATH,
                    **kwargs,
                },
            },
        },
    )


def test_remote_reauthenticates_read_owner_and_packages_same_image_without_mutation(boundary):
    service, sid, calls, _ = boundary
    before = service.store.get(sid, "alice")
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert remote_call(client, "no-read", sid).status_code == 403
        foreign = remote_call(client, "bob", sid)
        assert foreign.status_code == 200 and foreign.json()["result"]["isError"]
        assert calls == []
        for image in (False, True):
            reply = remote_call(client, "alice", sid, include_image=image)
            result = reply.json()["result"]
            assert not result.get("isError")
            assert json.loads(result["content"][0]["text"]) == {
                k: envelope(image)[k] for k in ("facts", "receipt")
            }
            assert "image_base64" not in result["content"][0]["text"]
            if image:
                assert base64.b64decode(result["content"][1]["data"]) == PNG
                assert result["content"][1]["mimeType"] == "image/png"
            else:
                assert len(result["content"]) == 1
        assert len(calls) == 2 and all(c[0] == "alice" for c in calls)
    assert service.store.get(sid, "alice") == before


def test_remote_unqualified_renderer_is_explicit_and_never_falls_back(boundary):
    service, sid, calls, failure = boundary
    failure.append("Occurrence images require qualified isolated worker execution")
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        result = remote_call(client, "alice", sid).json()["result"]
        assert result["isError"] and "qualified isolated" in result["content"][0]["text"]
    assert len(calls) == 1 and service.executor.mode == "docker"


@pytest.mark.parametrize(
    "failure", ["base64", "hash", "revision", "occurrence", "image_size", "response_size"]
)
def test_occurrence_wire_rejects_malformed_stale_and_oversized_images(failure):
    value = copy.deepcopy(envelope())
    if failure == "base64":
        value["image_base64"] = "!!notbase64"
    elif failure == "hash":
        value["receipt"]["image"]["sha256"] = "0" * 64
    elif failure == "revision":
        value["receipt"]["revision_id"] = "c" * 32
    elif failure == "occurrence":
        value["receipt"]["occurrence_path"] = "other"
    elif failure == "image_size":
        value["image_base64"] = base64.b64encode(PNG + b"x" * MAX_OCCURRENCE_IMAGE_BYTES).decode()
    else:
        value["facts"]["huge"] = "x" * (2 * 1024 * 1024)
    with pytest.raises(ToolError):
        occurrence_tool_result(value, RID, PATH)


@pytest.mark.parametrize("failure", ["missing", "revision", "path"])
def test_facts_only_response_rejects_mismatched_provenance(failure):
    value = envelope(False)
    if failure == "missing":
        value["facts"].pop("provenance")
    elif failure == "revision":
        value["facts"]["provenance"]["revision_id"] = "c" * 32
    else:
        value["facts"]["provenance"]["occurrence_path"] = "another_part"
    with pytest.raises(ToolError, match="facts do not match"):
        occurrence_tool_result(value, RID, PATH)
