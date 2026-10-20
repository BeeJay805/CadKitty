"""Actual SDK/owned HTTP transport tests; service/render boundary is mocked only."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
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
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server, occurrence_tool_result
from forge_mind.workspace.occurrence_focus import (
    OccurrenceFocus,
    focus_view_metadata,
    resolve_focus,
)
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

RID = "b" * 32
PATH = "assembly/guide"
SIGNATURE = "f" * 64
FOCUS = {"kind": "face", "signature": SIGNATURE, "view": "front", "padding": 2.5}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="
)
ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"


def envelope(focus=None, image=True):
    receipt = (
        {
            "revision_id": RID,
            "occurrence_path": PATH,
            "image": {"file": "inspection.png", "sha256": hashlib.sha256(PNG).hexdigest()},
        }
        if image
        else None
    )
    if focus is not None:
        topology = {
            "frame": "cad_z_up_mm",
            "entities": [
                {
                    "kind": "body",
                    "id": "b0",
                    "body_index": 0,
                    "signature": "1" * 64,
                    "body_signature": "1" * 64,
                    "min": [0, 0, 0],
                    "max": [10, 10, 10],
                },
                {
                    "kind": focus.kind,
                    "id": "selected",
                    "body_index": 0,
                    "signature": focus.signature,
                    "body_signature": "1" * 64,
                    "min": [4, 4, 8],
                    "max": [6, 6, 8],
                },
            ],
        }
        frame = resolve_focus(topology, 0, focus)
        receipt.update(focus=frame, focus_view=focus_view_metadata(frame))
    return {
        "facts": {
            "occurrence_path": PATH,
            "scope": "Saved facts only",
            "provenance": {"revision_id": RID, "occurrence_path": PATH, "body_index": 0},
        },
        "receipt": receipt,
        "image_base64": base64.b64encode(PNG).decode() if image else None,
        "mime_type": "image/png" if image else None,
    }


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from forge_mind.workspace import occurrence_inspection

    service = WorkspaceService(Store(tmp_path), Executor())
    monkeypatch.setattr(service, "start", lambda: pytest.fail("No CAD job"))
    sid = service.store.create("alice", "focus transport", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls = []
    mutation = []

    def inspect(svc, project, owner, revision, occurrence_path, **kwargs):
        svc.store.session_record(project, owner)
        svc.store.revision(project, revision)
        calls.append((owner, occurrence_path, kwargs))
        value = envelope(kwargs.get("focus"), kwargs["include_image"])
        if mutation:
            if mutation[0] == "drop":
                value["receipt"].pop("focus")
            else:
                value["receipt"]["focus_view"]["view"] = "rear"
        return value

    monkeypatch.setattr(occurrence_inspection, "inspect_occurrence", inspect)
    return service, sid, calls, mutation


def owned_app(service):
    def owner(authorization: str = Header()):
        return authorization

    app = FastAPI()
    app.include_router(router(service.store, service, owner))
    return app


def assert_image(result):
    assert not result.is_error, result.content
    assert base64.b64decode(result.content[1].data) == PNG
    assert result.content[1].mime_type == "image/png"
    assert json.loads(result.content[0].text) == result.structured_content
    assert "image_base64" not in result.content[0].text


def test_local_actual_sdk_focus_schema_image_and_legacy_omission(boundary, tmp_path):
    service, sid, calls, _ = boundary
    client = TestClient(owned_app(service))
    bodies = []

    def backend(request):
        bodies.append(json.loads(request.content))
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
            assert "focus" in tools["inspect_occurrence"].input_schema["properties"]
            assert "inspect_entity" not in tools and "inspect_geometry_view" not in tools
            arguments = {"project_id": sid, "revision_id": RID, "occurrence_path": PATH}
            focused = await sdk.call_tool("inspect_occurrence", {**arguments, "focus": FOCUS})
            assert_image(focused)
            expected = envelope(OccurrenceFocus.model_validate(FOCUS))
            assert focused.structured_content == {k: expected[k] for k in ("facts", "receipt")}
            assert focused.structured_content["receipt"]["focus"]["request"] == FOCUS
            legacy = await sdk.call_tool("inspect_occurrence", arguments)
            assert_image(legacy)
            assert legacy.structured_content == {k: envelope()[k] for k in ("facts", "receipt")}
            assert "focus" not in legacy.structured_content["receipt"]
            invalid = await sdk.call_tool(
                "inspect_occurrence", {**arguments, "focus": FOCUS, "include_image": False}
            )
            assert invalid.is_error

    asyncio.run(scenario())
    assert bodies[0]["focus"] == FOCUS and "focus" not in bodies[1]
    assert len(bodies) == 2 and len(calls) == 2
    assert calls[0][2]["focus"].model_dump(mode="json") == FOCUS and "focus" not in calls[1][2]
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["drop", "view"])
def test_local_sdk_rejects_worker_focus_loss_or_wrong_view(boundary, tmp_path, mutation):
    service, sid, _, mutations = boundary
    mutations.append(mutation)
    client = TestClient(owned_app(service))

    def backend(request):
        value = client.request(
            request.method,
            request.url.path,
            headers={"Authorization": "alice", "Content-Type": "application/json"},
            content=request.content,
        )
        return httpx.Response(value.status_code, content=value.content)

    async def scenario():
        async with Client(
            create_server(
                WorkspaceAdapter(
                    "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
                )
            )
        ) as sdk:
            result = await sdk.call_tool(
                "inspect_occurrence",
                {"project_id": sid, "revision_id": RID, "occurrence_path": PATH, "focus": FOCUS},
            )
            assert result.is_error and not any(item.type == "image" for item in result.content)

    asyncio.run(scenario())


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "no-read"}:
            return None
        return AccessToken(
            token=token,
            client_id="test",
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


def test_authenticated_remote_focus_image_owner_and_legacy_default(boundary):
    service, sid, calls, _ = boundary
    before = service.store.get(sid, "alice")
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert remote_call(client, "no-read", sid, focus=FOCUS).status_code == 403
        assert remote_call(client, "bob", sid, focus=FOCUS).json()["result"]["isError"]
        assert not calls
        result = remote_call(client, "alice", sid, focus=FOCUS).json()["result"]
        assert not result.get("isError")
        expected = envelope(OccurrenceFocus.model_validate(FOCUS))
        assert result["structuredContent"] == {k: expected[k] for k in ("facts", "receipt")}
        assert base64.b64decode(result["content"][1]["data"]) == PNG
        assert "image_base64" not in result["content"][0]["text"]
        legacy = remote_call(client, "alice", sid).json()["result"]
        assert legacy["structuredContent"] == {k: envelope()[k] for k in ("facts", "receipt")}
        assert remote_call(client, "alice", sid, focus=FOCUS, include_image=False).json()["result"][
            "isError"
        ]
        assert len(calls) == 2 and "focus" not in calls[1][2]
    assert service.store.get(sid, "alice") == before
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "focus",
    [
        {**FOCUS, "signature": "F" * 64},
        {**FOCUS, "kind": "body"},
        {**FOCUS, "padding": "2.5"},
        {**FOCUS, "padding": 0.1},
        {**FOCUS, "view": "free"},
        {**FOCUS, "owner": "bob"},
    ],
)
def test_http_strict_focus_schema_rejects_before_inspection(boundary, focus):
    service, sid, calls, _ = boundary
    response = TestClient(owned_app(service)).post(
        f"/workspace/sessions/{sid}/design/{RID}/inspect-occurrence",
        headers={"Authorization": "alice"},
        json={"occurrence_path": PATH, "focus": focus},
    )
    assert response.status_code == 422 and not calls


def test_http_focus_foreign_owner_and_facts_only_refuse_before_service(boundary):
    service, sid, calls, _ = boundary
    client = TestClient(owned_app(service))
    url = f"/workspace/sessions/{sid}/design/{RID}/inspect-occurrence"
    assert (
        client.post(
            url, headers={"Authorization": "bob"}, json={"occurrence_path": PATH, "focus": FOCUS}
        ).status_code
        == 404
    )
    assert (
        client.post(
            url,
            headers={"Authorization": "alice"},
            json={"occurrence_path": PATH, "focus": FOCUS, "include_image": False},
        ).status_code
        == 409
    )
    assert not calls


@pytest.mark.parametrize(
    "fault",
    [
        "omitted",
        "wrong_request",
        "wrong_body",
        "wrong_basis",
        "numeric_depth",
        "no_image",
        "unsolicited",
    ],
)
def test_shared_packaging_requires_exact_requested_focus_and_view(fault):
    focus = OccurrenceFocus.model_validate(FOCUS)
    value = copy.deepcopy(envelope(focus))
    if fault == "omitted":
        value["receipt"].pop("focus")
    elif fault == "wrong_request":
        value["receipt"]["focus"]["request"]["signature"] = "e" * 64
    elif fault == "wrong_body":
        value["receipt"]["focus"]["body_index"] = 1
    elif fault == "wrong_basis":
        value["receipt"]["focus_view"]["basis"][0][0] += 0.1
    elif fault == "numeric_depth":
        value["receipt"]["focus_view"]["all_selected_body_triangles_depth_tested"] = 1
    elif fault == "no_image":
        value["image_base64"] = None
        value["mime_type"] = None
    else:
        focus = None
    with pytest.raises(ToolError):
        occurrence_tool_result(value, RID, PATH, focus=focus)


@pytest.mark.parametrize("mutation", ["drop", "view"])
def test_remote_sdk_refuses_changed_worker_focus_before_image_delivery(boundary, mutation):
    service, sid, _, mutations = boundary
    mutations.append(mutation)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        result = remote_call(client, "alice", sid, focus=FOCUS).json()["result"]
        assert result["isError"] and not any(item["type"] == "image" for item in result["content"])
