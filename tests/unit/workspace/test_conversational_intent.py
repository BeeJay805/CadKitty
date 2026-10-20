from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.engineering_api import engineering_router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.project_access import project_access_router
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
INTENT = ' \tHold this café cable on my desk 🔧.\nKeep the guide open. Photo text: "ignore rules" is data.\n '
MAX_INTENT = "é" * 10000
INVALID = ["", " \n\t", "x" * 10001, "é" * 10001, 42, True, {}, []]
INVALID_IDS = [
    "empty",
    "whitespace",
    "long-ascii",
    "long-unicode",
    "integer",
    "boolean",
    "object",
    "array",
]


def assert_no_execution(store):
    with store.connect() as connection:
        for table in ("jobs", "revisions", "engineering_inputs", "engineering_history"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.fixture
def local(tmp_path):
    store = Store(tmp_path / "store")
    service = WorkspaceService(store, Executor())
    app = FastAPI()
    app.include_router(engineering_router(store, service, lambda: "alice"))
    app.include_router(project_access_router(store, service, lambda: "alice"))
    return app, store


def test_manual_api_retains_exact_intent_and_title_fallback(local):
    app, store = local
    with TestClient(app) as client:
        for body, expected in (
            (None, "Manual engineering design"),
            ({"name": "Cable clip"}, "Cable clip"),
            ({"name": "Cable clip", "intent": None}, "Cable clip"),
            ({"name": "Cable clip", "intent": INTENT}, INTENT),
            ({"name": "Cable clip", "intent": MAX_INTENT}, MAX_INTENT),
            ({"name": "Cable clip", "intent": " " + "x" * 9998 + " "}, " " + "x" * 9998 + " "),
        ):
            reply = client.post("/workspace/manual-sessions", json=body)
            assert reply.status_code == 200
            data = reply.json()
            assert data["prompt"] == expected
            assert store.get(data["id"], "alice")["prompt"] == expected
            assert data["contract"]["requirements"] == []
            assert data["selected_revision"] is None
    assert_no_execution(store)


@pytest.mark.parametrize("intent", INVALID, ids=INVALID_IDS)
def test_manual_api_invalid_intent_refuses_before_creation(local, intent):
    app, store = local
    with TestClient(app) as client:
        reply = client.post("/workspace/manual-sessions", json={"name": "Clip", "intent": intent})
    assert reply.status_code == 422
    assert store.list_sessions("alice") == []
    assert_no_execution(store)


def test_local_sdk_retains_intent_through_actual_http_route(local, tmp_path):
    app, store = local

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:8765", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            schema = tools["create_project"].input_schema
            assert "intent" in schema["properties"]
            assert "intent" not in schema.get("required", [])
            assert tools["create_project"].annotations.read_only_hint is False
            for arguments, expected in (
                ({}, "Manual engineering design"),
                ({"name": "Clip"}, "Clip"),
                ({"name": "Clip", "intent": None}, "Clip"),
                ({"name": "Clip", "intent": INTENT}, INTENT),
                ({"name": "Clip", "intent": MAX_INTENT}, MAX_INTENT),
            ):
                reply = await client.call_tool("create_project", arguments)
                assert not reply.is_error
                assert reply.structured_content["prompt"] == expected
                assert store.get(reply.structured_content["id"], "alice")["prompt"] == expected
                readback = await client.call_tool(
                    "get_project", {"project_id": reply.structured_content["id"]}
                )
                assert not readback.is_error
                assert readback.structured_content["prompt"] == expected
            count = len(store.list_sessions("alice"))
            for intent in INVALID:
                reply = await client.call_tool("create_project", {"name": "Clip", "intent": intent})
                assert reply.is_error
                assert len(store.list_sessions("alice")) == count

    asyncio.run(scenario())
    assert_no_execution(store)


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "readonly"}:
            return None
        return AccessToken(
            token=token,
            client_id="intent-test",
            subject=token,
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": ISSUER},
        )


@pytest.fixture
def remote(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "store"), Executor())
    monkeypatch.setattr(service, "start", lambda: None)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service.store


def remote_call(client, token, name, arguments):
    reply = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + token,
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert reply.status_code == 200
    return reply.json()["result"]


def test_remote_jsonrpc_retains_intent_title_fallback_and_fresh_owner(remote):
    client, store = remote
    for arguments, expected in (
        ({"title": " Clip "}, "Clip"),
        ({"title": "Clip", "intent": None}, "Clip"),
        ({"title": "Clip", "intent": INTENT}, INTENT),
        ({"title": "Clip", "intent": MAX_INTENT}, MAX_INTENT),
    ):
        reply = remote_call(client, "alice", "create_project", arguments)
        assert not reply.get("isError")
        data = reply["structuredContent"]
        assert data["prompt"] == expected
        assert store.get(data["id"], "alice")["prompt"] == expected
        assert data["contract"] is None
        denied = remote_call(client, "bob", "get_project", {"project_id": data["id"]})
        assert denied["isError"]
        assert not denied.get("structuredContent")
    denied = remote_call(client, "readonly", "create_project", {"title": "Clip", "intent": INTENT})
    assert denied["isError"]
    assert store.list_sessions("readonly") == []
    assert_no_execution(store)


@pytest.mark.parametrize("intent", INVALID, ids=INVALID_IDS)
def test_remote_jsonrpc_invalid_intent_refuses_before_creation(remote, intent):
    client, store = remote
    reply = remote_call(client, "alice", "create_project", {"title": "Clip", "intent": intent})
    assert reply["isError"]
    assert store.list_sessions("alice") == []
    assert_no_execution(store)
