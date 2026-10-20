"""Real HTTP/MCP admission checks; worker startup is stubbed and no CAD executes."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_scoped_authoring_transport import (
    ISSUER,
    RESOURCE,
    Verifier,
    call,
    result,
    snapshot,
)

from forge_mind.server.workspace_app import create_app as local_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store


def contract():
    return {
        "requirements": [
            {
                "id": "width",
                "text": "The block is 10 mm wide",
                "source_quote": "10 mm wide",
                "metric": "extent_x",
                "value": 10,
            }
        ]
    }


def plan():
    return {
        "objective": "Construct the requested block",
        "operation_plan": {
            "part_ids": ["main"],
            "frames": [{"id": "datum", "part_id": "main", "description": "Block local frame"}],
            "steps": [
                {
                    "id": "base",
                    "part_id": "main",
                    "feature_ids": ["body"],
                    "requirement_ids": ["width"],
                    "frame_id": "datum",
                    "expected_effect": "A block with the requested width",
                    "inspection_targets": ["Measure extent_x"],
                }
            ],
        },
    }


def document():
    return {
        "features": [
            {"id": "body", "name": "Block", "kind": "box", "params": {"x": 10, "y": 8, "z": 4}}
        ],
        "outputs": ["body"],
    }


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "remote"), Executor())
    service.executor._typed_container_verified = True  # Admission fixture, never execution.
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    app = remote_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service, starts


@pytest.fixture(params=["local_api", "local_sdk", "remote"])
def transport(request, hosted, tmp_path, monkeypatch):
    surface = request.param
    if surface == "remote":
        client, service, starts = hosted

        def invoke(tool, arguments):
            arguments = deepcopy(arguments)
            if tool == "create_project":
                arguments["title"] = arguments.pop("name", "Operation fixture")
            if tool == "submit_document":
                arguments.setdefault("contract", {})
            reply = result(call(client, "alice", tool, arguments))
            return (
                not reply.get("isError", False),
                reply.get("structuredContent"),
                "\n".join(item.get("text", "") for item in reply.get("content", [])),
            )

        return surface, service, "alice", starts, invoke

    service = WorkspaceService(Store(tmp_path / "local"), Executor("trusted_local"))
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    app = local_app(local=True, service=service)

    def invoke(tool, arguments):
        async def scenario():
            if surface == "local_sdk":
                adapter = WorkspaceAdapter(
                    "http://127.0.0.1:4321",
                    tmp_path / "exports",
                    transport=httpx.ASGITransport(app=app),
                )
                async with Client(local_server(adapter)) as sdk:
                    reply = await sdk.call_tool(tool, arguments)
                    return (
                        not reply.is_error,
                        reply.structured_content,
                        "\n".join(item.text for item in reply.content if item.type == "text"),
                    )
            sid = arguments.get("project_id")
            body = {key: value for key, value in arguments.items() if key != "project_id"}
            if tool == "create_project":
                path = "/workspace/manual-sessions"
            elif tool == "set_design_plan":
                path = f"/workspace/sessions/{sid}/design-plan"
                body["expected_current_revision"] = body.pop("revision_id", None)
                body["expected_input_version"] = service.store.engineering(sid)["version"]
            elif tool == "submit_document":
                path = f"/workspace/sessions/{sid}/document"
                body["expected_current_revision"] = body.get("revision_id")
                body["expected_input_version"] = service.store.engineering(sid)["version"]
            else:
                raise AssertionError(tool)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as api:
                reply = await api.post(path, json=body)
                return reply.status_code < 300, reply.json(), reply.text

        return asyncio.run(scenario())

    return surface, service, "local-owner", starts, invoke


def create_native(transport):
    _, service, owner, starts, invoke = transport
    ok, created, text = invoke(
        "create_project", {"name": "Operation fixture", "authoring_mode": "native_operations"}
    )
    assert ok, text
    sid = created["id"]
    assert service.store.session_record(sid, owner)["authoring_mode"] == "native_operations"
    assert starts == []
    return sid


def initialize(transport, sid):
    _, _, _, _, invoke = transport
    ok, binding, text = invoke(
        "set_design_plan",
        {"project_id": sid, "plan": plan(), "contract": contract(), "expected_plan_version": 0},
    )
    assert ok, text
    assert binding["plan_version"] == 1
    return binding


def test_native_requires_plan_and_contract_before_first_build(transport):
    _, service, owner, starts, invoke = transport
    sid = create_native(transport)
    before = snapshot(service.store)
    ok, _, text = invoke("submit_document", {"project_id": sid, "document": document()})
    assert not ok and "plan" in text.lower(), text
    assert snapshot(service.store) == before
    ok, _, text = invoke(
        "set_design_plan", {"project_id": sid, "plan": plan(), "expected_plan_version": 0}
    )
    assert not ok and "contract" in text.lower(), text
    assert snapshot(service.store) == before
    binding = initialize(transport, sid)
    persisted = service.store.session_record(sid, owner)
    assert persisted["plan_hash"] == binding["plan_hash"]
    assert persisted["plan_contract_hash"] == binding["contract_hash"]
    assert persisted["contract"] == binding["contract"]
    assert starts == []


def test_native_admission_binds_exact_saved_plan_without_worker_execution(transport):
    _, service, _, starts, invoke = transport
    sid = create_native(transport)
    binding = initialize(transport, sid)
    ok, queued, text = invoke(
        "submit_document", {"project_id": sid, "document": document(), "contract": contract()}
    )
    assert ok and queued["state"] == "queued", text
    with service.store.connect() as con:
        row = con.execute(
            "SELECT state,payload FROM jobs WHERE id=?", (queued["job_id"],)
        ).fetchone()
    assert row["state"] == "queued"
    assert json.loads(row["payload"])["plan_snapshot"] == binding
    assert starts == [True]
    before = snapshot(service.store)
    ok, _, text = invoke(
        "set_design_plan", {"project_id": sid, "plan": plan(), "expected_plan_version": 1}
    )
    assert not ok and "active" in text.lower(), text
    assert snapshot(service.store) == before
    assert starts == [True]


@pytest.mark.parametrize("stale", ["plan_version", "revision"])
def test_stale_plan_writes_preserve_the_saved_plan(transport, stale):
    _, service, _, starts, invoke = transport
    sid = create_native(transport)
    initialize(transport, sid)
    before = snapshot(service.store)
    arguments = {
        "project_id": sid,
        "plan": plan(),
        "expected_plan_version": 0 if stale == "plan_version" else 1,
    }
    if stale == "revision":
        arguments["revision_id"] = "a" * 32
    ok, _, text = invoke("set_design_plan", arguments)
    assert not ok and "changed" in text.lower(), text
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize("placement", ["suppressed", "unused_definition"])
def test_native_python_refusal_crosses_every_transport(transport, placement):
    _, service, _, starts, invoke = transport
    sid = create_native(transport)
    initialize(transport, sid)
    source = document()
    python = {
        "id": "hidden",
        "name": "Hidden Python",
        "kind": "python",
        "code": "result = b.Box(1, 2, 3)",
    }
    if placement == "suppressed":
        source["features"].append({**python, "suppressed": True})
    else:
        source = {
            "assembly": {
                "parts": [
                    {
                        "id": "main",
                        "name": "Block",
                        "features": source["features"],
                        "output": "body",
                    },
                    {
                        "id": "retained",
                        "name": "Unused definition",
                        "features": [python],
                        "output": "hidden",
                    },
                ],
                "roots": [{"id": "main", "definition": "main"}],
            }
        }
    before = snapshot(service.store)
    ok, _, text = invoke(
        "submit_document", {"project_id": sid, "document": source, "contract": contract()}
    )
    assert not ok and "hidden" in text and "native_operations" in text, text
    assert snapshot(service.store) == before
    assert starts == []


def test_native_rejects_per_document_contract_replacement(transport):
    _, service, _, starts, invoke = transport
    sid = create_native(transport)
    initialize(transport, sid)
    changed = contract()
    changed["requirements"][0]["value"] = 12
    before = snapshot(service.store)
    ok, _, text = invoke(
        "submit_document", {"project_id": sid, "document": document(), "contract": changed}
    )
    assert not ok and "frozen" in text.lower(), text
    assert snapshot(service.store) == before
    assert starts == []


def test_omitted_mode_preserves_legacy_create_serialization(transport):
    _, service, owner, starts, invoke = transport
    ok, created, text = invoke("create_project", {"name": "Legacy fixture"})
    assert ok, text
    persisted = service.store.session_record(created["id"], owner)
    assert "authoring_mode" not in persisted
    assert service.store.authoring_mode(created["id"]) == "hybrid"
    assert starts == []


def test_invalid_explicit_mode_cannot_create_project(transport):
    _, service, _, starts, invoke = transport
    before = snapshot(service.store)
    ok, _, text = invoke("create_project", {"name": "Invalid fixture", "authoring_mode": "native"})
    assert not ok and "authoring_mode" in text, text
    assert snapshot(service.store) == before
    assert starts == []


def test_legacy_hybrid_python_admission_needs_no_operation_plan(transport):
    _, service, _, starts, invoke = transport
    ok, created, text = invoke("create_project", {"name": "Legacy Python fixture"})
    assert ok, text
    source = {
        "features": [
            {"id": "body", "name": "Program", "kind": "python", "code": "result = b.Box(1, 2, 3)"}
        ],
        "outputs": ["body"],
    }
    ok, queued, text = invoke("submit_document", {"project_id": created["id"], "document": source})
    assert ok and queued["state"] == "queued", text
    with service.store.connect() as con:
        row = con.execute("SELECT payload FROM jobs WHERE id=?", (queued["job_id"],)).fetchone()
    assert "plan_snapshot" not in json.loads(row["payload"])
    assert starts == [True]


def test_local_http_stale_engineering_input_version_refuses_plan(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    app = local_app(local=True, service=service)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
        ) as api:
            created = await api.post(
                "/workspace/manual-sessions", json={"authoring_mode": "native_operations"}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["id"]
            service.store.save_engineering(
                sid, "local-owner", 0, service.store.engineering(sid)["inputs"]
            )
            before = snapshot(service.store)
            reply = await api.post(
                f"/workspace/sessions/{sid}/design-plan",
                json={
                    "plan": plan(),
                    "contract": contract(),
                    "expected_plan_version": 0,
                    "expected_input_version": 0,
                },
            )
            assert reply.status_code == 409 and "inputs changed" in reply.text, reply.text
            assert snapshot(service.store) == before

    asyncio.run(scenario())
    assert starts == []


@pytest.mark.parametrize("token", [None, "bob", "readonly"])
@pytest.mark.parametrize("tool", ["set_design_plan", "submit_document"])
def test_remote_owner_and_write_scope_protect_plan_and_geometry(hosted, token, tool):
    client, service, starts = hosted
    sid = service.store.create("alice", "Protected", None, authoring_mode="native_operations")["id"]
    arguments = {"project_id": sid}
    if tool == "set_design_plan":
        arguments.update(plan=plan(), contract=contract(), expected_plan_version=0)
    else:
        arguments.update(document=document(), contract=contract())
    before = snapshot(service.store)
    reply = call(client, token, tool, arguments)
    assert reply.status_code in (401, 403) or result(reply).get("isError", False), reply.text
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize("surface", ["local_api", "local_sdk"])
def test_local_plan_write_refuses_another_owner(surface, hosted, tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "foreign"), Executor("trusted_local"))
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    sid = service.store.create(
        "another-owner", "Protected", None, authoring_mode="native_operations"
    )["id"]
    app = local_app(local=True, service=service)
    before = snapshot(service.store)

    async def scenario():
        arguments = {
            "project_id": sid,
            "plan": plan(),
            "contract": contract(),
            "expected_plan_version": 0,
        }
        if surface == "local_sdk":
            adapter = WorkspaceAdapter(
                "http://127.0.0.1:4321",
                tmp_path / "exports",
                transport=httpx.ASGITransport(app=app),
            )
            async with Client(local_server(adapter)) as sdk:
                reply = await sdk.call_tool("set_design_plan", arguments)
                assert reply.is_error
        else:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as api:
                reply = await api.post(
                    f"/workspace/sessions/{sid}/design-plan",
                    json={key: value for key, value in arguments.items() if key != "project_id"},
                )
                assert reply.status_code in (404, 409)

    asyncio.run(scenario())
    assert snapshot(service.store) == before
    assert starts == []
