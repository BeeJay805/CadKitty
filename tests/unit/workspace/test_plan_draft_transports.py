"""Real owned draft dispatch across HTTP and both MCP paths, without CAD/model work."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_operation_authoring_transport import contract, plan
from tests.unit.workspace.test_scoped_authoring_transport import (
    ISSUER,
    RESOURCE,
    Verifier,
    call,
    result,
)

from forge_mind.server.workspace_app import create_app as local_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.models import Contract
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store


def database_state(store):
    with store.connect() as con:
        return {
            table: [tuple(row) for row in con.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in ("sessions", "jobs", "revisions", "engineering_inputs", "plan_drafts")
        }


@pytest.fixture(params=["http", "local_mcp", "remote_mcp"])
def transport(request, tmp_path, monkeypatch):
    remote = request.param == "remote_mcp"
    owner = "alice" if remote else "local-owner"
    service = WorkspaceService(
        Store(tmp_path / "store"), Executor("docker" if remote else "trusted_local")
    )
    # Draft intent requires no CAD qualification. No worker is started in these tests.
    service.executor._typed_container_verified = False
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    factory = service.client_factory
    sid = service.store.create(
        owner, "A block 10 mm wide", contract(), authoring_mode="native_operations"
    )["id"]
    if remote:
        app = remote_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            starts.clear()

            def invoke(tool, arguments, token="alice"):
                reply = result(call(client, token, tool, arguments))
                return (
                    not reply.get("isError", False),
                    reply.get("structuredContent"),
                    json.dumps(reply),
                )

            yield request.param, service, owner, sid, starts, invoke, factory
        return
    app = local_app(local=True, service=service)

    def invoke(tool, arguments, token=None):
        async def scenario():
            if request.param == "local_mcp":
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
            project = arguments["project_id"]
            prefix = f"/workspace/sessions/{project}/plan-draft"
            if tool == "read_plan_draft":
                method, path, body, params = "GET", prefix, None, arguments["query"]
            elif tool == "patch_plan_draft":
                method, path, body, params = "POST", prefix, arguments["patch"], None
            else:
                assert tool == "commit_plan_draft"
                method, path, body, params = "POST", prefix + "/commit", arguments["request"], None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as client:
                reply = await client.request(method, path, json=body, params=params)
                return reply.is_success, reply.json(), reply.text

        return asyncio.run(scenario())

    yield request.param, service, owner, sid, starts, invoke, factory


def complete_changes():
    source = plan()
    return {
        "objective": source["objective"],
        "part_ids": source["operation_plan"]["part_ids"],
        "operation_frames": source["operation_plan"]["frames"],
        "operation_steps": source["operation_plan"]["steps"],
    }


def create_draft(transport, changes=None):
    _, service, owner, sid, _, invoke, _ = transport
    patch = {
        "basis": service.store.edit_basis(sid, owner).model_dump(),
        "draft_version": 0,
        "changes": complete_changes() if changes is None else changes,
    }
    ok, value, text = invoke("patch_plan_draft", {"project_id": sid, "patch": patch})
    assert ok, text
    return value


def identity(draft):
    return {key: draft[key] for key in ("basis", "draft_version", "draft_hash")}


def test_draft_read_and_typed_correction_preserve_canonical_state(transport):
    _, service, owner, sid, starts, invoke, factory = transport
    canonical = service.store.session_record(sid, owner)
    draft = create_draft(transport)
    assert draft["status"] == "unadmitted_draft"
    assert draft["plan_applied"] is False and draft["cad_applied"] is False
    before_read = database_state(service.store)
    ok, page, text = invoke(
        "read_plan_draft",
        {"project_id": sid, "query": {"collection": "operation_steps", "record_id": "base"}},
    )
    assert ok, text
    assert page["draft_hash"] == draft["draft_hash"]
    assert [record["id"] for record in page["records"]] == ["base"]
    assert database_state(service.store) == before_read
    patch = {**identity(draft), "changes": {"verification": ["Measure the saved width"]}}
    ok, corrected, text = invoke("patch_plan_draft", {"project_id": sid, "patch": patch})
    assert ok, text
    assert corrected["draft_version"] == draft["draft_version"] + 1
    assert corrected["draft_hash"] != draft["draft_hash"]
    assert service.store.session_record(sid, owner) == canonical
    assert service.store.revisions(sid) == []
    assert starts == []
    assert service.client_factory is factory


def test_commit_admits_only_exact_intent_without_worker_startup(transport):
    _, service, owner, sid, starts, invoke, factory = transport
    draft = create_draft(transport)
    ok, committed, text = invoke(
        "commit_plan_draft", {"project_id": sid, "request": identity(draft)}
    )
    assert ok, text
    assert committed["applied"] is True and committed["plan_applied"] is True
    assert committed["cad_applied"] is False
    saved = service.store.session_record(sid, owner)
    assert saved["plan_hash"] == committed["plan_hash"]
    assert saved["plan_version"] == committed["plan_version"] == 1
    assert saved["contract"] == Contract.model_validate(contract()).model_dump()
    assert saved.get("selected_revision") is None and saved.get("accepted_revision") is None
    assert saved["design_plan"]["operation_plan"]["steps"][0]["id"] == "base"
    assert service.store.revisions(sid) == []
    assert starts == [] and service.client_factory is factory


def test_initial_fragment_is_unadmitted_and_invalid_commit_preserves_draft(transport):
    _, service, owner, sid, starts, invoke, _ = transport
    draft = create_draft(transport, {"objective": "An incomplete block plan"})
    canonical = service.store.session_record(sid, owner)
    ok, _, text = invoke("commit_plan_draft", {"project_id": sid, "request": identity(draft)})
    assert not ok, text
    assert service.store.session_record(sid, owner) == canonical
    ok, retained, text = invoke("read_plan_draft", {"project_id": sid, "query": {}})
    assert ok, text
    assert retained["draft_hash"] == draft["draft_hash"]
    assert retained["draft_version"] == draft["draft_version"]
    assert retained["validation_error"]
    assert service.store.revisions(sid) == [] and starts == []


def test_paging_and_read_queries_do_not_change_draft_or_start_work(transport):
    _, service, _, sid, starts, invoke, _ = transport
    create_draft(transport, {"objective": "Paged plan", "parts": [f"Part {i}" for i in range(75)]})
    before = database_state(service.store)
    ok, first, text = invoke(
        "read_plan_draft", {"project_id": sid, "query": {"collection": "parts", "limit": 50}}
    )
    assert ok and first["records"] == [f"Part {i}" for i in range(50)], text
    ok, second, text = invoke(
        "read_plan_draft",
        {
            "project_id": sid,
            "query": {"collection": "parts", "cursor": first["next_cursor"], "limit": 50},
        },
    )
    assert ok and second["records"] == [f"Part {i}" for i in range(50, 75)], text
    assert second["next_cursor"] is None
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize(
    "query", [{"cursor": -1}, {"limit": 0}, {"limit": 51}, {"collection": "raw_source"}]
)
def test_invalid_read_queries_reject_before_draft_mutation(transport, query):
    _, service, _, sid, starts, invoke, _ = transport
    create_draft(transport)
    before = database_state(service.store)
    ok, _, text = invoke("read_plan_draft", {"project_id": sid, "query": query})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize("tool", ["patch_plan_draft", "commit_plan_draft"])
def test_stale_draft_hash_and_edit_basis_refuse_all_transports(transport, tool):
    _, service, _, sid, starts, invoke, _ = transport
    draft = create_draft(transport)
    for field in ("draft_hash", "basis"):
        request = deepcopy(identity(draft))
        if field == "draft_hash":
            request[field] = "f" * 64
        else:
            request[field]["input_version"] += 1
        if tool == "patch_plan_draft":
            request["changes"] = {"verification": ["Do not apply stale intent"]}
        before = database_state(service.store)
        ok, _, text = invoke(
            tool, {"project_id": sid, "patch" if tool == "patch_plan_draft" else "request": request}
        )
        assert not ok, text
        assert database_state(service.store) == before
    assert starts == []


@pytest.mark.parametrize("tool", ["read_plan_draft", "patch_plan_draft", "commit_plan_draft"])
def test_foreign_owner_is_refused_without_read_or_write(transport, tool):
    _, service, _, _, starts, invoke, _ = transport
    foreign = service.store.create("other-owner", "Foreign plan", contract(), "native_operations")[
        "id"
    ]
    basis = service.store.edit_basis(foreign, "other-owner").model_dump()
    if tool == "read_plan_draft":
        arguments = {"project_id": foreign, "query": {}}
    elif tool == "patch_plan_draft":
        arguments = {
            "project_id": foreign,
            "patch": {"basis": basis, "draft_version": 0, "changes": complete_changes()},
        }
    else:
        arguments = {
            "project_id": foreign,
            "request": {"basis": basis, "draft_version": 1, "draft_hash": "f" * 64},
        }
    before = database_state(service.store)
    ok, _, text = invoke(tool, arguments)
    assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_external_draft_writes_refuse_active_job_but_reads_remain_available(transport):
    _, service, _, sid, starts, invoke, _ = transport
    draft = create_draft(transport)
    service.store.enqueue(sid, transport[2], {"max_seconds": 10})
    before = database_state(service.store)
    ok, _, text = invoke("read_plan_draft", {"project_id": sid, "query": {}})
    assert ok, text
    for tool, field, value in [
        (
            "patch_plan_draft",
            "patch",
            {**identity(draft), "changes": {"verification": ["Wait first"]}},
        ),
        ("commit_plan_draft", "request", identity(draft)),
    ]:
        ok, _, text = invoke(tool, {"project_id": sid, field: value})
        assert not ok and "active" in text.lower(), text
        assert database_state(service.store) == before
    assert starts == []


def test_remote_read_scope_allows_pages_and_refuses_both_writes(transport):
    surface, service, _, sid, starts, invoke, _ = transport
    if surface != "remote_mcp":
        pytest.skip("Remote OAuth scope boundary")
    draft = create_draft(transport)
    before = database_state(service.store)
    ok, _, text = invoke("read_plan_draft", {"project_id": sid, "query": {}}, token="readonly")
    assert ok, text
    for tool, field, value in [
        (
            "patch_plan_draft",
            "patch",
            {**identity(draft), "changes": {"verification": ["No write scope"]}},
        ),
        ("commit_plan_draft", "request", identity(draft)),
    ]:
        ok, _, text = invoke(tool, {"project_id": sid, field: value}, token="readonly")
        assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_local_sdk_advertises_exact_typed_arguments_and_read_write_hints(tmp_path):
    async def scenario():
        server = local_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321",
                tmp_path,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(500, json={"detail": "Must not dispatch"})
                ),
            )
        )
        async with Client(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            for name, field, readonly in [
                ("read_plan_draft", "query", True),
                ("patch_plan_draft", "patch", False),
                ("commit_plan_draft", "request", False),
            ]:
                tool = tools[name]
                assert set(tool.input_schema["properties"]) == {"project_id", field}
                assert tool.annotations.read_only_hint is readonly
                assert "jid" not in tool.input_schema["properties"]

    asyncio.run(scenario())
