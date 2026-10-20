"""Real HTTP and MCP admission boundaries; no model, native worker or CAD runs."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_input_application import INSTRUCTION, PROMPT, target
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
from forge_mind.workspace.input_application import pending
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store


@pytest.fixture(params=["http", "local_mcp", "remote_mcp"])
def boundary(request, tmp_path, monkeypatch):
    remote = request.param == "remote_mcp"
    owner = "alice" if remote else "local-owner"
    service = WorkspaceService(
        Store(tmp_path / "store"), Executor("docker" if remote else "trusted_local")
    )
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    sid = service.store.create(
        owner, PROMPT, target().model_dump(), authoring_mode="native_operations"
    )["id"]
    if remote:
        app = remote_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            starts.clear()

            def invoke(tool, arguments):
                reply = result(call(client, "alice", tool, arguments))
                return (
                    not reply.get("isError", False),
                    reply.get("structuredContent"),
                    json.dumps(reply),
                )

            yield service, owner, sid, starts, invoke
        return

    app = local_app(local=True, service=service)

    def invoke(tool, arguments):
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
            sid = arguments["project_id"]
            method, suffix, body = (
                ("GET", "overview", None)
                if tool == "get_project"
                else ("GET", "edit-basis", None)
                if tool == "get_edit_basis"
                else ("POST", "input-application/abandon", arguments["request"])
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as client:
                reply = await client.request(
                    method, f"/workspace/sessions/{sid}/{suffix}", json=body
                )
                return reply.is_success, reply.json(), reply.text

        return asyncio.run(scenario())

    yield service, owner, sid, starts, invoke


def interrupted(service, owner, sid):
    jid = service.store.enqueue(sid, owner, {"instruction": INSTRUCTION, "max_seconds": 60})
    job = service.store.claim()
    assert job and job["id"] == jid
    service.store.finish(jid, sid, "budget_exhausted")
    return job


def job_count(service):
    with service.store.connect() as con:
        return con.execute("SELECT count(*) FROM jobs").fetchone()[0]


def test_owner_can_read_pending_identity_and_abandon_without_work_or_target_change(boundary):
    service, owner, sid, starts, invoke = boundary
    job = interrupted(service, owner, sid)
    before = service.store.session_record(sid, owner)
    ok, overview, _ = invoke("get_project", {"project_id": sid})
    assert ok
    records = overview["input_application"]["pending"]
    assert len(records) == 1 and records[0]["input_id"] == job["input_application_id"]
    ok, basis, _ = invoke("get_edit_basis", {"project_id": sid})
    assert ok
    ok, receipt, _ = invoke(
        "abandon_input",
        {
            "project_id": sid,
            "request": {
                "input_id": records[0]["input_id"],
                "source_hash": records[0]["source_hash"],
                "basis": basis,
            },
        },
    )
    assert ok and receipt["status"] == "abandoned" and receipt["input_id"] == records[0]["input_id"]
    saved = service.store.session_record(sid, owner)
    assert saved["contract"] == before["contract"] == target().model_dump()
    assert saved.get("selected_revision") == before.get("selected_revision")
    assert saved.get("accepted_revision") == before.get("accepted_revision")
    assert saved["instructions"] == []
    assert saved["abandoned_input_applications"]
    ok, overview, _ = invoke("get_project", {"project_id": sid})
    assert ok and overview["input_application"]["pending"] == []
    assert starts == [] and job_count(service) == 1 and service.store.revisions(sid) == []


@pytest.mark.parametrize("bad", ["identity", "basis", "source", "extra"])
def test_stale_or_forged_abandonment_is_atomic(boundary, bad):
    service, owner, sid, starts, invoke = boundary
    job = interrupted(service, owner, sid)
    item = pending(service.store.session_record(sid, owner))
    assert item is not None
    body = {
        "input_id": job["input_application_id"],
        "source_hash": item["source_hash"],
        "basis": service.store.edit_basis(sid, owner).model_dump(),
    }
    if bad == "identity":
        body["input_id"] = "f" * 32
    elif bad == "basis":
        body["basis"]["source_fingerprint"] = "f" * 64
    elif bad == "source":
        body["source_hash"] = "f" * 64
    else:
        body["contract_application"] = {"pending": None}
    before = copy.deepcopy(service.store.session_record(sid, owner))
    ok, _, _ = invoke("abandon_input", {"project_id": sid, "request": body})
    assert not ok and service.store.session_record(sid, owner) == before
    assert starts == [] and job_count(service) == 1 and service.store.revisions(sid) == []


def test_abandonment_cannot_cross_owner_or_active_job_boundary(boundary):
    service, owner, sid, starts, invoke = boundary
    other = service.store.create("other-owner", PROMPT, target().model_dump())["id"]
    ok, _, _ = invoke("get_project", {"project_id": other})
    assert not ok
    jid = service.store.enqueue(sid, owner, {"instruction": INSTRUCTION})
    job = service.store.claim()
    assert job and job["id"] == jid
    item = pending(service.store.session_record(sid, owner))
    assert item is not None
    body = {
        "input_id": job["input_application_id"],
        "source_hash": item["source_hash"],
        "basis": service.store.edit_basis(sid, owner).model_dump(),
    }
    before = copy.deepcopy(service.store.session_record(sid, owner))
    ok, _, _ = invoke("abandon_input", {"project_id": sid, "request": body})
    assert not ok and service.store.session_record(sid, owner) == before
    assert starts == [] and job_count(service) == 1
