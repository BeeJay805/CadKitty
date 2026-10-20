"""Real HTTP and MCP dispatch with worker startup disabled; no CAD or model calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_operation_authoring_transport import contract, document, plan
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
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store

RID = "a" * 32


def seed(service, owner):
    frozen = Contract.model_validate(contract())
    sid = service.store.create(
        owner, "A block 10 mm wide", frozen.model_dump(), authoring_mode="native_operations"
    )["id"]
    binding = service.store.set_design_plan(sid, owner, plan(), expected_plan_version=0)
    source = DesignDocument.model_validate(document())
    directory = service.store.directory(sid, RID)
    directory.mkdir(parents=True)
    data = source.model_dump_json().encode()
    program = b"# Transport source fixture; never executed.\n"
    (directory / "document.json").write_bytes(data)
    (directory / "program.py").write_bytes(program)
    measurement = Measurement(
        revision_id=RID,
        contract_hash=frozen.fingerprint(),
        engine_fingerprint="f" * 64,
        geometry_valid=False,
        artifact_hashes={
            "document.json": hashlib.sha256(data).hexdigest(),
            "program.py": hashlib.sha256(program).hexdigest(),
        },
    )
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "code": program.decode(),
            "contract": frozen.model_dump(),
            "document": source.model_dump(mode="json"),
            "measurement": measurement.model_dump(),
            "operation_record": {"plan_hash": binding["plan_hash"], "plan_snapshot": binding},
        },
    )
    service.store.update(sid, selected_revision=RID, accepted_revision=RID)
    return sid


def delta(basis):
    return {
        "basis": basis,
        "changes": [
            {
                "append": [
                    {
                        "id": "hole",
                        "name": "Hole cutter",
                        "kind": "cylinder",
                        "params": {"radius": 1, "height": 6},
                    },
                    {
                        "id": "finished",
                        "name": "Cut result",
                        "kind": "cut",
                        "inputs": ["body", "hole"],
                    },
                ],
                "output": "finished",
            }
        ],
    }


@pytest.fixture(params=["http", "local_mcp", "remote_mcp"])
def transport(request, tmp_path, monkeypatch):
    remote = request.param == "remote_mcp"
    owner = "alice" if remote else "local-owner"
    service = WorkspaceService(
        Store(tmp_path / "store"), Executor("docker" if remote else "trusted_local")
    )
    if remote:
        service.executor._typed_container_verified = True  # Admission fixture, not qualification.
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    factory = service.client_factory
    sid = seed(service, owner)
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

            yield request.param, service, owner, sid, starts, invoke, factory, client
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
            if tool == "get_edit_basis":
                method, path, body = "GET", f"/workspace/sessions/{project}/edit-basis", None
            elif tool == "feature_edit":
                method, path = "POST", f"/workspace/sessions/{project}/feature-edit"
                body = {key: value for key, value in arguments.items() if key != "project_id"}
            elif tool == "plan_patch":
                method, path, body = (
                    "POST",
                    f"/workspace/sessions/{project}/plan-patch",
                    arguments["patch"],
                )
            else:
                method, path, body = "GET", f"/workspace/sessions/{project}", None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as client:
                reply = await client.request(method, path, json=body)
                return reply.is_success, reply.json(), reply.text

        return asyncio.run(scenario())

    yield request.param, service, owner, sid, starts, invoke, factory, None


def basis(transport):
    _, _, _, sid, _, invoke, _, _ = transport
    ok, value, text = invoke("get_edit_basis", {"project_id": sid})
    assert ok, text
    return value


def test_owned_basis_is_read_only_on_all_surfaces(transport):
    _, service, owner, _, starts, _, factory, _ = transport
    before = snapshot(service.store)
    value = basis(transport)
    assert value["revision_id"] == RID
    assert value["plan_version"] == 1
    assert value["input_version"] == 0
    assert snapshot(service.store) == before
    assert starts == []
    assert service.client_factory is factory
    assert service.store.edit_basis(transport[3], owner).model_dump() == value


def test_feature_delta_admission_retains_contract_and_reports_unapplied(transport):
    surface, service, owner, sid, starts, invoke, factory, _ = transport
    original = service.store.session_record(sid, owner)
    ok, value, text = invoke(
        "feature_edit", {"project_id": sid, "delta": delta(basis(transport)), "max_seconds": 25}
    )
    assert ok, text
    assert value["applied"] is False
    assert value["state"] == "queued"
    assert starts == [True]
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (value["job_id"],)).fetchone()[0]
        )
    assert payload["max_seconds"] == 25
    assert payload["document"]["outputs"] == ["finished"]
    assert [feature["id"] for feature in payload["document"]["features"]] == [
        "body",
        "hole",
        "finished",
    ]
    assert payload["contract"] == original["contract"]
    assert service.store.session_record(sid, owner)["accepted_revision"] == RID
    assert len(service.store.revisions(sid)) == 1
    assert service.client_factory is factory
    assert service.executor.mode == ("docker" if surface == "remote_mcp" else "trusted_local")


def test_plan_patch_applies_only_intent_and_starts_no_worker(transport):
    _, service, owner, sid, starts, invoke, factory, _ = transport
    patch = {"basis": basis(transport), "verification": ["Measure the saved output"]}
    ok, value, text = invoke("plan_patch", {"project_id": sid, "patch": patch})
    assert ok, text
    assert value["applied"] is True and value["plan_applied"] is True
    assert value["cad_applied"] is False
    assert value["plan_version"] == 2
    saved = service.store.session_record(sid, owner)
    assert saved["selected_revision"] == saved["accepted_revision"] == RID
    assert saved["design_plan"]["verification"] == patch["verification"]
    assert len(service.store.revisions(sid)) == 1
    assert starts == []
    assert service.client_factory is factory


@pytest.mark.parametrize("tool", ["feature_edit", "plan_patch"])
def test_stale_bases_reject_without_mutation_or_startup(transport, tool):
    _, service, _, sid, starts, invoke, _, _ = transport
    stale = deepcopy(basis(transport))
    stale["input_version"] = 1
    arguments = (
        {"project_id": sid, "delta": delta(stale)}
        if tool == "feature_edit"
        else {"project_id": sid, "patch": {"basis": stale, "verification": ["New intent"]}}
    )
    before = snapshot(service.store)
    ok, _, text = invoke(tool, arguments)
    assert not ok and "basis" in text.lower(), text
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize("seconds", [0, True])
def test_feature_budget_validation_runs_before_dispatch(transport, seconds):
    _, service, _, sid, starts, invoke, _, _ = transport
    arguments = {"project_id": sid, "delta": delta(basis(transport)), "max_seconds": seconds}
    before = snapshot(service.store)
    ok, _, text = invoke("feature_edit", arguments)
    assert not ok, text
    assert snapshot(service.store) == before
    assert starts == []


def test_foreign_owner_is_refused_before_read_or_edit(transport):
    _, service, _, _, starts, invoke, _, _ = transport
    foreign = service.store.create("other-owner", "Foreign owned fixture", Contract().model_dump())[
        "id"
    ]
    before = snapshot(service.store)
    ok, _, text = invoke("get_edit_basis", {"project_id": foreign})
    assert not ok, text
    assert snapshot(service.store) == before
    assert starts == []


def test_unqualified_docker_refuses_feature_edit(transport):
    surface, service, _, sid, starts, invoke, _, _ = transport
    if surface != "remote_mcp":
        pytest.skip("Remote isolated-worker qualification boundary")
    current = basis(transport)
    service.executor._typed_container_verified = False
    before = snapshot(service.store)
    ok, _, text = invoke("feature_edit", {"project_id": sid, "delta": delta(current)})
    assert not ok and "not qualified" in text, text
    assert snapshot(service.store) == before
    assert starts == []


def test_remote_readonly_scopes_and_budget_cap_and_receipt_readback(transport):
    surface, service, _, sid, starts, invoke, _, _ = transport
    if surface != "remote_mcp":
        pytest.skip("OAuth and hosted budget policy")
    ok, current, text = invoke("get_edit_basis", {"project_id": sid}, token="readonly")
    assert ok, text
    before = snapshot(service.store)
    for tool, arguments in [
        ("feature_edit", {"project_id": sid, "delta": delta(current)}),
        (
            "plan_patch",
            {"project_id": sid, "patch": {"basis": current, "verification": ["Review"]}},
        ),
    ]:
        assert not invoke(tool, arguments, token="readonly")[0]
    assert not invoke(
        "feature_edit", {"project_id": sid, "delta": delta(current), "max_seconds": 601}
    )[0]
    assert snapshot(service.store) == before
    assert starts == []
    receipt = {"action": "feature_edit", "applied": False, "code": "fixture_rejection"}
    service.store.update(sid, last_edit_result=receipt)
    ok, state, text = invoke("get_project", {"project_id": sid})
    assert ok and state["last_edit_result"] == receipt, text
