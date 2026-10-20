"""Owned question reads and atomic answer admission on HTTP and both MCP routes.

Only queue admission runs. Worker startup is stubbed, so no CAD or model executes.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_operation_authoring_transport import contract
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
from forge_mind.workspace.intent_attribution import IntentAttribution
from forge_mind.workspace.intent_attribution_service import record_proposal
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.models import Contract
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store

SOURCE = "Make a holder for a block. It is 10 mm wide."
RAW_ANSWER = "  The block is 10 mm wide. Give the holder clearance.\n"


def attribution():
    return IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "holder",
                    "label": "Holder",
                    "role": "designed",
                    "source_quote": "holder",
                },
                {
                    "id": "block",
                    "label": "Block",
                    "role": "reference",
                    "source_quote": "block",
                },
            ],
            "facts": [
                {
                    "id": "width_fact",
                    "candidate_subject_ids": ["holder", "block"],
                    "property": "width",
                    "value": 10.0,
                    "unit": "mm",
                    "source_quote": "10 mm wide",
                    "requirement_ids": ["width"],
                }
            ],
            "requirement_subjects": {"width": ["holder", "block"]},
            "issues": [
                {
                    "id": "width_subject",
                    "question": "Does the width describe the block or the holder?",
                    "reason": "The measurement subject is ambiguous.",
                    "fact_ids": ["width_fact"],
                    "requirement_ids": ["width"],
                }
            ],
        }
    )


def database_state(store):
    with store.connect() as con:
        return {
            table: [tuple(row) for row in con.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in ("sessions", "jobs", "revisions", "engineering_inputs", "plan_drafts")
        }


def saved_payload(store, jid):
    with store.connect() as con:
        return json.loads(con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0])


def propose(service, sid, owner, *, finish=True):
    jid = service.store.enqueue(sid, owner, {"max_seconds": 10})
    claimed = service.store.claim()
    assert claimed is not None and claimed["id"] == jid
    record = record_proposal(
        service.store,
        sid,
        owner,
        jid,
        attribution(),
        Contract.model_validate(contract()),
        SOURCE,
        service.store.edit_basis(sid, owner),
    )
    if finish:
        service.store.finish(jid, sid, "needs_input")
    return record


@pytest.fixture(params=["http", "local_mcp", "remote_mcp"])
def transport(request, tmp_path, monkeypatch):
    remote = request.param == "remote_mcp"
    owner = "alice" if remote else "local-owner"
    service = WorkspaceService(
        Store(tmp_path / "store"), Executor("docker" if remote else "trusted_local")
    )
    service.executor._typed_container_verified = True  # Admission fixture only.
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    factory = service.client_factory
    sid = service.store.create(owner, SOURCE, contract(), authoring_mode="native_operations")["id"]
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
                    tool_arguments = (
                        {"kind": arguments["query"]} if tool == "get_schema" else arguments
                    )
                    reply = await sdk.call_tool(tool, tool_arguments)
                    return (
                        not reply.is_error,
                        reply.structured_content,
                        "\n".join(item.text for item in reply.content if item.type == "text"),
                    )
            params = None
            if tool == "get_schema":
                method, path, body = "GET", "/workspace/authoring-schema", None
                params = arguments
            elif tool == "get_intent_attribution":
                prefix = f"/workspace/sessions/{arguments['project_id']}/intent-attribution"
                method, path, body = "GET", prefix, None
            else:
                assert tool == "resolve_intent"
                prefix = f"/workspace/sessions/{arguments['project_id']}/intent-attribution"
                method, path = "POST", prefix + "/resolve"
                body = {key: value for key, value in arguments.items() if key != "project_id"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as client:
                reply = await client.request(method, path, json=body, params=params)
                return reply.is_success, reply.json(), reply.text

        return asyncio.run(scenario())

    yield request.param, service, owner, sid, starts, invoke, factory


def resolution(record):
    return {
        "expected_version": record["version"],
        "draft_hash": record["draft_hash"],
        "answers": [{"issue_id": "width_subject", "answer": RAW_ANSWER}],
    }


def test_reader_preserves_full_record_and_starts_no_work(transport):
    _, service, owner, sid, starts, invoke, factory = transport
    before = database_state(service.store)
    ok, empty, text = invoke("get_intent_attribution", {"project_id": sid})
    assert ok and empty["record"] is None, text
    assert database_state(service.store) == before
    record = propose(service, sid, owner)
    before = database_state(service.store)
    ok, read, text = invoke("get_intent_attribution", {"project_id": sid})
    assert ok and read["record"] == record, text
    assert read["record"]["source_text"] == SOURCE
    assert read["record"]["attribution"] == attribution().model_dump()
    assert database_state(service.store) == before
    assert starts == [] and service.client_factory is factory


def test_resolution_queues_raw_answers_without_approving_or_rewriting_contract(transport):
    kind, service, owner, sid, starts, invoke, factory = transport
    record = propose(service, sid, owner)
    before = service.store.session_record(sid, owner)
    request = resolution(record)
    ok, queued, text = invoke("resolve_intent", {"project_id": sid, "resolution": request})
    assert ok and queued["state"] == "queued", text
    assert queued["id"] == sid
    payload = saved_payload(service.store, queued["job_id"])
    assert "intent_resolution" not in payload
    assert RAW_ANSWER in payload["instruction"]
    # Local transports preserve the actual finite policy when the answer omits a budget.
    prior_seconds = saved_payload(service.store, before["job_id"])["max_seconds"]
    assert prior_seconds == 10
    assert payload["max_seconds"] == (300 if kind == "remote_mcp" else prior_seconds)
    assert payload["max_actions"] is None
    saved = service.store.session_record(sid, owner)
    assert saved["intent_attribution"]["status"] == "answered"
    assert saved["intent_attribution"]["resolutions"] == request["answers"]
    assert saved["intent_attribution"]["attribution"] == record["attribution"]
    assert saved["contract"] == before["contract"]
    assert saved.get("design_plan") == before.get("design_plan")
    assert saved.get("selected_revision") == before.get("selected_revision")
    assert saved.get("accepted_revision") == before.get("accepted_revision")
    assert service.store.revisions(sid) == []
    assert starts == [True] and service.client_factory is factory


@pytest.mark.parametrize("field", ["expected_version", "draft_hash", "issue_id"])
def test_stale_identity_and_unknown_answers_are_atomic(transport, field):
    _, service, owner, sid, starts, invoke, _ = transport
    request = resolution(propose(service, sid, owner))
    if field == "expected_version":
        request[field] += 1
    elif field == "draft_hash":
        request[field] = "f" * 64
    else:
        request["answers"][0][field] = "unknown_question"
    before = database_state(service.store)
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": request})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"answers": []},
        {"answers": [{"issue_id": "width_subject", "answer": " \t\n"}]},
        {"expected_version": True},
        {"confirmed": True},
        {"instruction": "FORGE_UNTRUSTED_NESTED_INSTRUCTION_48210"},
        {
            "answers": [
                {"issue_id": "width_subject", "answer": RAW_ANSWER},
                {"issue_id": "width_subject", "answer": "A duplicate answer"},
            ]
        },
    ],
)
def test_invalid_or_author_confirmation_payload_cannot_queue(transport, invalid):
    _, service, owner, sid, starts, invoke, _ = transport
    request = {**resolution(propose(service, sid, owner)), **invalid}
    before = database_state(service.store)
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": request})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize("budget", [{"max_seconds": 0}, {"max_actions": 0}, {"max_actions": True}])
def test_invalid_budgets_reject_before_answer_mutation(transport, budget):
    _, service, owner, sid, starts, invoke, _ = transport
    request = resolution(propose(service, sid, owner))
    before = database_state(service.store)
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": request, **budget})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_resolution_cannot_supply_a_competing_instruction(transport):
    kind, service, owner, sid, starts, invoke, _ = transport
    request = resolution(propose(service, sid, owner))
    sentinel = "FORGE_UNTRUSTED_OUTER_INSTRUCTION_48210"
    before = database_state(service.store)
    canonical = service.store.session_record(sid, owner)
    ok, queued, text = invoke(
        "resolve_intent",
        {"project_id": sid, "resolution": request, "instruction": sentinel},
    )
    if kind == "http":
        assert not ok, text
    # The installed MCP SDK ignores unknown outer arguments before calling a
    # tool. Rejection or safe omission must both prevent instruction injection.
    # The nested IntentResolution remains closed and rejects competing fields.
    if not ok:
        assert database_state(service.store) == before and starts == []
        return
    assert kind in {"local_mcp", "remote_mcp"}
    assert queued["id"] == sid and queued["state"] == "queued", text
    payload = saved_payload(service.store, queued["job_id"])
    assert payload["instruction"] == f"User clarification (width_subject): {RAW_ANSWER}"
    saved = service.store.session_record(sid, owner)
    assert sentinel not in json.dumps(payload)
    assert sentinel not in json.dumps(saved)
    assert saved["intent_attribution"]["resolutions"] == request["answers"]
    assert saved["intent_attribution"]["status"] == "answered"
    for field in ("contract", "design_plan", "plan_hash", "selected_revision", "accepted_revision"):
        assert saved.get(field) == canonical.get(field)
    assert saved.get("accepted_revision") is None
    assert service.store.revisions(sid) == []
    assert len(database_state(service.store)["jobs"]) == len(before["jobs"]) + 1
    assert starts == [True]


def test_explicit_budgets_reach_existing_queue(transport):
    _, service, owner, sid, starts, invoke, _ = transport
    request = resolution(propose(service, sid, owner))
    ok, queued, text = invoke(
        "resolve_intent",
        {"project_id": sid, "resolution": request, "max_seconds": 600, "max_actions": 37},
    )
    assert ok, text
    payload = saved_payload(service.store, queued["job_id"])
    assert payload["max_seconds"] == 600 and payload["max_actions"] == 37
    assert starts == [True]


@pytest.mark.parametrize("tool", ["get_intent_attribution", "resolve_intent"])
def test_foreign_owner_cannot_read_or_resolve_questions(transport, tool):
    _, service, _, _, starts, invoke, _ = transport
    foreign = service.store.create("other-owner", SOURCE, contract(), "native_operations")["id"]
    record = propose(service, foreign, "other-owner")
    arguments = {"project_id": foreign}
    if tool == "resolve_intent":
        arguments["resolution"] = resolution(record)
    before = database_state(service.store)
    ok, _, text = invoke(tool, arguments)
    assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_changed_source_refuses_current_looking_resolution(transport):
    _, service, owner, sid, starts, invoke, _ = transport
    request = resolution(propose(service, sid, owner))
    service.store.update(sid, instructions=["The request changed after the question."])
    before = database_state(service.store)
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": request})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_missing_questions_cannot_accept_an_answer(transport):
    _, service, _, sid, starts, invoke, _ = transport
    before = database_state(service.store)
    ok, _, text = invoke(
        "resolve_intent",
        {
            "project_id": sid,
            "resolution": {
                "expected_version": 1,
                "draft_hash": "f" * 64,
                "answers": [{"issue_id": "width_subject", "answer": RAW_ANSWER}],
            },
        },
    )
    assert not ok, text
    assert database_state(service.store) == before and starts == []


def test_active_job_refuses_answers_but_keeps_questions_readable(transport):
    _, service, owner, sid, starts, invoke, _ = transport
    record = propose(service, sid, owner, finish=False)
    before = database_state(service.store)
    ok, read, text = invoke("get_intent_attribution", {"project_id": sid})
    assert ok and read["record"] == record, text
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": resolution(record)})
    assert not ok, text
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize("query", ["intent_attribution", "intent_resolution", "intent_conflict"])
def test_shared_schema_discovery_preserves_closed_typed_boundary(transport, query):
    kind, service, _, _, starts, invoke, _ = transport
    before = database_state(service.store)
    ok, value, text = invoke("get_schema", {"query": query})
    assert ok, text
    wrapper = value["schema"] if kind == "local_mcp" else value
    assert wrapper["name"] == query
    assert wrapper["schema"]["additionalProperties"] is False
    if query == "intent_resolution":
        assert set(wrapper["schema"]["properties"]) == {
            "expected_version",
            "draft_hash",
            "answers",
        }
    assert database_state(service.store) == before and starts == []


def test_remote_read_scope_is_available_but_cannot_resolve(transport):
    kind, service, owner, sid, starts, invoke, _ = transport
    if kind != "remote_mcp":
        pytest.skip("OAuth scope applies to the remote route")
    request = resolution(propose(service, sid, owner))
    before = database_state(service.store)
    ok, read, text = invoke("get_intent_attribution", {"project_id": sid}, "readonly")
    assert ok and read["record"]["status"] == "needs_input", text
    ok, _, text = invoke("resolve_intent", {"project_id": sid, "resolution": request}, "readonly")
    assert not ok, text
    assert database_state(service.store) == before and starts == []


@pytest.mark.parametrize("guard", ["unqualified", "hosted_time_limit"])
def test_remote_resolution_retains_execution_admission_controls(transport, guard):
    kind, service, owner, sid, starts, invoke, _ = transport
    if kind != "remote_mcp":
        pytest.skip("Hosted execution controls apply to the remote route")
    arguments = {"project_id": sid, "resolution": resolution(propose(service, sid, owner))}
    if guard == "unqualified":
        service.executor._typed_container_verified = False
    else:
        arguments["max_seconds"] = 601
    before = database_state(service.store)
    ok, _, text = invoke("resolve_intent", arguments)
    assert not ok, text
    assert database_state(service.store) == before and starts == []
