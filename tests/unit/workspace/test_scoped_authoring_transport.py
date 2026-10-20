from __future__ import annotations

import asyncio
import hashlib
import json
import time
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.auth.provider import AccessToken

from forge_mind.server.workspace_app import create_app as local_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.project_access import assembly_structure, design_relationships
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "a" * 32


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "readonly", "expired", "wrong-issuer", "wrong-resource"}:
            return None
        return AccessToken(
            token=token,
            client_id="scoped-test",
            subject="alice" if token == "readonly" else token,
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            resource=RESOURCE if token != "wrong-resource" else "https://other.test/mcp",
            expires_at=int(time.time()) + (-1 if token == "expired" else 600),
            claims={"iss": ISSUER if token != "wrong-issuer" else "https://other.test"},
        )


def document():
    return DesignDocument.model_validate(
        {
            "variables": {"width": 4, "gap": 1, "diameter": 2, "unrelated": 99},
            "relations": [{"name": "end", "operation": "multiply", "a": "gap", "b": 10}],
            "assembly": {
                "parts": [
                    {
                        "id": "body",
                        "name": "body",
                        "features": [
                            {
                                "id": "path",
                                "name": "path",
                                "kind": "curve",
                                "points": [[0, 0, 0], ["end", 0, 0]],
                            },
                            {
                                "id": "block",
                                "name": "block",
                                "kind": "box",
                                "params": {"x": 4, "y": 4, "z": 4},
                                "bindings": {"x": "width"},
                            },
                        ],
                        "output": "block",
                        "ports": [
                            {
                                "name": "axis",
                                "kind": "bore",
                                "origin": ["gap", 0, 0],
                                "diameter": "diameter",
                            }
                        ],
                        "production": {"role": "make", "process": "cnc"},
                    },
                    {
                        "id": "retained",
                        "name": "untouched",
                        "features": [
                            {
                                "id": "block",
                                "name": "block",
                                "kind": "box",
                                "params": {"x": 3, "y": 3, "z": 3},
                            }
                        ],
                        "output": "block",
                        "production": {"role": "reference"},
                    },
                ],
                "roots": [
                    {"id": "body", "definition": "body"},
                    {"id": "retained", "definition": "retained", "position": [10, 0, 0]},
                ],
            },
        }
    )


def seed(service, owner="alice", *, rid=RID, source=None, prompt="Scoped source fixture"):
    sid = service.store.create(owner, prompt, Contract().model_dump())["id"]
    doc = source or document()
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    data = doc.model_dump_json().encode()
    (directory / "document.json").write_bytes(data)
    program = b"# Source-only retained fixture; never executed.\n"
    (directory / "program.py").write_bytes(program)
    measurement = Measurement(
        revision_id=rid,
        contract_hash=Contract().fingerprint(),
        artifact_hashes={
            "document.json": hashlib.sha256(data).hexdigest(),
            "program.py": hashlib.sha256(program).hexdigest(),
        },
        engine_fingerprint="f" * 64,
        geometry_valid=False,
    )
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "document": doc.model_dump(mode="json"),
            "measurement": measurement.model_dump(),
        },
    )
    service.store.update(sid, selected_revision=rid, accepted_revision=rid)
    return sid, doc


def snapshot(store):
    with store.connect() as con:
        return {
            name: [tuple(row) for row in con.execute(f"SELECT * FROM {name} ORDER BY rowid")]
            for name in ("sessions", "jobs", "revisions", "engineering_inputs")
        }


def call(client, token, tool, args):
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
            "params": {"name": tool, "arguments": args},
        },
    )


def result(response):
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.fixture
def remote(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor())
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service, starts


def test_actual_local_api_and_sdk_and_remote_reads_preserve_values(remote, tmp_path, monkeypatch):
    client, service, starts = remote
    sid, doc = seed(service)
    before = snapshot(service.store)
    remote_part = result(
        call(
            client, "alice", "read_part", {"project_id": sid, "revision_id": RID, "part_id": "body"}
        )
    )["structuredContent"]
    assert remote_part["values"] == {"width": 4, "end": 10, "gap": 1, "diameter": 2}
    assert "unrelated" not in remote_part["values"]
    assert remote_part["part"] == doc.assembly.parts[0].model_dump(mode="json")
    assert (
        result(
            call(
                client,
                "readonly",
                "read_part",
                {"project_id": sid, "revision_id": RID, "part_id": "body"},
            )
        )["structuredContent"]
        == remote_part
    )
    assert result(
        call(client, "alice", "read_assembly_structure", {"project_id": sid, "revision_id": RID})
    )["structuredContent"] == json.loads(
        json.dumps({"revision_id": RID, **assembly_structure(doc)})
    )
    assert result(
        call(client, "alice", "get_design_relationships", {"project_id": sid, "revision_id": RID})
    )["structuredContent"] == {"revision_id": RID, **design_relationships(doc)}

    local_service = WorkspaceService(Store(tmp_path / "local"), Executor("trusted_local"))
    monkeypatch.setattr(local_service, "start", lambda: None)
    local_sid, _ = seed(local_service, "local-owner")
    app = local_app(local=True, service=local_service)
    with TestClient(app) as api:
        reply = api.get(f"/workspace/sessions/{local_sid}/design/{RID}/parts/body")
        assert reply.status_code == 200, reply.text
        assert reply.json() == remote_part
        assert (
            api.get(f"/workspace/sessions/{local_sid}/design/{RID}/parts/missing").status_code
            == 404
        )

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as sdk:
            for tool, args in [
                ("read_part", {"part_id": "body"}),
                ("read_assembly_structure", {}),
                ("get_design_relationships", {}),
            ]:
                returned = await sdk.call_tool(
                    tool, {"project_id": local_sid, "revision_id": RID, **args}
                )
                assert not returned.is_error
                expected = (
                    remote_part
                    if tool == "read_part"
                    else {
                        "revision_id": RID,
                        **(
                            assembly_structure(doc)
                            if tool == "read_assembly_structure"
                            else design_relationships(doc)
                        ),
                    }
                )
                assert returned.structured_content == json.loads(json.dumps(expected))

    asyncio.run(scenario())
    assert snapshot(service.store) == before
    assert starts == []


def test_project_exposes_real_edit_preconditions(remote):
    client, service, _ = remote
    sid, _ = seed(service)
    service.store.save_engineering(sid, "alice", 0, service.store.engineering(sid)["inputs"])
    service.store.update(sid, proposal={"revision_id": "b" * 32, "input_version": 999})
    state = result(call(client, "alice", "get_project", {"project_id": sid}))["structuredContent"]
    assert state["input_version"] == 1
    assert state["design_revision"] == "b" * 32
    assert state["selected_revision"] == RID


@pytest.mark.parametrize("tool", ["edit_variables", "edit_assembly"])
def test_unqualified_edits_refuse_without_any_store_or_worker_change(remote, tool):
    client, service, starts = remote
    sid, _ = seed(service)
    before = snapshot(service.store)
    extra = (
        {"variables": {"width": 6}}
        if tool == "edit_variables"
        else {"edit": {"variables": {"width": 6}}}
    )
    returned = result(
        call(
            client,
            "alice",
            tool,
            {"project_id": sid, "revision_id": RID, "input_version": 0, **extra},
        )
    )
    assert returned["isError"] and "not qualified" in returned["content"][0]["text"]
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize("tool", ["edit_variables", "edit_assembly"])
@pytest.mark.parametrize(
    "token", [None, "bob", "readonly", "expired", "wrong-issuer", "wrong-resource"]
)
def test_write_ownership_and_scope_errors_never_enqueue(remote, tool, token):
    client, service, starts = remote
    service.executor._typed_container_verified = True  # Admission control fixture only.
    sid, _ = seed(service)
    before = snapshot(service.store)
    extra = (
        {"variables": {"width": 6}}
        if tool == "edit_variables"
        else {"edit": {"variables": {"width": 6}}}
    )
    reply = call(
        client, token, tool, {"project_id": sid, "revision_id": RID, "input_version": 0, **extra}
    )
    assert reply.status_code in (401, 403) or result(reply)["isError"]
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize(
    "tool", ["read_part", "read_assembly_structure", "get_design_relationships"]
)
@pytest.mark.parametrize(
    "problem",
    ["foreign", "missing_revision", "tampered", "malformed_document", "oversize"],
)
def test_owned_reads_refuse_incomplete_or_changed_data(remote, monkeypatch, tool, problem):
    client, service, _ = remote
    sid, _ = seed(service)
    directory = service.store.directory(sid, RID)
    token = "alice"
    args = {"project_id": sid, "revision_id": RID}
    if tool == "read_part":
        args["part_id"] = "body"
    if problem == "foreign":
        token = "bob"
    elif problem == "missing_revision":
        args["revision_id"] = "b" * 32
    elif problem in ("tampered", "malformed_document"):
        (directory / "document.json").write_text("{}")
        if problem == "malformed_document":
            # Valid JSON but not an assembly; schema/operation refuses explicitly.
            value = {"variables": {"bad": "no"}}
            data = json.dumps(value).encode()
            (directory / "document.json").write_bytes(data)
            with service.store.connect() as con:
                row = json.loads(
                    con.execute(
                        "SELECT document FROM revisions WHERE session=?", (sid,)
                    ).fetchone()[0]
                )
                row["measurement"]["artifact_hashes"]["document.json"] = hashlib.sha256(
                    data
                ).hexdigest()
                con.execute(
                    "UPDATE revisions SET document=? WHERE session=?", (json.dumps(row), sid)
                )
    else:
        monkeypatch.setattr("forge_mind.workspace.remote_mcp.MAX_RESPONSE_BYTES", 10)
    before = snapshot(service.store)
    assert result(call(client, token, tool, args))["isError"]
    assert snapshot(service.store) == before


@pytest.mark.parametrize("tool", ["edit_variables", "edit_assembly"])
@pytest.mark.parametrize(
    "problem",
    [
        "stale_head",
        "stale_input",
        "tampered",
        "bad_patch",
        "unknown_variable",
        "zero_budget",
        "large_budget",
    ],
)
def test_edit_refusals_preserve_every_record(remote, tool, problem):
    client, service, starts = remote
    service.executor._typed_container_verified = True  # No native worker runs.
    sid, _ = seed(service)
    extra = (
        {"variables": {"width": 6}}
        if tool == "edit_variables"
        else {"edit": {"variables": {"width": 6}}}
    )
    args = {"project_id": sid, "revision_id": RID, "input_version": 0, **extra}
    if problem == "stale_head":
        service.store.update(sid, proposal={"revision_id": "b" * 32})
    elif problem == "stale_input":
        service.store.save_engineering(sid, "alice", 0, service.store.engineering(sid)["inputs"])
    elif problem == "tampered":
        (service.store.directory(sid, RID) / "document.json").write_text("{}")
    elif problem == "bad_patch":
        args.update(
            {"variables": {"width": "bad"}}
            if tool == "edit_variables"
            else {"edit": {"unexpected": True}}
        )
    elif problem == "unknown_variable":
        if tool == "edit_assembly":
            args["edit"] = {"roots": [{"id": "new", "definition": "missing"}]}
        else:
            args["variables"] = {"missing": 2}
    else:
        args["max_seconds"] = 0 if problem == "zero_budget" else 601
    before = snapshot(service.store)
    assert result(call(client, "alice", tool, args))["isError"]
    assert snapshot(service.store) == before
    assert starts == []


@pytest.mark.parametrize("tool", ["edit_variables", "edit_assembly"])
def test_remote_admission_retains_active_candidate_contract_and_unrelated_source(remote, tool):
    client, service, starts = remote
    service.executor._typed_container_verified = True  # Queue test; not runtime qualification.
    sid, doc = seed(service, prompt="Scoped source fixture. Set the overall X extent to 4 mm.")
    baseline, candidate = (
        {"requirements": []},
        {
            "requirements": [
                {
                    "id": "width",
                    "text": "width",
                    "source_quote": "Set the overall X extent to 4 mm",
                    "metric": "extent_x",
                    "comparator": "eq",
                    "value": 4,
                }
            ]
        },
    )
    service.store.update(
        sid,
        accepted_revision="b" * 32,
        proposal={"revision_id": RID, "contract": candidate},
        contract=baseline,
    )
    if tool == "edit_variables":
        extra = {"variables": {"width": 6}}
    else:
        part = doc.assembly.parts[0].model_dump(mode="json", exclude={"production"})
        part["ports"][0]["origin"] = ["end", 0, 0]
        extra = {"edit": {"parts": [part]}}
    returned = result(
        call(
            client,
            "alice",
            tool,
            {
                "project_id": sid,
                "revision_id": RID,
                "input_version": 0,
                "max_seconds": 600,
                **extra,
            },
        )
    )
    assert not returned.get("isError")
    with service.store.connect() as con:
        queued = json.loads(
            con.execute(
                "SELECT payload FROM jobs WHERE id=?", (returned["structuredContent"]["job_id"],)
            ).fetchone()[0]
        )
    assert queued["contract"] == candidate
    assert queued["baseline_contract"] == baseline
    assert queued["based_on"] == "b" * 32
    assert queued["expected_current_revision"] == RID
    assert queued["expected_input_version"] == 0
    assert queued["max_seconds"] == 600
    assert queued["preview_only"] is True
    actual = queued["document"]
    expected = deepcopy(doc.model_dump(mode="json"))
    if tool == "edit_variables":
        expected["variables"]["width"] = 6
    else:
        expected["assembly"]["parts"][0]["ports"][0]["origin"] = ["end", 0, 0]
    assert actual == expected
    assert service.store.session_record(sid, "alice")["contract"] == baseline
    assert starts == [True]


def test_remote_catalog_and_initialize_advertise_real_edit_preconditions(remote):
    client, _, _ = remote
    headers = {"Accept": "application/json, text/event-stream", "Authorization": "Bearer alice"}
    reply = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "scoped-proof", "version": "1"},
            },
        },
    )
    initialized = result(reply)
    assert initialized["serverInfo"]["version"] == "0.0.20"
    assert "design_revision and input_version" in initialized["instructions"]
    assert "read_assembly_structure" in initialized["instructions"]
    tools = result(
        client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
    )["tools"]
    catalog = {row["name"]: row for row in tools}
    for name in ("edit_variables", "edit_assembly"):
        assert catalog[name]["annotations"]["readOnlyHint"] is False
        schema = catalog[name]["inputSchema"]
        assert {"project_id", "revision_id", "input_version"} <= set(schema["required"])
        assert schema["properties"]["max_seconds"]["maximum"] == 600
        assert schema["properties"]["max_seconds"]["default"] == 300
    for name in ("read_part", "read_assembly_structure", "get_design_relationships"):
        assert catalog[name]["annotations"]["readOnlyHint"] is True
    assert "run_engineering_review" not in catalog


def test_unknown_part_is_explicit_and_does_not_change_records(remote):
    client, service, _ = remote
    sid, _ = seed(service)
    before = snapshot(service.store)
    returned = result(
        call(
            client,
            "alice",
            "read_part",
            {"project_id": sid, "revision_id": RID, "part_id": "missing"},
        )
    )
    assert returned["isError"]
    assert "Part definition not found" in returned["content"][0]["text"]
    assert snapshot(service.store) == before


@pytest.mark.parametrize("tool", ["edit_variables", "edit_assembly"])
def test_actual_local_sdk_writes_queue_only_the_intended_change(tmp_path, monkeypatch, tool):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    starts = []
    monkeypatch.setattr(service, "start", lambda: starts.append(True))
    sid, doc = seed(service, "local-owner")
    app = local_app(local=True, service=service)
    expected = deepcopy(doc.model_dump(mode="json"))
    if tool == "edit_variables":
        args = {"variables": {"width": 6}}
        expected["variables"]["width"] = 6
    else:
        part = doc.assembly.parts[0].model_dump(mode="json", exclude={"production"})
        part["ports"][0]["origin"] = ["end", 0, 0]
        args = {"input_version": 0, "edit": {"parts": [part]}}
        expected["assembly"]["parts"][0]["ports"][0]["origin"] = ["end", 0, 0]

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as sdk:
            returned = await sdk.call_tool(tool, {"project_id": sid, "revision_id": RID, **args})
            assert not returned.is_error, returned
            return returned.structured_content["job_id"]

    jid = asyncio.run(scenario())
    with service.store.connect() as con:
        queued = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
    assert queued["document"] == expected
    assert queued["expected_current_revision"] == RID
    assert queued["expected_input_version"] == 0
    assert queued["preview_only"] is True
    assert starts == [True]


def test_actual_remote_request_cap_prevents_large_edit_mutation(remote):
    client, service, starts = remote
    service.executor._typed_container_verified = True
    sid, _ = seed(service)
    before = snapshot(service.store)
    part = {
        "id": "body",
        "name": "body",
        "features": [
            {
                "id": "block",
                "name": "block",
                "kind": "python",
                "code": "#" + "x" * (1024 * 1024 + 1),
            }
        ],
        "output": "block",
    }
    reply = call(
        client,
        "alice",
        "edit_assembly",
        {"project_id": sid, "revision_id": RID, "input_version": 0, "edit": {"parts": [part]}},
    )
    assert reply.status_code == 413
    assert snapshot(service.store) == before
    assert starts == []


def test_queue_checks_the_current_head_again_at_admission(remote, monkeypatch):
    client, service, starts = remote
    service.executor._typed_container_verified = True
    sid, _ = seed(service)
    original = service.store.enqueue

    def concurrent_change(project_id, principal, payload):
        service.store.update(sid, proposal={"revision_id": "b" * 32})
        return original(project_id, principal, payload)

    monkeypatch.setattr(service.store, "enqueue", concurrent_change)
    returned = result(
        call(
            client,
            "alice",
            "edit_variables",
            {"project_id": sid, "revision_id": RID, "input_version": 0, "variables": {"width": 6}},
        )
    )
    assert returned["isError"]
    assert "Design changed before admission" in returned["content"][0]["text"]
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert service.store.session_record(sid, "alice")["proposal"] == {"revision_id": "b" * 32}
    assert starts == []


def test_actual_hosted_part_response_limit_preserves_source_and_records(remote):
    client, service, starts = remote
    raw = document().model_dump(mode="json")
    raw["assembly"]["parts"][0]["features"] = [
        {
            "id": "block",
            "name": "block",
            "kind": "python",
            "code": "#" + "x" * (2 * 1024 * 1024 + 1024) + "\nresult = b.Box(4, 4, 4)",
        }
    ]
    # Retained source only. No feature code is evaluated to create this fixture.
    sid, _ = seed(service, source=DesignDocument.model_validate(raw))
    path = service.store.directory(sid, RID) / "document.json"
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before = snapshot(service.store)
    returned = result(
        call(
            client, "alice", "read_part", {"project_id": sid, "revision_id": RID, "part_id": "body"}
        )
    )
    assert returned["isError"]
    assert "Response exceeds the hosted transfer limit" in returned["content"][0]["text"]
    structure = result(
        call(client, "alice", "read_assembly_structure", {"project_id": sid, "revision_id": RID})
    )
    assert not structure.get("isError")
    assert structure["structuredContent"]["parts"][0]["feature_count"] == 1
    assert snapshot(service.store) == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert starts == []


def relation_recovery_document():
    value = document().model_dump(mode="json")
    value["variables"].update(base_width=52, hole_spacing=36)
    value["relations"].extend(
        [
            {"name": "half_hole_spacing", "operation": "divide", "a": "hole_spacing", "b": 2},
            {"name": "left_hole_x", "operation": "multiply", "a": "half_hole_spacing", "b": -1},
            {"name": "mount_edge_offset", "operation": "add", "a": "base_width", "b": -36},
            {
                "name": "retained_dependent",
                "operation": "multiply",
                "a": "mount_edge_offset",
                "b": 2,
            },
        ]
    )
    value["delivery_guidance"] = {"print": "Retain these source-only fixture instructions."}
    return DesignDocument.model_validate(value)


def relation_recovery_patch():
    return {
        "relations": [
            {"name": "half_base_width", "operation": "divide", "a": "base_width", "b": 2},
            {
                "name": "mount_edge_offset",
                "operation": "add",
                "a": "half_base_width",
                "b": "left_hole_x",
            },
        ]
    }


def relation_recovery_setup(surface, remote, tmp_path, monkeypatch, *, foreign=False):
    if surface == "remote":
        client, service, starts = remote
        service.executor._typed_container_verified = True  # Admission fixture; no worker execution.
        owner = "alice"
    else:
        client = None
        service = WorkspaceService(Store(tmp_path / "local"), Executor("trusted_local"))
        starts = []
        monkeypatch.setattr(service, "start", lambda: starts.append(True))
        owner = "another-owner" if foreign else "local-owner"
    sid, doc = seed(
        service,
        owner,
        source=relation_recovery_document(),
        prompt="Scoped source fixture. Set the overall X extent to 4 mm.",
    )
    candidate = {
        "requirements": [
            {
                "id": "width",
                "text": "Retained width",
                "source_quote": "Set the overall X extent to 4 mm",
                "metric": "extent_x",
                "comparator": "eq",
                "value": 4,
            }
        ]
    }
    service.store.update(sid, proposal={"revision_id": RID, "contract": candidate})
    directory = service.store.directory(sid, RID)
    original_files = {f.name: f.read_bytes() for f in directory.iterdir() if f.is_file()}
    return client, service, starts, sid, doc, candidate, original_files


def relation_recovery_invoke(surface, client, service, sid, args, tmp_path, *, foreign=False):
    if surface == "remote":
        returned = result(call(client, "bob" if foreign else "alice", "edit_assembly", args))
        return (
            not returned.get("isError", False),
            returned.get("structuredContent"),
            "\n".join(item.get("text", "") for item in returned.get("content", [])),
        )
    app = local_app(local=True, service=service)

    async def scenario():
        if surface == "local_api":
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:4321"
            ) as api:
                returned = await api.post(
                    f"/workspace/sessions/{sid}/assembly-edit",
                    json={key: value for key, value in args.items() if key != "project_id"},
                )
                assert returned.status_code in (202, 409), returned.text
                return returned.status_code == 202, returned.json(), returned.text
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as sdk:
            returned = await sdk.call_tool("edit_assembly", args)
            return (
                not returned.is_error,
                returned.structured_content,
                "\n".join(item.text for item in returned.content if item.type == "text"),
            )

    return asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local_api", "local_sdk", "remote"])
def test_relation_dependency_replacement_queues_one_preserved_edit(
    surface, remote, tmp_path, monkeypatch
):
    client, service, starts, sid, doc, candidate, original_files = relation_recovery_setup(
        surface, remote, tmp_path, monkeypatch
    )
    before = snapshot(service.store)
    original = doc.model_dump(mode="json")
    args = {
        "project_id": sid,
        "revision_id": RID,
        "input_version": 0,
        "edit": relation_recovery_patch(),
    }
    ok, data, text = relation_recovery_invoke(surface, client, service, sid, args, tmp_path)
    assert ok, text
    with service.store.connect() as con:
        queued = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (data["job_id"],)).fetchone()[0]
        )
    actual = DesignDocument.model_validate(queued["document"])
    values = actual.values()
    assert values["half_base_width"] == 26
    assert values["mount_edge_offset"] == 8
    assert values["retained_dependent"] == 16
    assert [row.name for row in actual.relations] == [
        "end",
        "half_hole_spacing",
        "left_hole_x",
        "half_base_width",
        "mount_edge_offset",
        "retained_dependent",
    ]
    expected = deepcopy(original)
    expected["relations"] = [
        *original["relations"][:3],
        {"name": "half_base_width", "operation": "divide", "a": "base_width", "b": 2.0},
        {
            "name": "mount_edge_offset",
            "operation": "add",
            "a": "half_base_width",
            "b": "left_hole_x",
        },
        original["relations"][4],
    ]
    assert actual.model_dump(mode="json") == expected
    assert doc.model_dump(mode="json") == original
    assert queued["contract"] == candidate
    assert queued["baseline_contract"] == Contract().model_dump()
    assert queued["expected_current_revision"] == RID
    assert queued["expected_input_version"] == 0
    assert queued["based_on"] == RID
    assert queued["preview_only"] is True
    assert (
        service.store.session_record(sid, "alice" if surface == "remote" else "local-owner")[
            "contract"
        ]
        == Contract().model_dump()
    )
    after = snapshot(service.store)
    assert after["revisions"] == before["revisions"]
    assert len(after["jobs"]) == len(before["jobs"]) + 1
    assert after["engineering_inputs"] == before["engineering_inputs"]
    assert starts == [True]
    directory = service.store.directory(sid, RID)
    assert {f.name: f.read_bytes() for f in directory.iterdir() if f.is_file()} == original_files


@pytest.mark.parametrize("surface", ["local_api", "local_sdk", "remote"])
@pytest.mark.parametrize("problem", ["missing", "cycle", "foreign", "stale_head", "stale_input"])
def test_relation_recovery_refusal_preserves_all_records(
    surface, problem, remote, tmp_path, monkeypatch
):
    client, service, starts, sid, _, _, original_files = relation_recovery_setup(
        surface, remote, tmp_path, monkeypatch, foreign=problem == "foreign"
    )
    patch = relation_recovery_patch()
    if problem == "missing":
        patch["relations"][1]["a"] = "unavailable_width"
    elif problem == "cycle":
        patch["relations"][0]["a"] = "mount_edge_offset"
    elif problem == "stale_head":
        service.store.update(sid, proposal={"revision_id": "b" * 32})
    elif problem == "stale_input":
        service.store.save_engineering(
            sid,
            "alice" if surface == "remote" else "local-owner",
            0,
            service.store.engineering(sid)["inputs"],
        )
    before = snapshot(service.store)
    args = {"project_id": sid, "revision_id": RID, "input_version": 0, "edit": patch}
    ok, _, text = relation_recovery_invoke(
        surface, client, service, sid, args, tmp_path, foreign=problem == "foreign"
    )
    assert not ok
    if problem == "missing":
        assert "mount_edge_offset" in text and "unavailable_width" in text
    elif problem == "cycle":
        assert (
            any(word in text.lower() for word in ("cycle", "cyclic"))
            and "mount_edge_offset" in text
        )
    elif problem == "stale_head":
        assert "Design changed before admission" in text
    elif problem == "stale_input":
        assert "Engineering inputs changed before admission" in text
    else:
        assert ("Project not found" if surface == "remote" else "workspace not found") in text
    assert snapshot(service.store) == before
    assert starts == []
    directory = service.store.directory(sid, RID)
    assert {f.name: f.read_bytes() for f in directory.iterdir() if f.is_file()} == original_files
