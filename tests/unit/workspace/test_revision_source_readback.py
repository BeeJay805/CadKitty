"""Saved source readback through actual SDK transports; no worker or native CAD."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.revision_source import revision_source
from forge_mind.workspace.store import Store

ISSUER = "https://fixture.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
BASELINE = "a" * 32
CHILD = "b" * 32


def sha(data):
    return hashlib.sha256(data).hexdigest()


def saved_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", raising=False)
    service = WorkspaceService(Store(tmp_path / "store"), Executor())
    # No lifecycle starts or builds are required to read saved fixtures.
    monkeypatch.setattr(service, "start", lambda: None)
    sid = service.store.create("alice", "support", None)["id"]
    saved = {}
    for rid, width in ((BASELINE, 10), (CHILD, 15)):
        document = {
            "name": "Support ü\nwith literal notes",
            "production": None,
            "variables": {"width": width, "exact_integer": 2**60 + 1, "float_one": 1.0},
            "features": [
                {"id": "body", "name": "Body", "kind": "box", "params": {"x": 10, "y": 4, "z": 2}}
            ],
            "outputs": ["body"],
        }
        contract = {
            "requirements": [
                {
                    "id": "starting_width",
                    "text": "Chosen initial width",
                    "source_quote": "Initial width 10",
                    "source": "assumption",
                    "metric": "extent_x",
                    "value": 10.0,
                }
            ],
        }
        doc_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode()
        contract_bytes = json.dumps(contract, indent=1).encode()
        measurement = Measurement(
            revision_id=rid,
            contract_hash=Contract.model_validate(contract).fingerprint(),
            engine_fingerprint="saved-fixture-engine",
            geometry_valid=False,
            artifact_hashes={"document.json": sha(doc_bytes)},
        ).model_dump()
        record = {"id": rid, "contract": contract, "measurement": measurement}
        service.store.add_revision(sid, record)
        directory = service.store.directory(sid, rid)
        directory.mkdir(parents=True)
        for name, data in (
            ("document.json", doc_bytes),
            ("contract.json", contract_bytes),
            ("evidence.json", json.dumps(measurement).encode()),
        ):
            (directory / name).write_bytes(data)
        saved[rid] = (document, contract, doc_bytes, contract_bytes, measurement)
    # A newer project target must not replace either historical revision's target.
    service.store.update(sid, contract={"requirements": [], "coverage_notes": ["new target"]})
    return service, sid, saved


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "readonly"}:
            return None
        return AccessToken(
            token=token,
            client_id="source-read-test",
            subject="alice" if token == "readonly" else token,
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


@asynccontextmanager
async def connection(service, tmp_path, surface, foreign=False):
    if surface == "local":
        app = FastAPI()
        app.include_router(router(service.store, service, lambda: "bob" if foreign else "alice"))
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as sdk:
            yield sdk
    else:
        app = remote_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                trust_env=False,
                headers={"Authorization": "Bearer " + ("bob" if foreign else "readonly")},
            ) as http,
            Client(streamable_http_client(RESOURCE, http_client=http), mode="legacy") as sdk,
        ):
            yield sdk
    assert service.worker is None


def store_snapshot(service, sid):
    with service.store.connect() as con:
        return {
            table: con.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("sessions", "revisions", "jobs")
        }


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_sdk_complete_historical_source_targets_and_no_mutation(tmp_path, monkeypatch, surface):
    service, sid, saved = saved_fixture(tmp_path, monkeypatch)
    before = store_snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            assert sdk.server_info.version == ("0.0.19" if surface == "local" else "0.0.20")
            tool = {tool.name: tool for tool in (await sdk.list_tools()).tools}[
                "get_revision_source"
            ]
            assert tool.annotations.read_only_hint is True
            assert "get_revision_source" in sdk.instructions
            for rid in (BASELINE, CHILD):
                response = await sdk.call_tool(
                    "get_revision_source", {"project_id": sid, "revision_id": rid}
                )
                assert not response.is_error
                value = response.structured_content
                document, contract, doc_bytes, contract_bytes, measurement = saved[rid]
                assert set(value) == {
                    "project_id",
                    "revision_id",
                    "document",
                    "contract",
                    "document_sha256",
                    "contract_sha256",
                    "contract_fingerprint",
                    "scope",
                }
                assert value["project_id"] == sid and value["revision_id"] == rid
                assert value["document"] == document and value["contract"] == contract
                assert (
                    "units" not in value["document"] and "coverage_notes" not in value["contract"]
                )
                assert value["document"]["production"] is None
                assert value["document"]["variables"]["exact_integer"] == 2**60 + 1
                assert type(value["document"]["variables"]["float_one"]) is float
                assert value["contract"]["requirements"][0]["value"] == 10.0
                assert value["document_sha256"] == sha(doc_bytes)
                assert value["contract_sha256"] == sha(contract_bytes)
                assert value["contract_fingerprint"] == measurement["contract_hash"]
            if surface == "remote":
                old = await sdk.call_tool(
                    "get_document", {"project_id": sid, "revision_id": BASELINE}
                )
                assert not old.is_error and old.structured_content["document"]["units"] == "mm"
        async with connection(service, tmp_path, surface, foreign=True) as sdk:
            refusal = await sdk.call_tool(
                "get_revision_source", {"project_id": sid, "revision_id": CHILD}
            )
            assert refusal.is_error and not refusal.structured_content

    asyncio.run(scenario())
    assert store_snapshot(service, sid) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "fault", ["document", "contract", "missing", "malformed", "fingerprint", "symlink"]
)
def test_sdk_refuses_changed_missing_or_malformed_sources(tmp_path, monkeypatch, surface, fault):
    service, sid, _ = saved_fixture(tmp_path, monkeypatch)
    directory = service.store.directory(sid, CHILD)
    if fault == "document":
        (directory / "document.json").write_text("{}")
    elif fault == "contract":
        (directory / "contract.json").write_text("{}")
    elif fault == "missing":
        (directory / "contract.json").unlink()
    elif fault == "malformed":
        (directory / "contract.json").write_text("{broken")
    elif fault == "fingerprint":
        record = service.store.revision(sid, CHILD)
        record["measurement"]["contract_hash"] = "0" * 64
        with service.store.connect() as con:
            con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(record), CHILD))
        (directory / "evidence.json").write_text(json.dumps(record["measurement"]))
    else:
        path = directory / "contract.json"
        target = tmp_path / "linked-contract.json"
        path.replace(target)
        try:
            os.symlink(target, path)
        except OSError:
            pytest.skip("Symlink creation unavailable on this host")
    before = store_snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            reply = await sdk.call_tool(
                "get_revision_source", {"project_id": sid, "revision_id": CHILD}
            )
            assert reply.is_error and not reply.structured_content

    asyncio.run(scenario())
    assert store_snapshot(service, sid) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_sdk_aggregate_reply_capacity_not_individual_file_size(tmp_path, monkeypatch, surface):
    service, sid, saved = saved_fixture(tmp_path, monkeypatch)
    _, contract, doc_bytes, contract_bytes, measurement = saved[BASELINE]
    expected = revision_source(
        sid, BASELINE, doc_bytes, contract_bytes, measurement, record_contract=contract
    )
    bound = max(len(doc_bytes), len(contract_bytes), len(json.dumps(measurement).encode())) + 1
    assert bound < len(json.dumps(expected).encode())
    if surface == "local":
        monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", str(bound))
    else:
        monkeypatch.setattr("forge_mind.workspace.remote_mcp.MAX_RESPONSE_BYTES", bound)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            reply = await sdk.call_tool(
                "get_revision_source", {"project_id": sid, "revision_id": BASELINE}
            )
            assert reply.is_error

    asyncio.run(scenario())


def test_projection_keeps_raw_values_without_mutating_inputs(tmp_path, monkeypatch):
    _, sid, saved = saved_fixture(tmp_path, monkeypatch)
    _, contract, doc_bytes, contract_bytes, measurement = saved[BASELINE]
    before = deepcopy((measurement, contract))
    value = revision_source(
        sid, BASELINE, doc_bytes, contract_bytes, measurement, record_contract=contract
    )
    assert value["contract"]["requirements"][0]["value"] == 10.0
    assert isinstance(value["contract"]["requirements"][0]["value"], float)
    assert (measurement, contract) == before
    assert json.loads(json.dumps(value))["document"]["name"] == "Support ü\nwith literal notes"


@pytest.mark.parametrize(
    "bad",
    [
        b'{"name":"x","name":"y"}',
        b'{"variables":{"x":NaN}}',
        b'{"variables":{"x":1e999}}',
        b"[]",
        b"\xff",
    ],
)
def test_projection_refuses_invalid_json_even_with_matching_document_hash(
    tmp_path, monkeypatch, bad
):
    _, sid, saved = saved_fixture(tmp_path, monkeypatch)
    _, _, _, contract, measurement = saved[BASELINE]
    measurement["artifact_hashes"]["document.json"] = sha(bad)
    with pytest.raises(ValueError):
        revision_source(sid, BASELINE, bad, contract, measurement)


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("rid", ["c" * 32, "not-a-revision"])
def test_sdk_refuses_unknown_or_invalid_revision_without_job(tmp_path, monkeypatch, surface, rid):
    service, sid, _ = saved_fixture(tmp_path, monkeypatch)
    before = store_snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            reply = await sdk.call_tool(
                "get_revision_source", {"project_id": sid, "revision_id": rid}
            )
            assert reply.is_error and not reply.structured_content

    asyncio.run(scenario())
    assert store_snapshot(service, sid) == before
