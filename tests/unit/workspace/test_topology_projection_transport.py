"""Actual initialized SDK/ASGI saved-fact transport; no build or native rendering."""

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager

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
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.models import Measurement
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

ISSUER = "https://fixture.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "b" * 32


class Verifier:
    async def verify_token(self, token):
        if token not in ("alice", "bob"):
            return None
        return AccessToken(
            token=token,
            client_id="topology-test",
            subject=token,
            scopes=["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


def fixture(tmp_path, monkeypatch, *, minimal=False, malformed=False):
    service = WorkspaceService(Store(tmp_path / "store"), Executor())
    monkeypatch.setattr(service, "start", lambda: None)
    sid = service.store.create("alice", "saved topology", None)["id"]
    rows = []
    for body in (0, 1):
        for kind in ("body", "face", "edge"):
            rows.append(
                {
                    "id": f"{kind}-{body}",
                    "kind": kind,
                    "body_index": body,
                    "signature": f"exact-{kind}-{body}",
                    "body_signature": f"body-{body}",
                    "center": [0, 0, 0],
                    "min": [-1, -1, -1],
                    "max": [1, 1, 1],
                    "measure": 4,
                    "unit": {"body": "mm^3", "face": "mm^2", "edge": "mm"}[kind],
                    **({"polyline": [[0, 0, 0], [1, 1, 1]]} if kind == "edge" else {}),
                }
            )
    value = {
        "frame": "cad_z_up_mm",
        "entities": rows,
        "topology_counts": {"bodies": 2, "faces": 2, "edges": 2},
        "detail_policy": {"mode": "full_index"},
        "meshes": [],
    }
    if minimal:
        value = {"entities": [{"id": "legacy", "unknown": "preserved"}]}
    if malformed:
        value = {"entities": "invalid"}
    data = json.dumps(value, indent=2).encode()
    measurement = Measurement(
        revision_id=RID,
        contract_hash="saved",
        engine_fingerprint="fixture",
        geometry_valid=False,
        artifact_hashes={"topology.json": hashlib.sha256(data).hexdigest()},
    ).model_dump()
    service.store.add_revision(sid, {"id": RID, "contract": {}, "measurement": measurement})
    directory = service.store.directory(sid, RID)
    directory.mkdir(parents=True)
    (directory / "topology.json").write_bytes(data)
    (directory / "evidence.json").write_text(json.dumps(measurement))
    return service, sid, data, value


@asynccontextmanager
async def connection(service, tmp_path, surface, *, owner="alice"):
    if surface == "local":
        app = FastAPI()
        app.include_router(router(service.store, service, lambda: owner))
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(create_server(adapter)) as sdk:
            yield sdk
    else:
        app = create_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                trust_env=False,
                headers={"Authorization": "Bearer " + owner},
            ) as http,
            Client(streamable_http_client(RESOURCE, http_client=http), mode="legacy") as sdk,
        ):
            yield sdk
    assert service.worker is None


def records(service):
    with service.store.connect() as con:
        return {
            t: [dict(r) for r in con.execute(f"SELECT * FROM {t}")]
            for t in ("sessions", "revisions", "jobs")
        }


def test_actual_local_remote_compact_parity_and_saved_bytes(tmp_path, monkeypatch):
    service, sid, data, _ = fixture(tmp_path, monkeypatch)
    before = records(service)
    results = []

    async def scenario():
        for surface in ("local", "remote"):
            async with connection(service, tmp_path, surface) as sdk:
                tool = {t.name: t for t in (await sdk.list_tools()).tools}["get_topology"]
                assert tool.annotations.read_only_hint is True
                response = await sdk.call_tool(
                    "get_topology",
                    {
                        "project_id": sid,
                        "revision_id": RID,
                        "body_index": 1,
                        "kind": "edge",
                        "compact": True,
                        "limit": 1,
                    },
                )
                assert not response.is_error
                value = response.structured_content
                assert [r["id"] for r in value["entities"]] == ["edge-1"]
                assert value["entities"][0]["signature"] == "exact-edge-1"
                assert "polyline" not in value["entities"][0]
                assert value["total"] == value["returned"] == 1
                assert value["next_offset"] is None
                assert value["topology_sha256"] == hashlib.sha256(data).hexdigest()
                results.append(value)

    asyncio.run(scenario())
    assert results[0] == results[1]
    assert records(service) == before
    assert (service.store.directory(sid, RID) / "topology.json").read_bytes() == data


def test_actual_remote_no_new_options_preserves_historical_response(tmp_path, monkeypatch):
    service, sid, _, value = fixture(tmp_path, monkeypatch, minimal=True)

    async def scenario():
        async with connection(service, tmp_path, "remote") as sdk:
            result = await sdk.call_tool("get_topology", {"project_id": sid, "revision_id": RID})
            assert not result.is_error
            assert result.structured_content == {
                "revision_id": RID,
                "total": 1,
                "entities": value["entities"],
                "next_offset": None,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("failure", ["foreign", "stale", "malformed"])
def test_existing_owner_hash_and_invalid_artifact_boundaries(
    tmp_path, monkeypatch, surface, failure
):
    service, sid, _, _ = fixture(tmp_path, monkeypatch, malformed=failure == "malformed")
    path = service.store.directory(sid, RID) / "topology.json"
    if failure == "stale":
        path.write_bytes(b"{}")
    before = records(service)

    async def scenario():
        async with connection(
            service, tmp_path, surface, owner="bob" if failure == "foreign" else "alice"
        ) as sdk:
            response = await sdk.call_tool(
                "get_topology",
                {
                    "project_id": sid,
                    "revision_id": RID,
                    "compact": True,
                },
            )
            assert response.is_error

    asyncio.run(scenario())
    assert records(service) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "argument",
    [
        {"offset": -1},
        {"body_index": True},
        {"kind": "hole"},
        {"limit": 101},
        {"compact": 1},
    ],
)
def test_actual_sdk_rejects_invalid_query_arguments(tmp_path, monkeypatch, surface, argument):
    service, sid, _, _ = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool(
                "get_topology",
                {
                    "project_id": sid,
                    "revision_id": RID,
                    **argument,
                },
            )
            assert result.is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "paging",
    [
        {"offset": "0", "limit": 1.0},
        {"offset": 0.0, "limit": "1"},
    ],
)
def test_existing_paging_normalization_is_preserved(tmp_path, monkeypatch, surface, paging):
    service, sid, _, value = fixture(tmp_path, monkeypatch, minimal=True)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool(
                "get_topology",
                {
                    "project_id": sid,
                    "revision_id": RID,
                    **paging,
                },
            )
            assert not result.is_error
            assert result.structured_content == {
                "revision_id": RID,
                "total": 1,
                "entities": value["entities"],
                "next_offset": None,
            }

    asyncio.run(scenario())
