"""Initialized SDK/HTTP neighbors boundaries; native service is replaced by a fake."""

from __future__ import annotations

import asyncio
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
from mcp.server.mcpserver.exceptions import ToolError

from forge_mind.workspace import geometry_neighbors
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.geometry_neighbors_contract import (
    FRAME,
    MAX_RESPONSE_BYTES,
    SCOPE,
    NeighborSelector,
)
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import Store

ISSUER = "https://fixture.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "b" * 32


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "no-read"}:
            return None
        return AccessToken(
            token=token,
            client_id="neighbors-test",
            subject=token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


def arguments(sid, *, kind="face"):
    return {
        "project_id": sid,
        "revision_id": RID,
        "selector": {"occurrence_path": "assembly/second", "kind": kind, "signature": "3" * 64},
    }


def entity(kind, digest, identity, *, body=1):
    return {
        "id": identity,
        "kind": kind,
        "signature": digest * 64,
        "body_index": body,
        "body_signature": "c" * 64,
        "center": [0.0, 0.0, 0.0],
        "min": [-1.0, -1.0, 0.0],
        "max": [1.0, 1.0, 0.0],
        "measure": 4.0,
        "unit": "mm^2" if kind == "face" else "mm",
        "surface": "PLANE" if kind == "face" else None,
        "normal": [0.0, 0.0, 1.0] if kind == "face" else None,
        "curve": "LINE" if kind == "edge" else None,
    }


def observed_result(sid, selector, offset, limit):
    body = selector.body_index if selector.body_index is not None else 1
    neighbor_kind = "edge" if selector.kind == "face" else "face"
    neighbors = [
        {"entity": entity(neighbor_kind, "4", "neighbor-0", body=body), "incidence_count": 2},
        {"entity": entity(neighbor_kind, "5", "neighbor-1", body=body), "incidence_count": 1},
    ]
    return {
        "project_id": sid,
        "revision_id": RID,
        "operation": "topology_neighbors",
        "frame": FRAME,
        "units": "mm",
        "selector": {
            "requested": selector.model_dump(mode="json"),
            "body_index": body,
            "body_signature": "c" * 64,
        },
        "offset": offset,
        "limit": limit,
        "source_hashes": dict.fromkeys(("part.step", "topology.json", "document.json"), "e" * 64),
        "saved_engine_fingerprint": "1" * 64,
        "query_engine_fingerprint": "2" * 64,
        "focal": entity(selector.kind, "3", "focal", body=body),
        "relation": "face_boundary_edges" if selector.kind == "face" else "edge_incident_faces",
        "neighbors": neighbors[offset : offset + limit],
        "total_neighbors": 2,
        "total_incidences": 3,
        "next_offset": offset + limit if offset + limit < 2 else None,
        "native_completed": True,
        "coverage": "complete_selected_neighborhood",
        "scope": SCOPE,
    }


def fixture(tmp_path, monkeypatch, *, transform=None, failure=None):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Neighbor reads must not start jobs"))
    sid = service.store.create("alice", "saved neighbors", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls = []

    def inspect(
        actual_service, project_id, owner, revision_id, selector, *, offset, limit, timeout
    ):
        assert actual_service is service
        service.store.session_record(project_id, owner)
        service.store.revision(project_id, revision_id)
        calls.append((project_id, owner, revision_id, selector, offset, limit, timeout))
        if failure:
            raise failure
        value = observed_result(project_id, selector, offset, limit)
        if transform:
            transform(value)
        return value

    monkeypatch.setattr(geometry_neighbors, "inspect_topology_neighbors", inspect)
    return service, sid, calls


@asynccontextmanager
async def connection(service, tmp_path, surface, *, owner="alice"):
    if surface == "local":
        app = FastAPI()
        app.include_router(router(service.store, service, lambda: owner))
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(create_server(adapter), cache=None) as sdk:
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
            table: [dict(row) for row in con.execute(f"SELECT * FROM {table}")]
            for table in ("sessions", "revisions", "jobs")
        }


def test_initialized_sdk_catalog_parity_and_exact_default_dispatch(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)
    results, schemas = [], []

    async def scenario():
        for surface in ("local", "remote"):
            async with connection(service, tmp_path, surface) as sdk:
                tool = {t.name: t for t in (await sdk.list_tools()).tools}[
                    "inspect_topology_neighbors"
                ]
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.open_world_hint is False
                assert set(tool.input_schema["properties"]) == {
                    "project_id",
                    "revision_id",
                    "selector",
                    "offset",
                    "limit",
                }
                assert set(tool.input_schema["required"]) == {
                    "project_id",
                    "revision_id",
                    "selector",
                }
                assert tool.output_schema is not None
                assert "inspect_topology_neighbors" in sdk.instructions
                assert "seam" in tool.description
                schemas.append(tool.input_schema)
                result = await sdk.call_tool("inspect_topology_neighbors", arguments(sid))
                assert not result.is_error, result.content
                assert json.loads(result.content[0].text) == result.structured_content
                results.append(result.structured_content)

    asyncio.run(scenario())
    selector = NeighborSelector.model_validate(arguments(sid)["selector"])
    assert results == [observed_result(sid, selector, 0, 40)] * 2
    assert schemas[0] == schemas[1]
    assert calls == [(sid, "alice", RID, selector, 0, 40, 30)] * 2
    assert records(service) == before
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("kind", ["face", "edge"])
def test_unique_neighbor_pages_preserve_full_seam_incidence(tmp_path, monkeypatch, surface, kind):
    service, sid, _ = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid, kind=kind)
            args["selector"] = {"body_index": 0, "kind": kind, "signature": "3" * 64}
            args.update(offset=0, limit=1)
            first = await sdk.call_tool("inspect_topology_neighbors", args)
            assert not first.is_error, first.content
            value = first.structured_content
            assert value["total_neighbors"] == 2 and value["total_incidences"] == 3
            assert len(value["neighbors"]) == 1 and value["neighbors"][0]["incidence_count"] == 2
            assert value["next_offset"] == 1
            args["offset"] = 1
            second = await sdk.call_tool("inspect_topology_neighbors", args)
            assert not second.is_error, second.content
            assert second.structured_content["neighbors"][0]["incidence_count"] == 1
            assert second.structured_content["next_offset"] is None
            args["offset"] = 7
            empty = await sdk.call_tool("inspect_topology_neighbors", args)
            assert not empty.is_error, empty.content
            assert empty.structured_content["neighbors"] == []
            assert empty.structured_content["total_neighbors"] == 2
            assert empty.structured_content["next_offset"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_foreign_owner_refused_before_service_dispatch(tmp_path, monkeypatch, surface):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface, owner="bob") as sdk:
            result = await sdk.call_tool("inspect_topology_neighbors", arguments(sid))
            assert result.is_error and result.structured_content is None

    asyncio.run(scenario())
    assert calls == [] and records(service) == before


def test_remote_fresh_read_scope_is_required(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        app = create_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
            ) as http,
        ):
            response = await http.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer no-read",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "inspect_topology_neighbors", "arguments": arguments(sid)},
                },
            )
            assert response.status_code == 403

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_missing_revision_is_not_a_success(tmp_path, monkeypatch, surface):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["revision_id"] = "f" * 32
            response = await sdk.call_tool("inspect_topology_neighbors", args)
            assert response.is_error and "not found" in response.content[0].text

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "change",
    [
        {"selector": {}},
        {"selector": {"body_index": True, "kind": "face", "signature": "3" * 64}},
        {
            "selector": {
                "body_index": 0,
                "occurrence_path": "x",
                "kind": "edge",
                "signature": "3" * 64,
            }
        },
        {"selector": {"body_index": 0, "kind": "body", "signature": "3" * 64}},
        {"selector": {"body_index": 0, "kind": "face", "signature": "bad"}},
        {"offset": True},
        {"offset": "0"},
        {"offset": -1},
        {"limit": 0},
        {"limit": 101},
    ],
)
def test_invalid_selectors_and_paging_do_not_dispatch(tmp_path, monkeypatch, surface, change):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args.update(change)
            assert (await sdk.call_tool("inspect_topology_neighbors", args)).is_error

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "f" * 32),
        ("revision_id", "f" * 32),
        ("offset", 1),
        ("limit", 39),
        ("native_completed", False),
        ("scope", "physical fit approved"),
        ("relation", "edge_incident_faces"),
    ],
)
def test_invalid_or_substituted_response_never_returns_success(
    tmp_path, monkeypatch, surface, field, value
):
    service, sid, _ = fixture(
        tmp_path, monkeypatch, transform=lambda result: result.update({field: value})
    )

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("inspect_topology_neighbors", arguments(sid))
            assert result.is_error and result.structured_content is None

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_nested_selector_substitution_is_rejected(tmp_path, monkeypatch, surface):
    def substitute(value):
        value["selector"]["requested"]["occurrence_path"] = "assembly/foreign"

    service, sid, _ = fixture(tmp_path, monkeypatch, transform=substitute)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            assert (await sdk.call_tool("inspect_topology_neighbors", arguments(sid))).is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("Source hash changed"),
        ValueError("Neighbor deadline exhausted"),
        OSError("private/host/path"),
    ],
)
def test_failures_are_explicit_and_os_paths_are_hidden(tmp_path, monkeypatch, surface, failure):
    service, sid, _ = fixture(tmp_path, monkeypatch, failure=failure)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("inspect_topology_neighbors", arguments(sid))
            assert result.is_error
            assert "private/host/path" not in result.content[0].text

    asyncio.run(scenario())
    assert records(service) == before


def test_http_query_is_strict_owned_and_targets_existing_route(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(router(service.store, service, lambda: "alice"))
    before = records(service)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            path = f"/workspace/sessions/{sid}/design/{RID}/topology-neighbors"
            body = {"selector": arguments(sid)["selector"], "offset": 1, "limit": 1}
            assert (await http.post(path, json={**body, "extra": True})).status_code == 422
            assert (await http.post(path, json={**body, "offset": "1"})).status_code == 422
            result = await http.post(path, json=body)
            assert result.status_code == 200
            assert result.json()["neighbors"][0]["entity"]["id"] == "neighbor-1"
            assert (await http.post(path.replace(sid, "f" * 32), json=body)).status_code == 404

    asyncio.run(scenario())
    assert calls == [
        (sid, "alice", RID, NeighborSelector.model_validate(arguments(sid)["selector"]), 1, 1, 30)
    ]
    assert records(service) == before


@pytest.mark.parametrize(
    "content",
    [b"[]", b"not json", b"x" * (MAX_RESPONSE_BYTES + 1)],
    ids=["array", "invalid-json", "oversized"],
)
def test_local_decode_and_response_budget_fail_closed(tmp_path, content):
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )
    with pytest.raises(ToolError):
        asyncio.run(
            adapter.inspect_topology_neighbors(
                "a" * 32, RID, NeighborSelector(body_index=0, kind="face", signature="3" * 64)
            )
        )


def test_local_operator_limit_remains_active(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "100")
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b" " * 101)),
    )
    with pytest.raises(ToolError, match="size limit"):
        asyncio.run(
            adapter.inspect_topology_neighbors(
                "a" * 32, RID, NeighborSelector(body_index=0, kind="face", signature="3" * 64)
            )
        )
