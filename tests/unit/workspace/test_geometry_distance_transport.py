"""Initialized SDK and owned HTTP distance boundaries with a native-free service fake."""

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

from forge_mind.workspace import geometry_distance
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.geometry_distance_contract import (
    FRAME,
    MAX_RESPONSE_BYTES,
    SCOPE,
    DistanceSelector,
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
            client_id="distance-test",
            subject=token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


def arguments(sid):
    return {
        "project_id": sid,
        "revision_id": RID,
        "a": {"body_index": 0},
        "b": {"occurrence_path": "assembly/second"},
    }


def observed_result(sid, a, b):
    return {
        "project_id": sid,
        "revision_id": RID,
        "operation": "minimum_distance",
        "frame": FRAME,
        "units": "mm",
        "a": {
            "requested": a.model_dump(mode="json"),
            "body_index": a.body_index if a.body_index is not None else 0,
            "body_signature": "c" * 64,
            "face_signature": a.face_signature,
        },
        "b": {
            "requested": b.model_dump(mode="json"),
            "body_index": b.body_index if b.body_index is not None else 1,
            "body_signature": ("c" if b.body_index == 0 else "d") * 64,
            "face_signature": b.face_signature,
        },
        "source_hashes": dict.fromkeys(("part.step", "topology.json", "document.json"), "e" * 64),
        # Different saved/query engines are valid response identities, not a transport refusal.
        "saved_engine_fingerprint": "1" * 64,
        "query_engine_fingerprint": "2" * 64,
        "minimum_distance_mm": 2.0,
        "witness_a_mm": [0.0, 0.0, 0.0],
        "witness_b_mm": [2.0, 0.0, 0.0],
        "solution_count": 4,
        "native_completed": True,
        "native_inner_solution": False,
        "scope": SCOPE,
    }


def fixture(tmp_path, monkeypatch, transform=None, failure=None):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Distance must not start jobs"))
    sid = service.store.create("alice", "saved distance", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls = []

    def measure(actual_service, project_id, owner, revision_id, a, b, *, timeout):
        assert actual_service is service
        service.store.session_record(project_id, owner)
        service.store.revision(project_id, revision_id)
        calls.append((project_id, owner, revision_id, a, b, timeout))
        if failure:
            raise failure
        value = observed_result(project_id, a, b)
        if transform:
            transform(value)
        return value

    monkeypatch.setattr(geometry_distance, "measure_distance", measure)
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


def test_initialized_sdk_catalog_result_and_http_target_parity(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)
    results = []
    schemas = []

    async def scenario():
        for surface in ("local", "remote"):
            async with connection(service, tmp_path, surface) as sdk:
                tool = {t.name: t for t in (await sdk.list_tools()).tools}["measure_distance"]
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.open_world_hint is False
                assert set(tool.input_schema["properties"]) == {
                    "project_id",
                    "revision_id",
                    "a",
                    "b",
                }
                assert set(tool.input_schema["required"]) == {"project_id", "revision_id", "a", "b"}
                assert tool.output_schema is not None
                assert "measure_distance" in sdk.instructions
                assert "Zero does not distinguish" in tool.description
                schemas.append(tool.input_schema)
                result = await sdk.call_tool("measure_distance", arguments(sid))
                assert not result.is_error, result.content
                assert json.loads(result.content[0].text) == result.structured_content
                results.append(result.structured_content)

    asyncio.run(scenario())
    a = DistanceSelector(body_index=0)
    b = DistanceSelector(occurrence_path="assembly/second")
    assert results == [observed_result(sid, a, b)] * 2
    assert schemas[0] == schemas[1]
    assert calls == [(sid, "alice", RID, a, b, 30)] * 2
    assert records(service) == before
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_distinct_faces_of_one_body_are_transported(tmp_path, monkeypatch, surface):
    service, sid, _ = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args.update(
                a={"body_index": 0, "face_signature": "3" * 64},
                b={"body_index": 0, "face_signature": "4" * 64},
            )
            result = await sdk.call_tool("measure_distance", args)
            assert not result.is_error, result.content
            assert (
                result.structured_content["a"]["body_index"]
                == result.structured_content["b"]["body_index"]
                == 0
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_foreign_owner_refused_before_service_dispatch(tmp_path, monkeypatch, surface):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface, owner="bob") as sdk:
            result = await sdk.call_tool("measure_distance", arguments(sid))
            assert result.is_error
            assert result.structured_content is None

    asyncio.run(scenario())
    assert calls == []
    assert records(service) == before


def test_remote_fresh_read_scope_refused_before_service_dispatch(tmp_path, monkeypatch):
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
                    "params": {"name": "measure_distance", "arguments": arguments(sid)},
                },
            )
            assert response.status_code == 403

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_unknown_revision_is_not_a_measurement_success(tmp_path, monkeypatch, surface):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["revision_id"] = "f" * 32
            result = await sdk.call_tool("measure_distance", args)
            assert result.is_error
            assert "not found" in result.content[0].text

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_zero_preserves_native_facts_without_collision_classification(
    tmp_path, monkeypatch, surface
):
    def zero(value):
        value.update(
            minimum_distance_mm=0.0, witness_b_mm=[0.0, 0.0, 0.0], native_inner_solution=True
        )

    service, sid, _ = fixture(tmp_path, monkeypatch, transform=zero)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("measure_distance", arguments(sid))
            assert not result.is_error, result.content
            value = result.structured_content
            assert value["minimum_distance_mm"] == 0.0
            assert value["native_inner_solution"] is True
            assert value["scope"] == SCOPE
            assert not ({"collision", "overlap", "accepted"} & set(value))

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "selector",
    [
        {},
        {"body_index": True},
        {"body_index": "0"},
        {"body_index": -1},
        {"body_index": 0, "occurrence_path": "x"},
        {"occurrence_path": ""},
        {"body_index": 0, "face_signature": "bad"},
        {"body_index": 0, "extra": 1},
    ],
)
def test_sdk_selector_schema_refusal_has_no_dispatch(tmp_path, monkeypatch, surface, selector):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["a"] = selector
            assert (await sdk.call_tool("measure_distance", args)).is_error

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "f" * 32),
        ("revision_id", "f" * 32),
        ("frame", "local"),
        ("native_completed", False),
        ("minimum_distance_mm", 3.0),
        ("scope", "physical fit approved"),
    ],
)
def test_invalid_or_mismatched_results_never_return_success(
    tmp_path, monkeypatch, surface, field, value
):
    service, sid, _ = fixture(
        tmp_path, monkeypatch, transform=lambda result: result.update({field: value})
    )

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            response = await sdk.call_tool("measure_distance", arguments(sid))
            assert response.is_error
            assert response.structured_content is None

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_resolved_request_substitution_is_rejected(tmp_path, monkeypatch, surface):
    def replace(value):
        value["b"]["requested"]["occurrence_path"] = "assembly/foreign"

    service, sid, _ = fixture(tmp_path, monkeypatch, transform=replace)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            assert (await sdk.call_tool("measure_distance", arguments(sid))).is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("Source hash changed"),
        ValueError("Distance deadline exhausted"),
        OSError("private/host/path"),
    ],
)
def test_service_failures_are_explicit_and_os_paths_are_hidden(
    tmp_path, monkeypatch, surface, failure
):
    service, sid, _ = fixture(tmp_path, monkeypatch, failure=failure)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            response = await sdk.call_tool("measure_distance", arguments(sid))
            assert response.is_error
            assert "private/host/path" not in response.content[0].text

    asyncio.run(scenario())
    assert records(service) == before


def test_http_query_is_strict_owned_and_read_only(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(router(service.store, service, lambda: "alice"))
    before = records(service)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            path = f"/workspace/sessions/{sid}/design/{RID}/measure-distance"
            assert (
                await http.post(path, json={"a": {"body_index": True}, "b": {"body_index": 1}})
            ).status_code == 422
            assert (
                await http.post(
                    path, json={"a": {"body_index": 0}, "b": {"body_index": 1}, "extra": True}
                )
            ).status_code == 422
            response = await http.post(path, json={"a": {"body_index": 0}, "b": {"body_index": 1}})
            assert response.status_code == 200
            assert response.json()["revision_id"] == RID
            assert (
                await http.post(
                    path.replace(sid, "f" * 32),
                    json={"a": {"body_index": 0}, "b": {"body_index": 1}},
                )
            ).status_code == 404

    asyncio.run(scenario())
    assert len(calls) == 1
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
            adapter.measure_distance(
                "a" * 32, RID, DistanceSelector(body_index=0), DistanceSelector(body_index=1)
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
            adapter.measure_distance(
                "a" * 32, RID, DistanceSelector(body_index=0), DistanceSelector(body_index=1)
            )
        )
