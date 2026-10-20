from __future__ import annotations

import asyncio
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from mcp import Client
from tests.unit.workspace.test_remote_mcp import (
    call,
    remote,  # noqa: F401
    result,
)

from forge_mind.workspace.engineering_api import engineering_router
from forge_mind.workspace.hardware import search
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

CASES = [
    ({}, {}, "bearing"),
    ({"query": "washer"}, {}, "washer"),
    ({"query": ""}, {}, ""),
    ({"query": None}, {}, "bearing"),
    (
        {
            "family": "screw",
            "subtype": "socket_head_cap_screw",
            "thread_d": 3,
            "length": 20,
            "limit": 2,
        },
        {
            "family": "screw",
            "subtype": "socket_head_cap_screw",
            "thread_d": 3,
            "length": 20,
            "limit": 2,
        },
        "",
    ),
    ({"family": "washer", "query": "m3", "limit": 1}, {"family": "washer", "limit": 1}, "m3"),
    ({"family": "screw", "thread_d": 3.001}, {"family": "screw", "thread_d": 3.001}, ""),
]
INVALID = [
    {"limit": 0},
    {"limit": 501},
    {"limit": True},
    {"limit": 1.5},
    {"thread_d": 0},
    {"thread_d": -1},
    {"thread_d": 100001},
    {"thread_d": True},
    {"thread_d": "nan"},
    {"thread_d": "inf"},
    {"length": 0},
    {"length": -1},
    {"length": 100001},
    {"length": False},
    {"length": "nan"},
    {"query": "x" * 101},
    {"family": "x" * 101},
    {"subtype": "x" * 101},
]


@pytest.mark.parametrize("arguments,filters,query", CASES)
def test_local_sdk_uses_real_catalog_route_without_changing_results(
    tmp_path, arguments, filters, query
):
    app = FastAPI()
    app.include_router(engineering_router(Mock(), Mock(), lambda: "owner"))
    requests = []

    @app.middleware("http")
    async def capture(request, call_next):
        requests.append((request.method, request.url.path, dict(request.query_params)))
        return await call_next(request)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.ASGITransport(app=app)
        )
        async with Client(create_server(adapter)) as client:
            value = await client.call_tool("search_catalog", arguments)
            assert not value.is_error
            assert value.structured_content == {
                "results": jsonable_encoder(search(query, **filters))
            }

    asyncio.run(scenario())
    assert len(requests) == 1
    method, path, params = requests[0]
    assert (method, path) == ("GET", "/workspace/catalog")
    assert params["limit"] == str(arguments.get("limit", 30))
    if arguments.get("query") is None:
        assert "q" not in params
    else:
        assert params["q"] == arguments["query"]


@pytest.mark.parametrize("arguments,filters,query", CASES)
def test_remote_filtered_read_preserves_catalog_sources_and_needs_no_project(
    request, arguments, filters, query
):
    client, service = request.getfixturevalue("remote")
    value = result(call(client, "readonly", "search_catalog", arguments))
    assert not value.get("isError")
    assert value["structuredContent"] == {"results": jsonable_encoder(search(query, **filters))}
    assert service.store.list_sessions("readonly") == []


@pytest.mark.parametrize("arguments", INVALID)
def test_local_invalid_arguments_never_reach_backend(tmp_path, arguments):
    requests = []

    def backend(request):
        requests.append(request)
        raise AssertionError("Invalid tool arguments must fail before the catalog route")

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            assert (await client.call_tool("search_catalog", arguments)).is_error

    asyncio.run(scenario())
    assert requests == []


@pytest.mark.parametrize("arguments", INVALID)
def test_remote_invalid_arguments_never_search(request, monkeypatch, arguments):
    client, _ = request.getfixturevalue("remote")
    searched = []

    def forbidden(*args, **kwargs):
        searched.append((args, kwargs))
        raise AssertionError("Invalid tool arguments must fail before catalog search")

    monkeypatch.setattr("forge_mind.workspace.hardware.search", forbidden)
    value = result(call(client, "readonly", "search_catalog", arguments))
    assert value["isError"]
    assert searched == []


@pytest.mark.parametrize("token", [None, "wrong-resource", "wrong-issuer", "expired", "no-subject"])
def test_remote_denied_identity_never_searches(request, monkeypatch, token):
    client, _ = request.getfixturevalue("remote")
    searched = []

    def forbidden(*args, **kwargs):
        searched.append((args, kwargs))
        raise AssertionError("Unauthenticated callers must not search")

    monkeypatch.setattr("forge_mind.workspace.hardware.search", forbidden)
    response = call(client, token, "search_catalog")
    assert response.status_code in (401, 403) or result(response).get("isError")
    assert searched == []


def test_catalog_schemas_match_and_expose_read_only_bounded_filters(tmp_path, request):
    client, _ = request.getfixturevalue("remote")
    response = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer readonly",
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    remote_tool = next(
        tool for tool in result(response)["tools"] if tool["name"] == "search_catalog"
    )

    async def scenario():
        def forbidden(request):
            raise AssertionError("Tool discovery does not call the backend")

        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(forbidden)
        )
        async with Client(create_server(adapter)) as local:
            tool = next(
                tool for tool in (await local.list_tools()).tools if tool.name == "search_catalog"
            )
            assert tool.annotations.read_only_hint
            assert tool.input_schema == remote_tool["inputSchema"]
            properties = tool.input_schema["properties"]
            assert set(properties) == {"query", "family", "subtype", "thread_d", "length", "limit"}
            assert properties["query"]["default"] is None
            assert properties["limit"]["default"] == 30
            assert properties["limit"]["minimum"] == 1
            assert properties["limit"]["maximum"] == 500

    asyncio.run(scenario())
    assert remote_tool["annotations"]["readOnlyHint"] is True


def test_remote_transfer_limit_still_applies(request, monkeypatch):
    client, _ = request.getfixturevalue("remote")
    monkeypatch.setattr(
        "forge_mind.workspace.hardware.search",
        lambda *args, **kwargs: [{"source": "x" * (2 * 1024 * 1024)}],
    )
    value = result(call(client, "readonly", "search_catalog", {"limit": 500}))
    assert value["isError"]
    assert "transfer limit" in value["content"][0]["text"]
