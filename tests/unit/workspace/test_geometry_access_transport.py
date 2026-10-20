"""Initialized SDK and owned HTTP access boundaries; no native geometry executes."""

from __future__ import annotations

import asyncio
import json
import math
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

from forge_mind.workspace import geometry_access
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.geometry_access_contract import (
    FRAME,
    MAX_RESPONSE_BYTES,
    SCOPE,
    AccessQuery,
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
            client_id="access-test",
            subject=token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


def query_payload(mode="static", distance=12):
    motion = {"mode": mode}
    if mode != "static":
        motion["distance_mm"] = distance
    return {
        "cylinder": {
            "center_mm": [0, 0, 0],
            "axis": [0, 0, 1],
            "radial_reference": [1, 0, 0],
            "radius_mm": 2,
            "length_mm": 10,
        },
        "motion": motion,
        "obstacles": [{"occurrence_path": "assembly/stationary"}],
    }


def arguments(sid, **kwargs):
    return {"project_id": sid, "revision_id": RID, "query": query_payload(**kwargs)}


def observed_result(sid, query):
    """Canned wire facts for the analytic R2/L10 cases, not native evidence."""
    mode = query.motion.mode
    distance = 0 if mode == "static" else query.motion.distance_mm
    translation = [distance, 0, 0] if mode == "transverse" else [0, 0, distance]
    lo, hi = [-2, -2, -5], [2, 2, 5]
    if mode == "axial":
        lo[2] += min(0, distance)
        hi[2] += max(0, distance)
    if mode == "transverse":
        lo[0] += min(0, distance)
        hi[0] += max(0, distance)
    volume = (
        (10 + abs(distance)) * 4 * math.pi
        if mode == "axial"
        else 40 * math.pi + (40 * abs(distance) if mode == "transverse" else 0)
    )
    requested = query.obstacles[0].model_dump(mode="json")
    obstacle = {"requested": requested, "body_index": 0, "body_signature": "c" * 64}
    return {
        "project_id": sid,
        "revision_id": RID,
        "operation": "cylindrical_access",
        "frame": FRAME,
        "units": "mm",
        "query": query.model_dump(mode="json"),
        "evaluated_cylinder": {
            "center_mm": [0, 0, 0],
            "axis": [0, 0, 1],
            "frame_axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
            "radius_mm": 2,
            "length_mm": 10,
        },
        "evaluated_motion": {
            "mode": mode,
            "translation_mm": translation,
            "start_center_mm": [0, 0, 0],
            "end_center_mm": translation,
        },
        "obstacles": [obstacle],
        "inventory_body_count": 3,
        "omitted_body_indices": [1, 2],
        "source_hashes": dict.fromkeys(("part.step", "topology.json", "document.json"), "e" * 64),
        "saved_engine_fingerprint": "1" * 64,
        "query_engine_fingerprint": "2" * 64,
        "sweep": {
            "method": {
                "static": "static_cylinder",
                "axial": "axial_extended_cylinder",
                "transverse": "transverse_capsule_prism",
            }[mode],
            "volume_mm3": volume,
            "bbox_min_mm": lo,
            "bbox_max_mm": hi,
        },
        "observations": [
            {
                "obstacle": obstacle,
                "overlap_components": [],
                "overlap_volume_mm3": 0,
                "minimum_distance_mm": 1,
                "witness_sweep_mm": [hi[0], 0, 0],
                "witness_obstacle_mm": [hi[0] + 1, 0, 0],
                "solution_count": 1,
                "native_inner_solution": False,
                "native_completed": True,
                "relation": "separated",
            }
        ],
        "numerical_guards": {"distance_mm": 1e-6, "volume_mm3": 1e-6},
        "native_completed": True,
        "scope": SCOPE,
    }


def fixture(tmp_path, monkeypatch, *, transform=None, failure=None):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Access reads must not start jobs"))
    sid = service.store.create("alice", "saved access", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls = []

    def inspect(actual_service, project_id, owner, revision_id, query, *, timeout):
        assert actual_service is service
        service.store.session_record(project_id, owner)
        service.store.revision(project_id, revision_id)
        calls.append((project_id, owner, revision_id, query, timeout))
        if failure:
            raise failure
        value = observed_result(project_id, query)
        if transform:
            transform(value)
        return value

    monkeypatch.setattr(geometry_access, "inspect_cylindrical_access", inspect)
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


def test_initialized_sdk_catalog_schema_parity_and_exact_dispatch(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)
    results, schemas = [], []

    async def scenario():
        for surface in ("local", "remote"):
            async with connection(service, tmp_path, surface) as sdk:
                tool = {t.name: t for t in (await sdk.list_tools()).tools}[
                    "inspect_cylindrical_access"
                ]
                assert tool.annotations.read_only_hint is True
                assert tool.annotations.open_world_hint is False
                assert set(tool.input_schema["properties"]) == {
                    "project_id",
                    "revision_id",
                    "query",
                }
                assert set(tool.input_schema["required"]) == {"project_id", "revision_id", "query"}
                assert tool.output_schema is not None
                assert "radial_reference" in json.dumps(tool.input_schema)
                assert "inspect_cylindrical_access" in sdk.instructions
                assert "omitted" in tool.description and "physical-fit" in tool.description
                assert sdk.server_info.version == ("0.0.19" if surface == "local" else "0.0.20")
                schemas.append(tool.input_schema)
                result = await sdk.call_tool("inspect_cylindrical_access", arguments(sid))
                assert not result.is_error, result.content
                assert json.loads(result.content[0].text) == result.structured_content
                results.append(result.structured_content)

    asyncio.run(scenario())
    query = AccessQuery.model_validate(query_payload())
    assert results == [observed_result(sid, query)] * 2
    assert schemas[0] == schemas[1]
    assert calls == [(sid, "alice", RID, query, 30)] * 2
    assert records(service) == before
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "mode,distance", [("axial", 12), ("axial", -12), ("transverse", 12), ("transverse", -12)]
)
def test_signed_modes_keep_exact_public_request_and_effective_translation(
    tmp_path, monkeypatch, surface, mode, distance
):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool(
                "inspect_cylindrical_access", arguments(sid, mode=mode, distance=distance)
            )
            assert not result.is_error, result.content
            expected = [distance, 0, 0] if mode == "transverse" else [0, 0, distance]
            assert result.structured_content["evaluated_motion"]["translation_mm"] == expected
            assert result.structured_content["omitted_body_indices"] == [1, 2]

    asyncio.run(scenario())
    assert calls == [
        (sid, "alice", RID, AccessQuery.model_validate(query_payload(mode, distance)), 30)
    ]


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_foreign_owner_and_missing_revision_refuse_before_dispatch(tmp_path, monkeypatch, surface):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface, owner="bob") as sdk:
            assert (await sdk.call_tool("inspect_cylindrical_access", arguments(sid))).is_error
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["revision_id"] = "f" * 32
            result = await sdk.call_tool("inspect_cylindrical_access", args)
            assert result.is_error and "not found" in result.content[0].text

    asyncio.run(scenario())
    assert calls == [] and records(service) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "change",
    [
        {"motion": {"mode": "static", "distance_mm": 0}},
        {"motion": {"mode": "axial", "distance_mm": 0}},
        {"motion": {"mode": "transverse", "translation_mm": [0, 15, 1]}},
        {"obstacles": [{"body_index": True}]},
        {"obstacles": [{"body_index": 0, "occurrence_path": "assembly/stationary"}]},
        {"obstacles": []},
    ],
)
def test_invalid_public_shape_never_dispatches(tmp_path, monkeypatch, surface, change):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["query"].update(change)
            result = await sdk.call_tool("inspect_cylindrical_access", args)
            assert result.is_error and result.structured_content is None

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_multibyte_query_byte_budget_refuses_before_service_dispatch(
    tmp_path, monkeypatch, surface
):
    service, sid, calls = fixture(tmp_path, monkeypatch)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            args = arguments(sid)
            args["query"]["obstacles"] = [
                {"occurrence_path": "\u6d4b" * 2040 + str(index)} for index in range(16)
            ]
            result = await sdk.call_tool("inspect_cylindrical_access", args)
            assert result.is_error and result.structured_content is None

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "f" * 32),
        ("revision_id", "f" * 32),
        ("native_completed", False),
        ("scope", "physical fit approved"),
        ("omitted_body_indices", [1]),
        ("observations", []),
    ],
)
def test_response_identity_completion_and_inventory_refuse_substitution(
    tmp_path, monkeypatch, surface, field, value
):
    service, sid, _ = fixture(
        tmp_path, monkeypatch, transform=lambda result: result.update({field: value})
    )

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("inspect_cylindrical_access", arguments(sid))
            assert result.is_error and result.structured_content is None

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("nested", ["query", "frame", "relation"])
def test_valid_looking_nested_response_cannot_replace_request_or_classification(
    tmp_path, monkeypatch, surface, nested
):
    def change(value):
        if nested == "query":
            value["query"]["cylinder"]["radius_mm"] = 3
            value["evaluated_cylinder"]["radius_mm"] = 3
        elif nested == "frame":
            value["evaluated_cylinder"]["frame_axes"]["x"] = [0, 1, 0]
        else:
            value["observations"][0]["relation"] = "contact_or_unresolved"

    service, sid, _ = fixture(tmp_path, monkeypatch, transform=change)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            assert (await sdk.call_tool("inspect_cylindrical_access", arguments(sid))).is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("Access deadline exhausted"),
        ValueError("Source hash changed"),
        OSError("private/host/path"),
    ],
)
def test_service_refusal_is_not_empty_clearance_or_store_mutation(
    tmp_path, monkeypatch, surface, failure
):
    service, sid, _ = fixture(tmp_path, monkeypatch, failure=failure)
    before = records(service)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("inspect_cylindrical_access", arguments(sid))
            assert result.is_error and result.structured_content is None
            assert "private/host/path" not in result.content[0].text

    asyncio.run(scenario())
    assert records(service) == before


def test_http_query_targets_owned_route_and_preserves_wire_scope(tmp_path, monkeypatch):
    service, sid, calls = fixture(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(router(service.store, service, lambda: "alice"))
    before = records(service)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            path = f"/workspace/sessions/{sid}/design/{RID}/cylindrical-access"
            body = query_payload()
            assert (await http.post(path, json={**body, "extra": True})).status_code == 422
            result = await http.post(path, json=body)
            assert result.status_code == 200 and result.json()["scope"] == SCOPE
            assert result.json()["query"] == AccessQuery.model_validate(body).model_dump(
                mode="json"
            )
            assert (await http.post(path.replace(sid, "f" * 32), json=body)).status_code == 404

    asyncio.run(scenario())
    assert calls == [(sid, "alice", RID, AccessQuery.model_validate(query_payload()), 30)]
    assert records(service) == before


def test_remote_authenticated_read_scope_is_required(tmp_path, monkeypatch):
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
            result = await http.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer no-read",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "inspect_cylindrical_access", "arguments": arguments(sid)},
                },
            )
            assert result.status_code == 403

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize(
    "content",
    [b"[]", b"not json", b"x" * (MAX_RESPONSE_BYTES + 1)],
    ids=["array", "invalid-json", "oversized"],
)
def test_local_decode_and_external_response_budget_fail_closed(tmp_path, content):
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )
    with pytest.raises(ToolError):
        asyncio.run(
            adapter.inspect_cylindrical_access(
                "a" * 32, RID, AccessQuery.model_validate(query_payload())
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
            adapter.inspect_cylindrical_access(
                "a" * 32, RID, AccessQuery.model_validate(query_payload())
            )
        )
