from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from mcp import Client
from mcp.server import MCPServer

from forge_mind.workspace.mcp_server import READ, Identifier, WorkspaceAdapter, create_server

SID = "1" * 32
RID = "2" * 32
REPORT_ID = "3" * 32


def stored_report():
    return {
        "id": REPORT_ID,
        "revision_id": RID,
        "input_version": 0,
        "geometry_hash": "4" * 64,
        "input_hash": "5" * 64,
        "engine_fingerprint": "6" * 64,
        "frame": "cad_z_up_mm",
        "at": 1.0,
        "reviews": [
            {"category": "assembly_motion", "status": "unverified", "method": "seven native inputs"}
        ],
        "open_items": ['unknown "thread" and clearance\nµ'],
        "assembly_motion": {
            "status": "unverified",
            "coverage": "discrete_only",
            "errors": ["retained pose error"],
            "collisions": [
                {
                    "a": "body",
                    "b": "lid",
                    "overlap_mm3": 0.000000001,
                    "angle": 22.5,
                    "future_witness": {"exact_integer": 900719925474099312345, "float_one": 1.0},
                }
            ],
        },
        "geometry": {"facts": list(range(3000)), "errors": ["retained geometry error"]},
        "future_section": {"status": "fail", "witness": "not silently omitted"},
    }


def baseline_server(adapter):
    """Exercise the SDK's original dict-returning reader conversion independently."""
    server = MCPServer("Report reader baseline")

    @server.tool(annotations=READ)
    async def get_engineering_report(
        project_id: Identifier, report_id: Identifier
    ) -> dict[str, Any]:
        return dict(
            await adapter.json(
                "GET", f"/workspace/sessions/{project_id}/reports/{report_id}/report.json"
            )
        )

    return server


def test_sdk_full_default_explicit_full_and_output_schema_preserve_original_wire(tmp_path):
    original = stored_report()
    before = deepcopy(original)
    calls = []

    def backend(request):
        calls.append((request.method, request.url.path))
        assert request.method == "GET"
        assert request.url.path == f"/workspace/sessions/{SID}/reports/{REPORT_ID}/report.json"
        return httpx.Response(200, json=original)

    async def scenario():
        received = {}
        args = {"project_id": SID, "report_id": REPORT_ID}
        for name, builder in [("baseline", baseline_server), ("candidate", create_server)]:
            adapter = WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
            )
            async with Client(builder(adapter)) as client:
                catalog = {tool.name: tool for tool in (await client.list_tools()).tools}
                tool = catalog["get_engineering_report"]
                full = await client.call_tool("get_engineering_report", args)
                assert not full.is_error
                received[name] = {"tool": tool, "full": full.model_dump(mode="json", by_alias=True)}
                if name == "candidate":
                    explicit = await client.call_tool(
                        "get_engineering_report", {**args, "view": "full"}
                    )
                    assert explicit.model_dump(mode="json", by_alias=True) == received[name]["full"]
                    overview = await client.call_tool(
                        "get_engineering_report", {**args, "view": "overview"}
                    )
                    assert not overview.is_error
                    assert overview.structured_content["reviews"] == original["reviews"]
                    assert overview.structured_content["open_items"] == original["open_items"]
                    assert overview.structured_content["nested_failure_inventory_complete"] is False
                    assert "collisions" not in overview.content[0].text
                    assert len(overview.content) == 1
                    exact = await client.call_tool(
                        "get_engineering_report",
                        {**args, "view": "section", "section": "assembly_motion"},
                    )
                    assert not exact.is_error
                    assert exact.structured_content["value"] == original["assembly_motion"]
                    assert (
                        json.loads(json.dumps(exact.structured_content))["value"]
                        == original["assembly_motion"]
                    )
                    assert "section assembly_motion" in exact.content[0].text
        # This independent SDK baseline has a different server identity. Compare the
        # exact reader content/structure; the separate c21e receipt compares full wire.
        for field in ("content", "structuredContent", "isError", "resultType"):
            assert received["baseline"]["full"].get(field) == received["candidate"]["full"].get(
                field
            )
        baseline_tool = received["baseline"]["tool"]
        candidate_tool = received["candidate"]["tool"]
        assert baseline_tool.output_schema == candidate_tool.output_schema
        assert baseline_tool.annotations == candidate_tool.annotations
        new_input = deepcopy(candidate_tool.input_schema)
        del new_input["properties"]["view"]
        del new_input["properties"]["section"]
        assert new_input == baseline_tool.input_schema

    asyncio.run(scenario())
    assert len(calls) == 5
    assert original == before


def test_invalid_view_section_combinations_refuse_before_owned_reader(tmp_path):
    calls = []

    def backend(request):
        calls.append(request)
        raise AssertionError("Invalid selection must not reach backend")

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            for extra in [
                {"view": "unknown"},
                {"view": "section"},
                {"view": "overview", "section": "assembly_motion"},
                {"view": "full", "section": "geometry"},
                {"view": "section", "section": "future_section"},
                {"view": "section", "section": "../report.json"},
            ]:
                response = await client.call_tool(
                    "get_engineering_report", {"project_id": SID, "report_id": REPORT_ID, **extra}
                )
                assert response.is_error
                assert not response.structured_content

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("status", [401, 403, 404, 409])
@pytest.mark.parametrize(
    "view,section", [("full", None), ("overview", None), ("section", "assembly_motion")]
)
def test_report_views_preserve_owned_reader_authorization_and_stale_failures(
    tmp_path, status, view, section
):
    calls = []

    def backend(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == f"/workspace/sessions/{SID}/reports/{REPORT_ID}/report.json"
        return httpx.Response(status, json={"detail": "owner or immutable report check refused"})

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            args = {"project_id": SID, "report_id": REPORT_ID, "view": view}
            if section is not None:
                args["section"] = section
            response = await client.call_tool("get_engineering_report", args)
            assert response.is_error
            assert not response.structured_content

    asyncio.run(scenario())
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["wrong_report_id", "malformed_metadata", "absent_section"])
def test_malformed_views_fail_without_falling_back_to_full_report(tmp_path, mode):
    original = stored_report()
    section = "assembly_motion"
    if mode == "wrong_report_id":
        original["id"] = "f" * 32
    elif mode == "malformed_metadata":
        original["input_version"] = True
    else:
        section = "production"

    def backend(request):
        return httpx.Response(200, json=original)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            response = await client.call_tool(
                "get_engineering_report",
                {"project_id": SID, "report_id": REPORT_ID, "view": "section", "section": section},
            )
            assert response.is_error
            assert not response.structured_content

    asyncio.run(scenario())
