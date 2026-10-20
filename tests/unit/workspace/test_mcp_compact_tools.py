from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp import Client

from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.models import Finding, Measurement
from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE

SID, RID, JID = "a" * 32, "b" * 32, "c" * 32


def test_local_sdk_delivers_parameter_advice_without_backend_mutation(tmp_path):
    calls = []

    def backend(request):
        calls.append(request.method)
        raise AssertionError("Discovery must not submit or change CAD")

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            assert client.instructions.count(PARAMETRIC_DESIGN_GUIDANCE) == 1
            assert "Only ai_create_project and ai_edit_project" in client.instructions
            catalog = {tool.name for tool in (await client.list_tools()).tools}
            assert {"submit_document", "edit_variables", "get_design_relationships"} <= catalog

    asyncio.run(scenario())
    assert calls == []


def measurement():
    return Measurement(
        revision_id=RID,
        contract_hash="contract",
        engine_fingerprint="engine",
        artifact_hashes={"part.step": "d" * 64},
        geometry_valid=True,
        facts={"large_details": list(range(3000))},
        findings=[
            Finding(
                requirement_id="width",
                status="pass",
                measured=20,
                expected=20,
                method="native",
                reason="Matched",
            ),
            Finding(
                requirement_id="fit",
                status="fail",
                measured=0.3000000004,
                expected=0.3,
                method="native",
                reason="Outside frozen tolerance",
                locator=[1, 2, 3],
            ),
            Finding(
                requirement_id="strength",
                status="unverified",
                method="unsupported",
                reason="Physical testing required",
            ),
        ],
        errors=["retained error"],
        caveats=["retained caveat"],
    ).model_dump()


def test_compact_evidence_sdk_schema_exact_findings_and_existing_full_wire(tmp_path):
    original = measurement()
    calls = []

    def backend(request):
        calls.append((request.method, request.url.path))
        assert request.url.path == f"/workspace/sessions/{SID}/revisions/{RID}/evidence.json"
        return httpx.Response(200, json=original)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            catalog = {t.name: t for t in (await client.list_tools()).tools}
            tool = catalog["get_evidence_summary"]
            assert tool.annotations.read_only_hint
            assert "findings" in tool.output_schema["properties"]
            args = {"project_id": SID, "revision_id": RID}
            full = await client.call_tool("get_evidence", args)
            assert not full.is_error and full.structured_content == original
            assert json.loads(full.content[0].text) == original
            compact = await client.call_tool("get_evidence_summary", args)
            assert not compact.is_error
            data = compact.structured_content
            assert data["findings"] == original["findings"][1:]
            assert data["integrity_scope"] == "evidence_record"
            assert data["counts"] == {"pass": 1, "fail": 1, "unverified": 1}
            assert data["omitted_pass_count"] == 1 and not data["all_findings_included"]
            assert data["errors"] == original["errors"]
            assert data["caveats"] == original["caveats"]
            text = compact.content[0].text
            assert "0.3000000004" in text and "Physical testing required" in text
            assert "does not rehash artifacts" in text
            assert "large_details" not in text and "facts" not in data
            assert len(compact.model_dump_json()) < len(full.model_dump_json())
            all_rows = []
            for offset in range(3):
                page = await client.call_tool(
                    "get_evidence_summary",
                    {**args, "statuses": "all", "offset": offset, "limit": 1},
                )
                assert not page.is_error
                all_rows += page.structured_content["findings"]
            assert all_rows == original["findings"]

    asyncio.run(scenario())
    assert all(method == "GET" for method, _ in calls)
    assert original == measurement()


@pytest.mark.parametrize("failure", ["foreign", "missing", "stale", "revision", "malformed"])
def test_compact_evidence_preserves_boundary_errors(tmp_path, failure):
    value = measurement()
    status = {"foreign": 403, "missing": 404, "stale": 409}.get(failure, 200)
    if failure == "revision":
        value["revision_id"] = "d" * 32
    if failure == "malformed":
        value.pop("findings")

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321",
            tmp_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(status, json=value)),
        )
        async with Client(create_server(adapter)) as client:
            response = await client.call_tool(
                "get_evidence_summary", {"project_id": SID, "revision_id": RID}
            )
            assert response.is_error
            assert response.structured_content is None

    asyncio.run(scenario())


def test_wait_sdk_observes_matching_job_and_forwards_bounded_reads(tmp_path):
    reads = []

    def backend(request):
        assert request.method == "GET"
        assert request.url.path == f"/workspace/sessions/{SID}/overview"
        reads.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={
                "id": SID,
                "job_id": JID,
                "state": "running" if len(reads) == 1 else "needs_review",
                "design_revision": RID,
                "selected_revision": None,
                "input_version": 1,
            },
        )

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            catalog = {t.name: t for t in (await client.list_tools()).tools}
            tool = catalog["wait_for_job"]
            assert tool.annotations.read_only_hint
            assert "outcome" in tool.output_schema["properties"]
            response = await client.call_tool(
                "wait_for_job",
                {
                    "project_id": SID,
                    "job_id": JID,
                    "timeout_seconds": 2,
                },
            )
            assert not response.is_error
            data = response.structured_content
            assert data["outcome"] == "terminal" and data["state"] == "needs_review"
            assert data["target_job_id"] == data["current_job_id"] == JID
            assert data["snapshot"]["design_revision"] == RID
            assert data["snapshot"]["accepted_revision"] is None

    asyncio.run(scenario())
    assert len(reads) == 2
    assert 0 < reads[1]["read"] < reads[0]["read"] <= 2


@pytest.mark.parametrize("status", [403, 404, 500])
def test_wait_http_errors_are_not_completion_or_timeout(tmp_path, status):
    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321",
            tmp_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(status, json={})),
        )
        async with Client(create_server(adapter)) as client:
            response = await client.call_tool(
                "wait_for_job",
                {
                    "project_id": SID,
                    "job_id": JID,
                    "timeout_seconds": 0.05,
                },
            )
            assert response.is_error and response.structured_content is None

    asyncio.run(scenario())
