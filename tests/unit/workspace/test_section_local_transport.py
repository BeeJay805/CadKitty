"""Local SDK and actual HTTP route controls; section worker is mocked, not CAD proof."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json

import httpx
import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import SectionInspectionRequest, router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import (
    MAX_SECTION_IMAGE_BYTES,
    MAX_SECTION_RECEIPT_BYTES,
    MAX_SECTION_RESPONSE_BYTES,
    WorkspaceAdapter,
    create_server,
    section_tool_result,
)
from forge_mind.workspace.store import Store

RID = "b" * 32
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="
)


def envelope(sid, axis="z", offset=3.0, *, empty=False):
    return {
        "receipt": {
            "project_id": sid,
            "revision_id": RID,
            "axis": axis,
            "offset_mm": offset,
            "frame": "cad_z_up_mm_world",
            "mesh_sha256": "c" * 64,
            "document_sha256": "d" * 64,
            "section_status": "empty" if empty else "intersected",
            "polyline_count": 0 if empty else 2,
            "image": None
            if empty
            else {"file": "section.png", "sha256": hashlib.sha256(PNG).hexdigest()},
            "scope": "Whole saved revision mesh display; not verified closed cavities or physical approval.",
        },
        "image_base64": None if empty else base64.b64encode(PNG).decode(),
        "mime_type": None if empty else "image/png",
    }


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from forge_mind.workspace import section_inspection

    service = WorkspaceService(Store(tmp_path), Executor())
    monkeypatch.setattr(
        service, "start", lambda: pytest.fail("Section cannot start a generation job")
    )
    sid = service.store.create("alice", "saved hollow part", None)["id"]
    service.store.add_revision(sid, {"id": RID, "measurement": {"revision_id": RID}})
    calls, failures = [], []

    def inspect(svc, project, owner, revision, axis, offset_mm, *, timeout):
        svc.store.session_record(project, owner)
        svc.store.revision(project, revision)
        calls.append((project, owner, revision, axis, offset_mm, timeout))
        if failures:
            if isinstance(failures[0], Exception):
                raise failures[0]
            raise ValueError(failures[0])
        return envelope(project, axis, offset_mm, empty=offset_mm == 999.0)

    monkeypatch.setattr(section_inspection, "inspect_section", inspect)
    return service, sid, calls, failures


def app(service, fixed_owner=None):
    def owner(authorization: str = Header()):
        return authorization

    application = FastAPI()
    application.include_router(
        router(service.store, service, (lambda: fixed_owner) if fixed_owner else owner)
    )
    return application


def test_http_owner_first_exact_plane_empty_and_no_mutation(boundary):
    service, sid, calls, _ = boundary
    before = service.store.get(sid, "alice")
    url = f"/workspace/sessions/{sid}/design/{RID}/inspect-section"
    with TestClient(app(service)) as client:
        assert (
            client.post(
                url, headers={"Authorization": "bob"}, json={"axis": "x", "offset_mm": -2}
            ).status_code
            == 404
        )
        assert calls == []
        observed = client.post(
            url, headers={"Authorization": "alice"}, json={"axis": "x", "offset_mm": -2}
        )
        assert observed.status_code == 200 and observed.json() == envelope(sid, "x", -2)
        empty = client.post(
            url, headers={"Authorization": "alice"}, json={"axis": "y", "offset_mm": 999}
        )
        assert empty.status_code == 200 and empty.json() == envelope(sid, "y", 999, empty=True)
    assert calls == [(sid, "alice", RID, "x", -2.0, 30), (sid, "alice", RID, "y", 999.0, 30)]
    assert service.store.get(sid, "alice") == before
    with service.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"axis": "q", "offset_mm": 0},
        {"axis": "z", "offset_mm": True},
        {"axis": "z", "offset_mm": "3"},
        {"axis": "z", "offset_mm": None},
        {"axis": "z"},
        {"axis": "z", "offset_mm": 3, "owner": "bob"},
        {"axis": "z", "offset_mm": 3, "occurrence_path": "part"},
        {"axis": "z", "offset_mm": 3, "timeout": 1000},
    ],
)
def test_http_invalid_plane_and_extra_fields_refused_before_service(boundary, body):
    service, sid, calls, _ = boundary
    with TestClient(app(service, "alice")) as client:
        response = client.post(f"/workspace/sessions/{sid}/design/{RID}/inspect-section", json=body)
    assert response.status_code == 422 and calls == []


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf")])
def test_http_request_model_refuses_nonfinite_planes(offset):
    with pytest.raises(ValueError):
        SectionInspectionRequest(axis="z", offset_mm=offset)


def test_http_missing_revision_and_worker_refusals_stay_errors(boundary):
    service, sid, calls, failures = boundary
    with TestClient(app(service, "alice")) as client:
        body = {"axis": "z", "offset_mm": 3}
        assert (
            client.post(
                f"/workspace/sessions/{sid}/design/{'f' * 32}/inspect-section", json=body
            ).status_code
            == 404
        )
        assert calls == []
        for reason in (
            "Section source changed; evidence is stale",
            "Section images require qualified isolated execution",
            "Section cancelled or deadline exhausted",
        ):
            failures[:] = [reason]
            response = client.post(
                f"/workspace/sessions/{sid}/design/{RID}/inspect-section", json=body
            )
            assert response.status_code == 409 and response.json()["detail"] == reason


@pytest.mark.parametrize(
    "error", [FileNotFoundError("PRIVATE-HOST-PATH"), OSError("PRIVATE-HOST-PATH")]
)
def test_http_missing_or_unreadable_files_are_sanitized_refusals(boundary, error):
    service, sid, _, failures = boundary
    failures[:] = [error]
    before = service.store.get(sid, "alice")
    with TestClient(app(service, "alice")) as client:
        response = client.post(
            f"/workspace/sessions/{sid}/design/{RID}/inspect-section",
            json={"axis": "z", "offset_mm": 3},
        )
    assert response.status_code == 409
    assert response.json()["detail"] == "Section source or output files are unavailable"
    assert "PRIVATE-HOST-PATH" not in response.text
    assert service.store.get(sid, "alice") == before


def test_real_local_sdk_asgi_image_empty_receipt_and_schema(boundary, tmp_path):
    service, sid, calls, _ = boundary

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321",
            tmp_path / "unused-exports",
            transport=httpx.ASGITransport(app=app(service, "alice")),
        )
        async with Client(create_server(adapter), cache=None) as sdk:
            assert sdk.server_info.version == "0.0.19"
            assert "inspect_section" in sdk.instructions
            tool = {t.name: t for t in (await sdk.list_tools()).tools}["inspect_section"]
            assert tool.annotations.read_only_hint is True
            assert set(tool.input_schema["properties"]) == {
                "project_id",
                "revision_id",
                "axis",
                "offset_mm",
            }
            assert tool.input_schema["properties"]["axis"]["enum"] == ["x", "y", "z"]
            assert tool.output_schema is not None
            for axis, offset in (("z", 3), ("y", 999)):
                result = await sdk.call_tool(
                    "inspect_section",
                    {"project_id": sid, "revision_id": RID, "axis": axis, "offset_mm": offset},
                )
                assert not result.is_error, result.content
                assert result.structured_content == {
                    "receipt": envelope(sid, axis, offset, empty=offset == 999)["receipt"]
                }
                assert json.loads(result.content[0].text) == result.structured_content
                assert "image_base64" not in result.content[0].text
                if offset == 999:
                    assert len(result.content) == 1
                else:
                    assert result.content[1].mime_type == "image/png"
                    assert base64.b64decode(result.content[1].data, validate=True) == PNG

    asyncio.run(scenario())
    assert len(calls) == 2
    assert not (tmp_path / "unused-exports").exists()


@pytest.mark.parametrize(
    "field,value",
    [("axis", "q"), ("offset_mm", True), ("offset_mm", "3"), ("project_id", "invalid")],
)
def test_sdk_invalid_arguments_have_no_http_dispatch(tmp_path, field, value):
    calls = []

    def backend(request):
        calls.append(request)
        return httpx.Response(500)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter), cache=None) as sdk:
            args = {"project_id": "a" * 32, "revision_id": RID, "axis": "z", "offset_mm": 3}
            args[field] = value
            assert (await sdk.call_tool("inspect_section", args)).is_error

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "e" * 32),
        ("revision_id", "e" * 32),
        ("axis", "x"),
        ("offset_mm", 4.0),
        ("offset_mm", True),
        ("offset_mm", float("inf")),
        ("frame", "definition_local"),
        ("mesh_sha256", "bad"),
        ("document_sha256", "bad"),
        ("section_status", "unknown"),
        ("polyline_count", 0),
        ("polyline_count", True),
        ("polyline_count", 1.5),
        ("scope", ""),
        ("image", None),
    ],
)
def test_mismatched_or_malformed_receipt_is_not_displayed(field, value):
    observed = envelope("a" * 32)
    observed["receipt"][field] = value
    with pytest.raises(ToolError):
        section_tool_result(observed, "a" * 32, RID, "z", 3)


@pytest.mark.parametrize(
    "mutation",
    [
        "base64",
        "png",
        "hash",
        "file",
        "mime",
        "missing_image",
        "receipt_size",
        "image_size",
        "response_size",
        "empty_image",
        "empty_count",
        "missing_image_key",
    ],
)
def test_image_and_empty_invariants_and_byte_caps(mutation):
    observed = envelope("a" * 32, empty=mutation.startswith("empty"))
    if mutation == "base64":
        observed["image_base64"] = "!invalid!"
    elif mutation == "png":
        observed["image_base64"] = base64.b64encode(b"not png").decode()
    elif mutation == "hash":
        observed["receipt"]["image"]["sha256"] = "e" * 64
    elif mutation == "file":
        observed["receipt"]["image"]["file"] = "../section.png"
    elif mutation == "mime":
        observed["mime_type"] = "image/jpeg"
    elif mutation == "missing_image":
        observed["image_base64"] = None
    elif mutation == "receipt_size":
        observed["receipt"]["scope"] = "x" * (MAX_SECTION_RECEIPT_BYTES + 1)
    elif mutation == "image_size":
        image = PNG + b"x" * MAX_SECTION_IMAGE_BYTES
        observed["image_base64"] = base64.b64encode(image).decode()
        observed["receipt"]["image"]["sha256"] = hashlib.sha256(image).hexdigest()
    elif mutation == "response_size":
        observed["unexpected"] = "x" * (MAX_SECTION_RESPONSE_BYTES + 1)
    elif mutation == "empty_image":
        observed["image_base64"] = base64.b64encode(PNG).decode()
    elif mutation == "empty_count":
        observed["receipt"]["polyline_count"] = 1
    else:
        observed["receipt"].pop("image")
    with pytest.raises(ToolError):
        section_tool_result(observed, "a" * 32, RID, "z", 3)


def test_receipt_unknown_metadata_preserved_without_input_mutation():
    value = envelope("a" * 32)
    value["receipt"]["future_observation"] = {"exact": 9007199254740993, "note": "no cavity claim"}
    before = copy.deepcopy(value)
    result = section_tool_result(value, "a" * 32, RID, "z", 3)
    assert value == before
    assert result.structured_content["receipt"] == before["receipt"]


@pytest.mark.parametrize("data", [b"invalid json", b"[]", b"null"])
def test_adapter_invalid_json_is_an_explicit_failure(tmp_path, data):
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=data)),
    )
    with pytest.raises(ToolError, match="invalid section JSON"):
        asyncio.run(adapter.inspect_section("a" * 32, RID, "z", 3))


def test_adapter_response_cap_before_json_decode(tmp_path):
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * (MAX_SECTION_RESPONSE_BYTES + 1))
        ),
    )
    with pytest.raises(ToolError, match="size limit"):
        asyncio.run(adapter.inspect_section("a" * 32, RID, "z", 3))
