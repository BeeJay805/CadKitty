"""Authenticated section SDK transport controls; shared renderer is a narrow fixture."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError

from forge_mind.workspace import remote_mcp, section_inspection
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "a" * 32
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aCVcAAAAASUVORK5CYII="
)
SCOPE = (
    "Whole saved revision STL section in world CAD Z-up millimeters. Polylines are "
    "tessellated display intersections, not verified closed cavities or exact STEP topology. "
    "No full-wall thickness, passage continuity, sealing, strength, manufacturing suitability "
    "or physical approval is established."
)


class Verifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token not in {"alice", "bob", "no-read", "wrong-issuer"}:
            return None
        return AccessToken(
            token=token,
            subject=token,
            client_id="section-test",
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": "https://wrong.example" if token == "wrong-issuer" else ISSUER},
        )


def rpc(client, token, method, params):
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    return client.post(
        "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )


def call(client, token, sid, **options):
    return rpc(
        client,
        token,
        "tools/call",
        {
            "name": "inspect_section",
            "arguments": {
                "project_id": sid,
                "revision_id": RID,
                "axis": "z",
                "offset_mm": 4.0,
                **options,
            },
        },
    )


def result(response):
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.fixture
def remote(tmp_path: Path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor("docker"))
    sid = service.store.create("alice", "Saved hollow component", None)["id"]
    service.store.add_revision(
        sid, {"id": RID, "measurement": {"revision_id": RID, "artifact_hashes": {}}}
    )
    monkeypatch.setattr(
        service, "start", lambda: pytest.fail("Inspection cannot start a job worker")
    )
    actual_inspect = section_inspection.inspect_section
    observed = []
    receipt = {
        "project_id": sid,
        "revision_id": RID,
        "axis": "z",
        "offset_mm": 4.0,
        "frame": "cad_z_up_mm_world",
        "mesh_sha256": "b" * 64,
        "document_sha256": "c" * 64,
        "section_status": "intersected",
        "polyline_count": 2,
        "image": {"file": "section.png", "sha256": hashlib.sha256(PNG).hexdigest()},
        "scope": SCOPE,
    }
    value = {
        "receipt": receipt,
        "image_base64": base64.b64encode(PNG).decode("ascii"),
        "mime_type": "image/png",
    }

    def inspect(service, project_id, owner, revision_id, axis, offset_mm, *, timeout=30):
        service.store.session_record(project_id, owner)
        service.store.revision(project_id, revision_id)
        observed.append((project_id, owner, revision_id, axis, offset_mm, timeout))
        return copy.deepcopy(value)

    monkeypatch.setattr(section_inspection, "inspect_section", inspect)
    app = remote_mcp.create_app(
        service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service, sid, value, observed, actual_inspect


def test_intersected_receipt_and_real_image_have_exact_sdk_content(remote):
    client, service, sid, value, observed, _ = remote
    before = service.store.get(sid, "alice")
    reply = result(call(client, "alice", sid))
    assert not reply.get("isError")
    assert reply["structuredContent"] == {"receipt": value["receipt"]}
    assert reply["structuredContent"]["receipt"]["image"] == value["receipt"]["image"]
    assert [item["type"] for item in reply["content"]] == ["text", "image"]
    image = reply["content"][1]
    assert image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"], validate=True) == PNG
    assert value["image_base64"] not in reply["content"][0]["text"]
    assert observed == [(sid, "alice", RID, "z", 4.0, 30)]
    assert service.store.get(sid, "alice") == before


def test_empty_section_is_not_a_geometry_failure_or_image(remote):
    client, _, sid, value, _, _ = remote
    value["receipt"].update(section_status="empty", polyline_count=0, image=None)
    value.update(image_base64=None, mime_type=None)
    reply = result(call(client, "alice", sid))
    assert not reply.get("isError")
    assert reply["structuredContent"] == {"receipt": value["receipt"]}
    assert [item["type"] for item in reply["content"]] == ["text"]
    assert reply["structuredContent"]["receipt"]["image"] is None


@pytest.mark.parametrize("token", [None, "invalid", "no-read", "wrong-issuer", "bob"])
def test_fresh_auth_and_owner_denial_precede_service_dispatch(remote, token):
    client, _, sid, _, observed, _ = remote
    denied = call(client, token, sid)
    assert denied.status_code in {401, 403} or result(denied)["isError"]
    assert observed == []


@pytest.mark.parametrize(
    "options",
    [
        {"project_id": "../foreign"},
        {"revision_id": "invalid"},
        {"axis": "q"},
        {"axis": "Z"},
        {"offset_mm": "4"},
        {"offset_mm": True},
        {"offset_mm": None},
    ],
)
def test_invalid_plane_and_identifiers_refuse_before_dispatch(remote, options):
    client, _, sid, _, observed, _ = remote
    assert result(call(client, "alice", sid, **options))["isError"]
    assert observed == []


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), -float("inf")])
def test_public_sdk_rejects_nonfinite_plane(remote, monkeypatch, offset):
    _, service, sid, _, observed, _ = remote

    async def control():
        token = await Verifier().verify_token("alice")
        monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
        server = remote_mcp.create_server(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        with pytest.raises(ToolError):
            await server.call_tool(
                "inspect_section",
                {"project_id": sid, "revision_id": RID, "axis": "z", "offset_mm": offset},
            )

    asyncio.run(control())
    assert observed == []


def test_unknown_revision_returns_owned_unavailable(remote):
    client, _, sid, _, observed, _ = remote
    reply = result(call(client, "alice", sid, revision_id="d" * 32))
    assert reply["isError"] and "not found" in reply["content"][0]["text"]
    assert observed == []


def test_actual_service_refuses_unqualified_docker_without_worker(remote, monkeypatch):
    client, service, sid, _, observed, actual_inspect = remote
    assert not service.executor.typed_container_verified
    monkeypatch.setattr(section_inspection, "inspect_section", actual_inspect)
    reply = result(call(client, "alice", sid))
    assert reply["isError"]
    assert "qualified" in reply["content"][0]["text"] or "verified" in reply["content"][0]["text"]
    assert observed == [] and service.worker is None


@pytest.mark.parametrize("failure", ["stale", "timeout", "stop"])
def test_service_stale_deadline_and_stop_failures_stay_explicit(remote, monkeypatch, failure):
    client, _, sid, _, _, _ = remote

    def fail(*args, **kwargs):
        raise ValueError(
            {
                "stale": "Source hash changed",
                "timeout": "Section timed out",
                "stop": "Section cancelled",
            }[failure]
        )

    monkeypatch.setattr(section_inspection, "inspect_section", fail)
    reply = result(call(client, "alice", sid))
    assert reply["isError"]
    assert not any(item["type"] == "image" for item in reply["content"])


@pytest.mark.parametrize(
    "field,changed",
    [
        ("project_id", "d" * 32),
        ("revision_id", "d" * 32),
        ("axis", "x"),
        ("offset_mm", 5.0),
        ("frame", "local_part"),
        ("mesh_sha256", "invalid"),
        ("document_sha256", None),
        ("polyline_count", -1),
        ("section_status", "approved"),
    ],
)
def test_receipt_mismatch_is_never_an_image_success(remote, field, changed):
    client, _, sid, value, _, _ = remote
    value["receipt"][field] = changed
    reply = result(call(client, "alice", sid))
    assert reply["isError"]
    assert not any(item["type"] == "image" for item in reply["content"])


@pytest.mark.parametrize(
    "change",
    ["hash", "empty_with_image", "intersected_no_image", "mime", "png", "oversize", "receipt_size"],
)
def test_image_status_and_transfer_bounds_are_not_silent(remote, change):
    client, _, sid, value, _, _ = remote
    if change == "hash":
        value["receipt"]["image"]["sha256"] = "d" * 64
    elif change == "empty_with_image":
        value["receipt"].update(section_status="empty", polyline_count=0)
    elif change == "intersected_no_image":
        value.update(image_base64=None, mime_type=None)
    elif change == "mime":
        value["mime_type"] = "image/jpeg"
    elif change in {"png", "oversize"}:
        image = b"not PNG" if change == "png" else PNG + b"x" * (1024 * 1024)
        value["image_base64"] = base64.b64encode(image).decode("ascii")
        value["receipt"]["image"]["sha256"] = hashlib.sha256(image).hexdigest()
    else:
        value["receipt"]["scope"] = "x" * (64 * 1024 + 1)
    reply = result(call(client, "alice", sid))
    assert reply["isError"]
    assert not any(item["type"] == "image" for item in reply["content"])


def test_discovery_advertises_read_only_section_scope_and_new_version(remote):
    client, _, _, _, _, _ = remote
    initialized = result(
        rpc(
            client,
            "alice",
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "section-test", "version": "1"},
            },
        )
    )
    assert initialized["serverInfo"]["version"] == "0.0.20"
    assert "inspect_section" in initialized["instructions"]
    tools = result(rpc(client, "alice", "tools/list", {}))["tools"]
    section = next(tool for tool in tools if tool["name"] == "inspect_section")
    assert section["annotations"]["readOnlyHint"] is True
    assert set(section["inputSchema"]["properties"]) == {
        "project_id",
        "revision_id",
        "axis",
        "offset_mm",
    }
    assert section["inputSchema"]["properties"]["axis"]["enum"] == ["x", "y", "z"]
