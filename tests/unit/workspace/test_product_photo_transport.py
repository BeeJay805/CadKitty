"""Real SDK/HTTP transport controls; photo rendering is mocked, never launched."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import struct
import time
import zlib

import httpx
import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.auth.provider import AccessToken
from mcp.server.mcpserver.exceptions import ToolError

from forge_mind.workspace import mcp_server, product_photo, remote_mcp
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.product_photo_contract import FRAME, SCOPE
from forge_mind.workspace.store import Store

RID = "b" * 32
ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"


def png(width=1024, height=1024):
    """Construct a complete bounded fixture PNG without native or image imports."""

    def chunk(name, data):
        return (
            struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = (b"\0" + b"\x88\x99\xaa" * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


PNG = png()


def envelope(sid, preset="metal"):
    return {
        "receipt": {
            "project_id": sid,
            "revision_id": RID,
            "mesh_sha256": "c" * 64,
            "preset": preset,
            "frame": FRAME,
            "version": 1,
            "scope": SCOPE,
            "image": {
                "file": "photo.png",
                "sha256": hashlib.sha256(PNG).hexdigest(),
                "width": 1024,
                "height": 1024,
            },
            "rendering": {
                "versions": {"pyvista": "0.48.4", "vtk": "9.3.1"},
                "camera": {
                    "position": [[100, -100, 100], [0, 0, 0], [0, 0, 1]],
                    "view_angle_degrees": 35,
                    "distance_mm": 173,
                    "maximum_absolute_ndc": 0.8,
                    "projection": "perspective",
                },
                "texture": {"tile_mm": 12, "scope": "Shading only; no displacement"},
                "ssao": {"radius_mm": 0.25, "blur": True},
                "lights": [{"position": [10, -10, 20], "intensity": 1.0}],
                "shadow_method": "Native shadow maps and restrained SSAO",
                "geometry": {
                    "triangles": 2,
                    "triangle_sha256": "d" * 64,
                    "triangle_positions_unchanged": True,
                    "bounds_mm": [[-40, -25, -3], [40, 25, 3]],
                },
            },
        },
        "image_base64": base64.b64encode(PNG).decode("ascii"),
        "mime_type": "image/png",
    }


class Verifier:
    async def verify_token(self, token):
        if token not in {"alice", "bob", "no-read", "wrong-issuer"}:
            return None
        return AccessToken(
            token=token,
            subject=token,
            client_id="photo-test",
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": "https://wrong.example" if token == "wrong-issuer" else ISSUER},
        )


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    sid = service.store.create("alice", "saved CAD", None)["id"]
    measurement = {
        "revision_id": RID,
        "artifact_hashes": {"render.png": hashlib.sha256(PNG).hexdigest()},
    }
    service.store.add_revision(sid, {"id": RID, "measurement": measurement})
    directory = service.store.directory(sid, RID)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "render.png").write_bytes(PNG)
    (directory / "evidence.json").write_text(json.dumps(measurement), encoding="utf-8")
    monkeypatch.setattr(
        service, "start", lambda: pytest.fail("Photo must not start a generation job")
    )
    value, calls, failure = envelope(sid), [], []

    def photo(svc, project_id, owner, revision_id, preset, *, timeout):
        svc.store.session_record(project_id, owner)
        svc.store.revision(project_id, revision_id)
        calls.append((project_id, owner, revision_id, preset, timeout))
        if failure:
            raise failure[0]
        result = copy.deepcopy(value)
        if result["receipt"]["preset"] == "metal":
            result["receipt"]["preset"] = preset
        return result

    monkeypatch.setattr(product_photo, "get_product_photo", photo)
    return service, sid, value, calls, failure


def application(service, fixed_owner=None):
    def owner(authorization: str = Header()):
        return authorization

    app = FastAPI()
    app.include_router(
        router(service.store, service, (lambda: fixed_owner) if fixed_owner else owner)
    )
    return app


def sdk_server(surface, boundary, tmp_path, monkeypatch, owner="alice"):
    service, _, _, _, _ = boundary
    if surface == "local":
        adapter = mcp_server.WorkspaceAdapter(
            "http://127.0.0.1:4321",
            tmp_path / "unused-exports",
            transport=httpx.ASGITransport(app=application(service, owner)),
        )
        return mcp_server.create_server(adapter)
    token = asyncio.run(Verifier().verify_token(owner))
    monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
    return remote_mcp.create_server(
        service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )


def arguments(sid, preset="metal"):
    return {"project_id": sid, "revision_id": RID, "preset": preset}


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_real_sdk_photo_image_receipt_schema_and_preview_are_separate(
    surface, boundary, tmp_path, monkeypatch
):
    service, sid, _, calls, _ = boundary
    before = service.store.get(sid, "alice")
    server = sdk_server(surface, boundary, tmp_path, monkeypatch)

    async def scenario():
        async with Client(server, cache=None) as sdk:
            assert sdk.server_info.version == ("0.0.19" if surface == "local" else "0.0.20")
            assert "get_product_photo" in sdk.instructions
            tools = {tool.name: tool for tool in (await sdk.list_tools()).tools}
            photo = tools["get_product_photo"]
            assert photo.annotations.read_only_hint is True
            assert set(photo.input_schema["properties"]) == {"project_id", "revision_id", "preset"}
            assert set(photo.input_schema["required"]) == {"project_id", "revision_id", "preset"}
            assert photo.input_schema["properties"]["preset"]["enum"] == ["metal", "plastic"]
            assert photo.output_schema is not None
            for preset in ("metal", "plastic"):
                reply = await sdk.call_tool("get_product_photo", arguments(sid, preset))
                assert not reply.is_error, reply.content
                assert reply.structured_content == {"receipt": envelope(sid, preset)["receipt"]}
                assert json.loads(reply.content[0].text) == reply.structured_content
                assert "image_base64" not in reply.content[0].text
                assert [item.type for item in reply.content] == ["text", "image"]
                assert reply.content[1].mime_type == "image/png"
                assert base64.b64decode(reply.content[1].data, validate=True) == PNG
            preview = await sdk.call_tool("get_preview", {"project_id": sid, "revision_id": RID})
            assert not preview.is_error
            assert len(preview.content) == 1 and preview.content[0].type == "image"
            assert base64.b64decode(preview.content[0].data, validate=True) == PNG

    asyncio.run(scenario())
    assert calls == [(sid, "alice", RID, "metal", 30), (sid, "alice", RID, "plastic", 30)]
    assert service.store.get(sid, "alice") == before
    assert (service.store.directory(sid, RID) / "render.png").read_bytes() == PNG
    assert not (tmp_path / "unused-exports").exists()
    with service.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


def test_http_owned_route_and_unknown_revision_precede_dispatch(boundary):
    service, sid, _, calls, _ = boundary
    route = f"/workspace/sessions/{sid}/design/{RID}/product-photo"
    with TestClient(application(service)) as client:
        assert (
            client.post(
                route, headers={"Authorization": "bob"}, json={"preset": "metal"}
            ).status_code
            == 404
        )
        assert calls == []
        assert (
            client.post(
                route.replace(RID, "f" * 32),
                headers={"Authorization": "alice"},
                json={"preset": "metal"},
            ).status_code
            == 404
        )
        assert calls == []
        reply = client.post(route, headers={"Authorization": "alice"}, json={"preset": "plastic"})
        assert reply.status_code == 200 and reply.json() == envelope(sid, "plastic")
        assert (
            client.get(
                f"/workspace/sessions/{sid}/revisions/{RID}/photo.png",
                headers={"Authorization": "alice"},
            ).status_code
            == 404
        )
    assert calls == [(sid, "alice", RID, "plastic", 30)]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"preset": "steel"},
        {"preset": True},
        {"preset": None},
        {"preset": ["metal"]},
        {"preset": "metal", "owner": "bob"},
        {"preset": "metal", "timeout": 600},
    ],
)
def test_http_required_preset_and_extra_fields_are_refused(boundary, body):
    service, sid, _, calls, _ = boundary
    with TestClient(application(service, "alice")) as client:
        reply = client.post(f"/workspace/sessions/{sid}/design/{RID}/product-photo", json=body)
    assert reply.status_code == 422 and calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "change", ["missing-preset", "preset", "preset-type", "project", "revision"]
)
def test_sdk_invalid_request_never_dispatches(surface, boundary, tmp_path, monkeypatch, change):
    _, sid, _, calls, _ = boundary
    options = arguments(sid)
    if change == "missing-preset":
        options.pop("preset")
    elif change == "preset":
        options["preset"] = "steel"
    elif change == "preset-type":
        options["preset"] = ["metal"]
    else:
        options[change + "_id"] = "../invalid"
    server = sdk_server(surface, boundary, tmp_path, monkeypatch)

    async def scenario():
        async with Client(server, cache=None) as sdk:
            assert (await sdk.call_tool("get_product_photo", options)).is_error

    asyncio.run(scenario())
    assert calls == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("Optional photo runtime unavailable"),
        ValueError("Source evidence changed"),
        ValueError("Product photo deadline exhausted"),
        OSError("PRIVATE-PATH"),
    ],
)
def test_sdk_service_refusals_are_errors_without_an_image(
    surface, boundary, tmp_path, monkeypatch, failure
):
    service, sid, _, _, failures = boundary
    before = service.store.get(sid, "alice")
    failures[:] = [failure]
    server = sdk_server(surface, boundary, tmp_path, monkeypatch)

    async def scenario():
        async with Client(server, cache=None) as sdk:
            reply = await sdk.call_tool("get_product_photo", arguments(sid))
            assert reply.is_error and all(item.type != "image" for item in reply.content)
            assert "PRIVATE-PATH" not in str(reply.content)

    asyncio.run(scenario())
    assert service.store.get(sid, "alice") == before


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "change",
    [
        "project",
        "revision",
        "preset",
        "frame",
        "version",
        "hash",
        "mime",
        "base64",
        "png",
        "dimensions",
        "bytes",
        "receipt-cap",
        "envelope-cap",
        "geometry",
    ],
)
def test_sdk_malformed_or_oversized_image_never_returns_success(
    surface, boundary, tmp_path, monkeypatch, change
):
    _, sid, value, _, _ = boundary
    receipt = value["receipt"]
    if change in {"project", "revision"}:
        receipt[change + "_id"] = "e" * 32
    elif change == "preset":
        receipt["preset"] = "plastic"
    elif change == "frame":
        receipt["frame"] = "part_local"
    elif change == "version":
        receipt["version"] = True
    elif change == "hash":
        receipt["image"]["sha256"] = "e" * 64
    elif change == "mime":
        value["mime_type"] = "image/jpeg"
    elif change == "base64":
        value["image_base64"] = "@@@"
    elif change in {"png", "dimensions", "bytes"}:
        image = (
            b"not PNG"
            if change == "png"
            else png(1, 1)
            if change == "dimensions"
            else PNG + b"x" * (1024 * 1024)
        )
        value["image_base64"] = base64.b64encode(image).decode("ascii")
        receipt["image"]["sha256"] = hashlib.sha256(image).hexdigest()
    elif change == "receipt-cap":
        receipt["rendering"]["unexpected_large_note"] = "x" * (64 * 1024)
    elif change == "envelope-cap":
        value["image_base64"] = "x" * (2 * 1024 * 1024)
    else:
        receipt["rendering"]["geometry"]["triangle_positions_unchanged"] = False
    server = sdk_server(surface, boundary, tmp_path, monkeypatch)

    async def scenario():
        async with Client(server, cache=None) as sdk:
            reply = await sdk.call_tool("get_product_photo", arguments(sid))
            assert reply.is_error and all(item.type != "image" for item in reply.content)

    asyncio.run(scenario())


@pytest.mark.parametrize("preset", [["metal"], {"name": "metal"}, None])
def test_direct_local_adapter_refuses_unhashable_presets_without_http(tmp_path, preset):
    def fail(request):
        pytest.fail("Invalid preset must not reach HTTP")

    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(fail)
    )
    with pytest.raises(ToolError):
        asyncio.run(adapter.get_product_photo("a" * 32, RID, preset))


@pytest.mark.parametrize("token", [None, "invalid", "no-read", "wrong-issuer", "bob"])
def test_authenticated_streamable_sdk_owner_denial_before_dispatch(boundary, token):
    service, sid, _, calls, _ = boundary
    app = remote_mcp.create_app(
        service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )
    headers = {"Accept": "application/json, text/event-stream"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        reply = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_product_photo", "arguments": arguments(sid)},
            },
        )
    assert reply.status_code in {401, 403} or reply.json()["result"]["isError"]
    assert calls == []


def test_authenticated_streamable_sdk_positive_image(boundary):
    service, sid, _, calls, _ = boundary
    app = remote_mcp.create_app(
        service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        response = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer alice",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_product_photo", "arguments": arguments(sid)},
            },
        )
    assert response.status_code == 200
    reply = response.json()["result"]
    assert not reply.get("isError")
    assert reply["structuredContent"] == {"receipt": envelope(sid)["receipt"]}
    assert base64.b64decode(reply["content"][1]["data"], validate=True) == PNG
    assert calls == [(sid, "alice", RID, "metal", 30)]
