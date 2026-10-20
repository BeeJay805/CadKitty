"""Actual MCP/ASGI photo delivery with replayed PNGs, never a render worker."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import math
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_product_photo_transport import PNG, envelope
from tests.unit.workspace.test_project_delivery import RID as SAVED_RID
from tests.unit.workspace.test_project_delivery import saved_project
from tests.unit.workspace.test_project_export_local import PID, RID, SID, boundary, saved_package
from tests.unit.workspace.test_project_export_remote import RESOURCE, result, rpc
from tests.unit.workspace.test_project_members_remote import MemberVerifier, app

from forge_mind.workspace import mcp_server, project_delivery, project_photo, remote_mcp
from forge_mind.workspace.product_photo_contract import CAPTURED_SCOPE, OPPOSED_VIEW
from forge_mind.workspace.project_members import captured_cad_files
from forge_mind.workspace.project_photo_transport import validate_project_photo

PHOTO_ID = "f" * 32
ALTERNATE_ID = "e" * 32


def captured_photo(package, members, part_id="base", preset="plastic"):
    """Replay bounded image facts against real verified package member metadata."""
    row = next(row for row in members if row["part_id"] == part_id and row["format"] == "stl")
    value = envelope(package["project_id"], preset)
    value.update(status="available", photo_id=PHOTO_ID)
    value["receipt"].update(
        revision_id=package["revision_id"],
        package_id=package["package_id"],
        package_sha256=package["sha256"],
        part_id=part_id,
        file_id=row["file_id"],
        quantity=row["quantity"],
        mesh_sha256=row["sha256"],
        frame=row["frame"],
        scope=CAPTURED_SCOPE,
    )
    return value


def photo_pair(primary):
    """Replay a pair envelope without hiding intentional primary validation failures."""
    primary = copy.deepcopy(primary)
    if primary.get("status") != "available":
        return {
            "primary": primary,
            "alternate": {
                "view": OPPOSED_VIEW,
                "status": "unavailable",
                "code": "primary_unavailable",
                "reason": "Primary fixture is unavailable; alternate was skipped",
                "candidates": copy.deepcopy(primary["candidates"]),
            },
        }
    alternate = copy.deepcopy(primary)
    alternate.update(view=OPPOSED_VIEW, photo_id=ALTERNATE_ID)
    receipt = alternate["receipt"]
    receipt.update(view=OPPOSED_VIEW, version=2)
    bounds = receipt["rendering"]["geometry"]["bounds_mm"]
    center = [(lo + hi) / 2 for lo, hi in zip(*bounds, strict=True)]
    vector = [-1.25, 1.65, 1.6]
    length = math.hypot(*vector)
    outward = [item / length for item in vector]
    horizontal = math.hypot(*outward[:2])
    up = [-outward[0] * outward[2] / horizontal, -outward[1] * outward[2] / horizontal, horizontal]
    receipt["rendering"]["camera"].update(
        position=[[center[i] + 100 * outward[i] for i in range(3)], center, up],
        distance_mm=100.0,
        view_angle_degrees=35.0,
        maximum_absolute_ndc=0.8,
        projection="perspective",
    )
    return {"primary": primary, "alternate": alternate}


def local_boundary(receipt, data, value, *, status=200):
    transport, calls = boundary(receipt, data)
    application = transport.app

    @application.post("/workspace/sessions/{sid}/project-exports/{pid}/{digest}/photo")
    async def photo(sid: str, pid: str, digest: str, request: Request):
        options = await request.json()
        calls.append(("PHOTO", sid, pid, digest, options))
        delivered = value
        if options.get("include_alternate") is True and set(value) != {"primary", "alternate"}:
            delivered = photo_pair(value)
        return JSONResponse(delivered, status_code=status)

    @application.delete(
        "/workspace/sessions/{sid}/project-exports/{pid}/{digest}/photos/{photo_id}"
    )
    async def discard(sid: str, pid: str, digest: str, photo_id: str):
        calls.append(("DISCARD", sid, pid, digest, photo_id))
        return Response(status_code=204)

    return transport, calls


def local_call(adapter, **options):
    async def run():
        async with Client(mcp_server.create_server(adapter)) as client:
            return await client.call_tool(
                "export_project", {"project_id": SID, "revision_id": RID, **options}
            )

    return asyncio.run(run())


def test_local_explicit_photo_delivers_png_sidecars_and_exact_cad(tmp_path):
    receipt, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, receipt)
    value = captured_photo(receipt, members, "cap", "metal")
    transport, calls = local_boundary(receipt, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    response = local_call(
        adapter,
        include_stl=True,
        include_individual_files=True,
        photo_part_id="cap",
        photo_preset="metal",
    )
    assert not response.is_error
    delivered = response.structured_content
    assert Path(delivered["path"]).read_bytes() == data
    preview = delivered["preview"]
    assert preview["receipt"] == value["receipt"] and preview["receipt"]["quantity"] == 2
    assert Path(preview["path"]).is_absolute()
    assert Path(preview["path"]).read_bytes() == PNG
    assert json.loads(Path(preview["receipt_path"]).read_bytes()) == value["receipt"]
    assert base64.b64decode(response.content[1].data, validate=True) == PNG
    assert response.content[1].mime_type == "image/png"
    alternate = delivered["preview_alternate"]
    expected_alternate = photo_pair(value)["alternate"]
    assert alternate["receipt"] == expected_alternate["receipt"]
    assert Path(alternate["path"]).read_bytes() == PNG
    assert json.loads(Path(alternate["receipt_path"]).read_bytes()) == expected_alternate["receipt"]
    assert len(response.content) == 3
    assert base64.b64decode(response.content[2].data, validate=True) == PNG
    assert json.loads(response.content[0].text) == delivered
    assert "image_base64" not in json.dumps(delivered)
    assert calls[-1][-1] == {"preset": "metal", "part_id": "cap", "include_alternate": True}
    for member in delivered["individual_files"]:
        assert hashlib.sha256(Path(member["path"]).read_bytes()).hexdigest() == member["sha256"]


def test_local_structured_auto_refuses_single_definition_photo_even_if_unique(tmp_path):
    receipt, data = saved_package(tmp_path, True, unique_made=True)
    members = captured_cad_files(data, receipt)
    assert [row["part_id"] for row in members if row["format"] == "stl"] == ["base"]
    value = captured_photo(receipt, members)
    transport, calls = local_boundary(receipt, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    response = local_call(adapter, include_stl=True)
    assert response.is_error
    assert not any(item.type == "image" for item in response.content)
    assert not list(adapter.export_dir.glob("*.png"))
    assert not list(adapter.export_dir.glob("*.zip"))
    assert next(row[-1] for row in calls if row[0] == "PHOTO") == {
        "preset": "plastic",
        "part_id": None,
        "include_alternate": True,
    }


def test_local_none_is_exact_legacy_and_skips_photo(tmp_path):
    receipt, data = saved_package(tmp_path)
    transport, calls = local_boundary(receipt, data, {})
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    response = local_call(adapter, preview="none")
    assert not response.is_error and len(response.content) == 1
    metadata = response.structured_content
    assert {key: value for key, value in metadata.items() if key != "path"} == receipt
    assert json.loads(response.content[0].text) == metadata
    assert len(calls) == 2 and "preview" not in metadata


@pytest.mark.parametrize(
    "code", ["missing_stl", "execution_unqualified", "render_failed", "renderer_unavailable"]
)
def test_local_declared_absence_preserves_cad(tmp_path, code):
    receipt, data = saved_package(tmp_path, code != "missing_stl")
    value = {
        "status": "unavailable",
        "code": code,
        "reason": "Explicit optional absence",
        "candidates": ["base", "cap"],
    }
    transport, _ = local_boundary(receipt, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    response = local_call(adapter, include_stl=code != "missing_stl", photo_part_id="base")
    assert not response.is_error and response.structured_content["preview"] == value
    assert Path(response.structured_content["path"]).read_bytes() == data
    assert len(response.content) == 1
    assert response.structured_content["preview_alternate"] == photo_pair(value)["alternate"]


@pytest.mark.parametrize(
    "field",
    [
        "revision_id",
        "package_id",
        "package_sha256",
        "mesh_sha256",
        "preset",
        "frame",
        "quantity",
        "png",
    ],
)
def test_local_integrity_failure_refuses_and_cleans_only_new_files(tmp_path, field):
    receipt, data = saved_package(tmp_path, True)
    value = captured_photo(receipt, captured_cad_files(data, receipt))
    if field == "png":
        value["image_base64"] = base64.b64encode(PNG + b"changed").decode()
    else:
        value["receipt"][field] = {
            "quantity": True,
            "preset": "metal",
            "frame": "cad_z_up_mm_world",
        }.get(field, "0" * (64 if "sha256" in field else 32))
    transport, calls = local_boundary(receipt, data, value)
    exports = tmp_path / "exports"
    exports.mkdir()
    unrelated = exports / "unrelated.txt"
    unrelated.write_bytes(b"preserve")
    adapter = mcp_server.WorkspaceAdapter("http://127.0.0.1:4321", exports, transport=transport)
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert response.is_error
    assert list(exports.iterdir()) == [unrelated] and unrelated.read_bytes() == b"preserve"
    assert [row for row in calls if row[0] == "DISCARD"] == [
        ("DISCARD", SID, PID, receipt["sha256"], ALTERNATE_ID),
        ("DISCARD", SID, PID, receipt["sha256"], PHOTO_ID),
    ]


@pytest.mark.parametrize(
    "pattern", ["*.photo.png", "*.photo.json", "*.zip", "*-part-base-stl-*.stl"]
)
def test_local_same_inode_tamper_after_photo_write_refuses_publication(
    tmp_path, monkeypatch, pattern
):
    receipt, data = saved_package(tmp_path, True)
    value = captured_photo(receipt, captured_cad_files(data, receipt))
    transport, calls = local_boundary(receipt, data, value)
    exports = tmp_path / "exports"
    exports.mkdir()
    unrelated = exports / "unrelated.txt"
    unrelated.write_bytes(b"preserve")
    adapter = mcp_server.WorkspaceAdapter("http://127.0.0.1:4321", exports, transport=transport)
    real_sleep = asyncio.sleep
    modified = []

    async def tamper_at_checkpoint(delay, *args, **kwargs):
        # Wait until the PNG write, so every selected output already passed initial validation.
        if not modified and list(exports.glob("*.photo.png")):
            targets = list(exports.glob(pattern))
            if targets:
                target = targets[0]
                before = target.stat()
                original = target.read_bytes()
                with target.open("r+b") as file:
                    file.seek(len(original) - 1)
                    file.write(bytes([original[-1] ^ 1]))
                after = target.stat()
                assert (before.st_dev, before.st_ino, before.st_size) == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                )
                assert target.read_bytes() != original
                modified.append(target)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(mcp_server.asyncio, "sleep", tamper_at_checkpoint)
    response = local_call(
        adapter, include_stl=True, include_individual_files=True, photo_part_id="base"
    )
    assert modified and response.is_error
    assert "changed export file bytes" in response.content[0].text
    assert list(exports.iterdir()) == [unrelated] and unrelated.read_bytes() == b"preserve"
    assert [row for row in calls if row[0] == "DISCARD"] == [
        ("DISCARD", SID, PID, receipt["sha256"], ALTERNATE_ID),
        ("DISCARD", SID, PID, receipt["sha256"], PHOTO_ID),
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"preview": True},
        {"photo_preset": "wood"},
        {"photo_part_id": "../base"},
        {"photo_part_id": 7},
    ],
)
def test_local_invalid_options_refuse_before_http(tmp_path, options):
    transport, calls = local_boundary({}, b"", {})
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    response = local_call(adapter, **options)
    assert response.is_error and not calls


def jobs(service):
    with sqlite3.connect(service.store.root / "workspace.sqlite3") as connection:
        return connection.execute("select count(*) from jobs").fetchone()[0]


def replay_remote(monkeypatch):
    photos = {}
    discarded = []

    def create(
        service, sid, principal, package_id, digest, *, preset="plastic", part_id=None, timeout=15
    ):
        data, package = project_delivery.read_project_export(
            service, sid, principal, package_id, digest
        )
        value = captured_photo(
            package, captured_cad_files(data, package), part_id or "base", preset
        )
        photos[(sid, package_id, PHOTO_ID)] = copy.deepcopy(value)
        return value

    def create_pair(*args, **kwargs):
        value = photo_pair(create(*args, **kwargs))
        sid, package_id = args[1], args[3]
        for item in value.values():
            photos[(sid, package_id, item["photo_id"])] = copy.deepcopy(item)
        return value

    def read(service, sid, principal, package_id, digest, photo_id, image_sha256):
        project_delivery.read_project_export(service, sid, principal, package_id, digest)
        value = photos[(sid, package_id, photo_id)]
        assert value["receipt"]["image"]["sha256"] == image_sha256
        return base64.b64decode(value["image_base64"]), value["receipt"]

    def discard(service, sid, principal, package_id, photo_id):
        service.store.get(sid, principal)
        discarded.append((sid, package_id, photo_id))
        photos.pop((sid, package_id, photo_id), None)

    monkeypatch.setattr(project_photo, "create_project_photo", create)
    monkeypatch.setattr(project_photo, "create_project_photo_pair", create_pair)
    monkeypatch.setattr(project_photo, "read_project_photo", read)
    monkeypatch.setattr(project_photo, "discard_project_photo", discard)
    return photos, discarded


def remote_export(client, sid, **options):
    return result(
        rpc(
            client,
            "owner",
            "tools/call",
            {
                "name": "export_project",
                "arguments": {"project_id": sid, "revision_id": SAVED_RID, **options},
            },
        )
    )


def test_remote_explicit_photo_exact_image_and_durable_owned_resource(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    photos, _ = replay_remote(monkeypatch)
    before = jobs(service)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(
            client, sid, include_stl=True, photo_part_id="cap", photo_preset="metal"
        )
        assert not delivered.get("isError")
        metadata = delivered["structuredContent"]
        preview = metadata["preview"]
        assert preview["receipt"]["quantity"] == 2
        assert "path" not in json.dumps(metadata) and "image_base64" not in json.dumps(metadata)
        assert base64.b64decode(delivered["content"][1]["data"], validate=True) == PNG
        alternate = metadata["preview_alternate"]
        assert alternate["receipt"]["version"] == 2
        assert alternate["receipt"]["view"] == OPPOSED_VIEW
        assert len(delivered["content"]) == 3
        assert base64.b64decode(delivered["content"][2]["data"], validate=True) == PNG
        alternate_read = result(rpc(client, "owner", "resources/read", {"uri": alternate["uri"]}))[
            "contents"
        ][0]
        assert base64.b64decode(alternate_read["blob"], validate=True) == PNG
        uri = preview["uri"]
        fetched = result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]
        assert fetched["mimeType"] == "image/png" and base64.b64decode(fetched["blob"]) == PNG
        foreign = rpc(client, "foreign", "resources/read", {"uri": uri})
        assert foreign.json()["error"]["code"] == -32603
        assert (
            base64.b64decode(
                result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]["blob"]
            )
            == PNG
        )
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as restarted:
        assert (
            base64.b64decode(
                result(rpc(restarted, "owner", "resources/read", {"uri": uri}))["contents"][0][
                    "blob"
                ]
            )
            == PNG
        )
        value = photos[(sid, metadata["package_id"], PHOTO_ID)]
        value["image_base64"] = base64.b64encode(PNG + b"changed").decode()
        assert "error" in rpc(restarted, "owner", "resources/read", {"uri": uri}).json()
    assert jobs(service) == before == 0


def test_remote_missing_assembly_source_none_and_unqualified_no_worker(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "start", lambda: pytest.fail("No CAD jobs allowed"))
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        unavailable = remote_export(client, sid, include_stl=True)["structuredContent"]
        assert unavailable["preview"]["code"] == "source_unavailable"
        assert unavailable["preview"]["candidates"] == ["base", "cap"]
        assert unavailable["preview_alternate"]["code"] == "primary_unavailable"
        disabled = remote_export(client, sid, preview="none")["structuredContent"]
        assert "preview" not in disabled and "preview_alternate" not in disabled
        missing = remote_export(client, sid, photo_part_id="base")["structuredContent"]
        assert missing["preview"]["code"] == "missing_stl"
        assert missing["preview_alternate"]["code"] == "primary_unavailable"
        unqualified = remote_export(client, sid, include_stl=True, photo_part_id="base")
        assert unqualified["structuredContent"]["preview"]["code"] == "execution_unqualified"
        assert (
            unqualified["structuredContent"]["preview_alternate"]["code"] == "primary_unavailable"
        )
        invalid = remote_export(client, sid, include_stl=True, photo_part_id="guide")
        assert invalid.get("isError")
    assert jobs(service) == 0


def test_remote_cancel_during_photo_defers_cleanup_until_producer_completion(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    package_ids = []

    def block(svc, project, principal, package_id, digest, **options):
        package_ids.append(package_id)
        started.set()
        assert release.wait(5)
        return photo_pair(
            {
                "status": "unavailable",
                "code": "execution_unqualified",
                "reason": "Unqualified graphics",
                "candidates": ["base", "cap"],
            }
        )

    original_discard = project_delivery.discard_project_export

    def discard(*args):
        original_discard(*args)
        cleaned.set()

    monkeypatch.setattr(project_photo, "create_project_photo_pair", block)
    monkeypatch.setattr(project_delivery, "discard_project_export", discard)

    async def run():
        verifier = MemberVerifier()
        token = await verifier.verify_token("owner")
        server = remote_mcp.create_server(
            service,
            token_verifier=verifier,
            issuer_url="https://auth.example.test",
            resource_url=RESOURCE,
        )
        # The existing SDK transport cancellation controls use direct server dispatch and
        # patch only auth; production package/photo work and cleanup remain real.
        monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
        task = asyncio.create_task(
            server.call_tool(
                "export_project",
                {
                    "project_id": sid,
                    "revision_id": SAVED_RID,
                    "include_stl": True,
                    "photo_part_id": "base",
                },
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not cleaned.is_set()
        assert (
            service.store.root / "project-exports" / sid / package_ids[0] / "project.zip"
        ).exists()
        release.set()
        assert await asyncio.to_thread(cleaned.wait, 5)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_pure_photo_absence_cannot_hide_capture_mismatch(tmp_path):
    receipt, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, receipt)
    with pytest.raises(ValueError, match="absence"):
        validate_project_photo(
            {
                "status": "unavailable",
                "code": "missing_stl",
                "reason": "Invalid absence",
                "candidates": ["base", "cap"],
            },
            receipt,
            members,
            "plastic",
            "base",
        )


@pytest.mark.parametrize("mutation", ["short", "ihdr_length", "dimensions"])
def test_bound_hash_does_not_accept_malformed_png_header(tmp_path, mutation):
    receipt, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, receipt)
    value = captured_photo(receipt, members)
    malformed = (
        PNG[:24]
        if mutation == "short"
        else PNG[:8] + b"\0\0\0\x0c" + PNG[12:]
        if mutation == "ihdr_length"
        else PNG[:16] + b"\0\0\0\x01" + PNG[20:]
    )
    value["image_base64"] = base64.b64encode(malformed).decode()
    value["receipt"]["image"]["sha256"] = hashlib.sha256(malformed).hexdigest()
    with pytest.raises(ValueError, match="PNG"):
        validate_project_photo(value, receipt, members, "plastic", "base")
