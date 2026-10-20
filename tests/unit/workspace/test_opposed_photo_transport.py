"""Real local/remote SDK delivery with replayed images; no graphics or CAD worker."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import struct
import threading
import zlib
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from pydantic_core import to_json
from tests.unit.workspace.test_assembly_photo_transport import assembly_photo
from tests.unit.workspace.test_product_photo_transport import PNG, envelope
from tests.unit.workspace.test_project_delivery import RID as SAVED_RID
from tests.unit.workspace.test_project_delivery import saved_project
from tests.unit.workspace.test_project_export_local import RID, SID, saved_package
from tests.unit.workspace.test_project_export_remote import RESOURCE, result, rpc
from tests.unit.workspace.test_project_members_remote import MemberVerifier, app
from tests.unit.workspace.test_project_photo_delivery_transport import (
    PHOTO_ID,
    captured_photo,
    jobs,
    local_boundary,
    local_call,
    remote_export,
)

from forge_mind.workspace import mcp_server, project_photo, remote_mcp
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.product_photo_contract import (
    CAPTURED_SCOPE,
    OPPOSED_VIEW,
    opposed_request,
)
from forge_mind.workspace.project_members import captured_cad_files, captured_photo_context
from forge_mind.workspace.project_photo_transport import project_export_tool_result
from forge_mind.workspace.store import Store

ALTERNATE_ID = "e" * 32


def annotated_png(size=30):
    """Add a legal ancillary chunk to a complete fixture PNG, not a rendered image."""
    payload = b"fixture\0" + b"x" * size
    name = b"tEXt"
    chunk = (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", zlib.crc32(name + payload))
    )
    return PNG[:-12] + chunk + PNG[-12:]


ALTERNATE_PNG = annotated_png()


def apply_opposed_camera(receipt):
    bounds = receipt["rendering"]["geometry"]["bounds_mm"]
    center = [(lo + hi) / 2 for lo, hi in zip(*bounds, strict=True)]
    raw = [-1.25, 1.65, 1.6]
    length = math.hypot(*raw)
    outward = [item / length for item in raw]
    horizontal = math.hypot(*outward[:2])
    up = [-outward[0] * outward[2] / horizontal, -outward[1] * outward[2] / horizontal, horizontal]
    camera = receipt["rendering"]["camera"]
    camera.update(
        position=[[center[i] + 100 * outward[i] for i in range(3)], center, up],
        distance_mm=100.0,
        view_angle_degrees=35.0,
        maximum_absolute_ndc=0.8,
        projection="perspective",
    )


def pair_from_primary(primary, *, primary_png=PNG, alternate_png=ALTERNATE_PNG):
    primary = copy.deepcopy(primary)
    primary["image_base64"] = base64.b64encode(primary_png).decode()
    primary["receipt"]["image"]["sha256"] = hashlib.sha256(primary_png).hexdigest()
    alternate = copy.deepcopy(primary)
    alternate["photo_id"] = ALTERNATE_ID
    request = {
        key: value
        for key, value in primary["receipt"].items()
        if key not in {"scope", "image", "rendering"}
    }
    alternate["receipt"].update(opposed_request(request))
    apply_opposed_camera(alternate["receipt"])
    alternate["image_base64"] = base64.b64encode(alternate_png).decode()
    alternate["receipt"]["image"]["sha256"] = hashlib.sha256(alternate_png).hexdigest()
    alternate["view"] = OPPOSED_VIEW
    return {"primary": primary, "alternate": alternate}


def near_http_limit_pair(primary):
    """A legal HTTP pair that exceeds the SDK's duplicated metadata/image envelope."""
    primary_png = annotated_png(780_000)
    low, high = 0, 1_000_000
    target = 2 * 1024 * 1024 - 512
    while low < high:
        middle = (low + high + 1) // 2
        pair = pair_from_primary(
            primary, primary_png=primary_png, alternate_png=annotated_png(middle)
        )
        if len(json.dumps(pair, allow_nan=False).encode()) <= target:
            low = middle
        else:
            high = middle - 1
    return pair_from_primary(primary, primary_png=primary_png, alternate_png=annotated_png(low))


def local_pair(tmp_path, *, assembly=False, primary_png=PNG, alternate_png=ALTERNATE_PNG):
    package, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, package)
    primary = (
        assembly_photo(package, captured_photo_context(data, package))
        if assembly
        else captured_photo(package, members, "base", "plastic")
    )
    pair = pair_from_primary(primary, primary_png=primary_png, alternate_png=alternate_png)
    transport, calls = local_boundary(package, data, pair)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=transport
    )
    return adapter, calls, package, data, pair


@pytest.mark.parametrize("assembly", [False, True])
def test_default_export_delivers_two_exact_images_and_one_capture(tmp_path, assembly):
    adapter, calls, package, data, pair = local_pair(tmp_path, assembly=assembly)
    options = {} if assembly else {"photo_part_id": "base"}
    response = local_call(adapter, include_stl=True, include_individual_files=True, **options)
    assert not response.is_error
    metadata = response.structured_content
    assert len(response.content) == 3
    assert json.loads(response.content[0].text) == metadata
    assert Path(metadata["path"]).read_bytes() == data
    assert metadata["package_id"] == package["package_id"]
    for slot, image, expected in (
        ("preview", response.content[1], PNG),
        ("preview_alternate", response.content[2], ALTERNATE_PNG),
    ):
        photo = metadata[slot]
        assert (
            base64.b64decode(image.data, validate=True)
            == expected
            == Path(photo["path"]).read_bytes()
        )
        assert (
            json.loads(Path(photo["receipt_path"]).read_bytes())
            == pair["primary" if slot == "preview" else "alternate"]["receipt"]
        )
    assert metadata["preview"]["path"].endswith(".photo.png")
    assert metadata["preview_alternate"]["path"].endswith(".photo-opposed.png")
    assert metadata["preview_alternate"]["view"] == OPPOSED_VIEW
    assert len([call for call in calls if call[0] == "POST"]) == 1
    assert next(call[-1] for call in calls if call[0] == "PHOTO") == {
        "preset": "plastic",
        "part_id": None if assembly else "base",
        "include_alternate": True,
    }
    assert not [call for call in calls if call[0] == "DISCARD"]


def test_none_skips_pair_and_keeps_exact_legacy_metadata(tmp_path):
    adapter, calls, package, data, _ = local_pair(tmp_path)
    response = local_call(adapter, preview="none", include_stl=True)
    assert not response.is_error and len(response.content) == 1
    assert {
        key: value for key, value in response.structured_content.items() if key != "path"
    } == package
    assert Path(response.structured_content["path"]).read_bytes() == data
    assert not list(adapter.export_dir.glob("*.png"))
    assert not [call for call in calls if call[0] == "PHOTO"]


@pytest.mark.parametrize("code", ["render_failed", "renderer_unavailable", "execution_unqualified"])
def test_typed_alternate_absence_keeps_primary(tmp_path, code):
    adapter, _, _, data, pair = local_pair(tmp_path)
    pair["alternate"] = {
        "view": OPPOSED_VIEW,
        "status": "unavailable",
        "code": code,
        "reason": "Optional alternate graphics unavailable",
        "candidates": ["base", "cap"],
    }
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert not response.is_error and len(response.content) == 2
    assert Path(response.structured_content["path"]).read_bytes() == data
    assert response.structured_content["preview_alternate"] == pair["alternate"]
    assert base64.b64decode(response.content[1].data) == PNG


@pytest.mark.parametrize("change", ["view", "geometry", "source", "png"])
def test_alternate_integrity_failure_refuses_all_new_outputs(tmp_path, change):
    adapter, calls, _, _, pair = local_pair(tmp_path)
    prior = adapter.export_dir / "unrelated.png"
    prior.parent.mkdir()
    prior.write_bytes(b"preserve")
    alternate = pair["alternate"]
    if change == "view":
        alternate["receipt"]["view"] = "invented"
    elif change == "geometry":
        alternate["receipt"]["rendering"]["geometry"]["triangle_sha256"] = "0" * 64
    elif change == "source":
        alternate["receipt"]["mesh_sha256"] = "0" * 64
    else:
        alternate["image_base64"] = base64.b64encode(PNG).decode()
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert response.is_error and not any(item.type == "image" for item in response.content)
    assert list(adapter.export_dir.iterdir()) == [prior] and prior.read_bytes() == b"preserve"
    assert {call[-1] for call in calls if call[0] == "DISCARD"} == {PHOTO_ID, ALTERNATE_ID}


def test_complete_sdk_budget_drops_only_alternate(tmp_path):
    adapter, calls, _, data, pair = local_pair(tmp_path)
    large = near_http_limit_pair(pair["primary"])
    pair.clear()
    pair.update(large)
    large_primary = base64.b64decode(pair["primary"]["image_base64"])
    assert len(json.dumps(pair, allow_nan=False).encode()) < 2 * 1024 * 1024
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert not response.is_error and len(response.content) == 2
    assert len(response.model_dump_json(by_alias=True).encode()) <= 2 * 1024 * 1024
    assert base64.b64decode(response.content[1].data) == large_primary
    assert Path(response.structured_content["path"]).read_bytes() == data
    assert response.structured_content["preview_alternate"]["code"] == "delivery_budget_exceeded"
    assert [call[-1] for call in calls if call[0] == "DISCARD"] == [ALTERNATE_ID]
    assert not list(adapter.export_dir.glob("*.photo-opposed.*"))


def test_configured_file_budget_drops_only_alternate(tmp_path, monkeypatch):
    adapter, _, package, data, pair = local_pair(tmp_path)
    # A legal ZIP comment keeps file policy distinct from the smaller SDK envelope.
    buffer = io.BytesIO(data)
    with ZipFile(buffer, "a") as archive:
        archive.comment = b"saved transport fixture " + b"x" * 50_000
    data = buffer.getvalue()
    package.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for slot in ("primary", "alternate"):
        pair[slot]["receipt"]["package_sha256"] = package["sha256"]
    transport, calls = local_boundary(package, data, pair)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321", adapter.export_dir, transport=transport
    )
    primary_bytes = len(PNG) + len(
        json.dumps(pair["primary"]["receipt"], separators=(",", ":")).encode()
    )
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_EXPORT_BYTES", str(len(data) + primary_bytes + 10))
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert not response.is_error, response.content
    assert response.structured_content["preview_alternate"]["code"] == "delivery_budget_exceeded"
    assert [call[-1] for call in calls if call[0] == "DISCARD"] == [ALTERNATE_ID]


def test_configured_sdk_budget_drops_alternate_without_dropping_primary(tmp_path, monkeypatch):
    adapter, calls, _, _, _ = local_pair(tmp_path, alternate_png=annotated_png(90_000))
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "100000")
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert (
        not response.is_error and len(response.model_dump_json(by_alias=True).encode()) <= 100_000
    )
    assert response.structured_content["preview_alternate"]["code"] == "delivery_budget_exceeded"
    assert base64.b64decode(response.content[1].data) == PNG
    assert [call[-1] for call in calls if call[0] == "DISCARD"] == [ALTERNATE_ID]


def test_primary_alone_over_configured_budget_still_refuses(tmp_path, monkeypatch):
    adapter, calls, _, _, _ = local_pair(
        tmp_path, primary_png=annotated_png(50_000), alternate_png=annotated_png(60_000)
    )
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "60000")
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert response.is_error and not any(item.type == "image" for item in response.content)
    assert not list(adapter.export_dir.iterdir())
    assert {call[-1] for call in calls if call[0] == "DISCARD"} == {PHOTO_ID, ALTERNATE_ID}


def test_paired_absence_still_bounds_complete_sdk_metadata(tmp_path, monkeypatch):
    adapter, _, package, _, pair = local_pair(tmp_path)
    primary = {
        "status": "unavailable",
        "code": "execution_unqualified",
        "reason": "Optional graphics are unqualified",
        "candidates": ["base", "cap"],
    }
    alternate = {**primary, "view": OPPOSED_VIEW, "code": "primary_unavailable"}
    pair.clear()
    pair.update(primary=primary, alternate=alternate)
    metadata = {
        **package,
        "path": str(adapter.export_dir / f"{SID}-{RID}-project-{'a' * 12}.zip"),
        "preview": primary,
        "preview_alternate": alternate,
    }
    policy = max(len(to_json(metadata)), len(to_json(package))) + 100
    assert (
        len(project_export_tool_result(metadata).model_dump_json(by_alias=True).encode()) > policy
    )
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", str(policy))
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert response.is_error and not any(item.type == "image" for item in response.content)
    assert not list(adapter.export_dir.iterdir())


def test_alternate_budget_cleanup_failure_cannot_return_primary_success(tmp_path, monkeypatch):
    adapter, _, _, _, _ = local_pair(tmp_path, alternate_png=annotated_png(90_000))
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "100000")
    real_request = adapter.request

    async def request(method, path, **kwargs):
        if method == "DELETE" and path.endswith(ALTERNATE_ID):
            raise OSError("controlled alternate cleanup failure")
        return await real_request(method, path, **kwargs)

    monkeypatch.setattr(adapter, "request", request)
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert response.is_error and not any(item.type == "image" for item in response.content)
    assert "cleanup" in response.content[0].text
    assert not list(adapter.export_dir.iterdir())


def test_same_inode_alternate_tamper_refuses_publication(tmp_path, monkeypatch):
    adapter, calls, _, _, _ = local_pair(tmp_path)
    real_sleep = asyncio.sleep
    changed = []

    async def tamper(delay, *args, **kwargs):
        if not changed:
            paths = list(adapter.export_dir.glob("*.photo-opposed.png"))
            if paths:
                path = paths[0]
                before = path.stat().st_ino
                raw = path.read_bytes()
                with path.open("r+b") as handle:
                    handle.write(raw[:-1] + bytes([raw[-1] ^ 1]))
                assert path.stat().st_ino == before
                changed.append(path)
        await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(mcp_server.asyncio, "sleep", tamper)
    response = local_call(adapter, include_stl=True, photo_part_id="base")
    assert changed and response.is_error
    assert not list(adapter.export_dir.iterdir())
    assert {call[-1] for call in calls if call[0] == "DISCARD"} == {PHOTO_ID, ALTERNATE_ID}


def replay_worker(monkeypatch, *, large=False):
    calls = []

    def render(svc, source, request, *, timeout):
        calls.append((copy.deepcopy(request), timeout))
        value = envelope(request["project_id"], request["preset"])
        value["receipt"].update(request, scope=CAPTURED_SCOPE)
        opposite = "view" in request
        if large:
            original = copy.deepcopy(value)
            original.update(status="available", photo_id=PHOTO_ID)
            original["receipt"].pop("view", None)
            original["receipt"]["version"] = 1
            large_pair = near_http_limit_pair(original)
            image = base64.b64decode(
                large_pair["alternate" if opposite else "primary"]["image_base64"]
            )
        else:
            image = ALTERNATE_PNG if opposite else PNG
        value["receipt"]["image"]["sha256"] = hashlib.sha256(image).hexdigest()
        value["image_base64"] = base64.b64encode(image).decode()
        if opposite:
            apply_opposed_camera(value["receipt"])
        return value

    monkeypatch.setattr(project_photo, "render_mesh", render)
    return calls


def test_authenticated_remote_pair_resources_are_exact_owned_and_durable(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    calls = replay_worker(monkeypatch)
    before = jobs(service)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(client, sid, include_stl=True, photo_part_id="base")
        assert not delivered.get("isError") and len(delivered["content"]) == 3
        metadata = delivered["structuredContent"]
        assert json.loads(delivered["content"][0]["text"]) == metadata
        for index, key, expected in ((1, "preview", PNG), (2, "preview_alternate", ALTERNATE_PNG)):
            assert base64.b64decode(delivered["content"][index]["data"]) == expected
            uri = metadata[key]["uri"]
            owned = result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]
            assert base64.b64decode(owned["blob"]) == expected
            assert (
                rpc(client, "foreign", "resources/read", {"uri": uri}).json()["error"]["code"]
                == -32603
            )
            assert (
                base64.b64decode(
                    result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0][
                        "blob"
                    ]
                )
                == expected
            )
    restarted = WorkspaceService(Store(service.store.root), Executor("docker"))
    monkeypatch.setattr(restarted, "start", lambda: pytest.fail("Read cannot start a worker"))
    with TestClient(app(restarted), base_url=RESOURCE.removesuffix("/mcp")) as client:
        uri = metadata["preview_alternate"]["uri"]
        assert (
            base64.b64decode(
                result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]["blob"]
            )
            == ALTERNATE_PNG
        )
    assert len(calls) == 2 and all(0 < timeout <= 15 for _, timeout in calls)
    assert "view" not in calls[0][0] and calls[1][0]["view"] == OPPOSED_VIEW
    assert jobs(service) == before == 0


def test_remote_none_and_foreign_requests_do_not_call_pair(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        project_photo, "create_project_photo_pair", lambda *a, **k: pytest.fail("pair must not run")
    )
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        disabled = remote_export(client, sid, preview="none")
        assert not disabled.get("isError") and len(disabled["content"]) == 1
        assert "preview" not in disabled["structuredContent"]
        assert "preview_alternate" not in disabled["structuredContent"]
        denied = result(
            rpc(
                client,
                "foreign",
                "tools/call",
                {
                    "name": "export_project",
                    "arguments": {
                        "project_id": sid,
                        "revision_id": SAVED_RID,
                        "include_stl": True,
                        "photo_part_id": "base",
                    },
                },
            )
        )
        assert denied.get("isError") and not any(
            item["type"] == "image" for item in denied["content"]
        )
    assert jobs(service) == 0


def test_remote_complete_budget_keeps_primary_resource(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    replay_worker(monkeypatch, large=True)
    discarded = []
    real_discard = project_photo.discard_project_photo

    def discard(*args):
        discarded.append(args[-1])
        real_discard(*args)

    monkeypatch.setattr(project_photo, "discard_project_photo", discard)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(client, sid, include_stl=True, photo_part_id="base")
        assert not delivered.get("isError") and len(delivered["content"]) == 2
        metadata = delivered["structuredContent"]
        assert metadata["preview_alternate"]["code"] == "delivery_budget_exceeded"
        assert len(discarded) == 1 and discarded[0] != metadata["preview"]["photo_id"]
        uri = metadata["preview"]["uri"]
        assert base64.b64decode(
            result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]["blob"]
        ) == annotated_png(780_000)
    assert jobs(service) == 0


def test_remote_cleanup_failure_is_visible_after_invalid_alternate(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    replay_worker(monkeypatch)
    real_pair = project_photo.create_project_photo_pair
    real_discard = project_photo.discard_project_photo
    attempts = []

    def pair(*args, **kwargs):
        value = real_pair(*args, **kwargs)
        value["alternate"]["receipt"]["mesh_sha256"] = "0" * 64
        return value

    def discard(*args):
        attempts.append(args[-1])
        if len(attempts) == 1:
            raise OSError("controlled cleanup failure")
        real_discard(*args)

    monkeypatch.setattr(project_photo, "create_project_photo_pair", pair)
    monkeypatch.setattr(project_photo, "discard_project_photo", discard)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(client, sid, include_stl=True, photo_part_id="base")
        assert delivered.get("isError")
        assert "cleanup incomplete" in delivered["content"][0]["text"]
        assert "controlled cleanup failure" in delivered["content"][0]["text"]
        assert not any(item["type"] == "image" for item in delivered["content"])
    assert len(attempts) == 2 and jobs(service) == 0


def test_remote_cancel_waits_for_pair_producer_then_cleans_both(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    replay_worker(monkeypatch)
    ready, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    real_pair = project_photo.create_project_photo_pair
    real_discard = project_photo.discard_project_photo
    photo_ids, discarded = [], []

    def pair(*args, **kwargs):
        value = real_pair(*args, **kwargs)
        photo_ids.extend(value[key]["photo_id"] for key in ("primary", "alternate"))
        ready.set()
        assert release.wait(5)
        return value

    def discard(*args):
        real_discard(*args)
        discarded.append(args[-1])
        if len(discarded) == 2:
            cleaned.set()

    monkeypatch.setattr(project_photo, "create_project_photo_pair", pair)
    monkeypatch.setattr(project_photo, "discard_project_photo", discard)

    async def run():
        verifier = MemberVerifier()
        token = await verifier.verify_token("owner")
        server = remote_mcp.create_server(
            service,
            token_verifier=verifier,
            issuer_url="https://auth.example.test",
            resource_url=RESOURCE,
        )
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
        assert await asyncio.to_thread(ready.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not cleaned.is_set()
        release.set()
        assert await asyncio.to_thread(cleaned.wait, 5)

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert set(discarded) == set(photo_ids) and jobs(service) == 0
