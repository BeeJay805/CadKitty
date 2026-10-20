"""Local SDK/HTTP delivery of saved review bytes; no author or CAD worker runs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from mcp import Client
from tests.unit.workspace.test_mesh_handoff import fixture_bytes

from forge_mind.workspace.handoff import handoff_context
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.production import manufacture_bundle

SID, RID, PID = "a" * 32, "b" * 32, "c" * 32


def saved_package(tmp_path: Path, include_stl: bool = False, *, unique_made: bool = False):
    from forge_mind.workspace.project_delivery import _package_summary

    source = tmp_path / "source"
    document, hashes = fixture_bytes(source)
    if unique_made:
        # This pure saved-byte fixture leaves only base used, without a native build.
        raw = document.model_dump()
        raw["assembly"]["roots"] = [
            row for row in raw["assembly"]["roots"] if row["definition"] != "cap"
        ]
        next(part for part in raw["assembly"]["parts"] if part["id"] == "cap")["production"][
            "deliver_stl"
        ] = False
        document = type(document).model_validate(raw)
        document_bytes = json.dumps(document.model_dump()).encode()
        (source / "document.json").write_bytes(document_bytes)
        hashes["document.json"] = hashlib.sha256(document_bytes).hexdigest()
        manifest_path = source / "definition-meshes.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["document_sha256"] = hashes["document.json"]
        manifest["parts"].pop("cap")
        manifest_bytes = json.dumps(manifest).encode()
        manifest_path.write_bytes(manifest_bytes)
        hashes["definition-meshes.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    step = b"saved complete eight-body STEP fixture"
    (source / "part.step").write_bytes(step)
    hashes["part.step"] = hashlib.sha256(step).hexdigest()
    inputs = {"version": 2, "inputs": {"loads": []}}
    context = handoff_context(
        {
            "id": SID,
            "selected_revision": None,
            "accepted_revision": None,
            "proposal": {"revision_id": RID},
            "engineering": inputs,
            "source_fingerprint": "sources-1",
        },
        {
            "id": RID,
            "document": document.model_dump(),
            "measurement": {"engine_fingerprint": "measurement-engine"},
        },
        "export-engine",
    )
    package = manufacture_bundle(
        document,
        source,
        tmp_path / "saved.zip",
        hashes,
        sources={"fingerprint": "sources-1"},
        engineering_evidence={"revision_id": RID, "physical_readiness": "unverified"},
        revision_context=context,
        engineering_inputs=inputs,
        include_stl=include_stl,
        include_assembly_step=True,
    )
    data = package.read_bytes()
    receipt = {
        **_package_summary(package),
        "package_id": PID,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
    return receipt, data


def boundary(receipt, data, *, status=200):
    """ASGI simulates the owned route; shared package validation remains real."""
    app = FastAPI()
    calls = []

    @app.post("/workspace/sessions/{sid}/project-exports")
    async def create(sid: str, request: Request):
        calls.append(("POST", sid, await request.json()))
        return JSONResponse(
            receipt if status == 200 else {"detail": "Unknown owned project"}, status_code=status
        )

    @app.get("/workspace/sessions/{sid}/project-exports/{pid}/{digest}.zip")
    async def download(sid: str, pid: str, digest: str):
        calls.append(("GET", sid, pid, digest))
        return Response(data, media_type="application/zip")

    return httpx.ASGITransport(app=app), calls


def sdk_call(adapter, arguments=None):
    async def scenario():
        async with Client(create_server(adapter)) as client:
            return await client.call_tool(
                "export_project",
                {"preview": "none", **(arguments or {"project_id": SID, "revision_id": RID})},
            )

    return asyncio.run(scenario())


@pytest.mark.parametrize("include_stl", [False, True])
def test_real_sdk_and_asgi_deliver_verified_saved_project(tmp_path, include_stl):
    receipt, data = saved_package(tmp_path, include_stl)
    transport, calls = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)

    async def scenario():
        async with Client(create_server(adapter)) as client:
            assert client.server_info.version == "0.0.19"
            assert "export_project" in client.instructions
            tool = {tool.name: tool for tool in (await client.list_tools()).tools}["export_project"]
            assert tool.annotations.read_only_hint is False
            assert set(tool.input_schema["properties"]) == {
                "project_id",
                "revision_id",
                "include_stl",
                "include_import_assets",
                "include_individual_files",
                "preview",
                "photo_preset",
                "photo_part_id",
            }
            assert tool.input_schema["properties"]["include_stl"]["default"] is False
            result = await client.call_tool(
                "export_project",
                {
                    "project_id": SID,
                    "revision_id": RID,
                    "include_stl": include_stl,
                    "preview": "none",
                },
            )
            assert not result.is_error
            delivered = result.structured_content
            path = Path(delivered.pop("path"))
            assert delivered == receipt
            assert path.parent == adapter.export_dir
            assert path.read_bytes() == data
            with ZipFile(path) as archive:
                assert (
                    archive.read("assembly/part.step") == b"saved complete eight-body STEP fixture"
                )
                assert (
                    json.loads(archive.read("document.json"))["assembly"]["roots"][2]["definition"]
                    == "cap"
                )
            assert [(p["part_id"], p["quantity"]) for p in delivered["made_parts"]] == [
                ("base", 1),
                ("cap", 2),
            ]

    asyncio.run(scenario())
    assert calls == [
        (
            "POST",
            SID,
            {"revision_id": RID, "include_stl": include_stl, "include_import_assets": False},
        ),
        ("GET", SID, PID, receipt["sha256"]),
    ]


@pytest.mark.parametrize("option", ["true", 1, None])
def test_sdk_strict_boolean_refuses_before_http(tmp_path, option):
    transport, calls = boundary({}, b"")
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    result = sdk_call(adapter, {"project_id": SID, "revision_id": RID, "include_stl": option})
    assert result.is_error
    assert calls == []
    assert not adapter.export_dir.exists()


def test_wrong_owner_http_refusal_creates_no_local_file(tmp_path):
    transport, calls = boundary({}, b"", status=404)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    assert sdk_call(adapter).is_error
    assert len(calls) == 1
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "d" * 32),
        ("revision_id", "d" * 32),
        ("package_id", "../escape"),
        ("sha256", "INVALID"),
        ("snapshot_sha256", None),
        ("include_stl", 0),
        ("include_stl", True),
        ("bytes", True),
        ("bytes", 0),
        ("bytes", 64 * 1024 * 1024 + 1),
        ("status", "accepted"),
        ("manufacturing_ready", True),
        ("physical_readiness", "verified"),
        ("context", []),
        ("source_scope", []),
        ("made_parts", ["base"]),
        ("path", "C:/unexpected.zip"),
    ],
)
def test_malformed_receipt_refuses_before_download(tmp_path, field, value):
    receipt, data = saved_package(tmp_path)
    receipt[field] = value
    transport, calls = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    assert sdk_call(adapter).is_error
    assert len(calls) == 1
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("change", ["hash", "length"])
def test_download_mismatch_creates_no_local_file(tmp_path, change):
    receipt, data = saved_package(tmp_path)
    data = bytes([data[0] ^ 1]) + data[1:] if change == "hash" else data + b"extra"
    transport, calls = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    assert sdk_call(adapter).is_error
    assert len(calls) == 2
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("change", ["quantity", "context", "source"])
def test_offline_validation_rejects_rebound_metadata_and_removes_file(tmp_path, change):
    receipt, data = saved_package(tmp_path)
    receipt = copy.deepcopy(receipt)
    if change == "quantity":
        receipt["made_parts"][0]["quantity"] += 1
    elif change == "context":
        receipt["context"]["input_version"] += 1
    else:
        receipt["source_scope"]["omitted_import_asset_ids"] = ["d" * 32]
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    assert sdk_call(adapter).is_error
    assert list(adapter.export_dir.iterdir()) == []


def test_partial_write_failure_removes_only_created_export(tmp_path, monkeypatch):
    receipt, data = saved_package(tmp_path)
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    adapter.export_dir.mkdir()
    preserved = adapter.export_dir / "existing.zip"
    preserved.write_bytes(b"preserve")
    original_open = Path.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, payload):
            self.stream.write(payload[:20])
            raise OSError("disk full")

        def __exit__(self, *_):
            self.stream.close()

    def fail_write(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return (
            PartialWrite(stream) if mode == "xb" and path.parent == adapter.export_dir else stream
        )

    monkeypatch.setattr(Path, "open", fail_write)
    assert sdk_call(adapter).is_error
    assert list(adapter.export_dir.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"preserve"


def test_exclusive_filename_collision_preserves_existing_file(tmp_path, monkeypatch):
    import forge_mind.workspace.mcp_server as bridge

    receipt, data = saved_package(tmp_path)
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    adapter.export_dir.mkdir()
    preserved = adapter.export_dir / f"{SID}-{RID}-project-{'f' * 12}.zip"
    preserved.write_bytes(b"preserve")
    monkeypatch.setattr(bridge.uuid, "uuid4", lambda: SimpleNamespace(hex="f" * 32))
    assert sdk_call(adapter).is_error
    assert preserved.read_bytes() == b"preserve"


def test_linked_export_directory_refuses_without_write(tmp_path, monkeypatch):
    receipt, data = saved_package(tmp_path)
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    real_check = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == adapter.export_dir or real_check(path)
    )
    assert sdk_call(adapter).is_error
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("target", ["receipt", "download", "declared"])
def test_operator_transfer_limit_refuses_without_write(tmp_path, monkeypatch, target):
    receipt, data = saved_package(tmp_path)
    if target == "receipt":
        monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "20")
    else:
        limit = max(len(json.dumps(receipt).encode()) + 1, len(data) + 1)
        monkeypatch.setenv("FORGE_WORKSPACE_MCP_EXPORT_BYTES", str(limit))
        if target == "download":
            data += b"x" * limit
        else:
            receipt["bytes"] = limit + 1
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    assert sdk_call(adapter).is_error
    assert not adapter.export_dir.exists()


def test_cancellation_after_file_creation_removes_file_and_propagates(tmp_path, monkeypatch):
    import forge_mind.workspace.project_delivery as delivery

    receipt, data = saved_package(tmp_path)
    transport, _ = boundary(receipt, data)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)

    def cancel(*_):
        raise asyncio.CancelledError

    monkeypatch.setattr(delivery, "validate_project_export", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.export_project(SID, RID))
    assert list(adapter.export_dir.iterdir()) == []


def test_existing_artifact_export_still_uses_revision_evidence(tmp_path):
    data = b"existing assembled STEP"
    requests = []

    def backend(request):
        requests.append(request.url.path)
        return (
            httpx.Response(
                200,
                json={
                    "revision_id": RID,
                    "geometry_valid": True,
                    "artifact_hashes": {"part.step": hashlib.sha256(data).hexdigest()},
                },
            )
            if request.url.path.endswith("evidence.json")
            else httpx.Response(200, content=data)
        )

    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.MockTransport(backend)
    )
    result = asyncio.run(adapter.export(SID, RID, "step"))
    assert Path(result["path"]).read_bytes() == data
    assert result["geometry_valid"] is True
    assert len(requests) == 2
