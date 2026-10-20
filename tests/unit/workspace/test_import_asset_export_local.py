"""Strict opt-in forwarding and receipt binding through actual SDK/ASGI transport.

The archive verifier is replaced in these adapter tests; real dependency archive
contents are covered by producer/service controls and the separately owned proof.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response
from mcp import Client
from tests.unit.workspace.test_mesh_handoff import document_data, fixture_bytes
from tests.unit.workspace.test_project_export_local import boundary as saved_boundary

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.handoff import handoff_context
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.production import manufacture_bundle

SID, RID, PID = "a" * 32, "b" * 32, "c" * 32


def boundary(tmp_path, monkeypatch, flag):
    data = b"adapter transport fixture, not a validated archive"
    receipt = {
        "project_id": SID,
        "revision_id": RID,
        "package_id": PID,
        "include_stl": False,
        "include_import_assets": flag,
        "sha256": hashlib.sha256(data).hexdigest(),
        "snapshot_sha256": "d" * 64,
        "bytes": len(data),
        "status": "draft_candidate",
        "manufacturing_ready": False,
        "physical_readiness": "unverified",
        "context": {},
        "source_scope": {},
        "made_parts": [],
    }
    app, calls, validations = FastAPI(), [], []

    @app.post("/workspace/sessions/{sid}/project-exports")
    async def create(sid: str, request: Request):
        calls.append(("POST", await request.json()))
        return receipt

    @app.get("/workspace/sessions/{sid}/project-exports/{pid}/{digest}.zip")
    async def read(sid: str, pid: str, digest: str):
        calls.append(("GET", pid, digest))
        return Response(data, media_type="application/zip")

    def validate(path, returned):
        validations.append((path.read_bytes(), returned.copy()))

    monkeypatch.setattr("forge_mind.workspace.project_delivery.validate_project_export", validate)
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
    )
    return adapter, calls, receipt, validations


def call(adapter, option=False):
    # These fixtures exercise CAD/imported-asset delivery without a photo service.
    async def scenario():
        async with Client(create_server(adapter)) as client:
            return await client.call_tool(
                "export_project",
                {
                    "project_id": SID,
                    "revision_id": RID,
                    "include_import_assets": option,
                    "preview": "none",
                },
            )

    return asyncio.run(scenario())


@pytest.mark.parametrize("flag", [False, True])
def test_optin_is_forwarded_and_receipt_preserved(tmp_path, monkeypatch, flag):
    adapter, calls, receipt, validations = boundary(tmp_path, monkeypatch, flag)
    result = call(adapter, flag)
    assert not result.is_error
    delivered = result.structured_content
    path = Path(delivered.pop("path"))
    assert delivered == receipt
    assert calls[0] == (
        "POST",
        {"revision_id": RID, "include_stl": False, "include_import_assets": flag},
    )
    assert validations == [(path.read_bytes(), receipt)]


@pytest.mark.parametrize("option", ["true", 1, None])
def test_nonboolean_refused_before_http(tmp_path, monkeypatch, option):
    adapter, calls, _, validations = boundary(tmp_path, monkeypatch, False)
    assert call(adapter, option).is_error
    assert calls == [] and validations == []
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("returned", [False, "true", 1, None, "missing"])
def test_optin_receipt_mismatch_refused_before_download(tmp_path, monkeypatch, returned):
    adapter, calls, receipt, validations = boundary(tmp_path, monkeypatch, True)
    if returned == "missing":
        receipt.pop("include_import_assets")
    else:
        receipt["include_import_assets"] = returned
    assert call(adapter, True).is_error
    assert len(calls) == 1 and validations == []
    assert not adapter.export_dir.exists()


def test_legacy_default_receipt_remains_readable(tmp_path, monkeypatch):
    adapter, _, receipt, _ = boundary(tmp_path, monkeypatch, False)
    receipt.pop("include_import_assets")
    assert not call(adapter).is_error


def test_dependency_verifier_refusal_removes_only_own_partial_file(tmp_path, monkeypatch):
    adapter, _, _, _ = boundary(tmp_path, monkeypatch, True)
    adapter.export_dir.mkdir()
    existing = adapter.export_dir / "existing.zip"
    existing.write_bytes(b"preserve")

    def refuse(*_):
        raise ValueError("Saved imported dependency changed")

    monkeypatch.setattr("forge_mind.workspace.project_delivery.validate_project_export", refuse)
    result = call(adapter, True)
    assert result.is_error and "dependency changed" in result.content[0].text
    assert list(adapter.export_dir.iterdir()) == [existing]


def test_actual_saved_dependency_archive_through_sdk_and_http(tmp_path):
    """Real byte producer/verifier and SDK transport, without any native CAD build."""
    from forge_mind.workspace.project_delivery import _package_summary

    source = tmp_path / "source"
    _, hashes = fixture_bytes(source, selected=False)
    data = document_data(selected=False)
    asset_id = "e" * 32
    data["assembly"]["parts"][-1]["features"] = [
        {"id": "body", "name": "Owned input", "kind": "import", "asset_id": asset_id}
    ]
    document = DesignDocument.model_validate(data)
    input_bytes = b"ISO-10303-21; retained immutable test input; no native validity claim"
    files = {
        "document.json": document.model_dump_json(indent=2).encode(),
        "part.step": b"saved whole-assembly STEP bytes",
        f"input_{asset_id}.step": input_bytes,
        "assets.json": json.dumps(
            [
                {
                    "id": asset_id,
                    "name": "Reference",
                    "sha256": hashlib.sha256(input_bytes).hexdigest(),
                }
            ]
        ).encode(),
    }
    for name, value in files.items():
        (source / name).write_bytes(value)
        hashes[name] = hashlib.sha256(value).hexdigest()
    inputs = {"version": 0, "inputs": {}}
    context = handoff_context(
        {
            "id": SID,
            "engineering": inputs,
            "source_fingerprint": "input-source",
            "proposal": {"revision_id": RID},
        },
        {
            "id": RID,
            "document": document.model_dump(),
            "measurement": {"engine_fingerprint": "saved-engine"},
        },
        "export-engine",
    )
    package = manufacture_bundle(
        document,
        source,
        tmp_path / "saved.zip",
        hashes,
        sources={"fingerprint": "input-source"},
        engineering_evidence={"revision_id": RID, "physical_readiness": "unverified"},
        engineering_inputs=inputs,
        revision_context=context,
        include_assembly_step=True,
        include_import_assets=True,
    )
    package_bytes = package.read_bytes()
    receipt = {
        **_package_summary(package),
        "package_id": PID,
        "sha256": hashlib.sha256(package_bytes).hexdigest(),
        "bytes": len(package_bytes),
    }
    transport, calls = saved_boundary(receipt, package_bytes)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    result = call(adapter, True)
    assert not result.is_error
    delivered = result.structured_content
    assert delivered["source_scope"]["omitted_import_asset_ids"] == []
    assert delivered["source_scope"]["provided_import_asset_ids"] == [asset_id]
    with ZipFile(delivered["path"]) as archive:
        assert archive.read("document.json") == files["document.json"]
        assert archive.read(f"inputs/input_{asset_id}.step") == input_bytes
        assert archive.read("inputs/assets.json") == files["assets.json"]
        assert not any("guide" in name for name in archive.namelist() if name.startswith("make/"))
    assert calls[0] == (
        "POST",
        SID,
        {"revision_id": RID, "include_stl": False, "include_import_assets": True},
    )
