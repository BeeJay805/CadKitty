"""Saved flat CAD delivery through actual SDK dispatch; only graphics are replayed."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_product_photo_transport import PNG, envelope
from tests.unit.workspace.test_project_delivery import RID
from tests.unit.workspace.test_project_export_remote import RESOURCE, result, rpc
from tests.unit.workspace.test_project_members_remote import app, counts
from tests.unit.workspace.test_project_photo_delivery_transport import photo_pair, remote_export

from forge_mind.workspace import delivery_guidance, mcp_server, project_delivery, project_photo
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.product_photo_contract import CAPTURED_SCOPE, OPPOSED_VIEW, legacy_request
from forge_mind.workspace.store import Store


def flat_saved_project(tmp_path, monkeypatch, *, role="make"):
    """Pure byte fixture, not evidence of geometry or printing quality."""
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Delivery cannot start a worker"))
    sid = service.store.create("owner", "Saved flat part", None)["id"]
    raw = {
        "features": [{"id": "body", "name": "Body", "kind": "box"}],
        "outputs": ["body"],
    }
    if role is not None:
        raw["production"] = {"role": role}
    document = DesignDocument.model_validate(raw).model_dump(mode="json")
    # Deliberately retain whitespace which normalized document serialization would lose.
    document_bytes = (json.dumps(document, ensure_ascii=False, indent=3) + "\n").encode()
    files = {
        "program.py": b"# Saved fixture source; never execute.\n",
        "document.json": document_bytes,
        "part.step": b"Saved flat STEP fixture; not native geometry evidence",
        "part.stl": b"Saved flat STL fixture; never parsed by a renderer",
    }
    source = service.store.directory(sid, RID)
    source.mkdir(parents=True)
    for name, data in files.items():
        (source / name).write_bytes(data)
    contract = Contract()
    measurement = Measurement(
        revision_id=RID,
        geometry_valid=True,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="fixture-measurement-engine",
        artifact_hashes={name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    ).model_dump()
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "document": document,
            "contract": contract.model_dump(),
            "measurement": measurement,
        },
    )
    service.store.update(sid, selected_revision=RID)
    monkeypatch.setattr(project_delivery, "engine_fingerprint", lambda: "fixture-export-engine")
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "fixture-export-engine")
    return service, sid, source, files


def replay_renderer(monkeypatch, *, failure=None):
    """Leave capture, selection, validation and durable readback on production paths."""
    calls = []

    def render(service, mesh, request, *, timeout):
        assert timeout > 0
        calls.append((mesh, dict(request)))
        if failure == "unavailable":
            raise project_photo.PhotoUnavailableError(
                "renderer_unavailable", "Fixture graphics absent"
            )
        value = envelope(request["project_id"], request["preset"])
        value["receipt"] = {
            **request,
            "scope": CAPTURED_SCOPE,
            "image": value["receipt"]["image"],
            "rendering": value["receipt"]["rendering"],
        }
        if request.get("view") == OPPOSED_VIEW:
            value["receipt"] = photo_pair({"status": "available", **value})["alternate"]["receipt"]
        if failure == "tamper":
            value["receipt"]["mesh_sha256"] = "9" * 64
        return value

    monkeypatch.setattr(project_photo, "render_mesh", render)
    return calls


def local_adapter(service, tmp_path):
    application = FastAPI()
    application.include_router(product_router(service.store, service, lambda: "owner"))
    return mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=httpx.ASGITransport(app=application),
    )


def local_export(adapter, sid, **options):
    async def scenario():
        async with Client(mcp_server.create_server(adapter)) as client:
            return await client.call_tool(
                "export_project", {"project_id": sid, "revision_id": RID, **options}
            )

    return asyncio.run(scenario())


def assert_flat_capture(data, metadata, files, *, include_stl):
    with ZipFile(io.BytesIO(data)) as archive:
        expected = {"make/part.step", "make/part.stl"} if include_stl else {"make/part.step"}
        assert {name for name in archive.namelist() if name.startswith("make/")} == expected
        assert archive.read("assembly/part.step") == files["part.step"]
        assert archive.read("make/part.step") == files["part.step"]
        if include_stl:
            assert archive.read("make/part.stl") == files["part.stl"]
            assert archive.read("document.json") == files["document.json"]
        assert json.loads(archive.read("document.json")) == json.loads(files["document.json"])
        assert "definition-meshes.json" not in archive.namelist()
        assert "flat_mesh" not in json.loads(archive.read("manifest.json"))
    assert metadata["made_parts"] == [
        {
            "part_id": "part",
            "quantity": 1,
            "files": sorted(expected),
            "frame": "document_local",
        }
    ]
    assert len(data) == metadata["bytes"]
    assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
    assert metadata["manufacturing_ready"] is False
    assert metadata["physical_readiness"] == "unverified"


@pytest.mark.parametrize("explicit_part", [False, True])
def test_local_flat_make_same_capture_files_and_actual_image_content(
    tmp_path, monkeypatch, explicit_part
):
    service, sid, source, files = flat_saved_project(tmp_path, monkeypatch)
    calls = replay_renderer(monkeypatch)
    before = counts(service)
    adapter = local_adapter(service, tmp_path)
    options = {"photo_part_id": "part"} if explicit_part else {}
    response = local_export(
        adapter, sid, include_stl=True, include_individual_files=True, **options
    )
    assert not response.is_error
    metadata = response.structured_content
    data = Path(metadata["path"]).read_bytes()
    assert_flat_capture(data, metadata, files, include_stl=True)
    rows = metadata["individual_files"]
    assert {row["file_id"] for row in rows} == {"assembly-step", "part-part-step", "part-part-stl"}
    for row in rows:
        assert row["quantity"] == 1
        assert Path(row["path"]).read_bytes() == files["part." + row["format"]]
        if row["role"] == "make":
            assert row["frame"] == "document_local" and row["part_id"] == "part"
    preview = metadata["preview"]
    assert preview["status"] == "available"
    assert preview["receipt"]["frame"] == "document_local"
    assert preview["receipt"]["file_id"] == "part-part-stl"
    assert preview["receipt"]["mesh_sha256"] == hashlib.sha256(files["part.stl"]).hexdigest()
    assert preview["receipt"]["package_sha256"] == metadata["sha256"]
    assert Path(preview["path"]).read_bytes() == PNG
    assert base64.b64decode(response.content[1].data, validate=True) == PNG
    assert json.loads(response.content[0].text) == metadata
    alternate = metadata["preview_alternate"]
    assert alternate["view"] == alternate["receipt"]["view"] == OPPOSED_VIEW
    assert alternate["receipt"]["version"] == 2
    assert alternate["receipt"]["frame"] == "document_local"
    assert alternate["receipt"]["package_sha256"] == metadata["sha256"]
    assert alternate["receipt"]["mesh_sha256"] == preview["receipt"]["mesh_sha256"]
    assert alternate["photo_id"] != preview["photo_id"]
    assert Path(alternate["path"]).read_bytes() == PNG
    assert json.loads(Path(alternate["receipt_path"]).read_bytes()) == alternate["receipt"]
    assert len(response.content) == 3
    assert base64.b64decode(response.content[2].data, validate=True) == PNG
    assert len(calls) == 2 and all(mesh == files["part.stl"] for mesh, _ in calls)
    assert legacy_request(calls[1][1]) == calls[0][1]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == files
    assert counts(service) == before == {"jobs": 0, "revisions": 1}


def test_remote_flat_make_durable_exact_files_photo_and_fresh_owner_refusal(tmp_path, monkeypatch):
    service, sid, source, files = flat_saved_project(tmp_path, monkeypatch)
    calls = replay_renderer(monkeypatch)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        response = remote_export(client, sid, include_stl=True, include_individual_files=True)
        assert not response.get("isError")
        metadata = response["structuredContent"]
        fetched = result(rpc(client, "owner", "resources/read", {"uri": metadata["uri"]}))
        data = base64.b64decode(fetched["contents"][0]["blob"], validate=True)
        assert_flat_capture(data, metadata, files, include_stl=True)
        assert len(metadata["individual_files"]) == 3
        for row in metadata["individual_files"]:
            content = result(rpc(client, "owner", "resources/read", {"uri": row["uri"]}))
            raw = base64.b64decode(content["contents"][0]["blob"], validate=True)
            assert raw == files["part." + row["format"]]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
            assert row["quantity"] == 1 and "path" not in row
            assert "error" in rpc(client, "foreign", "resources/read", {"uri": row["uri"]}).json()
        preview = metadata["preview"]
        assert preview["receipt"]["frame"] == "document_local"
        assert base64.b64decode(response["content"][1]["data"], validate=True) == PNG
        photo_uri = preview["uri"]
        assert "error" in rpc(client, "foreign", "resources/read", {"uri": photo_uri}).json()
        assert "path" not in preview and "receipt_path" not in preview
        alternate = metadata["preview_alternate"]
        assert alternate["view"] == alternate["receipt"]["view"] == OPPOSED_VIEW
        assert alternate["receipt"]["frame"] == "document_local"
        assert alternate["receipt"]["package_sha256"] == metadata["sha256"]
        assert alternate["photo_id"] != preview["photo_id"]
        assert len(response["content"]) == 3
        assert base64.b64decode(response["content"][2]["data"], validate=True) == PNG
        alternate_uri = alternate["uri"]
        assert "error" in rpc(client, "foreign", "resources/read", {"uri": alternate_uri}).json()
    assert service.stop.is_set()
    fresh_service = WorkspaceService(Store(service.store.root), Executor("docker"))
    monkeypatch.setattr(
        fresh_service, "start", lambda: pytest.fail("Readback cannot start a worker")
    )
    with TestClient(app(fresh_service), base_url=RESOURCE.removesuffix("/mcp")) as restarted:
        for uri in (photo_uri, alternate_uri):
            content = result(rpc(restarted, "owner", "resources/read", {"uri": uri}))
            assert base64.b64decode(content["contents"][0]["blob"], validate=True) == PNG
    assert len(calls) == 2 and all(mesh == files["part.stl"] for mesh, _ in calls)
    assert legacy_request(calls[1][1]) == calls[0][1]
    assert {path.name: path.read_bytes() for path in source.iterdir()} == files
    assert counts(service) == {"jobs": 0, "revisions": 1}


@pytest.mark.parametrize("mode", ["step_only", "none", "unavailable"])
def test_local_flat_legacy_and_optional_graphics_keep_cad(tmp_path, monkeypatch, mode):
    service, sid, _, files = flat_saved_project(tmp_path, monkeypatch)
    calls = replay_renderer(monkeypatch, failure="unavailable" if mode == "unavailable" else None)
    adapter = local_adapter(service, tmp_path)
    options = {"include_stl": mode != "step_only"}
    if mode == "none":
        options["preview"] = "none"
    response = local_export(adapter, sid, **options)
    assert not response.is_error and len(response.content) == 1
    metadata = response.structured_content
    assert_flat_capture(
        Path(metadata["path"]).read_bytes(), metadata, files, include_stl=mode != "step_only"
    )
    if mode == "none":
        assert "preview" not in metadata
    else:
        assert metadata["preview"]["code"] == (
            "missing_stl" if mode == "step_only" else "renderer_unavailable"
        )
        assert metadata["preview"]["candidates"] == ["part"]
    assert len(calls) == (1 if mode == "unavailable" else 0)
    assert not list(adapter.export_dir.glob("*.png"))


@pytest.mark.parametrize("role", [None, "reference", "buy"])
def test_local_flat_stl_requires_explicit_make_without_rendering(tmp_path, monkeypatch, role):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch, role=role)
    calls = replay_renderer(monkeypatch)
    adapter = local_adapter(service, tmp_path)
    response = local_export(adapter, sid, include_stl=True)
    assert response.is_error and not calls
    assert not list(adapter.export_dir.glob("*.zip"))
    assert counts(service) == {"jobs": 0, "revisions": 1}


def test_local_flat_photo_integrity_failure_removes_own_delivery_preserves_prior(
    tmp_path, monkeypatch
):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    replay_renderer(monkeypatch, failure="tamper")
    adapter = local_adapter(service, tmp_path)
    adapter.export_dir.mkdir()
    prior = adapter.export_dir / "prior.stl"
    prior.write_bytes(b"preserve prior export")
    response = local_export(adapter, sid, include_stl=True, include_individual_files=True)
    assert response.is_error and list(adapter.export_dir.iterdir()) == [prior]
    assert prior.read_bytes() == b"preserve prior export"
    assert counts(service) == {"jobs": 0, "revisions": 1}


def test_local_flat_changed_saved_stl_refuses_capture_before_graphics(tmp_path, monkeypatch):
    service, sid, source, _ = flat_saved_project(tmp_path, monkeypatch)
    calls = replay_renderer(monkeypatch)
    (source / "part.stl").write_bytes(b"changed after saved measurement")
    adapter = local_adapter(service, tmp_path)
    response = local_export(adapter, sid, include_stl=True, include_individual_files=True)
    assert response.is_error and not calls
    assert not list(adapter.export_dir.glob("*.zip"))
    assert counts(service) == {"jobs": 0, "revisions": 1}
