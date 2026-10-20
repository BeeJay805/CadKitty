"""Owned discovery uses saved metadata and stat only, not CAD or content reads."""

from __future__ import annotations

import asyncio
import copy
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from mcp import Client

from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.project_access import project_access_router
from forge_mind.workspace.project_assets import list_project_assets
from forge_mind.workspace.store import Store, WorkspaceError

AID = "a" * 32


def fixture(tmp_path):
    store = Store(tmp_path / "store")
    sid = store.create("owner", "Fit an imported component", {})["id"]
    path = store.root / "assets" / sid / (AID + ".step")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed bytes are not verified by inventory")
    asset = {"id": AID, "name": "Imported motor", "sha256": "b" * 64, "secret": "hidden"}
    store.append_asset(sid, "owner", asset)
    return store, sid, path


def app_for(store):
    app = FastAPI()

    def owner(request: Request):
        return request.headers.get("X-Owner", "owner")

    app.include_router(project_access_router(store, SimpleNamespace(), owner))
    return app


def test_inventory_is_metadata_only_nonmutating_and_does_not_read_files(tmp_path, monkeypatch):
    store, sid, path = fixture(tmp_path)
    before = copy.deepcopy(store.session_record(sid, "owner"))
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("Content read during inventory"))
    monkeypatch.setattr(store, "get", lambda *_: pytest.fail("Expanded session during inventory"))
    result = list_project_assets(store, sid, "owner")
    assert result["assets"] == [
        {
            "asset_id": AID,
            "name": "Imported motor",
            "sha256": "b" * 64,
            "size_bytes": path.stat().st_size,
            "file_status": "available",
            "measurement_status": "not_measured_by_inventory",
        }
    ]
    assert result["content_verification"] == "stored_metadata_only"
    assert "units" in result["scope"] and "submit_document" in result["scope"]
    assert store.session_record(sid, "owner") == before


@pytest.mark.parametrize("state", ["missing", "directory", "stat_error"])
def test_unavailable_file_stays_explicit(tmp_path, monkeypatch, state):
    store, sid, path = fixture(tmp_path)
    if state == "stat_error":
        original = Path.lstat

        def broken(p):
            if p == path:
                raise PermissionError("Private host path must not escape")
            return original(p)

        monkeypatch.setattr(Path, "lstat", broken)
    else:
        path.unlink()
        if state == "directory":
            path.mkdir()
    row = list_project_assets(store, sid, "owner")["assets"][0]
    assert row["file_status"] == "unavailable" and row["size_bytes"] is None


@pytest.mark.parametrize("location", ["parent", "file"])
@pytest.mark.parametrize("link_kind", ["symlink", "reparse"])
def test_linked_paths_are_unavailable_without_following_them(
    tmp_path, monkeypatch, location, link_kind
):
    store, sid, path = fixture(tmp_path)
    selected = path.parent if location == "parent" else path
    original = Path.lstat

    def linked(p):
        metadata = original(p)
        if p == selected:
            return SimpleNamespace(
                st_mode=stat.S_IFLNK if link_kind == "symlink" else metadata.st_mode,
                st_file_attributes=0x400 if link_kind == "reparse" else 0,
                st_size=metadata.st_size,
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", linked)
    assert list_project_assets(store, sid, "owner")["assets"][0]["file_status"] == "unavailable"


def test_fresh_owner_recheck_refuses_revoked_access(tmp_path, monkeypatch):
    store, sid, _ = fixture(tmp_path)
    original = store.session_record
    calls = []

    def checked(project, owner):
        calls.append((project, owner))
        if len(calls) > 1:
            raise WorkspaceError("workspace not found")
        return original(project, owner)

    monkeypatch.setattr(store, "session_record", checked)
    with pytest.raises(WorkspaceError):
        list_project_assets(store, sid, "owner")
    assert calls == [(sid, "owner"), (sid, "owner")]


@pytest.mark.parametrize(
    "assets",
    [
        {},
        [{"id": "../escape", "name": "bad", "sha256": "b" * 64}],
        [{"id": AID, "name": "bad", "sha256": "invalid"}],
        [{"id": AID, "name": {}, "sha256": "b" * 64}],
        [{"id": AID, "name": "duplicate", "sha256": "b" * 64}] * 2,
        [{"id": f"{i:032x}", "name": "overflow", "sha256": "b" * 64} for i in range(101)],
    ],
)
def test_invalid_or_oversize_metadata_refuses_without_change(tmp_path, assets):
    store, sid, _ = fixture(tmp_path)
    store.update(sid, assets=assets)
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError):
        list_project_assets(store, sid, "owner")
    assert store.session_record(sid, "owner") == before


def test_names_are_bounded_and_only_declared_metadata_is_returned(tmp_path):
    store, sid, _ = fixture(tmp_path)
    store.update(sid, assets=[{"id": AID, "name": "M" * 10000, "sha256": "b" * 64}])
    assert list_project_assets(store, sid, "owner")["assets"][0]["name"] == "M" * 160


def test_http_foreign_and_unknown_projects_do_not_disclose_assets(tmp_path):
    store, sid, _ = fixture(tmp_path)
    client = TestClient(app_for(store))
    for project, owner in [(sid, "foreign"), ("f" * 32, "owner"), ("bad", "owner")]:
        response = client.get(f"/workspace/sessions/{project}/assets", headers={"X-Owner": owner})
        assert response.status_code == 404
        assert AID not in response.text


def test_actual_sdk_reads_real_http_route_and_advertises_readonly_discovery(tmp_path):
    store, sid, _ = fixture(tmp_path)
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=httpx.ASGITransport(app=app_for(store)),
    )

    async def scenario():
        async with Client(create_server(adapter)) as client:
            assert client.server_info.version == "0.0.19"
            assert "list_project_assets" in client.instructions
            tools = {t.name: t for t in (await client.list_tools()).tools}
            tool = tools["list_project_assets"]
            assert tool.annotations.read_only_hint is True
            assert set(tool.input_schema["properties"]) == {"project_id"}
            result = await client.call_tool("list_project_assets", {"project_id": sid})
            assert not result.is_error
            assert result.structured_content == list_project_assets(store, sid, "owner")
            bad = await client.call_tool("list_project_assets", {"project_id": "unknown"})
            assert bad.is_error

    asyncio.run(scenario())
    assert not adapter.export_dir.exists()
