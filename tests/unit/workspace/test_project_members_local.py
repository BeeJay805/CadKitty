"""Opt-in local files come from one verified project capture, without CAD work."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from tests.unit.workspace.test_project_export_local import PID, RID, SID, boundary, saved_package

from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server


def adapter_for(tmp_path, *, include_stl=False, status=200):
    receipt, content = saved_package(tmp_path, include_stl)
    transport, calls = boundary(receipt, content, status=status)
    adapter = WorkspaceAdapter("http://127.0.0.1:4321", tmp_path / "exports", transport=transport)
    return adapter, receipt, content, calls


def call(adapter, **options):
    async def scenario():
        async with Client(create_server(adapter)) as client:
            return await client.call_tool(
                "export_project",
                {"project_id": SID, "revision_id": RID, "preview": "none", **options},
            )

    return asyncio.run(scenario())


@pytest.mark.parametrize("include_stl", [False, True])
def test_sdk_opt_in_delivers_exact_members_from_two_http_requests(tmp_path, include_stl):
    adapter, receipt, content, calls = adapter_for(tmp_path, include_stl=include_stl)

    async def scenario():
        async with Client(create_server(adapter)) as client:
            tool = {tool.name: tool for tool in (await client.list_tools()).tools}["export_project"]
            option = tool.input_schema["properties"]["include_individual_files"]
            assert option["type"] == "boolean"
            assert option["default"] is False
            assert tool.annotations.read_only_hint is False
            return await client.call_tool(
                "export_project",
                {
                    "project_id": SID,
                    "revision_id": RID,
                    "include_stl": include_stl,
                    "include_individual_files": True,
                    "preview": "none",
                },
            )

    result = asyncio.run(scenario())
    assert not result.is_error
    delivered = result.structured_content
    assert Path(delivered["path"]).read_bytes() == content
    assert {
        key: value for key, value in delivered.items() if key not in {"path", "individual_files"}
    } == receipt
    rows = delivered["individual_files"]
    assert len(rows) == (5 if include_stl else 3)
    assert [
        (row["part_id"], row["quantity"], row["format"]) for row in rows if row["role"] == "make"
    ] == (
        [("base", 1, "step"), ("base", 1, "stl"), ("cap", 2, "step"), ("cap", 2, "stl")]
        if include_stl
        else [("base", 1, "step"), ("cap", 2, "step")]
    )
    assembly = [row for row in rows if row["role"] == "assembly"]
    assert len(assembly) == 1
    assert assembly[0]["part_id"] is None
    assert assembly[0]["quantity"] == 1
    assert assembly[0]["frame"] == "cad_z_up_mm_world"
    assert assembly[0]["archive_path"] == "assembly/part.step"
    assert all(row["frame"] == "definition_local" for row in rows if row["role"] == "make")
    with ZipFile(Path(delivered["path"])) as archive:
        for row in rows:
            path = Path(row["path"])
            payload = path.read_bytes()
            assert path.parent == adapter.export_dir
            assert payload == archive.read(row["archive_path"])
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    assert len({row["path"] for row in rows}) == len(rows)
    assert calls == [
        (
            "POST",
            SID,
            {"revision_id": RID, "include_stl": include_stl, "include_import_assets": False},
        ),
        ("GET", SID, PID, receipt["sha256"]),
    ]
    assert len(list(adapter.export_dir.iterdir())) == len(rows) + 1


@pytest.mark.parametrize("explicit", [False, True])
def test_default_response_and_saved_zip_are_unchanged(tmp_path, explicit):
    adapter, receipt, content, calls = adapter_for(tmp_path)
    result = call(adapter, **({"include_individual_files": False} if explicit else {}))
    assert not result.is_error
    delivered = result.structured_content
    path = Path(delivered.pop("path"))
    assert delivered == receipt
    assert path.read_bytes() == content
    assert list(adapter.export_dir.iterdir()) == [path]
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["true", 1, None])
def test_sdk_strict_option_refuses_before_http(tmp_path, value):
    adapter, _, _, calls = adapter_for(tmp_path)
    assert call(adapter, include_individual_files=value).is_error
    assert calls == []
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("value", ["true", 1, None])
def test_adapter_strict_option_refuses_before_http(tmp_path, value):
    adapter, _, _, calls = adapter_for(tmp_path)
    with pytest.raises(ToolError, match="Invalid project export"):
        asyncio.run(adapter.export_project(SID, RID, include_individual_files=value))
    assert calls == []


def test_foreign_refusal_creates_no_file(tmp_path):
    adapter, _, _, calls = adapter_for(tmp_path, status=404)
    assert call(adapter, include_individual_files=True).is_error
    assert len(calls) == 1
    assert not adapter.export_dir.exists()


@pytest.mark.parametrize("fault", ["member_read", "partial_write", "cancellation"])
def test_failure_after_first_member_removes_only_this_calls_files(tmp_path, monkeypatch, fault):
    import forge_mind.workspace.project_members as members

    adapter, _, _, calls = adapter_for(tmp_path)
    adapter.export_dir.mkdir()
    preserved = adapter.export_dir / "prior.stl"
    preserved.write_bytes(b"customer file")
    reads = 0
    original_read = members.read_captured_cad_file
    original_open = Path.open

    def failing_read(*args):
        nonlocal reads
        reads += 1
        if reads == 2 and fault != "partial_write":
            if fault == "cancellation":
                raise asyncio.CancelledError
            raise ValueError("captured member unavailable")
        return original_read(*args)

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, payload):
            self.stream.write(payload[:4])
            raise OSError("member disk full")

        def __exit__(self, *_):
            self.stream.close()

    def failing_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return (
            PartialWrite(stream)
            if mode == "xb" and path.suffix == ".step" and reads == 2 and fault == "partial_write"
            else stream
        )

    monkeypatch.setattr(members, "read_captured_cad_file", failing_read)
    monkeypatch.setattr(Path, "open", failing_open)
    if fault == "cancellation":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(adapter.export_project(SID, RID, include_individual_files=True))
    else:
        assert call(adapter, include_individual_files=True).is_error
    assert reads == 2
    assert list(adapter.export_dir.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"customer file"
    assert len(calls) == 2


def test_real_task_cancellation_at_member_checkpoint_rolls_back(tmp_path, monkeypatch):
    import forge_mind.workspace.project_members as members

    adapter, _, _, _ = adapter_for(tmp_path)
    original_read = members.read_captured_cad_file

    def cancel_current_task(*args):
        payload = original_read(*args)
        asyncio.current_task().cancel()
        return payload

    monkeypatch.setattr(members, "read_captured_cad_file", cancel_current_task)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.export_project(SID, RID, include_individual_files=True))
    assert list(adapter.export_dir.iterdir()) == []


def test_member_filename_collision_preserves_prior_file(tmp_path, monkeypatch):
    import forge_mind.workspace.mcp_server as bridge

    adapter, _, _, _ = adapter_for(tmp_path)
    adapter.export_dir.mkdir()
    preserved = adapter.export_dir / f"{SID}-{RID}-assembly-step-{'f' * 12}.step"
    preserved.write_bytes(b"preserve")
    monkeypatch.setattr(bridge.uuid, "uuid4", lambda: SimpleNamespace(hex="f" * 32))
    assert call(adapter, include_individual_files=True).is_error
    assert list(adapter.export_dir.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"preserve"


def test_directory_swapped_to_link_after_checkpoint_refuses_write_and_cleanup(
    tmp_path, monkeypatch
):
    import forge_mind.workspace.project_members as members

    adapter, _, content, calls = adapter_for(tmp_path)
    adapter.export_dir.mkdir()
    preserved = adapter.export_dir / "prior.stl"
    preserved.write_bytes(b"customer file")
    linked = False
    member_writes = []
    cleanup_attempts = []
    original_read = members.read_captured_cad_file
    original_is_symlink = Path.is_symlink
    original_open = Path.open
    original_unlink = Path.unlink

    def swap_after_checkpoint(*args):
        nonlocal linked
        payload = original_read(*args)
        linked = True
        return payload

    def is_symlink(path):
        return linked if path == adapter.export_dir else original_is_symlink(path)

    def record_open(path, mode="r", *args, **kwargs):
        if mode == "xb" and path.suffix in {".step", ".stl"}:
            member_writes.append(path)
        return original_open(path, mode, *args, **kwargs)

    def record_unlink(path, *args, **kwargs):
        cleanup_attempts.append(path)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(members, "read_captured_cad_file", swap_after_checkpoint)
    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(Path, "unlink", record_unlink)
    with pytest.raises(ToolError, match="cleanup incomplete") as error:
        asyncio.run(adapter.export_project(SID, RID, include_individual_files=True))
    assert "symlinks or junctions" in str(error.value)
    assert linked
    assert member_writes == []
    assert cleanup_attempts == []
    assert preserved.read_bytes() == b"customer file"
    remaining = list(adapter.export_dir.iterdir())
    new_files = [path for path in remaining if path != preserved]
    assert len(new_files) == 1
    assert new_files[0].suffix == ".zip"
    assert new_files[0].read_bytes() == content
    assert len(calls) == 2


def test_zip_plus_extracted_member_budget_refuses_without_partial_delivery(tmp_path, monkeypatch):
    adapter, _, content, calls = adapter_for(tmp_path, include_stl=True)
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_EXPORT_BYTES", str(len(content)))
    assert call(adapter, include_stl=True, include_individual_files=True).is_error
    assert list(adapter.export_dir.iterdir()) == []
    assert len(calls) == 2


def test_metadata_with_paths_obeys_response_budget(tmp_path, monkeypatch):
    adapter, receipt, _, calls = adapter_for(tmp_path)
    limit = len(json.dumps(receipt).encode("utf-8")) + 1
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", str(limit))
    assert call(adapter, include_individual_files=True).is_error
    assert list(adapter.export_dir.iterdir()) == []
    assert len(calls) == 2
