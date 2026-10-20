"""Authenticated SDK saved-member delivery; fixture CAD bytes never run a worker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import sqlite3
import threading
import time
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from tests.unit.workspace.test_project_delivery import RID, saved_project
from tests.unit.workspace.test_project_export_remote import ISSUER, RESOURCE, result, rpc

from forge_mind.workspace import project_delivery, remote_mcp


class MemberVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token not in {"owner", "foreign", "no-read"}:
            return None
        return AccessToken(
            token=token,
            client_id="member-sdk-test",
            subject=token,
            scopes=[] if token == "no-read" else ["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": ISSUER},
        )


def app(service):
    return remote_mcp.create_app(
        service, token_verifier=MemberVerifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )


def export(client, sid, **options):
    return rpc(
        client,
        "owner",
        "tools/call",
        {
            "name": "export_project",
            "arguments": {"project_id": sid, "revision_id": RID, "preview": "none", **options},
        },
    )


def read(client, uri, token="owner"):
    return rpc(client, token, "resources/read", {"uri": uri})


def counts(service):
    with sqlite3.connect(service.store.root / "workspace.sqlite3") as connection:
        return {
            name: connection.execute(f"select count(*) from {name}").fetchone()[0]
            for name in ("jobs", "revisions")
        }


@pytest.mark.parametrize("include_stl", [False, True])
def test_actual_sdk_members_match_same_saved_capture_and_survive_restart(
    tmp_path, monkeypatch, include_stl
):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    before = counts(service)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        initialized = result(
            rpc(
                client,
                "owner",
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "members-proof", "version": "1"},
                },
            )
        )
        assert "export_project" in initialized["instructions"]
        tools = result(rpc(client, "owner", "tools/list", {}))["tools"]
        tool = next(value for value in tools if value["name"] == "export_project")
        flag = tool["inputSchema"]["properties"]["include_individual_files"]
        assert flag["default"] is False and flag["type"] == "boolean"
        assert tool["annotations"]["readOnlyHint"] is True
        response = result(
            export(client, sid, include_stl=include_stl, include_individual_files=True)
        )
        assert not response.get("isError")
        captured = response["structuredContent"]
        data = base64.b64decode(
            result(read(client, captured["uri"]))["contents"][0]["blob"], validate=True
        )
        assert hashlib.sha256(data).hexdigest() == captured["sha256"]
        assert len(captured["individual_files"]) == (5 if include_stl else 3)
        with ZipFile(io.BytesIO(data)) as archive:
            for member in captured["individual_files"]:
                fetched = result(read(client, member["uri"]))["contents"][0]
                raw = base64.b64decode(fetched["blob"], validate=True)
                assert fetched["mimeType"] == "application/octet-stream"
                assert raw == archive.read(member["archive_path"])
                assert raw == (source / member["archive_path"].split("/")[-1]).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == member["sha256"]
                assert len(raw) == member["bytes"]
                assert captured["sha256"] in member["uri"]
                assert "path" not in member
                assert member["part_id"] in {None, "base", "cap"}
                assert member["quantity"] == (2 if member["part_id"] == "cap" else 1)
                assert member["frame"] == (
                    "cad_z_up_mm_world" if member["part_id"] is None else "definition_local"
                )
                assert member["role"] == ("assembly" if member["part_id"] is None else "make")
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as restarted:
        for member in captured["individual_files"]:
            assert "contents" in result(read(restarted, member["uri"]))
            assert "error" in read(restarted, member["uri"], "foreign").json()
    assert counts(service) == before == {"jobs": 0, "revisions": 1}
    assert not service.executor.typed_container_verified


def test_default_receipt_unchanged_and_explicit_false_does_not_read_members(tmp_path, monkeypatch):
    from forge_mind.workspace import project_members

    service, sid, _ = saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        project_members,
        "captured_cad_files",
        lambda *args: pytest.fail("Default cannot select members"),
    )
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        for options in ({}, {"include_individual_files": False}):
            response = result(export(client, sid, **options))["structuredContent"]
            _, saved = project_delivery.read_project_export(
                service, sid, "owner", response["package_id"], response["sha256"]
            )
            assert response == {
                **saved,
                "uri": f"cadkitty://projects/{sid}/exports/{saved['package_id']}/{saved['sha256']}.zip",
            }
            assert "individual_files" not in response


@pytest.mark.parametrize("value", [1, "true", None, [], {}])
def test_new_option_is_strict_before_capture(tmp_path, monkeypatch, value):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        assert result(export(client, sid, include_individual_files=value))["isError"]
    assert not (service.store.root / "project-exports").exists()
    assert counts(service) == {"jobs": 0, "revisions": 1}


@pytest.mark.parametrize("token", [None, "invalid", "no-read", "foreign"])
def test_member_read_reauthenticates_each_caller(tmp_path, monkeypatch, token):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        captured = result(export(client, sid, include_individual_files=True))["structuredContent"]
        denied = read(client, captured["individual_files"][0]["uri"], token)
        assert denied.status_code in {401, 403} or "error" in denied.json()


@pytest.mark.parametrize(
    "mutation",
    [
        "package_hash",
        "package_id",
        "project_id",
        "file_id",
        "bad_file_id",
        "zip",
        "missing_zip",
        "missing_member",
    ],
)
def test_member_refuses_changed_missing_or_malformed_resource(tmp_path, monkeypatch, mutation):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        captured = result(export(client, sid, include_individual_files=True))["structuredContent"]
        uri = captured["individual_files"][0]["uri"]
        package = service.store.root / "project-exports" / sid / captured["package_id"]
        if mutation == "package_hash":
            uri = uri.replace(captured["sha256"], "0" * 64)
        elif mutation == "package_id":
            uri = uri.replace(captured["package_id"], "f" * 32)
        elif mutation == "project_id":
            uri = uri.replace(sid, "malformed")
        elif mutation == "file_id":
            uri = uri.rsplit("/", 1)[0] + "/part-absent-step"
        elif mutation == "bad_file_id":
            uri = uri.rsplit("/", 1)[0] + "/invalid"
        elif mutation == "zip":
            (package / "project.zip").write_bytes(b"changed")
        elif mutation == "missing_zip":
            (package / "project.zip").unlink()
        else:
            old = (package / "project.zip").read_bytes()
            buffer = io.BytesIO()
            with ZipFile(io.BytesIO(old)) as original, ZipFile(buffer, "w") as changed:
                for name in original.namelist():
                    if name != "assembly/part.step":
                        changed.writestr(name, original.read(name))
            raw = buffer.getvalue()
            descriptor = json.loads((package / "receipt.json").read_bytes())
            descriptor.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            (package / "project.zip").write_bytes(raw)
            (package / "receipt.json").write_text(json.dumps(descriptor), encoding="utf-8")
            uri = uri.replace(captured["sha256"], descriptor["sha256"])
        assert "error" in read(client, uri).json()
    assert counts(service) == {"jobs": 0, "revisions": 1}


def test_metadata_budget_failure_cleans_only_new_unpublished_capture(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    with TestClient(app(service), base_url="http://127.0.0.1:8765") as client:
        original = result(export(client, sid))["structuredContent"]
        monkeypatch.setattr(
            remote_mcp, "MAX_RESPONSE_BYTES", len(json.dumps(original).encode()) + 50
        )
        assert result(export(client, sid, include_individual_files=True))["isError"]
    directories = list((service.store.root / "project-exports" / sid).iterdir())
    assert [entry.name for entry in directories] == [original["package_id"]]
    assert (directories[0] / "project.zip").exists()


def test_opt_in_cancel_waits_for_producer_and_discards_unpublished_capture(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    entered, finish, discarded = threading.Event(), threading.Event(), threading.Event()
    create, discard = (
        project_delivery.create_project_export,
        project_delivery.discard_project_export,
    )

    def delayed(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return create(*args, **kwargs)

    def record_discard(*args, **kwargs):
        discard(*args, **kwargs)
        discarded.set()

    monkeypatch.setattr(project_delivery, "create_project_export", delayed)
    monkeypatch.setattr(project_delivery, "discard_project_export", record_discard)

    async def control():
        token = await MemberVerifier().verify_token("owner")
        monkeypatch.setattr(remote_mcp, "get_access_token", lambda: token)
        server = remote_mcp.create_server(
            service, token_verifier=MemberVerifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        request = asyncio.create_task(
            server.call_tool(
                "export_project",
                {
                    "project_id": sid,
                    "revision_id": RID,
                    "include_individual_files": True,
                },
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert not discarded.is_set()
        finish.set()
        assert await asyncio.to_thread(discarded.wait, 5)
        await asyncio.sleep(0.01)

    asyncio.run(control())
    assert list((service.store.root / "project-exports" / sid).iterdir()) == []
    assert counts(service) == {"jobs": 0, "revisions": 1}
