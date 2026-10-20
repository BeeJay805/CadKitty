"""Exact capability reads over fixture bytes; no CAD, provider or listener jobs."""

from __future__ import annotations

import hashlib
import io
import time
from urllib.parse import urlsplit
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from tests.unit.workspace.test_project_delivery import RID, saved_project
from tests.unit.workspace.test_project_export_remote import ISSUER, result, rpc

from forge_mind.workspace.connector_downloads import ConnectorDownloads, project_package_reader
from forge_mind.workspace.remote_mcp import create_app
from forge_mind.workspace.store import WorkspaceError

ORIGIN = "https://cad.example.test"
RESOURCE = ORIGIN + "/mcp"
PROJECT, PACKAGE = "a" * 32, "b" * 32


@pytest.fixture
def fixture_package():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("make/part.stl", b"exact STL fixture")
    content = buffer.getvalue()
    checksum = hashlib.sha256(content).hexdigest()
    receipt = {
        "project_id": PROJECT,
        "package_id": PACKAGE,
        "sha256": checksum,
        "bytes": len(content),
    }
    members = [
        {
            "file_id": "part-stl",
            "archive_path": "make/part.stl",
            "sha256": hashlib.sha256(b"exact STL fixture").hexdigest(),
            "bytes": len(b"exact STL fixture"),
        }
    ]
    state = {
        "content": content,
        "receipt": receipt,
        "members": members,
        "owner": "alice",
        "now": 100.0,
    }

    def read(owner, project, package, digest):
        if (
            owner != state["owner"]
            or project != PROJECT
            or package != PACKAGE
            or digest != checksum
        ):
            raise WorkspaceError("Owner or source unavailable")
        return state["content"], state["receipt"], state["members"]

    adapter = ConnectorDownloads(
        origin=ORIGIN, readers={"project": read}, clock=lambda: state["now"]
    )
    return adapter, state, checksum


def mint(adapter, checksum, **changes):
    values = {
        "owner": "alice",
        "project_id": PROJECT,
        "package_id": PACKAGE,
        "package_hash": checksum,
        "authorization_expires_at": 1000.0,
    }
    return adapter.mint(**(values | changes))


def token_from(row):
    return urlsplit(row["download_url"]).path.rsplit("/", 1)[1]


def test_capabilities_deliver_only_bound_zip_and_exact_member(fixture_package):
    adapter, state, checksum = fixture_package
    links = mint(adapter, checksum)
    package = token_from(links["package"])
    member = token_from(links["individual_files"][0])
    assert len(package) == 43 and package != member
    assert package not in repr(adapter._grants)
    assert adapter.read(package)[0] == state["content"]
    assert adapter.read(member)[0] == b"exact STL fixture"
    assert adapter.read(member, authenticated_owner="alice")[0] == b"exact STL fixture"
    with pytest.raises(WorkspaceError):
        adapter.read(member, authenticated_owner="bob")
    with pytest.raises(WorkspaceError):
        mint(adapter, checksum, owner="bob")


@pytest.mark.parametrize(
    "mode",
    [
        "expired",
        "revoke",
        "owner_revoke",
        "package_revoke",
        "ownership_changed",
        "zip_changed",
        "member_removed",
        "member_hash_changed",
    ],
)
def test_existing_grants_refuse_revoked_or_changed_authority(fixture_package, mode):
    adapter, state, checksum = fixture_package
    links = mint(adapter, checksum)
    token = token_from(links["individual_files"][0])
    if mode == "expired":
        state["now"] = 400.0
    elif mode == "revoke":
        adapter.revoke(token)
    elif mode == "owner_revoke":
        adapter.revoke_owner("alice")
    elif mode == "package_revoke":
        adapter.revoke_package("alice", PROJECT, PACKAGE)
    elif mode == "ownership_changed":
        state["owner"] = "bob"
    elif mode == "zip_changed":
        state["content"] += b"changed"
    elif mode == "member_removed":
        state["members"] = []
    else:
        state["members"][0]["sha256"] = "0" * 64
    with pytest.raises(WorkspaceError):
        adapter.read(token)


def test_capability_lifetime_cannot_outlive_original_authorization(fixture_package):
    adapter, state, checksum = fixture_package
    links = mint(adapter, checksum, authorization_expires_at=120.0)
    assert links["package"]["expires_at"] == 120.0
    state["now"] = 120.0
    with pytest.raises(WorkspaceError):
        adapter.read(token_from(links["package"]))


@pytest.mark.parametrize(
    "path", ["../outside.stl", "/part.stl", "make/../part.stl", "make\\part.stl", "a\r\nheader.stl"]
)
def test_arbitrary_paths_cannot_receive_grants(fixture_package, path):
    adapter, state, checksum = fixture_package
    state["members"][0]["archive_path"] = path
    with pytest.raises(WorkspaceError):
        mint(adapter, checksum)
    assert not adapter._grants


def test_bounded_grants_refuse_without_evicting_active_links(fixture_package):
    adapter, _, checksum = fixture_package
    adapter.capacity = 2
    original = mint(adapter, checksum)
    with pytest.raises(WorkspaceError, match="capacity"):
        mint(adapter, checksum)
    assert adapter.read(token_from(original["package"]))[0]


def test_revocation_during_read_refuses_completed_bytes(fixture_package):
    adapter, _, checksum = fixture_package
    links = mint(adapter, checksum)
    token = token_from(links["package"])
    original = adapter.readers["project"]

    def revoked(*args):
        result = original(*args)
        adapter.revoke(token)
        return result

    adapter.readers["project"] = revoked
    with pytest.raises(WorkspaceError):
        adapter.read(token)


class Verifier:
    async def verify_token(self, token):
        if token not in {"owner", "foreign", "wrong-issuer", "expired"}:
            return None
        return AccessToken(
            token=token,
            client_id="fixture-client",
            subject=token,
            scopes=["cad:read"],
            resource=RESOURCE,
            expires_at=int(time.time()) + (-1 if token == "expired" else 600),
            claims={"iss": "https://wrong.test" if token == "wrong-issuer" else ISSUER},
        )


def test_opt_in_remote_export_returns_exact_browser_downloads(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    downloads = ConnectorDownloads(
        origin=ORIGIN, readers={"project": project_package_reader(service)}
    )
    app = create_app(
        service,
        token_verifier=Verifier(),
        issuer_url=ISSUER,
        resource_url=RESOURCE,
        downloads=downloads,
    )
    before = service.store.session_record(sid, "owner")
    with TestClient(app, base_url=ORIGIN) as client:
        exported = result(
            rpc(
                client,
                "owner",
                "tools/call",
                {
                    "name": "export_project",
                    "arguments": {
                        "project_id": sid,
                        "revision_id": RID,
                        "include_stl": True,
                        "include_individual_files": True,
                        "preview": "none",
                    },
                },
            )
        )
        assert not exported.get("isError"), exported
        metadata = exported["structuredContent"]
        assert metadata["uri"].startswith("cadkitty://")
        package = client.get(metadata["download_url"])
        assert package.status_code == 200
        assert hashlib.sha256(package.content).hexdigest() == metadata["sha256"]
        assert package.headers["cache-control"] == "no-store"
        assert package.headers["referrer-policy"] == "no-referrer"
        assert package.headers["content-disposition"] == 'attachment; filename="project.zip"'
        with ZipFile(io.BytesIO(package.content)) as archive:
            assert len(metadata["individual_files"]) == 5
            for member in metadata["individual_files"]:
                received = client.get(member["download_url"])
                assert received.status_code == 200
                assert received.content == archive.read(member["archive_path"])
        for other in ("foreign", "wrong-issuer", "expired", "invalid"):
            rejected = client.get(
                metadata["download_url"], headers={"Authorization": "Bearer " + other}
            )
            assert rejected.status_code == 404
            assert metadata["download_url"] not in rejected.text
        discoveries = client.get("/.well-known/oauth-protected-resource/mcp")
        assert discoveries.status_code == 200
        downloads.revoke_owner("owner")
        assert client.get(metadata["download_url"]).status_code == 404
    assert service.store.session_record(sid, "owner") == before


def test_download_route_is_absent_by_default(tmp_path, monkeypatch):
    service, _, _ = saved_project(tmp_path, monkeypatch)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url=ORIGIN) as client:
        assert client.get("/downloads/" + "a" * 43).status_code == 404
