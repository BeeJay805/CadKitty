"""Actual authenticated MCP transport with metadata/package fixtures, no CAD worker."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from tests.unit.workspace import test_project_export_remote as saved

from forge_mind.workspace import remote_mcp

packages = saved.packages
AID = "d" * 32


def asset_call(client, token, sid):
    return saved.rpc(
        client,
        token,
        "tools/call",
        {"name": "list_project_assets", "arguments": {"project_id": sid}},
    )


def test_assets_owned_metadata_only_missing_and_changed_bytes(packages, monkeypatch):
    service, sid, _, _, _, _, _ = packages
    path = service.store.root / "assets" / sid / (AID + ".step")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"ISO-10303-21; original")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    service.store.append_asset(
        sid,
        "alice",
        {
            "id": AID,
            "name": "owned housing",
            "sha256": original_hash,
            "private_path": "PRIVATE-HOST-PATH",
        },
    )
    service.store.append_asset(
        sid, "alice", {"id": "e" * 32, "name": "missing", "sha256": "f" * 64}
    )
    monkeypatch.setattr(
        service, "_load_revision", lambda *args: pytest.fail("Inventory must not parse a revision")
    )
    path.write_bytes(b"changed")
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        response = saved.result(asset_call(client, "readonly", sid))
        assert not response.get("isError")
        inventory = response["structuredContent"]
        assert inventory["project_id"] == sid
        assert inventory["content_verification"] == "stored_metadata_only"
        assert inventory["assets"][0] == {
            "asset_id": AID,
            "name": "owned housing",
            "sha256": original_hash,
            "size_bytes": 7,
            "file_status": "available",
            "measurement_status": "not_measured_by_inventory",
        }
        assert inventory["assets"][1]["file_status"] == "unavailable"
        assert inventory["assets"][1]["size_bytes"] is None
        assert "PRIVATE-HOST-PATH" not in json.dumps(response)
        assert saved.result(asset_call(client, "bob", sid))["isError"]
        empty = service.store.create("bob", "empty", None)["id"]
        assert saved.result(asset_call(client, "bob", empty))["structuredContent"]["assets"] == []
        assert saved.result(asset_call(client, "alice", sid))["structuredContent"] == inventory
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert not service.executor.typed_container_verified


@pytest.mark.parametrize("token", [None, "invalid", "no-read", "wrong-issuer"])
def test_assets_auth_denial(packages, token):
    service, sid, _, _, _, _, _ = packages
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        response = asset_call(client, token, sid)
        assert response.status_code in {401, 403} or saved.result(response)["isError"]


def test_assets_response_is_bounded(packages, monkeypatch):
    service, sid, _, _, _, _, _ = packages
    monkeypatch.setattr(remote_mcp, "MAX_RESPONSE_BYTES", 10)
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        response = saved.result(asset_call(client, "alice", sid))
        assert response["isError"]
        assert "transfer limit" in response["content"][0]["text"]


def enable_dependency_fixture(module, receipt):
    original_create = module.create_project_export

    def create(*args, include_stl=False, include_import_assets=False):
        result = original_create(*args, include_stl=include_stl)
        if include_import_assets:
            result["include_import_assets"] = True
            result["source_scope"] = {
                "representation": "saved_document_bytes",
                "independent_regeneration": "not_verified",
                "omitted_import_asset_ids": [],
                "provided_import_asset_ids": [AID],
                "input_map": {"file": "import-assets.json", "sha256": "f" * 64},
            }
            receipt.update({key: value for key, value in result.items() if key != "path"})
        return result

    module.create_project_export = create


def test_dependency_option_owned_readonly_and_restart_resource(packages):
    service, sid, module, receipt, _, _, _ = packages
    enable_dependency_fixture(module, receipt)
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        exported = saved.result(saved.call(client, "readonly", sid, include_import_assets=True))[
            "structuredContent"
        ]
        assert exported["include_import_assets"] is True
        assert exported["source_scope"]["provided_import_asset_ids"] == [AID]
        assert exported["source_scope"]["independent_regeneration"] == "not_verified"
        assert "PRIVATE-HOST-PATH" not in json.dumps(exported)
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        resource = saved.result(
            saved.rpc(client, "readonly", "resources/read", {"uri": exported["uri"]})
        )
        assert base64.b64decode(resource["contents"][0]["blob"], validate=True) == saved.ZIP
        assert (
            "error" in saved.rpc(client, "bob", "resources/read", {"uri": exported["uri"]}).json()
        )
    assert not service.executor.typed_container_verified


def test_default_and_explicit_false_keep_legacy_receipt(packages):
    service, sid, _, _, _, calls, _ = packages
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        implicit = saved.result(saved.call(client, "alice", sid))["structuredContent"]
        explicit = saved.result(saved.call(client, "alice", sid, include_import_assets=False))[
            "structuredContent"
        ]
        assert explicit == implicit
        assert "include_import_assets" not in explicit
    assert calls == [("create", "alice", False), ("create", "alice", False)]


@pytest.mark.parametrize("value", [1, 0, "true", "false", None])
def test_dependency_flag_strict_before_dispatch(packages, value):
    service, sid, _, _, _, calls, _ = packages
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        assert saved.result(saved.call(client, "alice", sid, include_import_assets=value))[
            "isError"
        ]
    assert calls == []


@pytest.mark.parametrize("reported", [None, False, 1, "true"])
def test_dependency_option_mismatch_cleans_unpublished_output(packages, reported):
    service, sid, module, _, directory, _, discarded = packages
    original_create = module.create_project_export

    def create(*args, include_import_assets=False, **kwargs):
        result = original_create(*args, **kwargs)
        if reported is not None:
            result["include_import_assets"] = reported
        return result

    module.create_project_export = create
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        assert saved.result(saved.call(client, "alice", sid, include_import_assets=True))["isError"]
    assert discarded.is_set() and not (directory / "package.zip").exists()


def test_current_advertisement_and_exact_options(packages):
    service, _, _, _, _, _, _ = packages
    with TestClient(saved.app(service), base_url="http://127.0.0.1:8765") as client:
        initialized = saved.result(
            saved.rpc(
                client,
                "alice",
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "import-test", "version": "1"},
                },
            )
        )
        assert initialized["serverInfo"]["version"] == "0.0.20"
        assert "list_project_assets" in initialized["instructions"]
        assert "include_import_assets=true" in initialized["instructions"]
        tools = saved.result(saved.rpc(client, "alice", "tools/list", {}))["tools"]
        inventory = next(tool for tool in tools if tool["name"] == "list_project_assets")
        export = next(tool for tool in tools if tool["name"] == "export_project")
        assert inventory["annotations"]["readOnlyHint"] is True
        assert export["annotations"]["readOnlyHint"] is True
        assert export["inputSchema"]["properties"]["include_import_assets"]["default"] is False
        assert export["inputSchema"]["properties"]["include_import_assets"]["type"] == "boolean"
