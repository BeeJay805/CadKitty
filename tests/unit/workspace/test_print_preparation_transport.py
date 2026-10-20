"""Transport and instruction freshness controls; fixture bytes are not CAD proof."""

from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client
from tests.unit.workspace.test_flat_project_delivery_transport import flat_saved_project
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_project_delivery import RID
from tests.unit.workspace.test_project_members_remote import counts

from forge_mind.workspace import delivery_guidance, mcp_server, print_setup_service
from forge_mind.workspace.delivery_guidance import GuidanceUpdate
from forge_mind.workspace.document import digest
from forge_mind.workspace.product_api import product_router


def test_saved_print_setup_invalidates_guidance_without_geometry_changes(tmp_path, monkeypatch):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        delivery_guidance, "engine_fingerprint", lambda: "fixture-measurement-engine"
    )
    before = counts(service)
    context = delivery_guidance.guidance_context(service, sid, "owner", RID)
    assert "print_setup_record_hash" not in context["basis"]
    delivery_guidance.update_guidance(
        service, sid, "owner", GuidanceUpdate.model_validate(instructions(context))
    )
    assert delivery_guidance.guidance_context(service, sid, "owner", RID)["status"] == "current"
    receipt = {
        "files": {"print/example.stl": {"part_id": "example", "quantity": 2, "sha256": "a" * 64}},
        "selected_candidates": {"example": "b" * 64},
        "package": {"sha256": "c" * 64},
        "result": {
            "parts": {
                "example": {
                    "recommended_candidate_hash": "b" * 64,
                    "candidates": [
                        {
                            "candidate_hash": "b" * 64,
                            "transform": "Fixture pose, not measured geometry",
                        }
                    ],
                }
            }
        },
    }
    setup = {"status": "current", "record": receipt, "record_hash": digest(receipt)}
    monkeypatch.setattr(print_setup_service, "_locked_print_setup", lambda *args: deepcopy(setup))
    service.store.update(sid, print_setup_records={RID: {"fixture": True}})
    changed = delivery_guidance.guidance_context(service, sid, "owner", RID)
    assert changed["status"] == "stale"
    assert changed["basis_hash"] != context["basis_hash"]
    assert changed["basis"]["print_setup_record_hash"] == digest(receipt)
    assert changed["facts"]["print_setup"]["files"] == receipt["files"]
    assert (
        changed["facts"]["print_setup"]["selected_poses"]["example"]["candidate_hash"] == "b" * 64
    )
    refreshed = delivery_guidance.update_guidance(
        service, sid, "owner", GuidanceUpdate.model_validate(instructions(changed))
    )
    assert refreshed["status"] == "current"
    setup["status"] = "stale"
    stale = delivery_guidance.guidance_context(service, sid, "owner", RID)
    with pytest.raises(ValueError, match="print setup is stale"):
        delivery_guidance.update_guidance(
            service, sid, "owner", GuidanceUpdate.model_validate(instructions(stale))
        )
    assert counts(service) == before


@pytest.mark.parametrize("corrupt", [False, True])
def test_actual_local_sdk_print_receipt_and_bytes_agree(tmp_path, monkeypatch, corrupt):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    data = b"Owned print ZIP byte fixture, not native geometry proof"
    package = "1" * 32
    checksum = hashlib.sha256(data).hexdigest()
    receipt = {
        "project_id": sid,
        "revision_id": RID,
        "package_id": package,
        "sha256": checksum,
        "bytes": len(data),
        "status": "prepared",
        "manufacturing_ready": False,
        "physical_readiness": "unverified",
    }
    monkeypatch.setattr(print_setup_service, "prepare_print_files", lambda *args: receipt)
    monkeypatch.setattr(
        print_setup_service,
        "read_print_export",
        lambda *args: (data + b"corrupt" if corrupt else data, receipt),
    )
    app = FastAPI()
    app.include_router(product_router(service.store, service, lambda: "owner"))

    def backend(request):
        with TestClient(app) as client:
            response = client.request(
                request.method,
                request.url.raw_path.decode("ascii"),
                content=request.content,
                headers={"Content-Type": "application/json"},
            )
        return httpx.Response(response.status_code, content=response.content)

    async def scenario():
        adapter = mcp_server.WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "downloads", transport=httpx.MockTransport(backend)
        )
        async with Client(mcp_server.create_server(adapter)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["inspect_print_setup"].annotations.read_only_hint is True
            assert tools["prepare_print_files"].annotations.read_only_hint is False
            assert tools["prepare_physical_pilot"].annotations.read_only_hint is True
            assert tools["record_physical_pilot"].annotations.read_only_hint is False
            result = await client.call_tool(
                "prepare_print_files", {"project_id": sid, "revision_id": RID}
            )
            if corrupt:
                assert result.is_error
                assert not list((tmp_path / "downloads").glob("*"))
            else:
                assert not result.is_error, result.content
                from pathlib import Path

                output = result.structured_content
                assert output is not None
                assert Path(output["path"]).read_bytes() == data
                assert output["sha256"] == checksum
                repeated = await client.call_tool(
                    "prepare_print_files", {"project_id": sid, "revision_id": RID}
                )
                assert not repeated.is_error, repeated.content
                again = repeated.structured_content
                assert again is not None
                assert again["path"] != output["path"]
                assert Path(again["path"]).read_bytes() == data
                assert again["sha256"] == checksum

    asyncio.run(scenario())


def test_typed_print_adapter_refuses_unqualified_or_unlisted_workers(tmp_path, monkeypatch):
    from forge_mind.workspace.executor import Executor

    executor = Executor()
    monkeypatch.setattr(
        executor,
        "_run_container",
        lambda *args, **kwargs: pytest.fail("Unqualified worker launched"),
    )
    with pytest.raises(ValueError, match="Unsupported"):
        executor.run_typed_tool(
            tmp_path, {}, module="arbitrary.module", timeout=1, cancelled=lambda: False
        )
    with pytest.raises(ValueError, match="qualified isolated"):
        executor.run_typed_tool(
            tmp_path,
            {},
            module="forge_mind.workspace.print_setup",
            timeout=1,
            cancelled=lambda: False,
        )


def test_remote_print_links_deliver_exact_all_made_bytes(tmp_path, monkeypatch):
    import io
    import time
    from zipfile import ZipFile

    from mcp.server.auth.provider import AccessToken
    from tests.unit.workspace.test_connector_downloads import ISSUER, ORIGIN, RESOURCE
    from tests.unit.workspace.test_print_setup_service import project
    from tests.unit.workspace.test_project_export_remote import result, rpc

    from forge_mind.workspace.connector_downloads import ConnectorDownloads, print_package_reader
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.remote_mcp import create_app

    service, sid, _, worker = project(tmp_path, monkeypatch)
    # Isolate transport behavior. The fixture worker uses a native-plane stub.
    service.executor = Executor("docker")
    monkeypatch.setattr(service.executor, "run_typed_tool", worker)
    receipt = print_setup_service.prepare_print_files(service, sid, "owner", RID)

    class Verifier:
        async def verify_token(self, token):
            if token not in {"owner", "readonly", "foreign"}:
                return None
            return AccessToken(
                token=token,
                subject="owner" if token == "readonly" else token,
                client_id="print-sdk-fixture",
                scopes=["cad:read"] + ([] if token == "readonly" else ["cad:write"]),
                expires_at=int(time.time()) + 600,
                resource=RESOURCE,
                claims={"iss": ISSUER},
            )

    downloads = ConnectorDownloads(origin=ORIGIN, readers={"print": print_package_reader(service)})
    app = create_app(
        service,
        token_verifier=Verifier(),
        issuer_url=ISSUER,
        resource_url=RESOURCE,
        downloads=downloads,
    )
    with TestClient(app, base_url=ORIGIN) as client:
        arguments = {"project_id": sid, "revision_id": RID}
        denied = result(
            rpc(
                client,
                "readonly",
                "tools/call",
                {"name": "prepare_print_files", "arguments": arguments},
            )
        )
        assert denied["isError"]
        assert "insufficient_scope" in denied["_meta"]["mcp/www_authenticate"][0]
        exported = result(
            rpc(
                client,
                "owner",
                "tools/call",
                {"name": "prepare_print_files", "arguments": arguments},
            )
        )
        assert not exported.get("isError"), exported
        metadata = exported["structuredContent"]
        assert metadata["sha256"] == receipt["sha256"]
        package = client.get(metadata["download_url"])
        assert package.status_code == 200
        assert hashlib.sha256(package.content).hexdigest() == receipt["sha256"]
        with ZipFile(io.BytesIO(package.content)) as archive:
            assert {row["archive_path"] for row in metadata["individual_files"]} == set(
                receipt["files"]
            )
            for member in metadata["individual_files"]:
                received = client.get(member["download_url"])
                assert received.status_code == 200
                assert received.content == archive.read(member["archive_path"])
        assert (
            client.get(
                metadata["download_url"], headers={"Authorization": "Bearer foreign"}
            ).status_code
            == 404
        )
        downloads.revoke_package("owner", sid, receipt["package_id"])
        assert client.get(metadata["download_url"]).status_code == 404
