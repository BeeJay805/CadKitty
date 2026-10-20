"""Guidance updates use owned metadata paths, never generation workers."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client
from mcp.server.auth.provider import AccessToken
from tests.unit.workspace.test_flat_project_delivery_transport import flat_saved_project
from tests.unit.workspace.test_project_delivery import RID
from tests.unit.workspace.test_project_export_remote import ISSUER, RESOURCE, result, rpc
from tests.unit.workspace.test_project_members_remote import counts

from forge_mind.workspace import delivery_guidance, mcp_server, remote_mcp
from forge_mind.workspace.project_access import project_access_router


def instructions(context):
    return {
        "revision_id": context["basis"]["revision_id"],
        "basis_hash": context["basis_hash"],
        "reviewed_current_facts": True,
        "guidance": {
            "print": "Printing requires a qualified material and settings.",
            "installation": "Check the mounting surface before installation.",
            "use_and_removal": "Test removal access before use.",
            "unresolved": "Physical fit and strength remain unverified.",
        },
    }


def test_http_and_actual_local_sdk_refresh_without_cad(tmp_path, monkeypatch):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        delivery_guidance, "engine_fingerprint", lambda: "fixture-measurement-engine"
    )
    before = counts(service)
    application = FastAPI()
    application.include_router(project_access_router(service.store, service, lambda: "owner"))
    with TestClient(application) as client:
        context = client.get(f"/workspace/sessions/{sid}/guidance").json()
        update = instructions(context)
        assert (
            client.post(f"/workspace/sessions/{sid}/guidance", json=update).json()["status"]
            == "current"
        )
        stale = {**update, "basis_hash": "0" * 64}
        assert client.post(f"/workspace/sessions/{sid}/guidance", json=stale).status_code == 409

    def backend(request):
        with TestClient(application) as client:
            response = client.request(
                request.method,
                request.url.raw_path.decode("ascii"),
                content=request.content,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
            )
        return httpx.Response(response.status_code, content=response.content)

    async def scenario():
        adapter = mcp_server.WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "downloads", transport=httpx.MockTransport(backend)
        )
        async with Client(mcp_server.create_server(adapter)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["get_guidance"].annotations.read_only_hint is True
            assert tools["update_guidance"].annotations.read_only_hint is False
            context = (
                await client.call_tool("get_guidance", {"project_id": sid})
            ).structured_content
            refreshed = await client.call_tool(
                "update_guidance", {"project_id": sid, "update": instructions(context)}
            )
            assert not refreshed.is_error, refreshed.content
            assert refreshed.structured_content["status"] == "current"

    asyncio.run(scenario())
    assert counts(service) == before
    assert service.store.get(sid, "owner").get("accepted_revision") is None


def test_http_explicit_pending_revision_does_not_change_selected_default(tmp_path, monkeypatch):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        delivery_guidance, "engine_fingerprint", lambda: "fixture-measurement-engine"
    )
    candidate = service.store.revision(sid, RID)
    proposal = {
        "revision_id": RID,
        "based_on": None,
        "eligible": False,
        "input_version": 0,
        "contract": candidate["contract"],
    }
    service.store.update(sid, selected_revision=None, accepted_revision=None, proposal=proposal)
    before = counts(service)
    app = FastAPI()
    app.include_router(project_access_router(service.store, service, lambda: "owner"))
    with TestClient(app) as client:
        assert client.get(f"/workspace/sessions/{sid}/guidance").status_code == 409
        response = client.get(f"/workspace/sessions/{sid}/guidance", params={"revision_id": RID})
        assert response.status_code == 200
        context = response.json()
        assert context["target_kind"] == "pending_candidate"
        assert context["basis"]["revision_id"] == context["design_revision"] == RID
        assert context["selected_revision"] is None
        assert context["accepted_revision"] is None
        saved = client.post(f"/workspace/sessions/{sid}/guidance", json=instructions(context))
        assert saved.status_code == 200
        assert saved.json()["status"] == "current"
        assert client.get(f"/workspace/sessions/{sid}/guidance").status_code == 409
        for revision_id in ("", "not-a-revision", "0" * 31, "0" * 33):
            refused = client.get(
                f"/workspace/sessions/{sid}/guidance", params={"revision_id": revision_id}
            )
            assert refused.status_code == 422
    after = service.store.get(sid, "owner")
    assert after["selected_revision"] is None
    assert after["accepted_revision"] is None
    assert after["proposal"] == proposal
    assert counts(service) == before


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_actual_sdk_explicit_guidance_refuses_invalid_revision_before_read(
    tmp_path, monkeypatch, surface
):
    from tests.unit.workspace.test_delivery_guidance_sdk import connection

    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    before = counts(service)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            for revision_id in ("", "not-a-revision", "0" * 31):
                response = await sdk.call_tool(
                    "get_guidance", {"project_id": sid, "revision_id": revision_id}
                )
                assert response.is_error

    asyncio.run(scenario())
    assert counts(service) == before
    assert service.store.get(sid, "owner").get("accepted_revision") is None


class GuidanceVerifier:
    async def verify_token(self, token):
        if token not in {"owner", "foreign", "readonly"}:
            return None
        return AccessToken(
            token=token,
            subject="owner" if token == "readonly" else token,
            client_id="guidance-controls",
            scopes=["cad:read"] + ([] if token == "readonly" else ["cad:write"]),
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": ISSUER},
        )


def test_authenticated_remote_sdk_preserves_scope_and_owner(tmp_path, monkeypatch):
    service, sid, _, _ = flat_saved_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        delivery_guidance, "engine_fingerprint", lambda: "fixture-measurement-engine"
    )
    before = counts(service)
    application = remote_mcp.create_app(
        service, token_verifier=GuidanceVerifier(), issuer_url=ISSUER, resource_url=RESOURCE
    )
    with TestClient(application, base_url=RESOURCE.removesuffix("/mcp")) as client:
        context = result(
            rpc(
                client,
                "owner",
                "tools/call",
                {"name": "get_guidance", "arguments": {"project_id": sid}},
            )
        )["structuredContent"]
        arguments = {"project_id": sid, "update": instructions(context)}
        for token in ("foreign", "readonly"):
            refused = result(
                rpc(
                    client, token, "tools/call", {"name": "update_guidance", "arguments": arguments}
                )
            )
            assert refused["isError"] is True
        saved = result(
            rpc(client, "owner", "tools/call", {"name": "update_guidance", "arguments": arguments})
        )
        assert not saved.get("isError")
        assert saved["structuredContent"]["status"] == "current"
        arguments["update"]["basis_hash"] = "0" * 64
        assert result(
            rpc(client, "owner", "tools/call", {"name": "update_guidance", "arguments": arguments})
        )["isError"]
    assert counts(service) == before
