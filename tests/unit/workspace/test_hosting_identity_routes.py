from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.connector_auth import require_owner
from forge_mind.workspace.connector_downloads import register_download_route
from forge_mind.workspace.hosting_budget import grant
from forge_mind.workspace.hosting_identity import canonical_principal
from forge_mind.workspace.hosting_store import HostingStore
from forge_mind.workspace.store import WorkspaceError


def test_website_auth_and_mcp_use_same_issuer_subject_owner(tmp_path, monkeypatch):
    from tests.unit.workspace.test_hosting_store import policy

    config = policy(tmp_path)
    monkeypatch.setenv("FORGE_WORKSPACE_DIR", str(config.root))
    monkeypatch.setenv("FORGE_WORKSPACE_EXECUTOR", "docker")
    monkeypatch.setenv("SUPABASE_URL", "https://auth.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "fixture-public-key")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "10")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "4")
    from forge_mind.server import workspace_app

    subject = ["alice"]
    monkeypatch.setattr(
        workspace_app.httpx,
        "get",
        lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"id": subject[0]}),
    )
    store = HostingStore(config)
    owner = canonical_principal(config.issuer, "alice")
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        grant(con, owner, "operator", 100)
    service = WorkspaceService(
        store,
        SimpleNamespace(
            mode="docker", typed_container_verified=True, customer_container_verified=True
        ),
    )
    monkeypatch.setattr(service, "start", lambda: None)
    client = TestClient(workspace_app.create_app(local=False, service=service))
    response = client.post(
        "/workspace/customer-projects",
        headers={"Authorization": "Bearer fixture"},
        json={"request_id": "a" * 32, "prompt": "A plate", "max_seconds": 10, "max_actions": 4},
    )
    assert response.status_code == 202, response.text
    sid = response.json()["id"]
    token = AccessToken(
        token="fixture",
        client_id="fixture",
        subject="alice",
        resource="https://cad.example/mcp",
        scopes=["cad:read"],
        expires_at=int(time.time()) + 60,
        claims={"iss": config.issuer},
    )
    validated_subject = require_owner(token, issuer=config.issuer, resource=token.resource)
    assert (
        store.session_record(sid, canonical_principal(config.issuer, validated_subject))["id"]
        == sid
    )
    with pytest.raises(WorkspaceError):
        store.session_record(sid, "alice")
    subject[0] = "bob"
    assert (
        client.get(
            "/workspace/sessions/" + sid, headers={"Authorization": "Bearer fixture"}
        ).status_code
        == 404
    )
    assert client.get("/workspace/sessions/" + sid).status_code == 401
    monkeypatch.setenv("SUPABASE_URL", "https://other.example")
    assert (
        client.get("/workspace/sessions", headers={"Authorization": "Bearer fixture"}).status_code
        == 503
    )


def test_authenticated_downloads_apply_mapping_only_after_validated_issuer():
    from tests.unit.workspace.test_connector_downloads import (
        ORIGIN,
        PACKAGE,
        PROJECT,
        RESOURCE,
        fixture_package,
        token_from,
    )

    downloads, state, checksum = fixture_package.__wrapped__()
    issuer = "https://auth.example/auth/v1"
    state["owner"] = canonical_principal(issuer, "alice")
    links = downloads.mint(
        owner=state["owner"],
        project_id=PROJECT,
        package_id=PACKAGE,
        package_hash=checksum,
        authorization_expires_at=1000,
    )
    capability = token_from(links["package"])

    class Verifier:
        async def verify_token(self, value):
            return AccessToken(
                token=value,
                client_id="fixture",
                subject="bob" if value == "foreign" else "alice",
                resource=RESOURCE,
                scopes=["cad:read"],
                expires_at=int(time.time()) + 60,
                claims={
                    "iss": "https://wrong.example/auth/v1" if value == "wrong-issuer" else issuer
                },
            )

    app = FastAPI()
    register_download_route(
        app,
        downloads,
        verifier=Verifier(),
        issuer=issuer,
        resource=RESOURCE,
        owner_transform=lambda subject: canonical_principal(issuer, subject),
    )
    client = TestClient(app, base_url=ORIGIN)
    assert (
        client.get(
            "/downloads/" + capability, headers={"Authorization": "Bearer owner"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/downloads/" + capability, headers={"Authorization": "Bearer foreign"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/downloads/" + capability, headers={"Authorization": "Bearer wrong-issuer"}
        ).status_code
        == 404
    )
    assert client.get("/downloads/" + capability).content == state["content"]
