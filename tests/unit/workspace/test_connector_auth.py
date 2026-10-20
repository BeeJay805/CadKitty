from __future__ import annotations

import asyncio
import time

import pytest
from mcp.server.auth.provider import AccessToken
from mcp.types import ToolAnnotations
from tests.unit.workspace.test_remote_mcp import (
    ISSUER,
    RESOURCE,
    call,
    result,
)
from tests.unit.workspace.test_remote_mcp import (
    remote as remote_fixture,
)

from forge_mind.workspace import connector_auth
from forge_mind.workspace.connector_auth import (
    ConnectorAuthError,
    ScopedMCPServer,
    auth_challenge,
    require_owner,
    scoped_tool,
)

remote = remote_fixture


def token(**changes):
    values = {
        "token": "private-token",
        "client_id": "client",
        "subject": "alice",
        "scopes": ["cad:read", "cad:write"],
        "resource": RESOURCE,
        "expires_at": int(time.time()) + 300,
        "claims": {"iss": ISSUER},
    }
    return AccessToken(**(values | changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"resource": "https://other.test/mcp"},
        {"claims": {"iss": "https://other.test"}},
        {"subject": None},
        {"expires_at": int(time.time()) - 1},
        {"expires_at": None},
    ],
)
def test_auth_does_not_turn_identity_errors_into_scope_success(changes):
    with pytest.raises(ConnectorAuthError) as caught:
        require_owner(token(**changes), issuer=ISSUER, resource=RESOURCE, write=True)
    response = auth_challenge(caught.value, RESOURCE)
    assert response.is_error
    assert "invalid_token" in response.meta["mcp/www_authenticate"][0]
    assert "private-token" not in response.model_dump_json()


def test_read_token_gets_exact_write_scope_challenge():
    read = token(scopes=["cad:read"])
    assert require_owner(read, issuer=ISSUER, resource=RESOURCE) == "alice"
    with pytest.raises(ConnectorAuthError) as caught:
        require_owner(read, issuer=ISSUER, resource=RESOURCE, write=True)
    challenge = auth_challenge(caught.value, RESOURCE).meta["mcp/www_authenticate"][0]
    assert 'error="insufficient_scope"' in challenge
    assert 'scope="cad:read cad:write"' in challenge
    assert (
        'resource_metadata="http://127.0.0.1:8765/.well-known/oauth-protected-resource/mcp"'
        in challenge
    )


def test_remote_scope_upgrade_refuses_before_project_creation(remote):
    client, service = remote
    denied = result(call(client, "readonly", "create_project", {"title": "Clamp"}))
    assert denied["isError"]
    assert "insufficient_scope" in denied["_meta"]["mcp/www_authenticate"][0]
    assert service.store.list_sessions("readonly") == []
    assert not result(call(client, "alice", "create_project", {"title": "Clamp"})).get("isError")


def test_all_remote_tools_advertise_matching_exact_oauth_scopes(remote):
    client, _ = remote
    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer alice", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    tools = result(response)["tools"]
    assert tools
    for tool in tools:
        scopes = ["cad:read"] if tool["annotations"]["readOnlyHint"] else ["cad:read", "cad:write"]
        expected = [{"type": "oauth2", "scopes": scopes}]
        assert tool["securitySchemes"] == tool["_meta"]["securitySchemes"] == expected


def test_generic_non_auth_tool_failure_keeps_existing_shape(remote):
    client, _ = remote
    failed = result(call(client, "alice", "get_project", {"project_id": "a" * 32}))
    assert failed["isError"]
    assert "mcp/www_authenticate" not in failed.get("_meta", {})


@pytest.mark.parametrize("supplied", [False, True])
def test_scoped_dispatch_reads_live_resolver_without_weakening_policy(monkeypatch, supplied):
    current = None
    dispatches = []

    def live_token():
        return current

    # Construct before changing the default resolver. A captured default function
    # would miss the later request, while the supplied resolver must be authoritative.
    monkeypatch.setattr(connector_auth, "get_access_token", lambda: None)
    server = ScopedMCPServer(
        "request-auth-test",
        issuer=ISSUER,
        resource=RESOURCE,
        access_token_getter=live_token if supplied else None,
    )
    if not supplied:
        monkeypatch.setattr(connector_auth, "get_access_token", live_token)

    @scoped_tool(server, annotations=ToolAnnotations(read_only_hint=True))
    async def read_owned() -> str:
        dispatches.append("read")
        return "owned read"

    @scoped_tool(server, annotations=ToolAnnotations(read_only_hint=False))
    async def write_owned() -> str:
        dispatches.append("write")
        return "owned write"

    async def control():
        nonlocal current
        denied = await server.call_tool("read_owned", {})
        assert denied.is_error
        assert "invalid_token" in denied.meta["mcp/www_authenticate"][0]
        assert dispatches == []

        current = token(scopes=["cad:read"])
        assert not (await server.call_tool("read_owned", {})).is_error
        denied = await server.call_tool("write_owned", {})
        assert denied.is_error
        assert "insufficient_scope" in denied.meta["mcp/www_authenticate"][0]
        assert dispatches == ["read"]

        for changes in (
            {"claims": {"iss": "https://other.test"}},
            {"resource": "https://other.test/mcp"},
            {"expires_at": int(time.time()) - 1},
            {"subject": None},
        ):
            current = token(**changes)
            denied = await server.call_tool("write_owned", {})
            assert denied.is_error
            assert "invalid_token" in denied.meta["mcp/www_authenticate"][0]
            assert dispatches == ["read"]

        current = token()
        assert not (await server.call_tool("write_owned", {})).is_error
        assert dispatches == ["read", "write"]

    asyncio.run(control())
