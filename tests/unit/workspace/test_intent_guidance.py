"""SDK exposure controls only; these do not prove model use or improved fit."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest
from mcp import Client
from mcp.types import InitializeResult
from tests.unit.workspace.test_remote_mcp import remote, result  # noqa: F401

from forge_mind.workspace.intent_guidance import FITTED_PART_INTENT_GUIDANCE
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.parametric_guidance import PARAMETRIC_DESIGN_GUIDANCE


def assert_instructions(instructions: str) -> None:
    assert instructions.count(FITTED_PART_INTENT_GUIDANCE) == 1
    assert instructions.count(PARAMETRIC_DESIGN_GUIDANCE) == 1
    assert instructions.endswith(PARAMETRIC_DESIGN_GUIDANCE + "\n\n" + FITTED_PART_INTENT_GUIDANCE)


def test_guidance_matches_exact_paired_host_paragraph() -> None:
    # Binds current canonical host advice; historical dialogue inputs remain immutable.
    paired = Path(__file__).resolve().parents[3] / "docs/host-instructions/fitted-part.txt"
    expected = (FITTED_PART_INTENT_GUIDANCE + "\n").encode("utf-8")
    assert paired.read_bytes() == expected
    assert hashlib.sha256(expected).hexdigest() == (
        "b177b5cf47fa0c40da730518988379b5b7e3bc78096ba4cdd0b98ffcadc4b2df"
    )


def test_local_sdk_initialize_exposes_once_without_backend_work(tmp_path: Path) -> None:
    def no_backend(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Initialize must not call the workspace or launch generation")

    async def scenario() -> None:
        server = create_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(no_backend)
            )
        )
        async with Client(server, mode="legacy") as client:
            assert_instructions(client.instructions)
            assert "Preserve failed and unverified evidence" in client.instructions

    asyncio.run(scenario())


def test_remote_sdk_initialize_exposes_once_and_keeps_staging_scope(
    request: pytest.FixtureRequest,
) -> None:
    client, service = request.getfixturevalue("remote")
    reply = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer alice",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "intent-guidance-test", "version": "1"},
            },
        },
    )
    initialized = InitializeResult.model_validate(result(reply))
    assert initialized.instructions is not None
    assert_instructions(initialized.instructions)
    assert "qualified isolated worker" in initialized.instructions
    assert "not a published ChatGPT connector" in initialized.instructions
    assert service.store.list_sessions("alice") == []
