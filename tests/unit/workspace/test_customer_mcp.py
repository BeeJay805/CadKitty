"""Customer transport/admission controls. All model and native execution is disabled."""

from __future__ import annotations

import base64
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from mcp.server.auth.provider import AccessToken
from tests.unit.workspace.test_remote_mcp import call, result

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.customer_mcp import (
    INSTRUCTIONS,
    MAX_PHOTO_BASE64,
    CustomerMCPReference,
    CustomerWorkflow,
)
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.remote_mcp import create_app, create_server
from forge_mind.workspace.store import Store, WorkspaceError

ISSUER = "https://auth.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"


class Verifier:
    def __init__(self, issuer=ISSUER):
        self.issuer = issuer

    async def verify_token(self, token):
        if token not in {"alice", "bob", "owner", "readonly"}:
            return None
        return AccessToken(
            token=token,
            client_id="customer-test",
            subject=token,
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            expires_at=int(time.time()) + 600,
            resource=RESOURCE,
            claims={"iss": self.issuer},
        )


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "16")
    executor = Executor("docker")
    executor._typed_container_verified = True
    executor._customer_container_verified = True
    service = WorkspaceService(Store(tmp_path), executor)
    monkeypatch.setattr(service, "start", lambda: None)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, service


def start(client, token="alice", **changes):
    return result(
        call(
            client,
            token,
            "start_customer_project",
            {"request_id": "a" * 32, "prompt": "Make a plate", "printable": False, **changes},
        )
    )


def test_short_request_replay_recovery_and_cross_owner_isolation(remote):
    client, service = remote
    first = start(client)["structuredContent"]
    replay = start(client)["structuredContent"]
    recovered = result(call(client, "alice", "recover_customer_request", {"request_id": "a" * 32}))[
        "structuredContent"
    ]
    assert first["id"] == replay["id"] == recovered["id"]
    assert first["job_id"] == replay["job_id"] == recovered["job_id"]
    assert start(client, prompt="Changed request")["isError"]
    assert result(call(client, "bob", "recover_customer_request", {"request_id": "a" * 32}))[
        "isError"
    ]
    other = start(client, token="bob")["structuredContent"]
    assert other["id"] != first["id"]
    for name in (
        "get_customer_progress",
        "get_customer_proposal",
        "get_customer_files",
        "cancel_job",
    ):
        assert result(call(client, "bob", name, {"project_id": first["id"]}))["isError"]
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_website_and_connector_share_the_same_customer_request_and_project(remote):
    client, service = remote
    original = start(client)["structuredContent"]
    app = FastAPI()

    def owner(authorization: str = Header()):
        return authorization

    app.include_router(router(service.store, service, owner))
    with TestClient(app) as website:
        reply = website.post(
            "/workspace/customer-projects",
            headers={"Authorization": "alice"},
            json={"request_id": "a" * 32, "prompt": "Make a plate", "printable": False},
        )
        assert reply.status_code == 202
        assert reply.json()["id"] == original["id"]
        assert reply.json()["job_id"] == original["job_id"]


@pytest.mark.parametrize("missing", ["typed", "customer", "budget"])
def test_missing_real_qualification_never_creates_customer_work(remote, monkeypatch, missing):
    client, service = remote
    if missing == "typed":
        service.executor._typed_container_verified = False
    elif missing == "customer":
        service.executor._customer_container_verified = False
    else:
        monkeypatch.delenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS")
    assert start(client)["isError"]
    assert service.store.list_sessions("alice") == []


def test_write_scope_and_caller_identity_are_not_parameters(remote):
    client, service = remote
    assert start(client, token="readonly")["isError"]
    spoofed = start(client, owner="bob")["structuredContent"]
    assert service.store.session_record(spoofed["id"], "alice")["id"] == spoofed["id"]
    assert start(client, request_id="b" * 32, approval_required="true")["isError"]
    assert service.store.list_sessions("readonly") == []
    assert service.store.list_sessions("bob") == []


def test_discovery_has_minimal_versioned_examples_and_existing_shared_tools(remote):
    client, _ = remote
    reply = result(call(client, "alice", "customer_workflow"))["structuredContent"]
    assert reply["workflow_version"] == "customer_workflow_v1"
    assert reply["tools"]["answers"] == "resolve_intent"
    assert reply["tools"]["cancel"] == "cancel_job"
    assert reply["photo_limit_bytes"] == 512 * 1024
    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer alice", "Accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tools = {item["name"]: item for item in result(response)["tools"]}
    schema = tools["start_customer_project"]["inputSchema"]
    assert set(schema["required"]) == {"request_id", "prompt"}
    assert "owner" not in schema["properties"]
    assert "contract" not in schema["properties"]
    assert tools["start_customer_project"]["annotations"]["readOnlyHint"] is False
    assert tools["get_customer_files"]["annotations"]["readOnlyHint"] is True


def test_current_progress_does_not_trust_author_completion_text(remote):
    client, service = remote
    sid = start(client)["structuredContent"]["id"]
    service.store.update(
        sid,
        state="needs_review",
        delivery_guidance_record={"guidance": {"use_and_removal": "Fully safe and approved"}},
        contract_verification_report={"status": "pass"},
    )
    reply = result(call(client, "alice", "get_customer_progress", {"project_id": sid}))[
        "structuredContent"
    ]
    assert reply["physical_readiness"] == "unverified"
    assert reply["cad_conformance"]["status"] == "unverified"
    assert reply["guidance"]["literal_procedures"] is None
    assert reply["file_capture_status"] == "unavailable"
    files = result(call(client, "alice", "get_customer_files", {"project_id": sid}))[
        "structuredContent"
    ]
    assert files["status"] == "not_ready"
    assert files["project"] is None


def test_exact_command_approval_reuses_shared_guards_and_replay(tmp_path, monkeypatch):
    from tests.unit.workspace.test_customer_construction import staged

    # This transport control fixes the synthetic engine identity while other
    # authorized lanes edit production files. Real source drift stays guarded.
    monkeypatch.setattr("forge_mind.workspace.engine_identity.engine_fingerprint", lambda: "f" * 64)
    store, service, sid = staged(tmp_path)
    service.executor = Executor("docker")
    service.executor._typed_container_verified = True
    service.executor._customer_container_verified = True
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "16")
    monkeypatch.setattr(service, "start", lambda: None)
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        proposal = result(call(client, "owner", "get_customer_proposal", {"project_id": sid}))[
            "structuredContent"
        ]
        assert proposal["version"] == 2
        approval = {"proposal_hash": proposal["proposal_hash"], "basis": proposal["basis"]}
        assert result(
            call(
                client,
                "bob",
                "approve_customer_proposal",
                {"project_id": sid, "approval": approval},
            )
        )["isError"]
        assert result(call(client, "owner", "resume_customer_project", {"project_id": sid}))[
            "isError"
        ]
        bad = {**approval, "proposal_hash": "f" * 64}
        assert result(
            call(client, "owner", "approve_customer_proposal", {"project_id": sid, "approval": bad})
        )["isError"]
        first_reply = result(
            call(
                client,
                "owner",
                "approve_customer_proposal",
                {"project_id": sid, "approval": approval},
            )
        )
        assert not first_reply.get("isError"), first_reply
        first = first_reply["structuredContent"]
        replay = result(
            call(
                client,
                "owner",
                "approve_customer_proposal",
                {"project_id": sid, "approval": approval},
            )
        )["structuredContent"]
        assert first["job_id"] == replay["job_id"]
        job = store.claim()
        assert job["max_seconds"] == 70
        assert job["max_actions"] == 7
        assert store.revisions(sid) == []


def test_stale_pending_source_refuses_approval_and_keeps_history(tmp_path, monkeypatch):
    from tests.unit.workspace.test_customer_construction import staged

    store, service, sid = staged(tmp_path)
    workflow = CustomerWorkflow(service)
    proposal = workflow.proposal("owner", sid)
    before = deepcopy(proposal)
    store.update(sid, instructions=["Change the requested geometry"])
    assert workflow.proposal("owner", sid)["status"] == "stale"
    service.executor = Executor("docker")
    service.executor._typed_container_verified = True
    service.executor._customer_container_verified = True
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "600")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "16")
    from forge_mind.workspace.customer_approval import ApprovalRequest

    with pytest.raises(WorkspaceError, match="source"):
        workflow.approve(
            "owner",
            sid,
            ApprovalRequest(proposal_hash=before["proposal_hash"], basis=before["basis"]),
        )
    assert store.revisions(sid) == []


def test_photo_envelope_refuses_oversized_or_invalid_base64():
    with pytest.raises(ValueError):
        CustomerMCPReference(
            filename="reference.png",
            media_type="image/png",
            content_base64="a" * (MAX_PHOTO_BASE64 + 1),
        )
    with pytest.raises(ValueError):
        CustomerMCPReference(
            filename="reference.png", media_type="image/png", content_base64="not-base64!"
        )
    reference = CustomerMCPReference(
        filename="reference.png",
        media_type="image/png",
        content_base64=base64.b64encode(b"small source").decode(),
        known_dimension={"label": "width", "value": 30.0, "unit": "mm"},
    )
    assert reference.known_dimension.value == 30


def test_explicit_host_identity_matches_website_and_preserves_legacy_default(tmp_path, monkeypatch):
    from forge_mind.workspace.hosting_identity import canonical_principal

    issuer = "https://auth.example.test/auth/v1"
    store = Store(tmp_path)
    store.host_config = SimpleNamespace(issuer=issuer)
    service = WorkspaceService(store, Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: None)
    with pytest.raises(ValueError, match="same canonical issuer"):
        create_server(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    app = create_app(
        service, token_verifier=Verifier(issuer), issuer_url=issuer, resource_url=RESOURCE
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        result(call(client, "alice", "create_project", {"title": "Owned metadata"}))
    assert len(store.list_sessions(canonical_principal(issuer, "alice"))) == 1
    assert store.list_sessions("alice") == []
    assert INSTRUCTIONS.startswith("For an ordinary customer request")


def test_raw_answers_are_hash_bound_and_null_preserves_original_finite_budget(remote):
    from tests.unit.workspace.test_intent_attribution_service import PROMPT, attribution, target

    from forge_mind.workspace.intent_attribution_service import record_proposal

    client, service = remote
    sid = start(client, prompt=PROMPT)["structuredContent"]["id"]
    first = service.store.claim()
    record = record_proposal(
        service.store,
        sid,
        "alice",
        first["id"],
        attribution(),
        target(),
        PROMPT,
        service.store.edit_basis(sid, "alice"),
    )
    service.store.finish(first["id"], sid, "needs_input")
    current = result(call(client, "alice", "get_customer_progress", {"project_id": sid}))[
        "structuredContent"
    ]
    assert current["questions"]["draft_hash"] == record["draft_hash"]
    raw = "The width belongs to my sample. Keep that exact wording."
    resolution = {
        "expected_version": record["version"],
        "draft_hash": "f" * 64,
        "answers": [{"issue_id": "width_subject", "answer": raw}],
    }
    assert result(
        call(
            client,
            "alice",
            "resolve_intent",
            {"project_id": sid, "resolution": resolution, "max_seconds": None},
        )
    )["isError"]
    resolution["draft_hash"] = record["draft_hash"]
    response = result(
        call(
            client,
            "alice",
            "resolve_intent",
            {"project_id": sid, "resolution": resolution, "max_seconds": None},
        )
    )
    assert not response.get("isError")
    job = service.store.claim()
    assert job["max_seconds"] == first["max_seconds"] == 600
    assert job["max_actions"] == first["max_actions"] == 16
    assert job["intent_answer_binding"]["answers"][0]["answer"] == raw
    assert service.store.revisions(sid) == []
