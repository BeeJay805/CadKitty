"""Owned immutable file and stale-selection controls, without CAD or model execution."""

from __future__ import annotations

import hashlib

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_mcp import CustomerWorkflow
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def captured(tmp_path, monkeypatch):
    store = Store(tmp_path)
    session = store.create_customer_project(
        "alice", {"request_id": "a" * 32, "prompt": "A plate", "printable": False}
    )
    sid = session["id"]
    rid = "b" * 32
    raw = b"exact immutable package bytes"
    receipt = {
        "project_id": sid,
        "revision_id": rid,
        "package_id": "c" * 32,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    binding = {"revision_id": rid, "engine_fingerprint": "d" * 64, "source_fingerprint": "e" * 64}
    store.update(
        sid,
        selected_revision=rid,
        automatic_delivery={
            "status": "delivered",
            "readback_verified": True,
            "job_id": session["job_id"],
            "receipt": receipt,
            **binding,
        },
    )
    service = WorkspaceService(store, Executor("docker"))
    monkeypatch.setattr(
        service, "start", lambda: pytest.fail("Reading saved bytes cannot start a worker")
    )
    monkeypatch.setattr(
        "forge_mind.workspace.customer_mcp.current_delivery_binding", lambda *_: dict(binding)
    )
    reads = []

    def read(_service, project, owner, package, digest):
        assert owner == "alice" and project == sid
        assert package == receipt["package_id"] and digest == receipt["sha256"]
        reads.append(owner)
        return raw, dict(receipt)

    monkeypatch.setattr("forge_mind.workspace.customer_mcp.read_project_export", read)
    return CustomerWorkflow(service), store, sid, receipt, binding, reads


def test_saved_file_reader_reuses_exact_immutable_identity_without_accepting(captured):
    workflow, store, sid, receipt, _binding, reads = captured
    before = store.session_record(sid, "alice")
    result = workflow.files("alice", sid)
    assert result["status"] == "captured_current"
    assert result["project"]["sha256"] == receipt["sha256"]
    assert result["project"]["resource_uri"].endswith(receipt["sha256"] + ".zip")
    assert result["physical_readiness"] == "unverified"
    assert store.session_record(sid, "alice") == before
    assert before.get("accepted_revision") is None
    assert reads == ["alice"]


def test_other_owner_never_reaches_the_package_reader(captured):
    workflow, _store, sid, _receipt, _binding, reads = captured
    with pytest.raises(WorkspaceError, match="not found"):
        workflow.files("bob", sid)
    assert reads == []


def test_changed_source_refuses_current_package_before_read(captured):
    workflow, _store, sid, _receipt, binding, reads = captured
    binding["source_fingerprint"] = "f" * 64
    result = workflow.files("alice", sid)
    assert result["status"] == "stale"
    assert result["project"] is None
    assert reads == []


def test_changed_inputs_during_readback_remove_current_links(captured, monkeypatch):
    workflow, _store, sid, _receipt, binding, _reads = captured
    saved = dict(binding)
    calls = iter([saved, {**saved, "source_fingerprint": "f" * 64}])
    monkeypatch.setattr(
        "forge_mind.workspace.customer_mcp.current_delivery_binding", lambda *_: next(calls)
    )
    result = workflow.files("alice", sid)
    assert result["status"] == "stale"
    assert result["project"] is None
    assert result["print"] is None


def test_file_byte_mismatch_is_an_error(captured, monkeypatch):
    workflow, _store, sid, receipt, _binding, _reads = captured
    monkeypatch.setattr(
        "forge_mind.workspace.customer_mcp.read_project_export", lambda *_: (b"wrong", receipt)
    )
    with pytest.raises(WorkspaceError, match="exact owned readback"):
        workflow.files("alice", sid)


def test_print_obligation_cannot_return_current_without_current_print_package(
    captured, monkeypatch
):
    workflow, store, sid, _receipt, _binding, _reads = captured
    store.update(sid, customer_printable=True)
    monkeypatch.setattr(
        "forge_mind.workspace.customer_mcp.print_setup_context",
        lambda *_: {"status": "stale", "record": {}},
    )
    result = workflow.files("alice", sid)
    assert result["status"] == "stale"
    assert result["print"] is None
