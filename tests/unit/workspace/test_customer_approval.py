from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event

import pytest
from tests.unit.workspace.test_operation_plan import contract, document, plan
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.construction_transaction import build_construction, prepare_construction
from forge_mind.workspace.customer_approval import (
    ApprovalRequest,
    CustomerApprovalRequiredError,
    ProposalRevisionRequest,
    approve_proposal,
    guard_construction,
    pause_after_plan,
    preserve_budget,
    read_proposal,
    revise_proposal,
)
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.input_application import pending
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import Store, WorkspaceError


def prepared(tmp_path):
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "a" * 32,
            "prompt": "Width 20 mm",
            "approval_required": True,
            "max_seconds": 100,
            "max_actions": 10,
        },
    )
    job = store.claim()
    sid = session["id"]
    service = WorkspaceService(store, Executor("trusted_local"))
    store.set_design_plan_for_job(sid, "owner", job["id"], plan(), 0, contract())
    return store, service, sid, job


def paused(tmp_path):
    store, service, sid, job = prepared(tmp_path)
    assert pause_after_plan(service, sid, "owner", job["id"])
    preserve_budget(store, sid, "owner", job["id"], 73.9, 8)
    store.finish(job["id"], sid, "needs_approval")
    return store, service, sid, job


def approval(store, sid):
    proposal = read_proposal(store, sid, "owner")
    return ApprovalRequest(proposal_hash=proposal["proposal_hash"], basis=proposal["basis"])


def test_requested_approval_blocks_native_build_before_any_cad(tmp_path):
    store, service, sid, _ = prepared(tmp_path)
    with pytest.raises(CustomerApprovalRequiredError):
        build_document(
            service, sid, "owner", document(), contract(), None, "build", 20, lambda: False
        )
    assert store.revisions(sid) == []
    proposal = read_proposal(store, sid, "owner")
    assert proposal["status"] == "pending"
    assert proposal["plan"] == plan().model_dump()


def test_approval_is_owner_and_basis_bound_with_durable_replay_and_remaining_budget(tmp_path):
    store, service, sid, first = paused(tmp_path)
    request = approval(store, sid)
    with pytest.raises(WorkspaceError, match="not found"):
        approve_proposal(store, sid, "other", request)
    jid = approve_proposal(store, sid, "owner", request)
    assert approve_proposal(Store(tmp_path), sid, "owner", request) == jid
    assert jid != first["id"]
    job = store.claim()
    assert job["id"] == jid
    assert job["max_seconds"] == 73
    assert job["max_actions"] == 8
    guard_construction(service, sid, "owner", store.plan_snapshot(sid, "owner"))
    changed = request.model_copy(update={"proposal_hash": "b" * 64})
    with pytest.raises(WorkspaceError, match="changed"):
        approve_proposal(store, sid, "owner", changed)
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2


def test_changed_source_or_plan_requires_fresh_review(tmp_path):
    store, _service, sid, _ = paused(tmp_path)
    request = approval(store, sid)
    store.update(sid, instructions=["Change the intended mounting surface"])
    assert read_proposal(store, sid, "owner")["status"] == "stale"
    with pytest.raises(WorkspaceError, match="source"):
        approve_proposal(store, sid, "owner", request)
    assert store.revisions(sid) == []


def test_no_budget_extension_through_approval_and_cancel_paused_plan(tmp_path):
    store, _, sid, _ = paused(tmp_path)
    request = approval(store, sid)
    record = store.session_record(sid, "owner")["customer_proposal"]
    record["remaining_seconds"] = 0
    store.update(sid, customer_proposal=record)
    with pytest.raises(WorkspaceError, match="exhausted"):
        approve_proposal(store, sid, "owner", request)
    store.cancel(sid, "owner")
    assert store.get(sid, "owner")["state"] == "cancelled"
    assert read_proposal(store, sid, "owner")["status"] == "stale"


def test_proposal_revision_keeps_history_and_replays_one_new_owner_job(tmp_path):
    store, _, sid, _ = paused(tmp_path)
    request = ProposalRevisionRequest(
        **approval(store, sid).model_dump(),
        request_id="c" * 32,
        instruction="Keep the 20 mm width and add rounded edges",
    )
    jid = revise_proposal(store, sid, "owner", request)
    assert revise_proposal(Store(tmp_path), sid, "owner", request) == jid
    with pytest.raises(WorkspaceError, match="different proposal revision"):
        revise_proposal(
            store, sid, "owner", request.model_copy(update={"instruction": "Other change"})
        )
    session = store.get(sid, "owner")
    assert session["approval_required"] is True
    assert session["customer_proposal_history"]
    assert session["instructions"][-1] == request.instruction
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
    assert payload["preview_only"] is False
    assert payload["delivery_requested"] is True


def test_needs_approval_skips_all_completion_capture_and_proof(tmp_path, monkeypatch):
    store, service, sid, job = prepared(tmp_path)
    assert pause_after_plan(service, sid, "owner", job["id"])
    store.update(sid, customer_approval_actions_left=8)
    monkeypatch.setattr(service, "_run_agent_job", lambda _: "needs_approval")

    def forbidden(*args, **kwargs):
        pytest.fail("Pending approval must not finalize, capture or verify")

    monkeypatch.setattr("forge_mind.workspace.completion.finalize_completion", forbidden)
    monkeypatch.setattr("forge_mind.workspace.native_delivery.capture_native_delivery", forbidden)
    monkeypatch.setattr("forge_mind.workspace.requirement_verification.verify_contract", forbidden)
    service.run_job(job)
    session = store.get(sid, "owner")
    assert session["state"] == "needs_approval"
    assert session["automatic_delivery"]["status"] == "incomplete"
    assert session["contract_verification_report"] is None
    assert session["resume_candidate"] is not None


def test_atomic_successor_pauses_without_publication_then_resumes_exact_approved_plan(
    tmp_path, monkeypatch
):
    """Synthetic publication proves lifecycle binding; no native geometry is inferred."""
    store, service, sid, _ = paused(tmp_path)
    approve_proposal(store, sid, "owner", approval(store, sid))
    job = store.claim()
    original = deepcopy(store.session_record(sid, "owner")["operation_plan_snapshot"])
    patch = PlanPatch(basis=store.edit_basis(sid, "owner"), verification=["Inspect the plate edge"])
    staged = prepare_construction(
        service, sid, "owner", job["id"], document(), patch, author_intent="Width 20 mm"
    )
    with pytest.raises(CustomerApprovalRequiredError):
        build_construction(
            service,
            sid,
            "owner",
            job["id"],
            staged,
            explanation="Requested successor",
            timeout=10,
            cancelled=lambda: False,
        )
    assert store.revisions(sid) == []
    assert store.session_record(sid, "owner")["operation_plan_snapshot"] == original
    proposal = read_proposal(store, sid, "owner")
    assert proposal["plan"] == staged.successor_snapshot["plan"]
    assert proposal["plan"] != original["plan"]
    preserve_budget(store, sid, "owner", job["id"], 60, 7)
    store.finish(job["id"], sid, "needs_approval")
    second = approve_proposal(store, sid, "owner", approval(store, sid))
    resumed = store.claim()
    assert resumed["id"] == second
    assert resumed["max_seconds"] == 60 and resumed["max_actions"] == 7
    staged = prepare_construction(
        service, sid, "owner", resumed["id"], document(), patch, author_intent="Width 20 mm"
    )
    from forge_mind.workspace.customer_construction import claim_construction

    with pytest.raises(CustomerApprovalRequiredError):
        guard_construction(service, sid, "owner", staged.successor_snapshot)
    assert claim_construction(store, sid, "owner", resumed["id"]) is not None
    guard_construction(service, sid, "owner", staged.successor_snapshot)
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    rid = "e" * 32
    revision = saved_revision(
        store, sid, rid, staged.document, contract(), staged.successor_snapshot
    )
    directory = store.directory(sid, rid)
    for name, value in (
        ("operation-plan.json", staged.successor_snapshot),
        ("operation-record.json", revision["operation_record"]),
    ):
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
        revision["measurement"]["artifact_hashes"][name] = file_sha256(directory / name)
    with store.connect() as con:
        con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))
    receipt = store.publish_construction(sid, "owner", resumed["id"], staged, rid)
    assert receipt["applied"] is True and receipt["cad_applied"] is True
    final = store.session_record(sid, "owner")
    assert final["operation_plan_snapshot"] == staged.successor_snapshot
    assert final["selected_revision"] == rid
    assert final.get("accepted_revision") is None


def pending_proposal(tmp_path):
    store, service, sid, first = prepared(tmp_path)
    store.finish(first["id"], sid, "needs_review")
    jid = store.enqueue(sid, "owner", {"instruction": "Add 4 mm thickness", "preview_only": False})
    job = store.claim()
    assert job["id"] == jid
    item = pending(store.session_record(sid, "owner"))
    assert item is not None
    assert pause_after_plan(service, sid, "owner", jid)
    preserve_budget(store, sid, "owner", jid, 50, 7)
    store.finish(jid, sid, "needs_approval")
    return store, sid, item


def test_revise_with_pending_owner_input_retains_exact_old_input_and_queues_once(tmp_path):
    store, sid, old = pending_proposal(tmp_path)
    body = ProposalRevisionRequest(
        **approval(store, sid).model_dump(),
        request_id="c" * 32,
        instruction="Use 5 mm thickness instead",
    )
    jid = revise_proposal(store, sid, "owner", body)
    assert revise_proposal(store, sid, "owner", body) == jid
    session = store.session_record(sid, "owner")
    archived = session["abandoned_input_applications"][-1]
    assert archived["input_id"] == old["input_id"]
    assert archived["snapshot"]["contract_application"]["pending"] == old
    assert archived["snapshot"]["instructions"][-1] == "Add 4 mm thickness"
    assert pending(session)["input_id"] != old["input_id"]
    assert session["instructions"][-1] == body.instruction
    assert session["approval_required"] is True and store.revisions(sid) == []


def test_revise_queue_failure_rolls_back_pending_abandonment_and_request_identity(tmp_path):
    store, sid, _ = pending_proposal(tmp_path)
    for index in range(16):
        other = store.create("other", f"Queued {index}", None)["id"]
        store.enqueue(other, "other", {})
    before = deepcopy(store.session_record(sid, "owner"))
    body = ProposalRevisionRequest(
        **approval(store, sid).model_dump(),
        request_id="c" * 32,
        instruction="Use 5 mm thickness instead",
    )
    with pytest.raises(WorkspaceError, match="queue is full"):
        revise_proposal(store, sid, "owner", body)
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM customer_operations").fetchone()[0] == 0


@pytest.mark.parametrize("ordering", ["approve_first", "cancel_first", "simultaneous"])
def test_approval_cancel_race_leaves_no_runnable_continuation_or_geometry(tmp_path, ordering):
    store, _, sid, _ = paused(tmp_path)
    body = approval(store, sid)
    barrier, completed = Barrier(2), Event()

    def approve():
        barrier.wait(timeout=5)
        if ordering == "cancel_first":
            assert completed.wait(5)
        try:
            return approve_proposal(store, sid, "owner", body)
        except WorkspaceError:
            return None
        finally:
            if ordering == "approve_first":
                completed.set()

    def cancel():
        barrier.wait(timeout=5)
        if ordering == "approve_first":
            assert completed.wait(5)
        try:
            store.cancel(sid, "owner")
        finally:
            if ordering == "cancel_first":
                completed.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        approved, cancelled = workers.submit(approve), workers.submit(cancel)
        jid = approved.result(timeout=10)
        cancelled.result(timeout=10)
    assert store.session_record(sid, "owner")["state"] == "cancelled"
    assert store.revisions(sid) == []
    with store.connect() as con:
        assert (
            con.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[
                0
            ]
            == 0
        )
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == (2 if jid else 1)
    if ordering == "cancel_first":
        assert jid is None
    elif ordering == "approve_first":
        assert jid is not None
