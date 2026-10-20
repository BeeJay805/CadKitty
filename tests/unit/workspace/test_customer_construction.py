"""Exact pre-CAD approval and durable claims; all geometry here is synthetic."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from tests.unit.workspace.test_customer_approval import approval, prepared
from tests.unit.workspace.test_operation_plan import contract, document
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.workspace.customer_approval import (
    CustomerApprovalRequiredError,
    approve_proposal,
    preserve_budget,
)
from forge_mind.workspace.customer_construction import (
    author_workflow_context,
    claim_construction,
    complete_construction,
    construction_finalization_allowed,
    guard_document,
    stage_construction,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import WorkspaceError


def staged(tmp_path):
    store, service, sid, job = prepared(tmp_path)
    with pytest.raises(CustomerApprovalRequiredError):
        stage_construction(
            service,
            sid,
            "owner",
            job["id"],
            document(),
            contract(),
            None,
            "Build the reviewed plate",
            remaining_seconds=70,
            remaining_actions=7,
        )
    preserve_budget(store, sid, "owner", job["id"], 70, 7)
    store.finish(job["id"], sid, "needs_approval")
    assert store.revisions(sid) == []
    return store, service, sid


def approved(tmp_path):
    store, service, sid = staged(tmp_path)
    body = approval(store, sid)
    jid = approve_proposal(store, sid, "owner", body)
    assert approve_proposal(store, sid, "owner", body) == jid
    assert store.claim()["id"] == jid
    return store, service, sid, jid


def test_exact_command_claim_and_publication_do_not_regenerate_or_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    store, service, sid, jid = approved(tmp_path)
    record = deepcopy(store.session_record(sid, "owner")["customer_proposal"])
    action = claim_construction(store, sid, "owner", jid)
    assert action["command_hash"] == record["command_hash"]
    assert author_workflow_context(store, sid, "owner", jid)["construction_authorized"] is True
    stage_construction(
        service, sid, "owner", jid, document(), contract(), None, "Build the reviewed plate"
    )
    target = record["command"]["target_snapshot"]
    guard_document(
        service, sid, "owner", document(), contract(), None, "Build the reviewed plate", target
    )
    with pytest.raises(WorkspaceError, match="already started"):
        guard_document(
            service, sid, "owner", document(), contract(), None, "Build the reviewed plate", target
        )
    rid = "e" * 32
    saved_revision(store, sid, rid, document(), contract(), target)
    store.update(sid, selected_revision=rid)
    complete_construction(store, sid, "owner", jid, rid)
    complete_construction(store, sid, "owner", jid, rid)
    assert claim_construction(store, sid, "owner", jid) == {
        "mode": "continue_published",
        "revision_id": rid,
        "execution_job_id": jid,
        "command_hash": record["command_hash"],
    }
    assert construction_finalization_allowed(store, sid, "owner", jid)
    execution = store.session_record(sid, "owner")["customer_proposal"]["execution"]
    assert execution["phase"] == "published" and execution["revision_id"] == rid
    assert (
        execution["staged_at"]
        <= execution["approved_at"]
        <= execution["claimed_at"]
        <= execution["build_started_at"]
        <= execution["completed_at"]
    )
    store.update(sid, instructions=["The owner changed the target after publication"])
    assert author_workflow_context(store, sid, "owner", jid)["status"] == "stale"
    assert not construction_finalization_allowed(store, sid, "owner", jid)


def published(tmp_path, monkeypatch):
    monkeypatch.setattr("forge_mind.workspace.measure.engine_fingerprint", lambda: "f" * 64)
    store, service, sid, jid = approved(tmp_path)
    target = store.session_record(sid, "owner")["customer_proposal"]["command"]["target_snapshot"]
    assert claim_construction(store, sid, "owner", jid)["mode"] == "execute_saved_command"
    guard_document(
        service, sid, "owner", document(), contract(), None, "Build the reviewed plate", target
    )
    rid = "e" * 32
    saved_revision(store, sid, rid, document(), contract(), target)
    store.update(sid, selected_revision=rid)
    complete_construction(store, sid, "owner", jid, rid)
    store.finish(jid, sid, "error")
    return store, sid, jid, rid


def test_published_resume_preserves_original_execution_and_current_publication(
    tmp_path, monkeypatch
):
    store, sid, original, rid = published(tmp_path, monkeypatch)
    before = deepcopy(store.session_record(sid, "owner")["customer_proposal"]["execution"])
    resumed = store.enqueue(sid, "owner", {"resume": True})
    assert store.claim()["id"] == resumed
    outcome = claim_construction(store, sid, "owner", resumed)
    assert outcome["mode"] == "continue_published" and outcome["revision_id"] == rid
    assert outcome["execution_job_id"] == original
    assert store.session_record(sid, "owner")["customer_proposal"]["execution"] == before
    assert construction_finalization_allowed(store, sid, "owner", resumed)
    assert not construction_finalization_allowed(store, sid, "owner", original)
    assert len(store.revisions(sid)) == 1


@pytest.mark.parametrize(
    "change", ["source", "engine", "selection", "cancel", "job", "seal", "revision", "bytes"]
)
def test_changed_published_continuation_cannot_claim_or_finalize(tmp_path, monkeypatch, change):
    store, sid, original, rid = published(tmp_path, monkeypatch)
    resumed = store.enqueue(sid, "owner", {"resume": True})
    assert store.claim()["id"] == resumed
    requested = resumed
    if change == "source":
        store.update(sid, instructions=["Changed owner target"])
    elif change == "engine":
        monkeypatch.setattr(
            "forge_mind.workspace.engine_identity.engine_fingerprint", lambda: "changed"
        )
    elif change == "selection":
        store.update(sid, selected_revision=rid)
    elif change == "cancel":
        store.cancel(sid, "owner")
    elif change == "job":
        requested = original
    elif change == "seal":
        with store.connect() as con:
            row = con.execute("SELECT payload FROM jobs WHERE id=?", (resumed,)).fetchone()
            payload = json.loads(row[0])
            payload["customer_construction_binding"]["command_hash"] = "0" * 64
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), resumed))
    elif change == "revision":
        with store.connect() as con:
            row = con.execute("SELECT document FROM revisions WHERE id=?", (rid,)).fetchone()
            revision = json.loads(row[0])
            revision["parent_id"] = "a" * 32
            con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))
    else:
        (store.directory(sid, rid) / "part.step").write_text("Changed artifact bytes")
    with pytest.raises(WorkspaceError):
        claim_construction(store, sid, "owner", requested)
    assert not construction_finalization_allowed(store, sid, "owner", requested)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "input",
        "engine",
        "command",
        "rehash",
        "cancel",
        "owner",
        "question",
        "asset",
        "plan",
    ],
)
def test_changed_or_cancelled_approval_cannot_claim_or_start(tmp_path, monkeypatch, change):
    store, _service, sid, jid = approved(tmp_path)
    owner = "owner"
    if change == "source":
        store.update(sid, instructions=["Change the physical target"])
    elif change == "input":
        with store.connect() as con:
            con.execute("INSERT INTO engineering_inputs VALUES (?,1,'{}')", (sid,))
    elif change == "engine":
        monkeypatch.setattr(
            "forge_mind.workspace.engine_identity.engine_fingerprint", lambda: "changed"
        )
    elif change in {"command", "rehash"}:
        record = store.session_record(sid, "owner")["customer_proposal"]
        record["command"]["explanation"] = "Different requested command"
        if change == "rehash":
            record["command_hash"] = digest(record["command"])
            record["proposal_hash"] = digest(
                {
                    key: record[key]
                    for key in ("basis", "plan", "contract", "source", "command_hash")
                }
            )
        store.update(sid, customer_proposal=record)
    elif change == "cancel":
        store.cancel(sid, "owner")
    elif change == "question":
        store.update(sid, intent_attribution={"status": "needs_input"})
    elif change == "asset":
        store.update(sid, assets=[{"id": "f" * 32, "sha256": "0" * 64}])
    elif change == "plan":
        store.update(sid, design_plan={"objective": "A different plan"})
    else:
        owner = "another-owner"
    with pytest.raises((WorkspaceError, ValueError)):
        claim_construction(store, sid, owner, jid)
    assert store.revisions(sid) == []
    assert not construction_finalization_allowed(store, sid, "owner", jid)


def test_unattempted_command_survives_guarded_resume_without_budget_extension(tmp_path):
    store, _service, sid, jid = approved(tmp_path)
    store.finish(jid, sid, "error")
    with pytest.raises(WorkspaceError, match="extend"):
        store.enqueue(sid, "owner", {"resume": True, "max_seconds": 71})
    next_id = store.enqueue(sid, "owner", {"resume": True})
    job = store.claim()
    assert job["id"] == next_id and job["max_seconds"] == 70 and job["max_actions"] == 7
    assert claim_construction(store, sid, "owner", next_id) is not None
    assert store.session_record(sid, "owner")["customer_proposal"]["approved_job_id"] == jid


def test_uncertain_claim_stops_resume_and_downstream_work(tmp_path):
    store, _service, sid, jid = approved(tmp_path)
    claim_construction(store, sid, "owner", jid)
    with pytest.raises(WorkspaceError, match="uncertain"):
        claim_construction(store, sid, "owner", jid)
    store.finish(jid, sid, "error")
    with pytest.raises(WorkspaceError, match="uncertain"):
        store.enqueue(sid, "owner", {"resume": True})
    assert not construction_finalization_allowed(store, sid, "owner", jid)
    assert store.revisions(sid) == []


def test_failed_claim_before_native_start_cannot_return_to_model_or_resume(tmp_path):
    store, _service, sid, jid = approved(tmp_path)
    claim_construction(store, sid, "owner", jid)
    complete_construction(store, sid, "owner", jid, None, error="Prebuild failed")
    execution = store.session_record(sid, "owner")["customer_proposal"]["execution"]
    assert (
        execution["phase"] == "not_applied" and execution["completed_at"] >= execution["claimed_at"]
    )
    with pytest.raises(WorkspaceError, match="uncertain"):
        claim_construction(store, sid, "owner", jid)
    store.finish(jid, sid, "error")
    with pytest.raises(WorkspaceError, match="uncertain"):
        store.enqueue(sid, "owner", {"resume": True})
    assert not construction_finalization_allowed(store, sid, "owner", jid)


def test_concurrent_claims_cannot_dispatch_twice(tmp_path):
    store, _service, sid, jid = approved(tmp_path)

    def attempt(_):
        try:
            return claim_construction(store, sid, "owner", jid)
        except WorkspaceError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1


def test_invalid_preflight_never_creates_a_reviewable_command(tmp_path):
    store, service, sid, job = prepared(tmp_path)
    with pytest.raises(WorkspaceError, match="contract differs"):
        stage_construction(
            service, sid, "owner", job["id"], document(), Contract(), None, "Invalid contract"
        )
    assert store.session_record(sid, "owner").get("customer_proposal") is None
    assert store.revisions(sid) == []
