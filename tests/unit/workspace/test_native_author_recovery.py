"""Saved native author work must survive interruptions without adopting acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.workspace.document import digest
from forge_mind.workspace.input_application import FIELD
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.product_sources import _append
from forge_mind.workspace.store import Store, WorkspaceError

BASELINE = "a" * 32
CANDIDATE = "b" * 32


def interrupted_author(
    root: Path,
    *,
    accepted: str = "null",
    state: str = "budget_exhausted",
    delivery: bool = True,
    candidate: str | None = CANDIDATE,
) -> tuple[Store, str, str]:
    store = Store(root)
    contract = Contract(
        requirements=[
            Requirement(
                id="width", text="Width 20 mm", source_quote="20 mm", metric="extent_x", value=20
            )
        ]
    ).model_dump()
    sid = store.create("owner", "Width 20 mm", contract, "native_operations")["id"]
    plan = DesignPlan(objective="Complete the plate").model_dump()
    binding = snapshot(plan, 2, contract)
    store.update(
        sid,
        # The saved target has already consumed this owner history.
        instructions=["Preserve the bore"],
        selected_revision=BASELINE if accepted == "accepted" else None,
        design_plan=plan,
        plan_version=2,
        plan_hash=binding["plan_hash"],
        plan_contract_hash=binding["contract_hash"],
        operation_plan_snapshot=binding,
    )
    if accepted != "absent":
        store.update(sid, accepted_revision=BASELINE if accepted == "accepted" else None)
    # A nonzero admitted engineering version must survive a resume.
    store.save_engineering(sid, "owner", 0, {})
    jid = store.enqueue(sid, "owner", {"delivery_requested": delivery})
    application = store.session_record(sid, "owner")[FIELD]
    assert application["pending"] is None
    assert application["applied"]["source_count"] == 1
    assert application["applied"]["source_hash"] == digest(
        "Width 20 mm\nUser revision instruction: Preserve the bore"
    )
    assert (
        application["applied"]["contract_hash"] == Contract.model_validate(contract).fingerprint()
    )
    assert store.claim()["id"] == jid
    store.update(sid, selected_revision=candidate)
    if state == "needs_review":
        store.update(sid, incomplete_finish_job_id=jid)
    store.finish(jid, sid, state)
    return store, sid, jid


@pytest.mark.parametrize("state", ["cancelled", "error", "budget_exhausted", "needs_review"])
def test_native_non_preview_terminal_interruptions_keep_current_author_work(tmp_path, state):
    store, sid, _ = interrupted_author(tmp_path, state=state)
    session = store.session_record(sid, "owner")
    assert session["state"] == state
    assert session["selected_revision"] == CANDIDATE
    assert session["accepted_revision"] is None
    assert session["resume_candidate"]["revision_id"] == CANDIDATE
    assert session["resume_candidate"]["contract"] == session["contract"]
    assert session["resume_candidate"]["instructions"] == ["Preserve the bore"]
    assert session["resume_candidate"]["plan_snapshot"] == session["operation_plan_snapshot"]


@pytest.mark.parametrize("accepted", ["absent", "null", "accepted"])
@pytest.mark.parametrize("delivery", [False, True])
def test_resume_preserves_native_mode_acceptance_and_delivery(tmp_path, accepted, delivery):
    store, sid, _ = interrupted_author(tmp_path, accepted=accepted, delivery=delivery)
    saved = store.session_record(sid, "owner")
    reopened = Store(tmp_path)
    # The API currently sends preview_only=True. Saved execution mode is authoritative.
    jid = reopened.enqueue(
        sid,
        "owner",
        {
            "resume": True,
            "preview_only": True,
            "max_actions": 4,
            "delivery_requested": not delivery,
        },
    )
    queued = reopened.session_record(sid, "owner")
    assert queued["selected_revision"] == CANDIDATE
    assert ("accepted_revision" in queued) == ("accepted_revision" in saved)
    assert queued.get("accepted_revision") == saved.get("accepted_revision")
    job = reopened.claim()
    assert job["id"] == jid
    assert job["revision_id"] == CANDIDATE
    assert job["contract"] == saved["contract"]
    assert job["resume_instructions"] == saved["instructions"]
    assert job["preview_only"] is False
    assert job["delivery_requested"] is delivery
    assert job["max_actions"] == 4


def test_dead_native_author_keeps_candidate_and_original_input_version(tmp_path, monkeypatch):
    store, sid, _ = interrupted_author(tmp_path)
    jid = store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    assert store.claim()["id"] == jid
    monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda _: True))
    assert Store(tmp_path).claim() is None
    session = store.session_record(sid, "owner")
    assert session["state"] == "interrupted"
    assert session["selected_revision"] == CANDIDATE
    assert session["accepted_revision"] is None
    assert session["resume_candidate"]["input_version"] == 1
    assert session["resume_candidate"]["revision_id"] == CANDIDATE


def test_interrupt_does_not_rebind_admitted_engineering_version(tmp_path):
    store, sid, _ = interrupted_author(tmp_path)
    jid = store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    assert store.claim()["id"] == jid
    # Simulate changed external state. Owner APIs normally reject this during work.
    with store.connect() as con:
        con.execute("UPDATE engineering_inputs SET version=2 WHERE session=?", (sid,))
    store.finish(jid, sid, "error")
    assert store.session_record(sid, "owner")["resume_candidate"]["input_version"] == 1
    with pytest.raises(WorkspaceError, match="changed"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True})


def test_plan_only_native_author_can_resume_without_inventing_a_revision(tmp_path):
    store, sid, _ = interrupted_author(tmp_path, accepted="absent", candidate=None)
    assert store.session_record(sid, "owner")["resume_candidate"]["revision_id"] is None
    store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    job = store.claim()
    assert job["revision_id"] is None
    assert job["preview_only"] is False
    assert "accepted_revision" not in store.session_record(sid, "owner")


@pytest.mark.parametrize(
    "changed",
    [
        "selected",
        "accepted_value",
        "accepted_presence",
        "plan",
        "plan_hash",
        "plan_contract_hash",
        "plan_version",
        "plan_snapshot",
        "proposal",
        "contract",
        "instructions",
        "engineering",
        "sources",
    ],
)
def test_native_resume_refuses_changed_saved_author_inputs(tmp_path, changed):
    store, sid, _ = interrupted_author(tmp_path, accepted="absent")
    saved = store.session_record(sid, "owner")
    if changed == "selected":
        store.update(sid, selected_revision=BASELINE)
    elif changed == "accepted_value":
        store.update(sid, accepted_revision=BASELINE)
    elif changed == "accepted_presence":
        store.update(sid, accepted_revision=None)
    elif changed == "plan":
        store.update(sid, design_plan={**saved["design_plan"], "objective": "Different plate"})
    elif changed == "plan_hash":
        store.update(sid, plan_hash="0" * 64)
    elif changed == "plan_contract_hash":
        store.update(sid, plan_contract_hash="0" * 64)
    elif changed == "plan_version":
        store.update(sid, plan_version=3)
    elif changed == "plan_snapshot":
        store.update(
            sid, operation_plan_snapshot={**saved["operation_plan_snapshot"], "version": 9}
        )
    elif changed == "proposal":
        store.update(sid, proposal={"revision_id": BASELINE})
    elif changed == "contract":
        store.update(sid, contract={**saved["contract"], "coverage_notes": ["Changed target"]})
    elif changed == "instructions":
        store.update(sid, instructions=["Remove the bore"])
    elif changed == "engineering":
        store.save_engineering(sid, "owner", 1, {"assumptions": []})
    else:
        _append(store, sid, "owner", {"kind": "artifact", "data": {"id": "source"}})
    before = store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match=r"changed|stale"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    assert store.session_record(sid, "owner") == before


@pytest.mark.parametrize(
    "override", [{"instruction": "Remove the bore"}, {"parameters": {"width": 99}}]
)
def test_native_resume_is_not_an_implicit_edit(tmp_path, override):
    store, sid, _ = interrupted_author(tmp_path)
    with pytest.raises(WorkspaceError, match=r"resume|Resume"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True, **override})


def test_completed_native_author_clears_old_checkpoint(tmp_path):
    store, sid, _ = interrupted_author(tmp_path)
    jid = store.enqueue(sid, "owner", {"resume": True, "preview_only": True})
    assert store.claim()["id"] == jid
    store.finish(jid, sid, "contract_satisfied")
    assert store.session_record(sid, "owner").get("resume_candidate") is None
    with pytest.raises(WorkspaceError, match="No interrupted candidate"):
        store.enqueue(sid, "owner", {"resume": True, "preview_only": True})


def test_unrelated_hybrid_non_preview_interruptions_remain_unchanged(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Plate", Contract().model_dump())["id"]
    jid = store.enqueue(sid, "owner", {})
    assert store.claim()["id"] == jid
    store.update(sid, selected_revision=CANDIDATE)
    store.finish(jid, sid, "budget_exhausted")
    assert store.session_record(sid, "owner").get("resume_candidate") is None


@pytest.mark.parametrize("payload", [{"engineering_operation": "review"}, {"document": {}}])
def test_typed_operations_never_get_native_author_checkpoints(tmp_path, payload):
    store, sid, _ = interrupted_author(tmp_path)
    store.update(sid, resume_candidate=None)
    # Inject a synthetic typed payload after ordinary admission. No CAD is executed.
    jid = store.enqueue(sid, "owner", {})
    with store.connect() as con:
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
    assert store.claim()["id"] == jid
    store.finish(jid, sid, "error")
    assert store.session_record(sid, "owner").get("resume_candidate") is None
