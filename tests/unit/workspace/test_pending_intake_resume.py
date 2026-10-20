"""Pending pre-CAD owner input resumes without changing its authority or answers."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_intent_attribution_service import (
    OWNER,
    resolution,
    terminal_pending,
)

from forge_mind.workspace.store import WorkspaceError


def failed_intake(tmp_path, *, legacy=False):
    store, sid, _, record = terminal_pending(tmp_path, delivery_requested=True)
    store.enqueue(sid, OWNER, {"intent_resolution": resolution(record), "preview_only": False})
    job = store.claim()
    assert job is not None
    store.finish(job["id"], sid, "needs_review")
    if legacy:
        store.update(sid, resume_candidate=None)
    return store, sid


@pytest.mark.parametrize("legacy", [False, True])
def test_failed_intake_resumes_exact_pending_owner_answers_without_cad(tmp_path, legacy):
    store, sid = failed_intake(tmp_path, legacy=legacy)
    before = store.get(sid, OWNER)
    pending = deepcopy(before["contract_application"]["pending"])
    if not legacy:
        assert before["resume_candidate"]["revision_id"] is None
    jid = store.enqueue(sid, OWNER, {"resume": True})
    after = store.get(sid, OWNER)
    assert after["job_id"] == jid
    assert after["instructions"] == before["instructions"]
    assert after["contract_application"]["pending"] == pending
    assert after["intent_attribution"] == before["intent_attribution"]
    assert not after["revisions"] and after["selected_revision"] is None
    assert pending["owner_answers"] and pending["input_id"]
    with pytest.raises(WorkspaceError, match="active job"):
        store.enqueue(sid, OWNER, {"resume": True})


@pytest.mark.parametrize(
    "change",
    ["source", "input", "plan", "contract", "pending_id", "owner", "instruction", "question"],
)
def test_stranded_intake_resume_rejects_changed_authority(tmp_path, change):
    store, sid = failed_intake(tmp_path, legacy=True)
    before = store.get(sid, OWNER)
    owner = OWNER
    payload = {"resume": True}
    if change == "source":
        store.update(sid, instructions=[*before["instructions"], "Change the source"])
    elif change == "input":
        with store.connect() as con:
            con.execute(
                "INSERT INTO engineering_inputs(session,version,document) VALUES (?,1,'{}')", (sid,)
            )
    elif change == "plan":
        store.update(sid, plan_version=1, plan_hash="a" * 64)
    elif change == "contract":
        store.update(sid, contract={"requirements": []})
    elif change == "pending_id":
        application = deepcopy(before["contract_application"])
        application["pending"]["input_id"] = "f" * 32
        store.update(sid, contract_application=application)
    elif change == "owner":
        owner = "other-owner"
    elif change == "instruction":
        payload["instruction"] = "Invent a new edit"
    else:
        attribution = deepcopy(before["intent_attribution"])
        attribution["status"] = "needs_input"
        # Rebuild the exact record identity, as a real newly raised question does.
        from forge_mind.workspace.document import digest

        attribution["draft_hash"] = digest(
            {key: value for key, value in attribution.items() if key != "draft_hash"}
        )
        store.update(sid, intent_attribution=attribution)
    with store.connect() as con:
        jobs = con.execute("SELECT COUNT(*) FROM jobs WHERE session=?", (sid,)).fetchone()[0]
    with pytest.raises(WorkspaceError):
        store.enqueue(sid, owner, payload)
    with store.connect() as con:
        assert (
            con.execute("SELECT COUNT(*) FROM jobs WHERE session=?", (sid,)).fetchone()[0] == jobs
        )
    assert store.get(sid, OWNER)["resume_candidate"] is None


def test_missing_pending_input_does_not_create_a_resume_checkpoint(tmp_path):
    store, sid = failed_intake(tmp_path, legacy=True)
    store.update(sid, contract_application={"applied": None, "pending": None})
    with pytest.raises(WorkspaceError, match="No interrupted candidate"):
        store.enqueue(sid, OWNER, {"resume": True})
