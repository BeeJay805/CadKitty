"""A preview may use a draft baseline without inventing explicit acceptance."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_input_application import claim, create, target
from tests.unit.workspace.test_input_continuation import (
    answer,
    ask,
    identity_revision,
    payload,
    width_plan,
)
from tests.unit.workspace.test_revision_bound_edits import (
    admitted_job,
    authority,
    built_candidate,
    inline_publication,
)
from tests.unit.workspace.test_revision_bound_edits import (
    edits as feature_fixture,
)

from forge_mind.workspace import operation_recovery
from forge_mind.workspace.handoff import handoff_context
from forge_mind.workspace.input_application import FIELD, baseline
from forge_mind.workspace.models import Contract
from forge_mind.workspace.product_sources import _append
from forge_mind.workspace.store import EditConflictError, Store, WorkspaceError

BASELINE = "a" * 32
CANDIDATE = "b" * 32
OTHER = "c" * 32
IDENTITY = "preview_acceptance_identity"
ACCEPTANCE_FIELDS = (
    "accepted_revision",
    "accepted_operation_plan_snapshot",
    "accepted_operation_plan_revision",
)


def test_preview_admission_does_not_accept_a_selected_draft(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Make a holder", Contract().model_dump())["id"]
    store.update(sid, selected_revision=BASELINE)
    assert "accepted_revision" not in store.session_record(sid, "owner")
    store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Add a hole"})
    session = store.session_record(sid, "owner")
    assert session["selected_revision"] == BASELINE
    assert "accepted_revision" not in session
    context = handoff_context(
        {**session, "engineering": {"version": 0}, "source_fingerprint": "source"},
        {"id": BASELINE, "document": {}, "measurement": {}},
        "engine",
    )
    assert context["accepted_revision"] is None
    assert context["matches_accepted_revision"] is False


def acceptance(session):
    return {
        key: {"present": key in session, "value": deepcopy(session.get(key))}
        for key in ACCEPTANCE_FIELDS
    }


def stored_state(store, sid):
    session = deepcopy(store.session_record(sid, "owner"))
    for key in ("state", "pending_finish", "authority_conflict_job_id", "incomplete_finish_job_id"):
        session.pop(key, None)
    return session


def raw_session(store, sid, change):
    with store.connect() as con:
        value = json.loads(
            con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()[0]
        )
        change(value)
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(value), sid))


def workspace(tmp_path, state="absent", *, native=True):
    store, sid = create(tmp_path, native=native)
    store.update(sid, instructions=["Known consumed baseline source."])
    plan = store.set_design_plan(sid, "owner", width_plan(40), 0)
    identity_revision(store, sid, BASELINE, binding=plan, contract=target())
    identity_revision(store, sid, OTHER, binding=plan, contract=target())
    fields = {"selected_revision": BASELINE}
    if state == "null":
        fields.update(
            accepted_revision=None,
            accepted_operation_plan_snapshot=None,
            accepted_operation_plan_revision=None,
        )
    elif state in {"explicit", "different"}:
        accepted = BASELINE if state == "explicit" else OTHER
        fields.update(
            accepted_revision=accepted,
            accepted_operation_plan_snapshot=plan,
            accepted_operation_plan_revision=accepted,
        )
    store.update(sid, **fields)
    return store, sid, plan


def candidate_work(store, sid, job):
    store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], target(50))
    plan = store.set_design_plan_for_job(
        sid, "owner", job["id"], width_plan(50), 1, contract=target(50)
    )
    identity_revision(store, sid, CANDIDATE, binding=plan, contract=target(50))
    store.update(sid, selected_revision=CANDIDATE)
    return plan


def finish_preview(store, sid, job, outcome, proposal=None):
    store.finish(job["id"], sid, outcome)
    store.complete_preview(sid, job["id"], job.get("based_on"), proposal)


@pytest.mark.parametrize(
    "native,state",
    [(False, "absent"), (True, "absent"), (True, "null"), (True, "explicit"), (True, "different")],
)
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "error",
        "budget_exhausted",
        "queued_cancel",
        "running_cancel",
        "dead",
        "needs_input",
    ],
)
def test_every_preview_path_preserves_acceptance_and_existing_working_baseline(
    tmp_path, monkeypatch, native, state, outcome
):
    store, sid, plan_a = workspace(tmp_path, state, native=native)
    original = deepcopy(store.session_record(sid, "owner"))
    identity = acceptance(original)
    expected_baseline = original.get("accepted_revision", original["selected_revision"])
    jid = store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Make it 50 mm wide."})
    admitted = payload(store, jid)
    assert admitted[IDENTITY] == identity
    assert admitted["based_on"] == expected_baseline
    assert acceptance(store.session_record(sid, "owner")) == identity
    if outcome == "queued_cancel":
        store.cancel(sid, "owner")
    else:
        job = store.claim()
        assert job and job["id"] == jid
        plan_b = candidate_work(store, sid, job)
        candidate_application = deepcopy(store.session_record(sid, "owner")[FIELD])
        if outcome == "dead":
            monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
            assert store.claim() is None
        elif outcome == "needs_input":
            ask(store, sid, job, target(50))
            finish_preview(store, sid, job, "needs_input")
        elif outcome == "success":
            current = store.session_record(sid, "owner")
            finish_preview(
                store,
                sid,
                job,
                "needs_review",
                {
                    "revision_id": CANDIDATE,
                    "based_on": expected_baseline,
                    "input_version": 0,
                    "eligible": True,
                    "contract": target(50).model_dump(),
                    "instructions": current["instructions"],
                },
            )
        else:
            if outcome == "running_cancel":
                store.cancel(sid, "owner")
            finish_preview(store, sid, job, "cancelled" if outcome == "running_cancel" else outcome)
    current = store.session_record(sid, "owner")
    assert acceptance(current) == identity
    assert current["selected_revision"] == expected_baseline
    assert current["contract"] == original["contract"]
    assert current[FIELD] == original[FIELD]
    if outcome != "needs_input":
        assert current["instructions"] == original["instructions"]
    if native and outcome not in {"success", "needs_input"}:
        assert store.plan_snapshot(sid, "owner") == plan_a
    if outcome in {"error", "budget_exhausted", "running_cancel", "dead"}:
        checkpoint = current["resume_candidate"]
        assert checkpoint["revision_id"] == CANDIDATE
        assert checkpoint["contract"] == target(50).model_dump()
        assert checkpoint[FIELD] == candidate_application
        if native:
            assert checkpoint["plan_snapshot"] == plan_b
    with store.connect() as con:
        assert con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] not in {
            "queued",
            "running",
        }


@pytest.mark.parametrize("state", ["absent", "null", "explicit", "different"])
def test_only_later_explicit_acceptance_promotes_candidate_and_its_plan(tmp_path, state):
    store, sid, _ = workspace(tmp_path, state)
    original = deepcopy(store.session_record(sid, "owner"))
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    plan_b = candidate_work(store, sid, job)
    instructions = store.session_record(sid, "owner")["instructions"]
    finish_preview(
        store,
        sid,
        job,
        "needs_review",
        {
            "revision_id": CANDIDATE,
            "based_on": job["based_on"],
            "input_version": 0,
            "eligible": True,
            "contract": target(50).model_dump(),
            "instructions": instructions,
        },
    )
    assert acceptance(store.session_record(sid, "owner")) == acceptance(original)
    store.accept_proposal(sid, "owner", CANDIDATE, job["based_on"], 0)
    accepted = store.session_record(sid, "owner")
    assert accepted["accepted_revision"] == accepted["selected_revision"] == CANDIDATE
    assert accepted["accepted_operation_plan_snapshot"] == plan_b
    assert accepted["accepted_operation_plan_revision"] == CANDIDATE
    assert accepted["contract"] == target(50).model_dump()
    assert accepted["instructions"] == instructions


def change_acceptance(store, sid, change):
    field = {
        "pointer_presence": "accepted_revision",
        "pointer_value": "accepted_revision",
        "plan_presence": "accepted_operation_plan_snapshot",
        "plan_value": "accepted_operation_plan_snapshot",
        "plan_revision_presence": "accepted_operation_plan_revision",
        "plan_revision_value": "accepted_operation_plan_revision",
    }[change]

    def mutate(value):
        if change.endswith("presence"):
            value.pop(field)
        else:
            value[field] = (
                {"literal": "New acceptance plan"} if field.endswith("snapshot") else OTHER
            )
        value.update(selected_revision=OTHER, instructions=["Newer owner state."])
        value[FIELD] = baseline(value)

    raw_session(store, sid, mutate)


@pytest.mark.parametrize(
    "change",
    [
        "pointer_presence",
        "pointer_value",
        "plan_presence",
        "plan_value",
        "plan_revision_presence",
        "plan_revision_value",
    ],
)
@pytest.mark.parametrize("phase", ["queued_cancel", "complete", "dead"])
def test_stale_acceptance_identity_never_rolls_back_newer_authority(
    tmp_path, monkeypatch, change, phase
):
    store, sid, _ = workspace(tmp_path, "explicit")
    jid = store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Make it 50 mm wide."})
    if phase != "queued_cancel":
        job = store.claim()
        assert job and job["id"] == jid
        candidate_work(store, sid, job)
        if phase == "complete":
            store.finish(jid, sid, "needs_review")
    change_acceptance(store, sid, change)
    newer = stored_state(store, sid)
    if phase == "queued_cancel":
        store.cancel(sid, "owner")
    elif phase == "dead":
        monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
        assert store.claim() is None
    else:
        store.complete_preview(sid, jid, job["based_on"], None)
    assert stored_state(store, sid) == newer
    with store.connect() as con:
        assert con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] not in {
            "queued",
            "running",
        }


def test_changed_acceptance_before_claim_or_owned_write_refuses_dispatch_and_publication(tmp_path):
    store, sid, _ = workspace(tmp_path, "explicit")
    jid = store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Make it 50 mm wide."})
    store.update(sid, accepted_revision=OTHER)
    newer = stored_state(store, sid)
    assert store.claim() is None
    assert stored_state(store, sid) == newer
    with store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] == "needs_review"
        )


def test_changed_acceptance_refuses_owned_contract_application_and_releases_lease(tmp_path):
    store, sid, _ = workspace(tmp_path, "explicit")
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    store.update(sid, accepted_operation_plan_snapshot=None)
    newer = stored_state(store, sid)
    with pytest.raises(EditConflictError, match="Acceptance authority changed"):
        store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], target(50))
    assert stored_state(store, sid) == newer
    store.finish(job["id"], sid, "error")
    assert stored_state(store, sid) == newer
    with store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (job["id"],)).fetchone()[0] == "error"
        )


def test_old_preview_finish_cannot_change_a_newer_session_job(tmp_path):
    store, sid, _ = workspace(tmp_path, "explicit")
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    store.update(sid, job_id="e" * 32, state="idle", accepted_revision=OTHER)
    newer = deepcopy(store.session_record(sid, "owner"))
    store.finish(job["id"], sid, "needs_review")
    assert store.session_record(sid, "owner") == newer
    with store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == "needs_review"
        )


@pytest.mark.parametrize("original,replacement", [(1, True), (1, 1.0), (False, 0)])
def test_acceptance_plan_identity_distinguishes_json_scalar_types(tmp_path, original, replacement):
    store, sid, _ = workspace(tmp_path, "explicit")
    store.update(sid, accepted_operation_plan_snapshot={"literal": original})
    jid = store.enqueue(sid, "owner", {"preview_only": True, "instruction": "Make it 50 mm wide."})
    store.update(sid, accepted_operation_plan_snapshot={"literal": replacement})
    newer = stored_state(store, sid)
    store.cancel(sid, "owner")
    assert stored_state(store, sid) == newer
    assert payload(store, jid)[IDENTITY]["accepted_operation_plan_snapshot"]["value"] == {
        "literal": original
    }


@pytest.mark.parametrize("outcome", ["cancelled", "dead", "error"])
def test_legacy_preview_without_identity_preserves_current_acceptance_fields(
    tmp_path, monkeypatch, outcome
):
    store, sid, plan_a = workspace(tmp_path)
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    candidate_work(store, sid, job)
    legacy = payload(store, job["id"])
    legacy.pop(IDENTITY)
    with store.connect() as con:
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(legacy), job["id"]))
    store.update(
        sid,
        accepted_revision=OTHER,
        accepted_operation_plan_snapshot=plan_a,
        accepted_operation_plan_revision=OTHER,
    )
    identity = acceptance(store.session_record(sid, "owner"))
    if outcome == "dead":
        monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
        assert store.claim() is None
    else:
        if outcome == "cancelled":
            store.cancel(sid, "owner")
        finish_preview(store, sid, job, outcome)
    assert acceptance(store.session_record(sid, "owner")) == identity


@pytest.mark.parametrize(
    "value", [None, {}, {"accepted_revision": {"present": True, "value": CANDIDATE}}]
)
@pytest.mark.parametrize("retry", [False, True])
def test_caller_cannot_inject_preview_acceptance_identity(tmp_path, value, retry):
    store, sid, _ = workspace(tmp_path)
    before = deepcopy(store.session_record(sid, "owner"))
    request = {"retry_operation": "d" * 32} if retry else {"preview_only": True}
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, "owner", {**request, IDENTITY: value})
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_caller_cannot_inject_working_baseline_source_fingerprint(tmp_path):
    store, sid, _ = workspace(tmp_path)
    before = deepcopy(store.session_record(sid, "owner"))
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, "owner", {"preview_only": True, "baseline_source_fingerprint": "forged"})
    assert store.session_record(sid, "owner") == before


def test_retry_reconstruction_cannot_supply_a_foreign_acceptance_identity(tmp_path, monkeypatch):
    store, sid, _ = workspace(tmp_path)
    before = deepcopy(store.session_record(sid, "owner"))
    forged = acceptance(before)
    forged["accepted_revision"] = {"present": True, "value": CANDIDATE}
    monkeypatch.setattr(
        operation_recovery,
        "retry_payload",
        lambda *_: ({"preview_only": True, IDENTITY: forged}, None),
    )
    with pytest.raises(EditConflictError, match="Acceptance authority changed"):
        store.enqueue(sid, "owner", {"retry_operation": "d" * 32})
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_question_answer_cancel_and_resume_preserve_absent_acceptance(tmp_path):
    store, sid, _ = workspace(tmp_path)
    original = acceptance(store.session_record(sid, "owner"))
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    candidate_work(store, sid, job)
    record = ask(store, sid, job, target(50))
    finish_preview(store, sid, job, "needs_input")
    assert acceptance(store.session_record(sid, "owner")) == original
    answered = store.enqueue(sid, "owner", answer(record))
    assert acceptance(store.session_record(sid, "owner")) == original
    store.cancel(sid, "owner")
    recovered = deepcopy(store.session_record(sid, "owner")["resume_candidate"])
    assert recovered["revision_id"] == CANDIDATE
    resumed = claim(store, sid, resume=True, preview_only=True)
    assert resumed["revision_id"] == CANDIDATE
    assert resumed[IDENTITY] == original
    assert acceptance(store.session_record(sid, "owner")) == original
    assert payload(store, answered)[IDENTITY] == original
    assert len(store.session_record(sid, "owner")[FIELD]["pending"]["owner_answers"]) == 1


def null_preview(store, sid, outcome):
    job = claim(store, sid, preview_only=True, instruction="Make it 50 mm wide.")
    plan_b = candidate_work(store, sid, job)
    record = ask(store, sid, job, target(50)) if outcome == "needs_input" else None
    current = store.session_record(sid, "owner")
    proposal = {
        "revision_id": CANDIDATE,
        "based_on": None,
        "input_version": 0,
        "eligible": True,
        "contract": target(50).model_dump(),
        "instructions": current["instructions"],
    }
    finish_preview(
        store, sid, job, "needs_input" if record else "needs_review", None if record else proposal
    )
    return job, plan_b, record


@pytest.mark.parametrize("outcome", ["success", "needs_input"])
def test_explicit_null_followup_reuses_exact_working_plan_without_accepted_cache(tmp_path, outcome):
    store, sid, plan_a = workspace(tmp_path, "null")
    original = acceptance(store.session_record(sid, "owner"))
    _, plan_b, record = null_preview(store, sid, outcome)
    paused = store.session_record(sid, "owner")
    assert paused["selected_revision"] is None and paused["contract"] == target().model_dump()
    assert store.plan_snapshot(sid, "owner") == plan_b
    assert acceptance(paused) == original
    request = answer(record) if record else {"preview_only": True, "revision_id": CANDIDATE}
    jid = store.enqueue(sid, "owner", request)
    followup = payload(store, jid)
    assert followup["baseline_plan_snapshot"] == plan_a
    assert followup["baseline_contract"] == target().model_dump()
    assert acceptance(store.session_record(sid, "owner")) == original
    store.cancel(sid, "owner")
    current = store.session_record(sid, "owner")
    assert current["contract"] == target().model_dump()
    assert store.plan_snapshot(sid, "owner") == plan_a
    assert acceptance(current) == original
    assert current["resume_candidate"]["revision_id"] == CANDIDATE


@pytest.mark.parametrize(
    "change", ["job", "source", "owner_prefix", "prompt", "contract", "snapshot", "legacy_source"]
)
def test_working_plan_fallback_refuses_mismatched_admission_provenance(tmp_path, change):
    store, sid, _ = workspace(tmp_path, "null")
    job, _, _ = null_preview(store, sid, "success")
    previous = payload(store, job["id"])
    if change == "job":
        store.update(sid, job_id="d" * 32)
    elif change == "source":
        _append(
            store, sid, "owner", {"kind": "artifact", "data": {"id": "e" * 32, "kind": "fixture"}}
        )
    elif change == "owner_prefix":
        store.update(sid, instructions=["Changed consumed owner history."])
    elif change == "prompt":
        store.update(sid, prompt="A different owner request")
    else:
        if change == "contract":
            previous["baseline_contract"] = target(60).model_dump()
        elif change == "snapshot":
            previous["baseline_plan_snapshot"]["plan_hash"] = "f" * 64
        else:
            previous.pop("baseline_source_fingerprint")
        with store.connect() as con:
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(previous), job["id"]))
    before = deepcopy(store.session_record(sid, "owner"))
    expected_error = "Applied owner input snapshot changed" if change == "prompt" else "baseline"
    with pytest.raises(WorkspaceError, match=expected_error):
        store.enqueue(sid, "owner", {"preview_only": True, "revision_id": CANDIDATE})
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("death", [False, True])
def test_published_feature_preview_preserves_absent_acceptance_and_existing_plan_cache(
    tmp_path, monkeypatch, death
):
    fixture = feature_fixture.__wrapped__(tmp_path, monkeypatch)
    service, sid, _ = fixture
    raw_session(service.store, sid, lambda value: value.pop("accepted_revision"))
    identity = acceptance(service.store.session_record(sid, "owner"))
    job, candidate, _ = inline_publication(fixture)
    assert acceptance(service.store.session_record(sid, "owner")) == identity
    if death:
        monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
        assert service.store.claim() is None
    else:
        finish_preview(service.store, sid, job, "error")
    current = service.store.session_record(sid, "owner")
    assert acceptance(current) == identity
    assert current["proposal"]["revision_id"] == candidate["id"]
    assert current["proposal"]["eligible"] is False


def test_focused_publication_refuses_changed_accepted_plan_without_rollback(tmp_path, monkeypatch):
    fixture = feature_fixture.__wrapped__(tmp_path, monkeypatch)
    service, sid, _ = fixture
    job = admitted_job(fixture)
    candidate = built_candidate(fixture, job)
    service.store.update(sid, accepted_operation_plan_snapshot=None)
    newer = authority(service.store, sid)
    with pytest.raises(EditConflictError, match="Acceptance authority changed"):
        service.store.publish_feature_edit(
            sid, "owner", job["id"], job["feature_edit_basis"], candidate["id"]
        )
    assert authority(service.store, sid) == newer
    finish_preview(service.store, sid, job, "needs_review")
    assert authority(service.store, sid) == newer
    with service.store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == "needs_review"
        )
