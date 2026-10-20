"""Continuation ownership and recovery; no model, native work or geometry proof."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_functional_guidance import pending as saved_guidance
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_input_application import checkpoint, claim, create, target
from tests.unit.workspace.test_mesh_handoff import RID

from forge_mind.workspace import delivery_guidance as guidance
from forge_mind.workspace import input_application
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.input_application import FIELD, baseline, pending, source
from forge_mind.workspace.intent_attribution import IntentAttribution, IntentIssue
from forge_mind.workspace.intent_attribution_service import record_proposal, request_text
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.store import Store, WorkspaceError

CONTINUATION = "admitted_input_continuation"
BASE, CANDIDATE, LATEST = (character * 32 for character in "abc")
RAW_ANSWER = "The block is 50 mm wide.\nKeep this literal answer. 中文"


def identity_revision(store, sid, rid, *, binding=None, contract=None):
    """Store a typed identity only; there is no measurement or CAD execution."""
    document = DesignDocument.model_validate(
        {
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "kind": "box",
                    "params": {"x": 10, "y": 8, "z": 4},
                }
            ],
            "outputs": ["body"],
        }
    )
    revision = {"id": rid, "document": document.model_dump(mode="json")}
    if binding is not None:
        revision["operation_record"] = {"plan_snapshot": binding}
    if contract is not None:
        revision["contract"] = contract.model_dump()
    store.add_revision(sid, revision)


def width_plan(width):
    return DesignPlan.model_validate(
        {
            "objective": f"Keep the requested width of {width} mm",
            "operation_plan": {
                "part_ids": ["block"],
                "steps": [
                    {
                        "id": "body",
                        "part_id": "block",
                        "feature_ids": ["body"],
                        "requirement_ids": ["width"],
                        "expected_effect": f"Construct the requested {width} mm block",
                    }
                ],
            },
        }
    )


def ask(store, sid, job, contract):
    return record_proposal(
        store,
        sid,
        "owner",
        job["id"],
        IntentAttribution(
            issues=[
                IntentIssue(
                    id="subject",
                    question="Which subject does the owner mean?",
                    reason="The source needs an owner clarification.",
                    requirement_ids=[contract.requirements[0].id],
                )
            ]
        ),
        contract,
        request_text(store.session_record(sid, "owner")),
        store.edit_basis(sid, "owner"),
    )


def answer(record, raw=RAW_ANSWER):
    return {
        "intent_resolution": {
            "expected_version": record["version"],
            "draft_hash": record["draft_hash"],
            "answers": [{"issue_id": "subject", "answer": raw}],
        }
    }


def paused(tmp_path, *, native=True, preview=False, candidate=False):
    store, sid = create(tmp_path, initial=not candidate, native=native)
    if candidate:
        binding = store.set_design_plan(sid, "owner", width_plan(40), 0)
        identity_revision(store, sid, BASE, binding=binding, contract=target())
        identity_revision(store, sid, CANDIDATE, binding=binding, contract=target())
        store.update(sid, selected_revision=BASE, accepted_revision=BASE)
    job = claim(store, sid, preview_only=preview)
    if candidate:
        store.update(sid, selected_revision=CANDIDATE)
    record = ask(store, sid, job, target())
    checkpoint(store, sid, job, "needs_input")
    return store, sid, record, deepcopy(store.session_record(sid, "owner"))


def payload(store, jid):
    with store.connect() as con:
        return json.loads(con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0])


def state_snapshot(store, sid):
    with store.connect() as con:
        jobs = [tuple(row) for row in con.execute("SELECT * FROM jobs ORDER BY id")]
    return deepcopy(store.session_record(sid, "owner")), jobs


def assert_transfer(store, sid, jid, before):
    current = store.session_record(sid, "owner")
    provenance = payload(store, jid)[CONTINUATION]
    predecessors = {"resume_candidate": before["resume_candidate"]}
    if before.get("proposal") is not None:
        predecessors["proposal"] = before["proposal"]
    assert provenance["version"] == 1
    assert provenance["predecessors"] == predecessors
    assert current.get("resume_candidate") is None
    assert current.get("proposal") is None
    seed, staged = before["resume_candidate"], provenance["target"]
    for key in ("revision_id", "based_on", "contract", "input_version", "source_fingerprint"):
        assert staged[key] == seed[key]
    assert staged.get("plan_snapshot") == seed.get("plan_snapshot")
    assert staged["instructions"] == current["instructions"]
    assert staged[FIELD] == current[FIELD]
    assert staged[FIELD]["pending"] is not None
    assert source({**current, **staged}) == {
        key: staged[FIELD]["pending"][key] for key in ("source_count", "source_hash")
    }
    return deepcopy(provenance)


def interpret_and_apply(store, sid, job, contract):
    session = store.session_record(sid, "owner")
    record_proposal(
        store,
        sid,
        "owner",
        job["id"],
        IntentAttribution(),
        contract,
        request_text(session),
        store.edit_basis(sid, "owner"),
    )
    store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], contract)
    return store.session_record(sid, "owner")


@pytest.mark.parametrize(
    "native,preview,candidate",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
        (False, True, True),
    ],
)
def test_answer_transfers_literal_checkpoint_and_clears_authority_while_job_active(
    tmp_path, native, preview, candidate
):
    store, sid, record, before = paused(
        tmp_path, native=native, preview=preview, candidate=candidate
    )
    original = deepcopy(record)
    jid = store.enqueue(sid, "owner", answer(record))
    provenance = assert_transfer(store, sid, jid, before)
    staged = deepcopy(pending(store.session_record(sid, "owner")))
    job = store.claim()
    assert job and job["id"] == jid
    saved = interpret_and_apply(store, sid, job, target(50))
    assert saved[FIELD]["pending"] is None
    assert saved[FIELD]["applied"]["input_id"] == staged["input_id"]
    assert saved[FIELD]["applied"]["source_hash"] == source(saved)["source_hash"]
    assert saved[FIELD]["applied"]["owner_answers"] == staged["owner_answers"]
    assert staged["owner_answers"][0]["answers"] == answer(record)["intent_resolution"]["answers"]
    assert len(saved["instructions"]) == len(before["resume_candidate"]["instructions"]) + 1
    assert RAW_ANSWER in saved["instructions"][-1]
    assert guidance._input_authority(saved, saved.get("selected_revision"))["status"] == "clear"
    assert payload(store, jid)[CONTINUATION] == provenance
    assert record == original and original in saved["intent_attribution_history"]
    with store.connect() as con:
        assert con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] == "running"
    assert not saved["usage"]
    if not candidate:
        assert not store.revisions(sid)


def test_current_guidance_can_bind_after_answer_during_owned_job(tmp_path, monkeypatch):
    service, sid, *_ = saved_guidance.__wrapped__(tmp_path, monkeypatch)
    store = service.store
    store.update(sid, selected_revision=RID, proposal=None)
    store.update(sid, **{FIELD: baseline(store.session_record(sid, "owner"))})
    original_history = deepcopy(store.session_record(sid, "owner")["delivery_guidance_history"])
    contract = Contract.model_validate(store.session_record(sid, "owner")["contract"])
    first = claim(store, sid)
    record = ask(store, sid, first, contract)
    checkpoint(store, sid, first, "needs_input")
    job = claim(store, sid, **answer(record, "Keep the existing support requirement. 中文"))
    interpret_and_apply(store, sid, job, contract)
    current = guidance.guidance_context(service, sid, "owner", RID)
    assert current["input_authority"]["status"] == "clear"
    request = instructions(current)
    request["instruction_plan"] = {
        "version": 1,
        "fact_refs": current["instruction_catalog"]["required_refs"],
        "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
    }
    saved = guidance.update_guidance(
        service, sid, "owner", guidance.GuidanceUpdate.model_validate(request)
    )
    assert saved["status"] == "current"
    assert saved["instruction_binding"]["status"] == "bound"
    assert saved["physical_readiness"] == "unverified"
    assert saved["history"][:-1] == original_history[RID]
    assert store.session_record(sid, "owner")["accepted_revision"] is None
    with store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == "running"
        )
    assert len(store.revisions(sid)) == 1
    assert not store.session_record(sid, "owner")["usage"]


@pytest.mark.parametrize("rejection", ["stale_question", "after_stage"])
def test_rejected_answer_admission_preserves_checkpoint_and_all_database_writes(
    tmp_path, monkeypatch, rejection
):
    store, sid, record, _ = paused(tmp_path, preview=True)
    request = answer(record)
    if rejection == "stale_question":
        request["intent_resolution"]["expected_version"] += 1
    else:
        original_stage = input_application.stage

        def reject_after_stage(*args, **kwargs):
            original_stage(*args, **kwargs)
            raise WorkspaceError("Fixture rejection after staging")

        monkeypatch.setattr(input_application, "stage", reject_after_stage)
    before = state_snapshot(store, sid)
    with pytest.raises(WorkspaceError):
        store.enqueue(sid, "owner", request)
    assert state_snapshot(store, sid) == before


def test_same_id_with_different_pending_source_is_not_a_consumed_alias(tmp_path):
    store, sid, record, before = paused(tmp_path)
    competing = deepcopy(before["resume_candidate"])
    competing["instructions"] = ["A separate pending owner source."]
    competing[FIELD]["pending"].update(source({**before, **competing}))
    store.update(sid, proposal=competing)
    assert store.edit_basis(sid, "owner").model_dump() == record["basis"]
    assert pending(competing)["input_id"] == pending(before["resume_candidate"])["input_id"]
    assert pending(competing)["source_hash"] != pending(before["resume_candidate"])["source_hash"]
    snapshot = state_snapshot(store, sid)
    with pytest.raises(WorkspaceError):
        store.enqueue(sid, "owner", answer(record))
    assert state_snapshot(store, sid) == snapshot


@pytest.mark.parametrize("value", [None, {"version": 1, "predecessors": {}, "target": {}}])
def test_client_cannot_inject_continuation_provenance(tmp_path, value):
    store, sid, record, _ = paused(tmp_path)
    before = state_snapshot(store, sid)
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, "owner", {**answer(record), CONTINUATION: value})
    assert state_snapshot(store, sid) == before


def test_selection_write_version_cannot_be_replaced_by_caller(tmp_path):
    store, sid = create(tmp_path)
    before = state_snapshot(store, sid)
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.update(sid, selection_version=100, selected_revision=BASE)
    assert state_snapshot(store, sid) == before
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, "owner", {"selection_version": 100})
    assert state_snapshot(store, sid) == before


@pytest.mark.parametrize("native", [False, True])
def test_explicit_same_selection_before_worker_start_retains_owner_selection(tmp_path, native):
    store, sid, record, _ = paused(tmp_path, native=native, preview=True, candidate=True)
    jid = store.enqueue(sid, "owner", answer(record))
    assert store.session_record(sid, "owner")["selected_revision"] == BASE
    # A deliberate same-ID selection is different from no selection yet.
    store.update(sid, selected_revision=BASE)
    store.cancel(sid, "owner")
    saved = store.session_record(sid, "owner")
    assert saved["resume_candidate"]["revision_id"] == BASE
    assert saved["accepted_revision"] == BASE
    assert payload(store, jid)[CONTINUATION]["target"]["revision_id"] == CANDIDATE


@pytest.mark.parametrize("native", [False, True])
def test_queued_answer_cancel_retains_candidate_not_accepted_baseline(tmp_path, native):
    store, sid, record, before = paused(tmp_path, native=native, preview=True, candidate=True)
    jid = store.enqueue(sid, "owner", answer(record))
    provenance = assert_transfer(store, sid, jid, before)
    staged = deepcopy(pending(store.session_record(sid, "owner")))
    assert store.session_record(sid, "owner")["selected_revision"] == BASE
    store.cancel(sid, "owner")
    saved = store.session_record(sid, "owner")
    recovered = deepcopy(saved["resume_candidate"])
    assert saved["selected_revision"] == saved["accepted_revision"] == BASE
    assert recovered["revision_id"] == CANDIDATE
    assert recovered[FIELD]["pending"] == staged
    assert recovered["instructions"] == provenance["target"]["instructions"]
    assert saved["contract"] == before["contract"]
    assert saved[FIELD] == before[FIELD]
    assert payload(store, jid)[CONTINUATION] == provenance
    resumed = claim(store, sid, resume=True, preview_only=True)
    assert resumed["revision_id"] == CANDIDATE
    assert resumed["input_application_id"] == staged["input_id"]
    assert resumed[CONTINUATION]["predecessors"]["resume_candidate"] == recovered
    assert pending(store.session_record(sid, "owner"))["owner_answers"] == staged["owner_answers"]


def test_preview_answer_cancel_and_resume_preserve_distinct_candidate_contract_and_plan(tmp_path):
    store, sid = create(tmp_path)
    baseline_contract, candidate_contract = target(), target(50)
    plan_a = store.set_design_plan(sid, "owner", width_plan(40), 0)
    identity_revision(store, sid, BASE, binding=plan_a, contract=baseline_contract)
    store.update(sid, selected_revision=BASE, accepted_revision=BASE)
    acceptance_keys = (
        "accepted_revision",
        "accepted_operation_plan_snapshot",
        "accepted_operation_plan_revision",
    )
    original = store.session_record(sid, "owner")
    acceptance_before = {key: deepcopy(original[key]) for key in acceptance_keys if key in original}
    first = claim(store, sid, instruction="Make it 50 mm wide.", preview_only=True)
    store.apply_input_contract(
        sid, "owner", first["id"], first["input_application_id"], candidate_contract
    )
    plan_b = store.set_design_plan_for_job(
        sid, "owner", first["id"], width_plan(50), 1, contract=candidate_contract
    )
    assert plan_b["plan_hash"] != plan_a["plan_hash"]
    assert plan_b["contract_hash"] != plan_a["contract_hash"]
    identity_revision(store, sid, CANDIDATE, binding=plan_b, contract=candidate_contract)
    store.update(sid, selected_revision=CANDIDATE)
    record = ask(store, sid, first, candidate_contract)
    checkpoint(store, sid, first, "needs_input")
    paused_state = deepcopy(store.session_record(sid, "owner"))
    seed = paused_state["resume_candidate"]
    assert seed["revision_id"] == CANDIDATE
    assert seed["contract"] == candidate_contract.model_dump()
    assert seed["plan_snapshot"] == plan_b
    assert seed[FIELD]["pending"] is None
    assert paused_state["contract"] == baseline_contract.model_dump()
    assert {
        key: paused_state[key] for key in acceptance_keys if key in paused_state
    } == acceptance_before
    assert first["baseline_plan_snapshot"] == plan_a
    jid = store.enqueue(sid, "owner", answer(record))
    provenance = assert_transfer(store, sid, jid, paused_state)
    staged = deepcopy(provenance["target"])
    assert staged["contract"] == candidate_contract.model_dump()
    assert staged["plan_snapshot"] == plan_b
    assert staged["instructions"] == [
        "Make it 50 mm wide.",
        "User clarification (subject): " + RAW_ANSWER,
    ]
    assert staged[FIELD]["pending"]["owner_answers"][0]["source_hash"] == record["source_hash"]
    store.cancel(sid, "owner")
    cancelled = store.session_record(sid, "owner")
    recovered = deepcopy(cancelled["resume_candidate"])
    for key in ("revision_id", "contract", "plan_snapshot", "instructions", FIELD):
        assert recovered[key] == staged[key]
    assert cancelled["selected_revision"] == cancelled["accepted_revision"] == BASE
    assert cancelled["contract"] == baseline_contract.model_dump()
    assert store.plan_snapshot(sid, "owner") == plan_a
    assert {key: cancelled[key] for key in acceptance_keys if key in cancelled} == acceptance_before
    resumed = claim(store, sid, resume=True, preview_only=True)
    current = store.session_record(sid, "owner")
    assert resumed["revision_id"] == CANDIDATE
    assert resumed["contract"] == candidate_contract.model_dump()
    assert resumed["baseline_contract"] == baseline_contract.model_dump()
    assert resumed["baseline_plan_snapshot"] == plan_a
    assert resumed[CONTINUATION]["predecessors"]["resume_candidate"] == recovered
    assert resumed[CONTINUATION]["target"]["plan_snapshot"] == plan_b
    assert current["contract"] == candidate_contract.model_dump()
    assert store.plan_snapshot(sid, "owner") == plan_b
    assert current["accepted_revision"] == BASE
    assert {key: current[key] for key in acceptance_keys if key in current} == acceptance_before
    assert current["instructions"] == staged["instructions"]
    assert pending(current) == staged[FIELD]["pending"]
    assert len(pending(current)["owner_answers"]) == 1
    assert store.revision(sid, BASE)["contract"] == baseline_contract.model_dump()
    assert store.revision(sid, BASE)["operation_record"]["plan_snapshot"] == plan_a
    assert payload(store, jid)[CONTINUATION] == provenance


@pytest.mark.parametrize("native", [False, True])
def test_death_after_claim_before_candidate_selection_keeps_admitted_target(
    tmp_path, monkeypatch, native
):
    store, sid, record, before = paused(tmp_path, native=native, preview=True, candidate=True)
    job = claim(store, sid, **answer(record))
    assert store.session_record(sid, "owner")["selected_revision"] == BASE
    staged = deepcopy(pending(store.session_record(sid, "owner")))
    monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
    assert store.claim() is None
    saved = store.session_record(sid, "owner")
    assert saved["state"] == "interrupted"
    assert saved["selected_revision"] == saved["accepted_revision"] == BASE
    assert saved["resume_candidate"]["revision_id"] == CANDIDATE
    assert saved["resume_candidate"][FIELD]["pending"] == staged
    assert (
        saved["resume_candidate"]["instructions"]
        == payload(store, job["id"])[CONTINUATION]["target"]["instructions"]
    )
    assert saved["contract"] == before["contract"]
    assert saved[FIELD] == before[FIELD]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("outcome", ["cancelled", "dead"])
def test_explicit_restore_back_to_baseline_is_not_replaced_by_inherited_candidate(
    tmp_path, monkeypatch, native, outcome
):
    store, sid, record, before = paused(tmp_path, native=native, preview=True, candidate=True)
    raw = "Restore the accepted block, retaining its 40 mm target."
    job = claim(store, sid, **answer(record, raw))
    provenance = deepcopy(payload(store, job["id"])[CONTINUATION])
    assert provenance["target"]["revision_id"] == CANDIDATE
    # The real author loop loads B, can explicitly restore A, then saves A again at finalization.
    store.update(sid, selected_revision=CANDIDATE)
    applied = deepcopy(interpret_and_apply(store, sid, job, target())[FIELD])
    store.update(sid, selected_revision=BASE)
    store.update(sid, selected_revision=BASE)
    if outcome == "dead":
        monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
        assert store.claim() is None
    else:
        store.cancel(sid, "owner")
        checkpoint(store, sid, job, outcome)
    saved = store.session_record(sid, "owner")
    recovered = deepcopy(saved["resume_candidate"])
    assert recovered["revision_id"] == BASE
    assert recovered["contract"] == target().model_dump()
    assert recovered[FIELD] == applied
    assert recovered["instructions"] == provenance["target"]["instructions"]
    assert raw in recovered["instructions"][-1]
    assert saved["selected_revision"] == saved["accepted_revision"] == BASE
    assert saved["contract"] == before["contract"]
    assert saved[FIELD] == before[FIELD]
    assert payload(store, job["id"])[CONTINUATION] == provenance
    resumed = claim(store, sid, resume=True, preview_only=True)
    assert resumed["revision_id"] == BASE
    assert resumed["input_application_id"] == applied["applied"]["input_id"]
    assert resumed[CONTINUATION]["predecessors"]["resume_candidate"] == recovered


@pytest.mark.parametrize("native,preview", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("outcome", ["cancelled", "budget_exhausted", "dead"])
def test_running_recovery_keeps_latest_selected_candidate_and_applied_answer(
    tmp_path, monkeypatch, native, preview, outcome
):
    store, sid, record, before = paused(tmp_path, native=native, preview=preview, candidate=True)
    job = claim(store, sid, **answer(record))
    provenance = deepcopy(payload(store, job["id"])[CONTINUATION])
    applied = interpret_and_apply(store, sid, job, target(50))[FIELD]
    identity_revision(store, sid, LATEST)
    store.update(sid, selected_revision=LATEST)
    if outcome == "dead":
        monkeypatch.setattr(Store, "_owner_dead", staticmethod(lambda row: True))
        assert store.claim() is None
    else:
        if outcome == "cancelled":
            store.cancel(sid, "owner")
        checkpoint(store, sid, job, outcome)
    saved = store.session_record(sid, "owner")
    recovered = deepcopy(saved["resume_candidate"])
    assert recovered["revision_id"] == LATEST
    assert recovered[FIELD] == applied and recovered[FIELD]["pending"] is None
    assert recovered["contract"] == target(50).model_dump()
    assert recovered["instructions"] == provenance["target"]["instructions"]
    assert saved["accepted_revision"] == BASE
    assert saved["selected_revision"] == (BASE if preview else LATEST)
    if preview:
        assert saved["contract"] == before["contract"]
        assert saved[FIELD] == before[FIELD]
    assert payload(store, job["id"])[CONTINUATION] == provenance
    resumed = claim(store, sid, resume=True, preview_only=preview)
    assert resumed["revision_id"] == LATEST
    assert resumed["input_application_id"] == applied["applied"]["input_id"]
    before_repeat = state_snapshot(store, sid)
    store.apply_input_contract(
        sid, "owner", resumed["id"], resumed["input_application_id"], target(50)
    )
    store.apply_input_contract(
        sid, "owner", resumed["id"], resumed["input_application_id"], target(50)
    )
    assert state_snapshot(store, sid) == before_repeat


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_explicit_pending_resume_transfers_once_and_application_is_idempotent(
    tmp_path, native, preview
):
    store, sid = create(tmp_path, native=native)
    first = claim(store, sid, instruction="Make it 50 mm wide.", preview_only=preview)
    seed = deepcopy(checkpoint(store, sid, first))
    before = deepcopy(store.session_record(sid, "owner"))
    resumed = claim(store, sid, resume=True, preview_only=preview)
    provenance = assert_transfer(store, sid, resumed["id"], before)
    assert provenance["predecessors"]["resume_candidate"] == seed
    assert resumed["input_application_id"] == seed[FIELD]["pending"]["input_id"]
    store.apply_input_contract(
        sid, "owner", resumed["id"], resumed["input_application_id"], target(50)
    )
    after_first = state_snapshot(store, sid)
    store.apply_input_contract(
        sid, "owner", resumed["id"], resumed["input_application_id"], target(50)
    )
    assert state_snapshot(store, sid) == after_first
    saved = store.session_record(sid, "owner")
    assert saved["instructions"] == ["Make it 50 mm wide."]
    assert guidance._input_authority(saved, None)["status"] == "clear"
    assert payload(store, resumed["id"])[CONTINUATION] == provenance
    assert not store.revisions(sid)
    assert not saved["usage"]
