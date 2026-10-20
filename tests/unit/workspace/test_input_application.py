"""Input consumption recovery; no provider, native worker, or CAD execution."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.input_application import FIELD, SERVER_FIELDS, baseline, pending
from forge_mind.workspace.intent_attribution import IntentAttribution, IntentIssue
from forge_mind.workspace.intent_attribution_service import record_proposal, request_text
from forge_mind.workspace.models import Contract, ContractEdit, DesignPlan, Requirement
from forge_mind.workspace.store import Store, WorkspaceError

OWNER = "owner"
PROMPT = "Make a block 40 mm wide."
INSTRUCTION = "Make it 50 mm wide."


def target(width=40, *, source="explicit"):
    return Contract(
        requirements=[
            Requirement(
                id="width",
                text=f"Width {width} mm",
                source_quote=f"{width} mm",
                source=source,
                metric="extent_x",
                value=width,
                position_tolerance=0.001,
            )
        ]
    )


def create(tmp_path, *, initial=False, native=True):
    store = Store(tmp_path)
    sid = store.create(
        OWNER,
        PROMPT,
        None if initial else target().model_dump(),
        authoring_mode="native_operations" if native else None,
    )["id"]
    return store, sid


def claim(store, sid, **payload):
    jid = store.enqueue(sid, OWNER, payload)
    job = store.claim()
    assert job and job["id"] == jid
    return job


def checkpoint(store, sid, job, state="budget_exhausted"):
    store.finish(job["id"], sid, state)
    if job.get("preview_only"):
        store.complete_preview(sid, job["id"], job.get("based_on"), None)
    return store.session_record(sid, OWNER)["resume_candidate"]


def question(store, sid, job):
    return record_proposal(
        store,
        sid,
        OWNER,
        job["id"],
        IntentAttribution(
            issues=[
                IntentIssue(
                    id="width_subject",
                    question="Which body has this width?",
                    reason="The subject is unclear.",
                    requirement_ids=["width"],
                )
            ]
        ),
        target(),
        request_text(store.session_record(sid, OWNER)),
        store.edit_basis(sid, OWNER),
    )


def answer(record):
    return {
        "intent_resolution": {
            "expected_version": record["version"],
            "draft_hash": record["draft_hash"],
            "answers": [{"issue_id": "width_subject", "answer": "The block is 50 mm wide."}],
        }
    }


def test_admission_stages_once_before_worker_and_resume_preserves_identity(tmp_path):
    store, sid = create(tmp_path)
    jid = store.enqueue(sid, OWNER, {"instruction": INSTRUCTION})
    saved = store.session_record(sid, OWNER)
    original = deepcopy(pending(saved))
    assert saved["contract"] == target().model_dump()
    assert saved["instructions"] == [INSTRUCTION]
    assert original["mode"] == "delta"
    job = store.claim()
    assert job and job["id"] == jid
    interrupted = checkpoint(store, sid, job)
    assert interrupted[FIELD]["pending"] == original
    resumed = claim(store, sid, resume=True)
    assert resumed["input_application_id"] == original["input_id"]
    assert resumed["input_instructions"] == [INSTRUCTION]
    assert "instruction" not in resumed
    assert store.session_record(sid, OWNER)["instructions"] == [INSTRUCTION]


def test_dead_worker_before_inference_retains_pending_input(tmp_path, monkeypatch):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    identity = job["input_application_id"]
    monkeypatch.setattr(store, "_owner_dead", lambda row: True)
    assert store.claim() is None
    saved = store.session_record(sid, OWNER)
    assert saved["resume_candidate"][FIELD]["pending"]["input_id"] == identity
    resumed = claim(store, sid, resume=True)
    assert resumed["input_application_id"] == identity


@pytest.mark.parametrize("native", [False, True])
def test_resumed_delta_runs_contract_edit_before_authoring_without_replaying_cad(
    tmp_path, monkeypatch, native
):
    store, sid = create(tmp_path, native=native)
    job = claim(store, sid, instruction=INSTRUCTION)
    checkpoint(store, sid, job)
    resumed = claim(store, sid, resume=True, max_actions=0, max_seconds=60)
    seen = []

    class Client:
        model = "scripted-input-recovery"

        def generate_structured(self, **kwargs):
            seen.append(kwargs)
            assert kwargs["output_model"] is ContractEdit
            return ContractEdit.model_validate(target(50).model_dump()), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    state = service._run_agent_job(resumed)
    assert state != "error"
    saved = store.session_record(sid, OWNER)
    assert len(seen) == 1 and INSTRUCTION in seen[0]["user"]
    assert saved["contract"] == target(50, source="request").model_dump()
    assert pending(saved) is None
    assert saved[FIELD]["applied"]["input_id"] == resumed["input_application_id"]
    assert saved["instructions"] == [INSTRUCTION]
    assert not store.revisions(sid)
    checkpoint(store, sid, resumed)
    again = claim(store, sid, resume=True)
    assert again["input_application_id"] == resumed["input_application_id"]
    assert pending(store.session_record(sid, OWNER)) is None
    assert store.session_record(sid, OWNER)["instructions"] == [INSTRUCTION]


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_owner_answer_identity_survives_interruption_and_new_job(tmp_path, initial, preview):
    store, sid = create(tmp_path, initial=initial)
    job = claim(store, sid, preview_only=preview)
    record = question(store, sid, job)
    original = deepcopy(record)
    checkpoint(store, sid, job, "needs_input")
    resolved = claim(store, sid, **answer(record))
    item = deepcopy(pending(store.session_record(sid, OWNER)))
    assert item["mode"] == ("full" if initial else "delta")
    owner_binding = item["owner_answers"][0]
    assert owner_binding["question_version"] == record["version"]
    assert owner_binding["question_draft_hash"] == record["draft_hash"]
    assert owner_binding["answers"] == answer(record)["intent_resolution"]["answers"]
    checkpoint(store, sid, resolved)
    resumed = claim(store, sid, resume=True, preview_only=preview)
    restored = pending(store.session_record(sid, OWNER))
    assert restored["input_id"] == item["input_id"] == resumed["input_application_id"]
    assert restored["owner_answers"] == item["owner_answers"]
    assert len(store.session_record(sid, OWNER)["instructions"]) == 1
    assert record == original


@pytest.mark.parametrize("change", ["source", "target", "plan", "input_id", "cancel", "owner"])
def test_late_or_stale_result_cannot_publish_target_or_applied_cursor(tmp_path, change):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    if change == "source":
        store.update(sid, instructions=["Changed history"])
    elif change == "target":
        store.update(sid, contract=target(60).model_dump())
    elif change == "plan":
        store.update(sid, plan_version=1, plan_hash="f" * 64)
    elif change == "cancel":
        store.cancel(sid, OWNER)
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError):
        store.apply_input_contract(
            sid,
            "other-owner" if change == "owner" else OWNER,
            job["id"],
            "f" * 32 if change == "input_id" else job["input_application_id"],
            target(50),
        )
    assert store.session_record(sid, OWNER) == before


def test_pending_input_refuses_competing_edits_and_keeps_history(tmp_path):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    checkpoint(store, sid, job)
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError, match="pending owner input"):
        store.enqueue(sid, OWNER, {"instruction": "Make it 60 mm wide."})
    assert store.session_record(sid, OWNER) == before


@pytest.mark.parametrize("field", sorted(SERVER_FIELDS))
def test_application_metadata_is_server_owned(tmp_path, field):
    store, sid = create(tmp_path)
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, OWNER, {field: {}})
    assert store.session_record(sid, OWNER)["job_id"] is None


def test_preview_restores_baseline_and_acceptance_promotes_candidate_binding(tmp_path):
    store, sid = create(tmp_path, native=False)
    job = claim(store, sid, preview_only=True, instruction=INSTRUCTION)
    store.apply_input_contract(sid, OWNER, job["id"], job["input_application_id"], target(50))
    candidate = store.session_record(sid, OWNER)
    rid = "b" * 32
    store.finish(job["id"], sid, "needs_review")
    store.complete_preview(
        sid,
        job["id"],
        None,
        {
            "revision_id": rid,
            "based_on": None,
            "input_version": 0,
            "eligible": True,
            "contract": candidate["contract"],
            "instructions": candidate["instructions"],
        },
    )
    restored = store.session_record(sid, OWNER)
    assert restored["contract"] == target().model_dump()
    assert restored["instructions"] == []
    assert restored[FIELD]["applied"]["source_count"] == 0
    assert restored["proposal"][FIELD] == candidate[FIELD]
    store.accept_proposal(sid, OWNER, rid, None, 0)
    accepted = store.session_record(sid, OWNER)
    assert accepted["contract"] == candidate["contract"]
    assert accepted["instructions"] == candidate["instructions"]
    assert accepted[FIELD] == candidate[FIELD]


@pytest.mark.parametrize("supplied", [False, True])
def test_supplied_target_origin_differs_from_inherited_candidate(tmp_path, supplied):
    store, sid = create(tmp_path, native=False)
    rid = "b" * 32
    candidate = {"prompt": PROMPT, "contract": target(50).model_dump(), "instructions": []}
    store.update(
        sid,
        proposal={"revision_id": rid, **candidate, FIELD: baseline(candidate)},
    )
    payload = {"preview_only": True, "revision_id": rid, "instruction": "Inspect the block."}
    if supplied:
        payload["contract"] = target(60).model_dump()
    job = claim(store, sid, **payload)
    item = pending(store.session_record(sid, OWNER))
    assert item["mode"] == ("supplied" if supplied else "delta")
    assert item["origin"] == ("supplied" if supplied else "inherited")
    assert item["base_contract_hash"] == target(50).fingerprint()
    assert job["contract"] == target(60 if supplied else 50).model_dump()


def test_ambiguous_legacy_checkpoint_cannot_assume_history_was_applied(tmp_path):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    snapshot = checkpoint(store, sid, job)
    snapshot.pop(FIELD)
    store.update(sid, resume_candidate=snapshot, contract_application=None)
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError, match="Legacy interrupted owner input"):
        store.enqueue(sid, OWNER, {"resume": True})
    assert store.session_record(sid, OWNER) == before


def test_noop_application_is_atomic_idempotent_and_keeps_contract_serialization(tmp_path):
    store, sid = create(tmp_path)
    canonical = target().model_dump()
    job = claim(store, sid, instruction="Inspect the block.")
    for _ in range(2):
        store.apply_input_contract(sid, OWNER, job["id"], job["input_application_id"], target())
    saved = store.session_record(sid, OWNER)
    assert saved["contract"] == canonical
    assert Contract.model_validate(saved["contract"]).fingerprint() == target().fingerprint()
    assert pending(saved) is None
    assert saved[FIELD]["applied"]["source_count"] == 1


def test_initial_full_input_advances_only_with_canonical_plan_admission(tmp_path):
    store, sid = create(tmp_path, initial=True)
    job = claim(store, sid)
    record_proposal(
        store,
        sid,
        OWNER,
        job["id"],
        IntentAttribution(),
        target(),
        PROMPT,
        store.edit_basis(sid, OWNER),
    )
    assert store.session_record(sid, OWNER)["contract"] is None
    assert pending(store.session_record(sid, OWNER))["mode"] == "full"
    plan = DesignPlan.model_validate(
        {
            "objective": "Make the requested block",
            "operation_plan": {
                "part_ids": ["block"],
                "steps": [
                    {
                        "id": "body",
                        "part_id": "block",
                        "feature_ids": ["body"],
                        "requirement_ids": ["width"],
                        "expected_effect": "Make the block body",
                    }
                ],
            },
        }
    )
    store.set_design_plan_for_job(
        sid, OWNER, job["id"], plan, expected_plan_version=0, contract=target()
    )
    saved = store.session_record(sid, OWNER)
    assert saved["contract"] == target().model_dump()
    assert pending(saved) is None
    assert saved[FIELD]["applied"]["input_id"] == job["input_application_id"]
    assert not store.revisions(sid)


@pytest.mark.parametrize("preview", [False, True])
def test_queued_cancellation_checkpoints_input_and_restores_preview_baseline(tmp_path, preview):
    store, sid = create(tmp_path)
    store.enqueue(sid, OWNER, {"instruction": INSTRUCTION, "preview_only": preview})
    identity = pending(store.session_record(sid, OWNER))["input_id"]
    store.cancel(sid, OWNER)
    saved = store.session_record(sid, OWNER)
    assert saved["resume_candidate"][FIELD]["pending"]["input_id"] == identity
    assert saved["instructions"] == ([] if preview else [INSTRUCTION])
    resumed = claim(store, sid, resume=True, preview_only=preview)
    assert resumed["input_application_id"] == identity
    assert store.session_record(sid, OWNER)["instructions"] == [INSTRUCTION]


def test_supplied_pending_target_survives_resume_without_becoming_its_old_base(tmp_path):
    store, sid = create(tmp_path, native=False)
    job = claim(store, sid, contract=target(60).model_dump())
    checkpoint(store, sid, job)
    resumed = claim(store, sid, resume=True)
    assert resumed["contract"] == target(60).model_dump()
    store.apply_input_contract(
        sid, OWNER, resumed["id"], resumed["input_application_id"], target(60)
    )
    assert store.session_record(sid, OWNER)["contract"] == target(60).model_dump()


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_explicit_abandon_retains_raw_input_and_allows_new_edit(tmp_path, initial, preview):
    store, sid = create(tmp_path, initial=initial)
    job = claim(store, sid, instruction=INSTRUCTION, preview_only=preview)
    checkpoint(store, sid, job)
    before = store.session_record(sid, OWNER)
    snapshot = deepcopy(before["resume_candidate"])
    identity = snapshot[FIELD]["pending"]["input_id"]
    receipt = store.abandon_input(
        sid,
        OWNER,
        identity,
        store.edit_basis(sid, OWNER),
        snapshot[FIELD]["pending"]["source_hash"],
    )
    saved = store.session_record(sid, OWNER)
    assert receipt == {"input_id": identity, "status": "abandoned", "retained_history": True}
    assert saved["contract"] == before["contract"]
    assert saved["selected_revision"] == before["selected_revision"]
    assert saved["instructions"] == []
    assert pending(saved) is None and saved["resume_candidate"] is None
    archive = saved["abandoned_input_applications"][0]
    assert archive["snapshot"]["instructions"] == [INSTRUCTION]
    assert archive["snapshot"][FIELD] == snapshot[FIELD]
    assert not store.revisions(sid)
    next_job = claim(store, sid, instruction="Make it 60 mm wide.")
    assert next_job["input_application_id"] != identity


def test_abandon_question_preserves_raw_question_and_owner_resolution_history(tmp_path):
    store, sid = create(tmp_path, initial=True)
    job = claim(store, sid)
    record = question(store, sid, job)
    checkpoint(store, sid, job, "needs_input")
    resolved = claim(store, sid, **answer(record))
    snapshot = checkpoint(store, sid, resolved)
    identity = snapshot[FIELD]["pending"]["input_id"]
    answered = deepcopy(store.session_record(sid, OWNER)["intent_attribution"])
    store.abandon_input(
        sid,
        OWNER,
        identity,
        store.edit_basis(sid, OWNER),
        snapshot[FIELD]["pending"]["source_hash"],
    )
    saved = store.session_record(sid, OWNER)
    assert "intent_attribution" not in saved
    assert answered in saved["intent_attribution_history"]
    assert record in saved["intent_attribution_history"]
    assert saved["abandoned_input_applications"][0]["intent_attribution"] == answered
    assert (
        saved["abandoned_input_applications"][0]["snapshot"][FIELD]["pending"]["owner_answers"]
        == snapshot[FIELD]["pending"]["owner_answers"]
    )


@pytest.mark.parametrize("checkpoint_present", [False, True])
def test_legacy_second_resume_payload_does_not_prove_owner_history_was_applied(
    tmp_path, checkpoint_present
):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    snapshot = checkpoint(store, sid, job)
    snapshot.pop(FIELD)
    store.update(
        sid, resume_candidate=snapshot if checkpoint_present else None, contract_application=None
    )
    with store.connect() as con:
        con.execute(
            "UPDATE jobs SET payload=? WHERE id=?",
            (
                json.dumps(
                    {
                        "resume": True,
                        "contract": target().model_dump(),
                        "resume_instructions": [INSTRUCTION],
                    }
                ),
                job["id"],
            ),
        )
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError, match="Legacy interrupted owner input"):
        store.enqueue(sid, OWNER, {"resume": True})
    assert store.session_record(sid, OWNER) == before


def test_stale_pre_answer_abandon_request_cannot_discard_the_newer_answer(tmp_path):
    store, sid = create(tmp_path, initial=True)
    job = claim(store, sid)
    record = question(store, sid, job)
    old = checkpoint(store, sid, job, "needs_input")
    old_basis = store.edit_basis(sid, OWNER)
    identity = old[FIELD]["pending"]["input_id"]
    resolved = claim(store, sid, **answer(record))
    current = checkpoint(store, sid, resolved)
    assert current[FIELD]["pending"]["input_id"] == identity
    assert store.edit_basis(sid, OWNER) == old_basis
    assert current[FIELD]["pending"]["source_hash"] != old[FIELD]["pending"]["source_hash"]
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError, match="source changed"):
        store.abandon_input(sid, OWNER, identity, old_basis, old[FIELD]["pending"]["source_hash"])
    assert store.session_record(sid, OWNER) == before


@pytest.mark.parametrize("limited", [False, True])
def test_late_success_keeps_pending_but_unlimited_work_can_apply(tmp_path, monkeypatch, limited):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION, max_actions=0, max_seconds=None)
    now = [1000.0]
    monkeypatch.setattr("forge_mind.workspace.agent.time.monotonic", lambda: now[0])
    job["author_deadline"] = now[0] + 1 if limited else None

    class Client:
        model = "scripted-late-input"

        def generate_structured(self, **kwargs):
            assert kwargs["output_model"] is ContractEdit
            now[0] += 10
            return ContractEdit.model_validate(target(50).model_dump()), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    state = service._run_agent_job(job)
    saved = store.session_record(sid, OWNER)
    assert len(saved["usage"]) == 1
    expected = target() if limited else target(50, source="request")
    assert saved["contract"] == expected.model_dump()
    if limited:
        assert state == "budget_exhausted"
        assert pending(saved)["input_id"] == job["input_application_id"]
        assert saved[FIELD]["applied"]["source_count"] == 0
        checkpoint(store, sid, job)
        resumed = claim(store, sid, resume=True)
        assert resumed["input_application_id"] == job["input_application_id"]
    else:
        assert pending(saved) is None


def test_typed_candidate_target_carries_its_own_history_and_application(tmp_path):
    store, sid = create(tmp_path, native=False)
    rid = "b" * 32
    candidate = {
        "prompt": PROMPT,
        "contract": target(50).model_dump(),
        "instructions": [INSTRUCTION],
    }
    application = baseline(candidate)
    store.update(sid, proposal={"revision_id": rid, **candidate, FIELD: application})
    job = claim(store, sid, preview_only=True, revision_id=rid, document={})
    assert job["target_contract_application"] == application
    assert job["target_instructions"] == [INSTRUCTION]
    assert (
        job["baseline_contract_application"]["applied"]["contract_hash"] == target().fingerprint()
    )
    # This is the sanctioned write used by the real typed-document worker.
    store.update(sid, contract=target(50).model_dump())
    saved = store.session_record(sid, OWNER)
    assert saved[FIELD] == application
    assert saved["instructions"] == [INSTRUCTION]


@pytest.mark.parametrize("replacement", [{"instruction": INSTRUCTION}, {"contract": {}}])
def test_resume_cannot_replace_its_saved_input_or_target(tmp_path, replacement):
    store, sid = create(tmp_path)
    job = claim(store, sid, instruction=INSTRUCTION)
    checkpoint(store, sid, job)
    with pytest.raises(WorkspaceError, match="Resume cannot replace"):
        store.enqueue(sid, OWNER, {"resume": True, **replacement})


def test_legacy_empty_history_checkpoint_keeps_its_distinct_compatibility_boundary(tmp_path):
    store, sid = create(tmp_path, native=False)
    job = claim(store, sid)
    store.finish(job["id"], sid, "error")
    source_fingerprint = store.get(sid, OWNER)["source_fingerprint"]
    store.update(
        sid,
        contract_application=None,
        resume_candidate={
            "revision_id": None,
            "based_on": None,
            "contract": target().model_dump(),
            "instructions": [],
            "input_version": 0,
            "source_fingerprint": source_fingerprint,
        },
    )
    resumed = claim(store, sid, resume=True)
    assert resumed["contract"] == target().model_dump()
    assert pending(store.session_record(sid, OWNER)) is None
    assert store.session_record(sid, OWNER)[FIELD]["applied"]["source_count"] == 0
