"""Pure saved-state admission checks; no model, CAD, or worker execution."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.document_service import finish_preview
from forge_mind.workspace.intent_attribution import IntentAttribution, IntentIssue
from forge_mind.workspace.intent_attribution_service import (
    read_attribution,
    record_conflict,
    record_proposal,
    request_text,
)
from forge_mind.workspace.job_wait import wait_for_job
from forge_mind.workspace.models import Contract, DesignPlan, Measurement, Requirement
from forge_mind.workspace.project_access import project_overview
from forge_mind.workspace.store import Store, WorkspaceError

OWNER = "alice"
PROMPT = "Make a carrier for a sample. Its body is 45 mm wide."
BASE = "a" * 32
DRAFT = "b" * 32


def target():
    return Contract(
        requirements=[
            Requirement(id="width", text="The body is 45 mm wide", source_quote="45 mm wide")
        ]
    )


def attribution():
    return IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "carrier",
                    "label": "Carrier",
                    "role": "designed",
                    "source_quote": "carrier",
                },
                {
                    "id": "sample",
                    "label": "Sample",
                    "role": "reference",
                    "source_quote": "sample",
                },
            ],
            "facts": [
                {
                    "id": "quoted_width",
                    "candidate_subject_ids": ["carrier", "sample"],
                    "property": "width",
                    "value": 45.0,
                    "unit": "mm",
                    "source_quote": "45 mm wide",
                    "requirement_ids": ["width"],
                }
            ],
            "issues": [
                {
                    "id": "width_subject",
                    "question": "Which body has this width?",
                    "reason": "The pronoun has two possible subjects.",
                    "fact_ids": ["quoted_width"],
                    "requirement_ids": ["width"],
                }
            ],
        }
    )


def plan():
    return DesignPlan.model_validate(
        {
            "objective": "Make the carrier with separate reference intent",
            "operation_plan": {
                "part_ids": ["carrier"],
                "steps": [
                    {
                        "id": "body",
                        "part_id": "carrier",
                        "feature_ids": ["body"],
                        "requirement_ids": ["width"],
                        "expected_effect": "Construct the carrier body",
                    }
                ],
            },
        }
    )


def saved_revision(store, sid, rid, binding):
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
    measurement = Measurement(
        revision_id=rid,
        contract_hash=target().fingerprint(),
        engine_fingerprint="f" * 64,
        artifact_hashes={},
        geometry_valid=False,
    )
    store.add_revision(
        sid,
        {
            "id": rid,
            "document": document.model_dump(mode="json"),
            "contract": target().model_dump(),
            "measurement": measurement.model_dump(),
            "operation_record": {"plan_snapshot": binding},
        },
    )


def pending(tmp_path, *, preview=False, geometry=False, delivery_requested=None):
    store = Store(tmp_path)
    sid = store.create(
        OWNER,
        PROMPT,
        target().model_dump() if geometry else None,
        authoring_mode="native_operations",
    )["id"]
    if geometry:
        # This fixture represents already-consumed history on its accepted base.
        store.update(sid, instructions=["Keep both subjects distinct."])
        binding = store.set_design_plan(sid, OWNER, plan(), expected_plan_version=0)
        saved_revision(store, sid, BASE, binding)
        store.update(sid, selected_revision=BASE, accepted_revision=BASE)
    payload = {"preview_only": preview}
    if delivery_requested is not None:
        payload["delivery_requested"] = delivery_requested
    jid = store.enqueue(sid, OWNER, payload)
    job = store.claim()
    assert job and job["id"] == jid
    if geometry:
        saved_revision(store, sid, DRAFT, binding)
        store.update(sid, selected_revision=DRAFT)
    record = record_proposal(
        store,
        sid,
        OWNER,
        jid,
        attribution(),
        target(),
        request_text(store.session_record(sid, OWNER)),
        store.edit_basis(sid, OWNER),
    )
    return store, sid, job, record


def resolution(record, answer="The width belongs to the sample."):
    return {
        "expected_version": record["version"],
        "draft_hash": record["draft_hash"],
        "answers": [{"issue_id": "width_subject", "answer": answer}],
    }


def terminal_pending(tmp_path, **kwargs):
    store, sid, job, record = pending(tmp_path, **kwargs)
    store.finish(job["id"], sid, "needs_input")
    if job.get("preview_only"):
        store.complete_preview(sid, job["id"], job.get("based_on"), None)
    return store, sid, job, record


def test_exact_numeric_quote_with_unresolved_subject_blocks_construction(tmp_path):
    store, sid, job, record = pending(tmp_path)
    assert record["status"] == "needs_input"
    assert record["draft_hash"] == digest(
        {key: value for key, value in record.items() if key != "draft_hash"}
    )
    assert record["attribution"]["facts"][0]["value"] == 45.0
    assert record["attribution"]["facts"][0]["unit"] == "mm"
    with pytest.raises(WorkspaceError, match="clarification"):
        store.set_design_plan_for_job(
            sid, OWNER, job["id"], plan(), expected_plan_version=0, contract=target()
        )
    store.finish(job["id"], sid, "needs_input")
    with pytest.raises(WorkspaceError, match="clarification"):
        store.enqueue(sid, OWNER, {"instruction": "Just proceed"})
    saved = store.session_record(sid, OWNER)
    assert saved["contract"] is None
    assert not saved.get("design_plan")
    assert not store.revisions(sid)


@pytest.mark.parametrize("invalid", ["owner", "job", "worker", "basis", "source"])
def test_proposal_authority_is_checked_before_any_sidecar_write(tmp_path, invalid):
    store = Store(tmp_path)
    sid = store.create(OWNER, PROMPT, None, authoring_mode="native_operations")["id"]
    jid = store.enqueue(sid, OWNER, {})
    assert store.claim()
    basis = store.edit_basis(sid, OWNER)
    if invalid == "worker":
        with store.connect() as con:
            con.execute("UPDATE jobs SET worker_pid=-1 WHERE id=?", (jid,))
    if invalid == "basis":
        basis = basis.model_copy(update={"input_version": 1})
    with pytest.raises(WorkspaceError):
        record_proposal(
            store,
            sid,
            "bob" if invalid == "owner" else OWNER,
            "f" * 32 if invalid == "job" else jid,
            attribution(),
            target(),
            PROMPT + " changed" if invalid == "source" else PROMPT,
            basis,
        )
    assert read_attribution(store, sid, OWNER)["record"] is None


def test_author_cannot_erase_questions_or_supply_an_owner_answer(tmp_path):
    store, sid, job, record = pending(tmp_path)
    clear = IntentAttribution()
    with pytest.raises(WorkspaceError, match="clarification"):
        record_proposal(
            store, sid, OWNER, job["id"], clear, target(), PROMPT, store.edit_basis(sid, OWNER)
        )
    with pytest.raises(WorkspaceError, match="active job"):
        store.enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
    assert read_attribution(store, sid, OWNER)["record"] == record


def test_owner_answer_admits_one_job_and_preserves_raw_text_and_contract(tmp_path):
    store, sid, _, record = terminal_pending(tmp_path)
    answer = "  Sample only.\nKeep 45 mm unchanged. \u03b1  "
    jid = store.enqueue(sid, OWNER, {"intent_resolution": resolution(record, answer)})
    saved = store.session_record(sid, OWNER)
    answered = read_attribution(store, sid, OWNER)["record"]
    assert saved["job_id"] == jid
    assert saved["state"] == "queued"
    assert saved["contract"] is None
    assert not saved.get("design_plan")
    assert answered["status"] == "answered"
    assert answered["resolutions"] == [{"issue_id": "width_subject", "answer": answer}]
    assert answered["proposed_contract"] == record["proposed_contract"]
    assert answered["attribution"] == record["attribution"]
    assert saved["intent_attribution_history"] == [record]
    queued = store.claim()
    assert queued and queued["id"] == jid
    assert "intent_resolution" not in queued
    assert queued["instruction"] == "User clarification (width_subject): " + answer
    assert record["attribution"]["issues"][0]["question"] not in queued["instruction"]
    # Answer receipt alone does not authorize an unchanged model interpretation.
    with pytest.raises(WorkspaceError, match="request"):
        record_proposal(
            store,
            sid,
            OWNER,
            jid,
            IntentAttribution(),
            target(),
            PROMPT,
            store.edit_basis(sid, OWNER),
        )
    store.update(sid, instructions=[queued["instruction"]])
    interpreted = record_proposal(
        store,
        sid,
        OWNER,
        jid,
        IntentAttribution(),
        target(),
        request_text(store.session_record(sid, OWNER)),
        store.edit_basis(sid, OWNER),
    )
    assert interpreted["status"] == "interpreted"
    assert interpreted["version"] == 3
    assert store.session_record(sid, OWNER)["contract"] is None


@pytest.mark.parametrize(
    "invalid", ["owner", "version", "hash", "missing", "extra", "duplicate", "source", "contract"]
)
def test_stale_or_incomplete_resolution_never_consumes_questions(tmp_path, invalid):
    store, sid, job, record = terminal_pending(tmp_path)
    supplied = resolution(record)
    if invalid == "version":
        supplied["expected_version"] += 1
    elif invalid == "hash":
        supplied["draft_hash"] = "0" * 64
    elif invalid == "missing":
        supplied["answers"] = []
    elif invalid == "extra":
        supplied["answers"].append({"issue_id": "other", "answer": "Extra"})
    elif invalid == "duplicate":
        supplied["answers"] *= 2
    elif invalid == "source":
        store.update(sid, instructions=["New source"])
    elif invalid == "contract":
        store.update(sid, contract=Contract(coverage_notes=["Changed target"]).model_dump())
    before = store.session_record(sid, OWNER)
    with pytest.raises(ValueError):
        store.enqueue(sid, "bob" if invalid == "owner" else OWNER, {"intent_resolution": supplied})
    assert store.session_record(sid, OWNER) == before
    assert read_attribution(store, sid, OWNER)["record"] == record
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs WHERE session=?", (sid,)).fetchone()[0] == 1
        assert (
            con.execute("SELECT state FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == "needs_input"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("instruction", "Proceed"),
        ("contract", {}),
        ("document", {}),
        ("resume", True),
        ("delivery_requested", True),
    ],
)
def test_resolution_cannot_smuggle_another_operation(tmp_path, field, value):
    store, sid, _, record = terminal_pending(tmp_path)
    with pytest.raises(WorkspaceError, match="competing"):
        store.enqueue(sid, OWNER, {"intent_resolution": resolution(record), field: value})
    assert read_attribution(store, sid, OWNER)["record"] == record


def test_queue_failure_rolls_back_the_answer_record(tmp_path):
    store, sid, _, record = terminal_pending(tmp_path)
    for index in range(16):
        other = store.create(OWNER, f"Other request {index}", None)["id"]
        store.enqueue(other, OWNER, {})
    before = store.session_record(sid, OWNER)
    with pytest.raises(WorkspaceError, match="queue is full"):
        store.enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
    assert store.session_record(sid, OWNER) == before
    assert read_attribution(store, sid, OWNER)["record"] == record


def test_two_owner_answers_cannot_queue_two_jobs(tmp_path):
    store, sid, _, record = terminal_pending(tmp_path)

    def admit(_):
        try:
            return Store(tmp_path).enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
        except WorkspaceError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        admitted = list(pool.map(admit, range(2)))
    assert sum(jid is not None for jid in admitted) == 1
    assert read_attribution(store, sid, OWNER)["record"]["status"] == "answered"


def test_late_conflict_retains_mapping_and_all_open_questions(tmp_path):
    store, sid, job, record = pending(tmp_path, geometry=True)
    new_issue = IntentIssue(
        id="retained_subject",
        question="Does the old subject assignment still apply?",
        reason="The saved target and the later request disagree.",
        requirement_ids=["width"],
    )
    before = store.session_record(sid, OWNER)
    added = record_conflict(
        store,
        sid,
        OWNER,
        job["id"],
        [new_issue],
        target(),
        request_text(before),
        store.edit_basis(sid, OWNER),
    )
    assert added["attribution"]["facts"] == record["attribution"]["facts"]
    assert len(added["attribution"]["issues"]) == 2
    after = store.session_record(sid, OWNER)
    for key in ("contract", "design_plan", "selected_revision", "accepted_revision"):
        assert after[key] == before[key]
    assert after["intent_attribution_history"] == [record]


@pytest.mark.parametrize("outcome", ["needs_input", "cancelled", "interrupted"])
def test_preview_pause_keeps_baseline_and_resolvable_draft(tmp_path, monkeypatch, outcome):
    store, sid, job, record = pending(tmp_path, preview=True, geometry=True)
    exports = store.directory(sid, BASE)
    exports.mkdir(parents=True)
    sentinel = exports / "saved-export.step"
    sentinel.write_text("Existing immutable export", encoding="utf-8")
    before = store.session_record(sid, OWNER)
    assert before["selected_revision"] == DRAFT
    if outcome == "interrupted":
        with store.connect() as con:
            con.execute(
                "UPDATE jobs SET deadline=0,worker_pid=NULL,worker_created=NULL,worker_host=NULL WHERE id=?",
                (job["id"],),
            )
        assert store.claim() is None
    else:
        if outcome == "cancelled":
            store.cancel(sid, OWNER)
        store.finish(job["id"], sid, "needs_input")

        def no_validation(*args, **kwargs):
            pytest.fail("Intent pause must not run new candidate validation")

        monkeypatch.setattr(
            "forge_mind.workspace.document_service.protection_results", no_validation
        )
        finish_preview(SimpleNamespace(store=store), job)
    saved = store.session_record(sid, OWNER)
    assert saved["state"] == outcome
    assert saved["selected_revision"] == saved["accepted_revision"] == BASE
    assert saved["contract"] == before["contract"]
    assert saved["instructions"] == before["instructions"]
    assert saved["resume_candidate"]["revision_id"] == DRAFT
    assert saved["proposal"]["revision_id"] == DRAFT
    assert saved["proposal"]["eligible"] is False
    assert sentinel.read_text(encoding="utf-8") == "Existing immutable export"
    assert store.edit_basis(sid, OWNER).model_dump() == record["basis"]
    jid = store.enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
    next_job = store.claim()
    assert next_job and next_job["id"] == jid
    assert next_job["preview_only"] is True
    assert next_job["revision_id"] == DRAFT
    assert next_job["based_on"] == BASE


def test_legacy_conflict_needs_no_invented_subject_map(tmp_path):
    store = Store(tmp_path)
    sid = store.create(OWNER, PROMPT, target().model_dump(), authoring_mode="native_operations")[
        "id"
    ]
    jid = store.enqueue(sid, OWNER, {})
    assert store.claim()
    record = record_conflict(
        store,
        sid,
        OWNER,
        jid,
        [
            IntentIssue(
                id="legacy_width",
                question="Which subject?",
                reason="A later conflict",
                requirement_ids=["width"],
            )
        ],
        target(),
        PROMPT,
        store.edit_basis(sid, OWNER),
    )
    assert record["status"] == "needs_input"
    assert record["attribution"]["subjects"] == record["attribution"]["facts"] == []
    assert store.session_record(sid, OWNER)["contract"] == target().model_dump()


def test_reader_overview_and_wait_expose_questions_without_starting_work(tmp_path):
    store, sid, job, record = terminal_pending(tmp_path)
    before = deepcopy(store.session_record(sid, OWNER))
    assert read_attribution(store, sid, OWNER)["record"] == record
    overview = project_overview(store, sid, OWNER)
    summary = overview["intent_attribution"]
    assert summary["version"] == record["version"]
    assert summary["draft_hash"] == record["draft_hash"]
    assert summary["status"] == "needs_input"
    assert summary["issue_count"] == len(summary["questions"]) == 1
    assert summary["subject_count"] == 2 and summary["fact_count"] == 1
    assert "proposed_contract" not in summary
    with pytest.raises(WorkspaceError):
        read_attribution(store, sid, "bob")

    async def read(_):
        return overview

    waited = asyncio.run(wait_for_job(read, job["id"]))
    assert waited.outcome == "terminal" and waited.state == "needs_input"
    assert store.session_record(sid, OWNER) == before


def test_clear_interpretation_keeps_existing_contract_hash_and_proceeds(tmp_path):
    store = Store(tmp_path)
    frozen = target()
    sid = store.create(OWNER, PROMPT, frozen.model_dump(), authoring_mode="native_operations")["id"]
    jid = store.enqueue(sid, OWNER, {})
    assert store.claim()
    record = record_proposal(
        store, sid, OWNER, jid, IntentAttribution(), frozen, PROMPT, store.edit_basis(sid, OWNER)
    )
    assert record["status"] == "interpreted"
    store.set_design_plan_for_job(sid, OWNER, jid, plan(), expected_plan_version=0, contract=frozen)
    assert (
        Contract.model_validate(store.session_record(sid, OWNER)["contract"]).fingerprint()
        == frozen.fingerprint()
    )


@pytest.mark.parametrize("forged", [True, False, None])
@pytest.mark.parametrize("with_resolution", [False, True])
def test_caller_cannot_forge_reconciliation_marker(tmp_path, forged, with_resolution):
    store, sid, job, record = terminal_pending(tmp_path)
    before = deepcopy(store.session_record(sid, OWNER))
    payload = {"reconcile_intent": forged}
    if with_resolution:
        payload["intent_resolution"] = resolution(record)
    with pytest.raises(WorkspaceError, match="server-owned"):
        store.enqueue(sid, OWNER, payload)
    assert store.session_record(sid, OWNER) == before
    with store.connect() as con:
        jobs = con.execute("SELECT id,state FROM jobs WHERE session=?", (sid,)).fetchall()
    assert len(jobs) == 1 and tuple(jobs[0]) == (job["id"], "needs_input")


@pytest.mark.parametrize(
    "preview,prior_delivery",
    [(False, True), (True, True), (True, False), (True, None), (True, 1)],
)
def test_resolution_derives_reconciliation_and_preserves_only_requested_delivery(
    tmp_path, preview, prior_delivery
):
    store, sid, _, record = terminal_pending(
        tmp_path, preview=preview, geometry=True, delivery_requested=prior_delivery
    )
    before = deepcopy(store.session_record(sid, OWNER))
    jid = store.enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
    queued = store.claim()
    assert queued and queued["id"] == jid
    assert queued["reconcile_intent"] is True
    if prior_delivery is True:
        assert queued["delivery_requested"] is True
    else:
        assert "delivery_requested" not in queued
    assert queued["preview_only"] is preview
    assert queued["revision_id"] == DRAFT
    if preview:
        assert queued["contract"] == target().model_dump()
        assert queued["based_on"] == BASE
    saved = store.session_record(sid, OWNER)
    assert saved["intent_attribution"]["status"] == "answered"
    for field in ("contract", "design_plan", "plan_hash", "selected_revision", "accepted_revision"):
        assert saved.get(field) == before.get(field)
    assert len(store.revisions(sid)) == 2
