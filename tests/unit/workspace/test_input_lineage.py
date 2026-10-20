"""Host delta consumption and saved snapshot authority, without CAD/provider calls."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_construction_transaction import (
    RESULT,
    candidate,
    prepare,
)
from tests.unit.workspace.test_construction_transaction import (
    construction as construction,
)
from tests.unit.workspace.test_input_application import claim, create, target

from forge_mind.workspace.delivery_guidance import _input_authority
from forge_mind.workspace.document import digest
from forge_mind.workspace.input_application import FIELD, baseline, copy_to_proposal
from forge_mind.workspace.input_lineage import FIELD as LINEAGE
from forge_mind.workspace.input_lineage import validated
from forge_mind.workspace.intent_attribution import IntentAttribution
from forge_mind.workspace.intent_attribution_service import record_proposal, request_text
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.store import Store, WorkspaceError


def interpret(store, sid, job, contract):
    return record_proposal(
        store,
        sid,
        "owner",
        job["id"],
        IntentAttribution(),
        contract,
        request_text(store.session_record(sid, "owner")),
        store.edit_basis(sid, "owner"),
    )


def initial(tmp_path):
    store, sid = create(tmp_path, native=False)
    job = claim(store, sid)
    interpret(store, sid, job, target())
    store.finish(job["id"], sid, "needs_review")
    return store, sid


def consume(store, sid, width=50):
    job = claim(store, sid, instruction=f"Make it {width} mm wide.")
    before = deepcopy(store.session_record(sid, "owner"))
    store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], target(width))
    return job, before, store.session_record(sid, "owner")


def historical_status(document):
    # Exclude old sibling snapshots: they must not accidentally satisfy the old
    # source comparison instead of checking the selected consumption receipt.
    return _input_authority({**document, "proposal": None, "resume_candidate": None}, "f" * 32)


def test_exact_staged_delta_keeps_historical_interpretation_without_false_block(tmp_path):
    store, sid = initial(tmp_path)
    original = deepcopy(store.session_record(sid, "owner")["intent_attribution"])
    job, pending, saved = consume(store, sid)
    assert job["input_application_id"] == pending[FIELD]["pending"]["input_id"]
    assert pending[FIELD]["pending"]["mode"] == "delta"
    assert historical_status(pending)["status"] == "blocked"
    assert saved["intent_attribution"] == original
    assert saved[FIELD]["applied"]["input_id"] == job["input_application_id"]
    assert saved[FIELD]["pending"] is None
    assert historical_status(saved)["status"] == "clear"
    receipt = validated(saved, saved[FIELD], original)
    assert receipt["steps"][0]["job_id"] == job["id"]
    assert receipt["root"]["source_count"] == 0
    assert receipt["steps"][0]["after"]["contract_hash"] == target(50).fingerprint()
    assert original["contract_hash"] == target().fingerprint()


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("changed_contract", [False, True])
def test_publication_and_terminal_recovery_keep_exact_consumed_application(
    construction, preview, changed_contract
):
    service, sid, first_job, source = construction
    store = service.store
    frozen = Contract.model_validate(store.session_record(sid, "owner")["contract"])
    interpret(store, sid, first_job, frozen)
    original = deepcopy(store.session_record(sid, "owner")["intent_attribution"])
    store.finish(first_job["id"], sid, "needs_review")
    instruction = (
        "Make it 12 mm wide." if changed_contract else "Keep the width; make the body deeper."
    )
    job = claim(store, sid, instruction=instruction, preview_only=preview)
    resulting = target(12) if changed_contract else frozen
    store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], resulting)
    applied = deepcopy(store.session_record(sid, "owner")[FIELD])
    if changed_contract:
        state = store.session_record(sid, "owner")
        store.set_design_plan_for_job(
            sid,
            "owner",
            job["id"],
            DesignPlan.model_validate(state["design_plan"]),
            state["plan_version"],
        )
        assert store.session_record(sid, "owner")[FIELD] == applied
        source = source.model_copy(deep=True)
        source.features[0].params["x"] = 12
    fixture = service, sid, job, source
    prepared = prepare(fixture)
    candidate(fixture, prepared)
    store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    published = store.session_record(sid, "owner")
    assert published["selected_revision"] == RESULT
    assert published[FIELD] == applied
    assert published["intent_attribution"] == original
    assert Contract.model_validate(published["contract"]).fingerprint() == resulting.fingerprint()
    assert historical_status(published)["status"] == "clear"
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        )
    assert payload["target_contract_application"] == applied
    assert payload["target_instructions"] == published["instructions"]
    store.finish(job["id"], sid, "budget_exhausted")
    if preview:
        store.complete_preview(sid, job["id"], job.get("based_on"), None)
    terminal = store.session_record(sid, "owner")
    recovered = terminal["proposal"] if preview else terminal["resume_candidate"]
    assert recovered[FIELD] == applied
    assert recovered["revision_id"] == RESULT
    assert _input_authority(terminal, RESULT)["status"] == "clear"


def test_two_edits_and_selected_saved_snapshots_retain_scope(tmp_path):
    store, sid = initial(tmp_path)
    first_job, _, first = consume(store, sid)
    store.finish(first_job["id"], sid, "needs_review")
    _, _, second = consume(store, sid, 60)
    assert len(second[FIELD][LINEAGE]["steps"]) == 2
    assert second["intent_attribution"] == first["intent_attribution"]
    assert historical_status(second)["status"] == "clear"
    proposal = {
        "revision_id": "b" * 32,
        "contract": deepcopy(second["contract"]),
        "instructions": list(second["instructions"]),
    }
    copy_to_proposal(second, proposal)
    assert proposal[FIELD] == second[FIELD]
    selected = {**second, "proposal": proposal, "resume_candidate": None}
    assert _input_authority(selected, proposal["revision_id"])["status"] == "clear"
    # The normal preview restoration helper copies its exact admitted baseline.
    restored = deepcopy(second)
    Store._restore_preview_working(
        restored,
        {
            "baseline_contract": first["contract"],
            "baseline_instructions": first["instructions"],
            "baseline_contract_application": first[FIELD],
        },
    )
    assert restored[FIELD] == first[FIELD]
    assert historical_status(restored)["status"] == "clear"


@pytest.mark.parametrize("failure", ["cancel", "changed_candidate"])
def test_rejected_publication_keeps_consumption_and_plan_atomic(construction, failure):
    service, sid, first_job, source = construction
    store = service.store
    frozen = Contract.model_validate(store.session_record(sid, "owner")["contract"])
    interpret(store, sid, first_job, frozen)
    store.finish(first_job["id"], sid, "needs_review")
    job = claim(store, sid, instruction="Keep the width; make the body deeper.")
    store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], frozen)
    fixture = service, sid, job, source
    prepared = prepare(fixture)
    candidate(fixture, prepared)
    if failure == "cancel":
        store.cancel(sid, "owner")
    else:
        (store.directory(sid, RESULT) / "document.json").write_text("changed", encoding="utf-8")
    before = deepcopy(store.session_record(sid, "owner"))
    with store.connect() as con:
        job_before = con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
    with pytest.raises(WorkspaceError):
        store.publish_construction(sid, "owner", job["id"], prepared, RESULT)
    assert store.session_record(sid, "owner") == before
    with store.connect() as con:
        assert (
            con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == job_before
        )
    assert before[FIELD][LINEAGE]["steps"]


def test_resumed_staged_delta_retains_original_consumption_identity(tmp_path):
    store, sid = initial(tmp_path)
    first = claim(store, sid, instruction="Make it 50 mm wide.")
    store.finish(first["id"], sid, "budget_exhausted")
    resumed = claim(store, sid, resume=True)
    assert resumed["input_origin"] == "inherited"
    assert resumed["input_application_id"] == first["input_application_id"]
    store.apply_input_contract(
        sid, "owner", resumed["id"], resumed["input_application_id"], target(50)
    )
    saved = store.session_record(sid, "owner")
    assert historical_status(saved)["status"] == "clear"
    assert saved[FIELD][LINEAGE]["steps"][0]["job_id"] == resumed["id"]


@pytest.mark.parametrize(
    "changed", ["source", "contract", "session", "intent", "step", "origin", "count", "pending"]
)
def test_changed_authority_or_receipt_never_clears(tmp_path, changed):
    store, sid = initial(tmp_path)
    _, _, saved = consume(store, sid)
    altered = deepcopy(saved)
    receipt = altered[FIELD][LINEAGE]
    if changed == "source":
        altered["prompt"] += " changed original source"
    elif changed == "contract":
        altered["contract"] = target(60).model_dump()
    elif changed == "session":
        altered["id"] = "f" * 32
    elif changed == "intent":
        altered["intent_attribution"]["status"] = "needs_input"
        altered["intent_attribution"]["draft_hash"] = digest(
            {
                key: value
                for key, value in altered["intent_attribution"].items()
                if key != "draft_hash"
            }
        )
    elif changed == "step":
        receipt["steps"][0]["job_id"] = "f" * 32
    elif changed == "origin":
        receipt["steps"][0]["origin"] = "supplied"
    elif changed == "count":
        receipt["root"]["source_count"] = False
    else:
        altered[FIELD]["pending"] = {"input_id": "e" * 32}
    before = deepcopy(altered)
    assert historical_status(altered)["status"] == "blocked"
    assert altered == before  # Reading never repairs or mints a receipt.


def test_missing_old_receipt_cannot_be_reconstructed_from_source_prefix(tmp_path):
    store, sid = initial(tmp_path)
    _, _, saved = consume(store, sid)
    saved[FIELD].pop(LINEAGE)
    assert historical_status(saved)["status"] == "blocked"
    before = deepcopy(saved)
    historical_status(saved)
    assert saved == before


@pytest.mark.parametrize("branch", ["proposal", "resume_candidate"])
@pytest.mark.parametrize("change", ["corrupt", "missing"])
def test_unchanged_baseline_cannot_authorize_different_target(tmp_path, branch, change):
    store, sid = initial(tmp_path)
    baseline_snapshot = deepcopy(store.session_record(sid, "owner"))
    _, _, saved = consume(store, sid)
    proposal = {
        "revision_id": "b" * 32,
        "contract": saved["contract"],
        "instructions": saved["instructions"],
        FIELD: deepcopy(saved[FIELD]),
    }
    selected = {**baseline_snapshot, branch: proposal}
    assert _input_authority(selected, proposal["revision_id"])["status"] == "clear"
    if change == "corrupt":
        proposal[FIELD][LINEAGE]["steps"][0]["job_id"] = "e" * 32
    else:
        proposal[FIELD].pop(LINEAGE)
    assert _input_authority(selected, proposal["revision_id"])["status"] == "blocked"
    # The original baseline retains its own exact-source authority.
    assert _input_authority(selected, "a" * 32)["status"] == "clear"


@pytest.mark.parametrize("change", ["cancel", "payload", "source", "basis"])
def test_failed_consumption_rolls_back_contract_and_receipt(tmp_path, change):
    store, sid = initial(tmp_path)
    job = claim(store, sid, instruction="Make it 50 mm wide.")
    if change == "cancel":
        store.cancel(sid, "owner")
    elif change == "payload":
        with store.connect() as con:
            payload = json.loads(
                con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            )
            payload["input_origin"] = "supplied"
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
    elif change == "source":
        store.update(sid, prompt="Modified original request")
    else:
        store.update(sid, plan_version=7, plan_hash="f" * 64)
    before = deepcopy(store.session_record(sid, "owner"))
    with pytest.raises(WorkspaceError):
        store.apply_input_contract(sid, "owner", job["id"], job["input_application_id"], target(50))
    assert store.session_record(sid, "owner") == before


def test_baseline_preserves_only_exact_application_tuple(tmp_path):
    store, sid = initial(tmp_path)
    _, _, saved = consume(store, sid)
    retained = baseline(saved)
    assert retained == saved[FIELD] and retained is not saved[FIELD]
    retained[LINEAGE]["steps"].clear()
    assert saved[FIELD][LINEAGE]["steps"]
    for altered in (
        {**saved, "contract": target(60).model_dump()},
        {**saved, "instructions": [*saved["instructions"], "Another owner edit"]},
    ):
        fresh = baseline(altered)
        assert LINEAGE not in fresh
        assert fresh["applied"]["input_id"] is None


def test_new_complete_interpretation_replaces_historical_lineage_need(tmp_path):
    store, sid = initial(tmp_path)
    job, _, _ = consume(store, sid)
    store.finish(job["id"], sid, "needs_review")
    second = claim(store, sid, instruction="Make it 60 mm wide.")
    interpret(store, sid, second, target(60))
    store.apply_input_contract(
        sid, "owner", second["id"], second["input_application_id"], target(60)
    )
    saved = store.session_record(sid, "owner")
    assert LINEAGE not in saved[FIELD]
    assert historical_status(saved)["status"] == "clear"
