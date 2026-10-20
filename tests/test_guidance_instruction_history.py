"""Guidance input, literal history and offline binding controls; no native/model work."""

from __future__ import annotations

from copy import deepcopy

import pytest

from forge_mind.workspace import delivery_guidance as guidance
from forge_mind.workspace.document import digest
from forge_mind.workspace.handoff import _verify_guidance, handoff_context
from forge_mind.workspace.input_application import FIELD, baseline
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import WorkspaceError
from tests.unit.workspace.test_functional_guidance import pending as pending_fixture
from tests.unit.workspace.test_guidance_transport import instructions
from tests.unit.workspace.test_mesh_handoff import RID


@pytest.fixture
def saved(tmp_path, monkeypatch):
    service, sid, *_ = pending_fixture.__wrapped__(tmp_path, monkeypatch)
    return service, sid


def update(
    context, *, text="Literal authored proposal. 中文\nUnknown notes remain literal.", plan=True
):
    value = instructions(context)
    value["guidance"] = dict.fromkeys(value["guidance"], text)
    if plan:
        value["instruction_plan"] = {
            "version": 1,
            "fact_refs": context["instruction_catalog"]["required_refs"],
            "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
        }
    return guidance.GuidanceUpdate.model_validate(value)


def test_same_revision_updates_preserve_literal_records_and_hash_chain(saved):
    service, sid = saved
    context = guidance.guidance_context(service, sid, "owner", RID)
    original = deepcopy(context["record"])
    first = guidance.update_guidance(service, sid, "owner", update(context))
    assert first["instruction_binding"]["status"] == "bound"
    assert first["record"]["instruction_plan_hash"] == digest(first["record"]["instruction_plan"])
    assert first["record"]["input_authority_hash"] == digest(first["input_authority"]["identity"])
    second = guidance.update_guidance(
        service, sid, "owner", update(first, text="Second literal update.")
    )
    assert second["history_status"]["status"] == "verified"
    entries = second["history"]
    assert [entry["version"] for entry in entries] == [1, 2, 3]
    assert entries[0]["record"] == original
    assert entries[1]["record"] == first["record"]
    assert entries[2]["record"] == second["record"]
    for index, entry in enumerate(entries):
        assert entry["revision_id"] == RID
        assert entry["record_hash"] == digest(entry["record"])
        assert entry["predecessor_hash"] == (digest(entries[index - 1]) if index else None)
    assert entries[1]["record"]["guidance"]["unresolved"] == update(context).guidance.unresolved
    assert second["history_status"]["head_hash"] == digest(entries[-1])
    assert service.store.session_record(sid, "owner")["accepted_revision"] is None


def test_legacy_current_record_is_seeded_literally_before_replacement(saved):
    service, sid = saved
    session = service.store.session_record(sid, "owner")
    legacy = deepcopy(session["delivery_guidance_records"][RID])
    legacy.pop("input_authority_hash", None)
    legacy["unknown_legacy_notes"] = {"raw": ["Do not discard me.", None, 7]}
    service.store.update(sid, delivery_guidance_history={}, delivery_guidance_records={RID: legacy})
    context = guidance.guidance_context(service, sid, "owner", RID)
    assert context["history_status"]["status"] == "legacy_unverified"
    current = guidance.update_guidance(service, sid, "owner", update(context, plan=False))
    assert current["instruction_binding"]["status"] == "legacy_unverified"
    assert current["history"][0]["record"] == legacy
    assert [entry["version"] for entry in current["history"]] == [1, 2]
    assert current["record"]["guidance"] == update(context, plan=False).guidance.model_dump()


def test_nonfinite_history_is_reported_unverified_without_hashing_it_as_proof():
    entry = {
        "version": 1,
        "predecessor_hash": None,
        "revision_id": RID,
        "record": {"invalid": float("nan")},
        "record_hash": "0" * 64,
    }
    assert guidance.guidance_history_status([entry], RID, entry["record"])["status"] == "unverified"


def test_unknown_instruction_reference_is_refused_before_history_or_record_writes(saved):
    service, sid = saved
    context = guidance.guidance_context(service, sid, "owner", RID)
    request = update(context)
    assert request.instruction_plan is not None
    request.instruction_plan.fact_refs.append("prepared:print/foreign.stl")
    before = deepcopy(service.store.session_record(sid, "owner"))
    with pytest.raises(WorkspaceError, match=r"prepared:print/foreign\.stl"):
        guidance.update_guidance(service, sid, "owner", request)
    assert service.store.session_record(sid, "owner") == before


@pytest.mark.parametrize("fault", ["record", "hash", "version", "predecessor"])
def test_tampered_history_is_unverified_on_read_and_refused_on_write(saved, fault):
    service, sid = saved
    context = guidance.guidance_context(service, sid, "owner", RID)
    current = guidance.update_guidance(service, sid, "owner", update(context))
    history = deepcopy(service.store.session_record(sid, "owner")["delivery_guidance_history"])
    entry = history[RID][0]
    if fault == "record":
        entry["record"]["guidance"]["print"] = "Substituted history."
    elif fault == "hash":
        entry["record_hash"] = "0" * 64
    elif fault == "version":
        entry["version"] = True
    else:
        entry["predecessor_hash"] = "0" * 64
    service.store.update(sid, delivery_guidance_history=history)
    before = deepcopy(service.store.session_record(sid, "owner"))
    observed = guidance.guidance_context(service, sid, "owner", RID)
    assert observed["history_status"]["status"] == "unverified"
    assert observed["history"] == history[RID]
    assert observed["instruction_binding"]["status"] == "invalid"
    with pytest.raises(WorkspaceError, match="history"):
        guidance.update_guidance(service, sid, "owner", update(current))
    assert service.store.session_record(sid, "owner") == before


@pytest.mark.parametrize("location", ["current", "proposal", "resume_candidate"])
def test_pending_owner_input_in_any_branch_refuses_publication(saved, location):
    service, sid = saved
    service.store.update(sid, selected_revision=RID, proposal=None)
    session = service.store.session_record(sid, "owner")
    application = baseline(session)
    application["pending"] = {"input_id": "a" * 32, "source_hash": "b" * 64}
    if location == "current":
        service.store.update(sid, **{FIELD: application})
    else:
        service.store.update(
            sid,
            **{
                location: {
                    "revision_id": "c" * 32,
                    "contract": session["contract"],
                    "instructions": session.get("instructions", []),
                    FIELD: application,
                }
            },
        )
    before = deepcopy(service.store.session_record(sid, "owner"))
    context = guidance.guidance_context(service, sid, "owner", RID)
    assert context["input_authority"]["status"] == "blocked"
    assert context["instruction_binding"]["status"] == "blocked_input"
    assert context["status"] != "current"
    with pytest.raises(WorkspaceError):
        guidance.update_guidance(service, sid, "owner", update(context))
    assert service.store.session_record(sid, "owner") == before


def test_unresolved_owner_clarification_blocks_without_trusting_prose(saved):
    service, sid = saved
    record = {"status": "needs_input", "source_text": "owner", "source_hash": digest("owner")}
    record["proposed_contract"] = Contract().model_dump()
    record["contract_hash"] = Contract().fingerprint()
    record["draft_hash"] = digest(record)
    service.store.update(sid, intent_attribution=record)
    context = guidance.guidance_context(service, sid, "owner", RID)
    assert context["input_authority"]["status"] == "blocked"
    assert "Owner clarification remains unresolved" in context["input_authority"]["reasons"]
    with pytest.raises(WorkspaceError, match="Owner input"):
        guidance.update_guidance(service, sid, "owner", update(context))


def test_active_job_with_applied_input_can_save_but_new_consumed_input_stales_old_basis(saved):
    service, sid = saved
    service.store.update(sid, selected_revision=RID, proposal=None)
    session = service.store.session_record(sid, "owner")
    service.store.update(sid, **{FIELD: baseline(session)})
    context = guidance.guidance_context(service, sid, "owner", RID)
    old_request = update(context)
    jid = service.store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": 30})
    queued_state = service.store.session_record(sid, "owner")["state"]
    claimed = service.store.claim()
    assert claimed is not None and claimed["id"] == jid
    current = guidance.update_guidance(service, sid, "owner", old_request)
    assert current["instruction_binding"]["status"] == "bound"
    with service.store.connect() as con:
        assert (
            con.execute("SELECT state FROM jobs WHERE id=? AND session=?", (jid, sid)).fetchone()[0]
            == "running"
        )
    assert service.store.session_record(sid, "owner")["state"] == queued_state == "queued"
    # A consumed new owner source must be distinct even when the Contract and CAD stay equal.
    changed = deepcopy(service.store.session_record(sid, "owner"))
    changed["instructions"] = ["Known consumed owner source for this fixture."]
    service.store.update(sid, instructions=changed["instructions"], **{FIELD: baseline(changed)})
    newer = guidance.guidance_context(service, sid, "owner", RID)
    assert newer["input_authority"]["status"] == "clear"
    assert newer["basis_hash"] != current["basis_hash"]
    assert newer["status"] == "stale"
    with pytest.raises(WorkspaceError, match="inputs changed"):
        guidance.update_guidance(service, sid, "owner", old_request)


def test_same_consumed_preview_target_keeps_instruction_binding_after_acceptance(saved):
    service, sid = saved
    context = guidance.guidance_context(service, sid, "owner", RID)
    pending = guidance.update_guidance(service, sid, "owner", update(context))
    proposal = deepcopy(service.store.session_record(sid, "owner")["proposal"])
    proposal["eligible"] = True  # Controlled metadata transition; fixture bytes are not CAD proof.
    service.store.update(sid, proposal=proposal)
    service.store.accept_proposal(sid, "owner", RID, None, 0)
    current = guidance.guidance_context(service, sid, "owner", RID)
    assert pending["target_kind"] == "pending_candidate"
    assert current["target_kind"] == "selected_candidate"
    assert current["input_authority"]["identity"] == pending["input_authority"]["identity"]
    assert current["basis_hash"] == pending["basis_hash"]
    assert current["status"] == "current"
    assert current["instruction_binding"]["status"] == "bound"
    assert current["history"] == pending["history"]


def test_instruction_part_catalog_is_complete_and_missing_quantities_stay_unknown():
    parts = [
        {
            "id": f"part_{index}",
            "name": f"Literal part {index}",
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "output": "body",
            "production": {"role": "make"},
        }
        for index in range(40)
    ]
    document = {
        "assembly": {
            "parts": parts,
            "roots": [
                {"id": f"occ_{index}", "definition": part["id"]} for index, part in enumerate(parts)
            ],
        }
    }
    revision = {"document": document, "measurement": {"artifact_hashes": {}, "facts": {}}}
    unknown = guidance._facts(revision)
    assert len(unknown["parts"]) == 32
    assert len(unknown["instruction_parts"]) == 40
    assert all(row["exported_occurrence_quantity"] is None for row in unknown["instruction_parts"])
    revision["measurement"]["facts"] = {
        "assembly": {"occurrences": [{"definition": part["id"]} for part in parts]}
    }
    known = guidance._facts(revision)
    assert known["instruction_parts"][-1]["part_id"] == "part_39"
    assert known["instruction_parts"][-1]["exported_occurrence_quantity"] == 1
    flat = {
        "document": {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "production": {"role": "make"},
        },
        "measurement": {"facts": {}},
    }
    assert guidance._facts(flat)["instruction_parts"][0]["exported_occurrence_quantity"] is None
    flat["measurement"]["facts"]["solid_count"] = 1
    assert guidance._facts(flat)["instruction_parts"][0]["exported_occurrence_quantity"] == 1


@pytest.mark.parametrize(
    "field",
    ["instruction_catalog", "instruction_binding", "history", "history_status", "input_authority"],
)
def test_offline_handoff_recomputes_instruction_and_history_fields(saved, field):
    service, sid = saved
    context = guidance.guidance_context(service, sid, "owner", RID)
    current = guidance.update_guidance(service, sid, "owner", update(context))
    session = service.store.get(sid, "owner")
    revision = service.store.revision(sid, RID)
    captured = handoff_context(session, revision, "f" * 64)
    captured.update(
        guidance_basis=current["basis"],
        guidance_basis_hash=current["basis_hash"],
        guidance_context_hash=digest(current),
    )
    assert _verify_guidance(current, captured, {}, revision["document"]) == "current"
    altered = deepcopy(current)
    if field == "instruction_catalog":
        altered[field]["facts"]["physical:readiness"]["text"] = "Physical use approved."
    elif field == "instruction_binding":
        altered[field]["steps"][0]["text"] = "Physical use approved."
    elif field == "history":
        altered[field][0]["record"]["guidance"]["print"] = "Changed old note."
    elif field == "history_status":
        altered[field]["head_hash"] = "0" * 64
    else:
        altered[field]["status"] = "blocked"
    captured["guidance_context_hash"] = digest(altered)  # Repair the outer hash only.
    with pytest.raises(ValueError, match="guidance"):
        _verify_guidance(altered, captured, {}, revision["document"])
