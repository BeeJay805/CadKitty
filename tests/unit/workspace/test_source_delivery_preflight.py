"""Rejected raw interpretation stays repairable without native or model execution."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.workspace.test_customer_interpretation import SOURCE, wire
from tests.unit.workspace.test_customer_references import reference
from tests.unit.workspace.test_source_contract_draft_lifecycle import run_source_intake

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_interpretation import validate_server_interpretation_proposal
from forge_mind.workspace.document import digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract
from forge_mind.workspace.plan_intent import PlanIntent
from forge_mind.workspace.source_delivery_contract_draft import (
    PhotoSourceContractDraftV2,
    SourceDeliveryContractDraft,
    lower_source_delivery_draft,
    preflight_source_delivery_draft,
)
from forge_mind.workspace.store import Store


def retained_plate():
    return json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("customer-plate-lowering001.json")
        .read_text(encoding="utf-8-sig")
    )


def malformed(kind):
    raw = wire().model_dump(mode="json")
    target = raw["nominal_targets"][0]
    if kind == "quote":
        target["criterion"]["source_quote"] = SOURCE
    elif kind == "fact":
        target["field_bindings"][0]["terms"][0]["fact_id"] = "unknown"
    elif kind == "subject":
        target["criterion"]["subject_id"] = "unknown"
    elif kind == "frame":
        target["criterion"]["center_mm"][0] = 1.0
    elif kind == "unit":
        raw["fact_groups"][0]["unit"] = "cm"
    elif kind == "operand":
        target["field_bindings"][0]["terms"][0]["operation"] = "negative"
    elif kind == "count":
        target["hole_count_fact_id"] = "width"
    elif kind == "binding":
        target["field_bindings"].pop()
    return SourceDeliveryContractDraft.model_validate(raw)


def test_retained_plate_initial_failure_is_diagnosed_without_changing_its_quote():
    retained = retained_plate()
    proposal = SourceDeliveryContractDraft.model_validate(retained["raw_response"])
    before = proposal.model_dump(mode="json")
    with pytest.raises(ValueError, match="exact original source duty quote"):
        lower_source_delivery_draft(proposal, retained["source"])
    result = preflight_source_delivery_draft(proposal, retained["source"])
    assert result.draft is None and result.compilation["status"] == "invalid_interpretation"
    assert result.compilation["diagnostics"][0]["code"] == "invalid_source_lowering"
    assert "exact original source duty quote" in result.compilation["diagnostics"][0]["message"]
    assert result.source_proposal["raw_response"] == before
    assert proposal.model_dump(mode="json") == before
    assert all(
        row.critical and row.metric == "unmeasured" and row.verification is None
        for row in result.retention_draft.contract.requirements
    )


@pytest.mark.parametrize(
    "kind", ["quote", "fact", "subject", "frame", "unit", "operand", "count", "binding"]
)
def test_all_deterministic_lowering_failures_share_one_source_wire_correction(
    tmp_path, monkeypatch, kind
):
    rejected = malformed(kind)
    corrected = wire()
    before = deepcopy(rejected.model_dump(mode="json"))
    with pytest.raises(ValueError):
        lower_source_delivery_draft(rejected, SOURCE)
    case = run_source_intake(
        tmp_path, monkeypatch, SOURCE, [before, corrected.model_dump(mode="json")]
    )
    assert len(case.calls) == 2 and all(
        call["output_model"] is SourceDeliveryContractDraft for call in case.calls
    )
    compilation = case.saved["intent_compilation"]
    assert compilation["status"] == "ready", case.saved["events"]
    assert [item["status"] for item in compilation["attempts"]] == [
        "invalid_interpretation",
        "ready",
    ]
    assert compilation["attempts"][0]["source_proposal"]["raw_response"] == before
    assert compilation["source_proposal"]["raw_response"] == corrected.model_dump(mode="json")
    canonical = lower_source_delivery_draft(corrected, SOURCE)
    assert compilation["source_proposal"]["lowered_draft_hash"] == digest(
        canonical.model_dump(mode="json")
    )
    validate_server_interpretation_proposal(
        case.saved, compilation["job_id"], SOURCE, canonical.contract, canonical.attribution
    )
    assert case.saved["intent_attribution"]["status"] == "interpreted"
    assert case.saved["intent_attribution"]["proposed_contract"] == canonical.contract.model_dump(
        mode="json"
    )
    assert all(
        row.critical and row.metric == "unmeasured" for row in canonical.contract.requirements[:-1]
    )
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert rejected.model_dump(mode="json") == before


def test_retained_plate_scripted_correction_uses_existing_parent_quote_and_replays(
    tmp_path, monkeypatch
):
    retained = retained_plate()
    corrected = deepcopy(retained["raw_response"])
    # This is an explicit scripted model response, not a host-selected repair.
    parent = corrected["requirements"][0]
    corrected["nominal_targets"][0]["criterion"]["source_quote"] = next(
        row["text"] for row in corrected["quotes"] if row["id"] == parent["quote_ref"]
    )
    case = run_source_intake(
        tmp_path, monkeypatch, retained["source"], [retained["raw_response"], corrected]
    )
    assert len(case.calls) == 2
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready"
    proposal = SourceDeliveryContractDraft.model_validate(corrected)
    canonical = lower_source_delivery_draft(proposal, retained["source"])
    validate_server_interpretation_proposal(
        case.saved, record["job_id"], retained["source"], canonical.contract, canonical.attribution
    )
    assert len(canonical.contract.requirements) == 8
    assert all(
        row.critical and row.metric == "unmeasured" and row.verification is None
        for row in canonical.contract.requirements[:7]
    )
    assert not canonical.contract.requirements[7].critical
    assert not case.saved["revisions"]


def test_second_invalid_source_response_is_retained_and_ends_without_admission(
    tmp_path, monkeypatch
):
    raw = malformed("quote").model_dump(mode="json")
    case = run_source_intake(tmp_path, monkeypatch, SOURCE, [raw, raw])
    assert case.state == "needs_review" and len(case.calls) == 2
    record = case.saved["intent_compilation"]
    assert record["status"] == "invalid_interpretation" and len(record["attempts"]) == 2
    assert all(item["source_proposal"]["raw_response"] == raw for item in record["attempts"])
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


@pytest.mark.parametrize(
    "change", ["drop_target", "rename_target", "drop_parent", "duty", "critical", "fact"]
)
def test_correction_cannot_launder_targets_source_duties_or_valid_facts(change):
    initial = malformed("quote")
    first = preflight_source_delivery_draft(initial, SOURCE)
    raw = wire().model_dump(mode="json")
    if change == "drop_target":
        raw["nominal_targets"] = []
    elif change == "rename_target":
        raw["nominal_targets"][0]["requirement_id"] = "other"
    elif change == "drop_parent":
        raw["canonical_requirements"].pop()
        raw["requirement_order"].pop()
    elif change == "duty":
        raw["canonical_requirements"][1]["requirement"]["text"] = "No load duty"
    elif change == "critical":
        raw["canonical_requirements"][1]["requirement"]["critical"] = False
    elif change == "fact":
        raw["fact_groups"][0]["values"][0]["value"] = 50
    result = preflight_source_delivery_draft(
        SourceDeliveryContractDraft.model_validate(raw), SOURCE, previous=first
    )
    assert result.compilation["status"] == "invalid_interpretation"


def test_valid_nominal_source_keeps_one_extraction_call(tmp_path, monkeypatch):
    raw = wire().model_dump(mode="json")
    case = run_source_intake(tmp_path, monkeypatch, SOURCE, [raw])
    assert len(case.calls) == 1 and case.saved["intent_compilation"]["status"] == "ready"


@pytest.mark.parametrize("change", ["none", "drop", "reclassify"])
def test_source_wire_correction_keeps_original_digital_scope(tmp_path, monkeypatch, change):
    source = SOURCE + " Provide STEP."
    initial = malformed("quote").model_dump(mode="json")
    initial["quotes"].append({"id": "digital_quote", "text": "Provide STEP."})
    initial["digital_requirements"] = [
        {
            "id": "step",
            "source": "request",
            "critical": True,
            "quote_ref": "digital_quote",
            "subject_ids": [],
            "deliverable": "project_step",
        }
    ]
    initial["requirement_order"].append("step")
    corrected = deepcopy(initial)
    corrected["nominal_targets"] = wire().model_dump(mode="json")["nominal_targets"]
    if change != "none":
        corrected["digital_requirements"] = []
        if change == "drop":
            corrected["requirement_order"].remove("step")
        else:
            corrected["canonical_requirements"].append(
                {
                    "requirement": {
                        "id": "step",
                        "text": "Provide STEP.",
                        "source_quote": "Provide STEP.",
                    },
                    "subject_ids": [],
                }
            )
    case = run_source_intake(tmp_path, monkeypatch, source, [initial, corrected])
    record = case.saved["intent_compilation"]
    assert len(case.calls) == 2
    assert record["attempts"][0]["source_proposal"]["source_only_requirement_ids"] == ["step"]
    if change == "none":
        assert record["status"] == "ready"
        assert record["source_proposal"]["source_only_requirement_ids"] == ["step"]
        draft = lower_source_delivery_draft(
            SourceDeliveryContractDraft.model_validate(corrected), source
        )
        validate_server_interpretation_proposal(
            case.saved, record["job_id"], source, draft.contract, draft.attribution
        )
    else:
        assert case.state == "needs_review" and record["status"] == "invalid_interpretation"
        assert any(
            item["code"] == "changed_source_proposal"
            for item in record["attempts"][-1]["diagnostics"]
        )
        assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def scripted_intake(
    tmp_path,
    monkeypatch,
    responses,
    *,
    durations=None,
    cancel_after=None,
    mutate=None,
    startup=False,
    photo=False,
):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    if photo:
        sid = service.store.create_customer_project(
            "owner",
            {
                "request_id": "a" * 32,
                "prompt": SOURCE + " Use this photo only as context.",
                "reference": reference(known_dimension=None),
                "approval_required": True,
            },
        )["id"]
    else:
        sid = service.store.create("owner", SOURCE, None, authoring_mode="native_operations")["id"]
        service.store.enqueue(sid, "owner", {})
    job = service.store.claim()
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock.now)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight control cannot start geometry, workers or provider services")

    class Client:
        model = "scripted-source-preflight"
        timeout_s = None

        def respond(self, kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            assert index < len(responses), "One correction allowance only"
            clock.now += durations[index] if durations else 1.0
            if cancel_after == len(calls):
                service.stop.set()
            if mutate:
                mutate(service, sid, len(calls))
            response = responses[index]
            if isinstance(response, Exception):
                raise response
            payload = (
                response.model_dump(mode="json") if hasattr(response, "model_dump") else response
            )
            if photo:
                payload = contextual_photo_wire(payload).model_dump(mode="json")
            return kwargs["output_model"].model_validate(payload), LLMUsage(self.model, 11, 7)

        def generate_structured(self, **kwargs):
            assert "image_png" not in kwargs
            return self.respond(kwargs)

        def generate_structured_with_image(self, **kwargs):
            assert photo and kwargs["image_png"]
            return self.respond(kwargs)

    service.client_factory = lambda **kwargs: Client()
    for target, name in (
        (service, "start"),
        (service, "build"),
        (service.executor, "build"),
        (service.executor, "run_typed_tool"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    state = service._run_agent_job(
        {**job, "max_actions": 1 if startup else 0, "max_seconds": 100, "author_deadline": 1100.0}
    )
    return SimpleNamespace(
        service=service, sid=sid, saved=service.store.get(sid, "owner"), calls=calls, state=state
    )


def contextual_photo_wire(raw):
    """Represent the retained noncopy duties in the fresh photo schema."""
    payload = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else deepcopy(raw)
    payload["photo_reference"] = {
        "role": "context",
        "source_quote": "Use this photo only as context.",
    }
    payload.pop("photo_nominal_claims", None)
    payload["photo_requirements"] = []
    return PhotoSourceContractDraftV2.model_validate(payload)


def test_corrected_source_reaches_real_atomic_contract_and_plan_admission(tmp_path, monkeypatch):
    canonical = lower_source_delivery_draft(wire(), SOURCE)
    startup = PlanIntent(
        objective="Preserve the plate and physical load duty",
        parts=[
            {
                "id": "plate",
                "role": "make",
                "purpose": "Requested plate",
                "requirement_ids": [row.id for row in canonical.contract.requirements],
            }
        ],
        first_part_id="plate",
    )
    case = scripted_intake(
        tmp_path, monkeypatch, [malformed("quote"), wire(), startup], startup=True
    )
    assert [call["output_model"] for call in case.calls] == [
        SourceDeliveryContractDraft,
        SourceDeliveryContractDraft,
        PlanIntent,
    ]
    assert Contract.model_validate(case.saved["contract"]) == canonical.contract, case.saved[
        "events"
    ]
    assert case.saved["plan_version"] == 1 and case.saved["design_plan"]
    assert all(row["critical"] for row in case.saved["contract"]["requirements"][:-1])
    assert not case.saved["revisions"] and case.saved.get("accepted_revision") is None


@pytest.mark.parametrize(
    "stop", ["cancel_first", "cancel_correction", "exhaust_first", "exhaust_correction"]
)
@pytest.mark.parametrize("photo", [False, True])
def test_cancel_or_exhaustion_keeps_raw_attempts_without_admission(
    tmp_path, monkeypatch, stop, photo
):
    cancel = 1 if stop == "cancel_first" else 2 if stop == "cancel_correction" else None
    durations = (
        [101.0, 1.0]
        if stop == "exhaust_first"
        else [1.0, 101.0]
        if stop == "exhaust_correction"
        else None
    )
    case = scripted_intake(
        tmp_path,
        monkeypatch,
        [malformed("quote"), wire()],
        durations=durations,
        cancel_after=cancel,
        photo=photo,
    )
    expected_calls = 1 if stop.endswith("first") else 2
    assert len(case.calls) == expected_calls
    assert case.state in {"cancelled", "budget_exhausted"}
    assert len(case.saved["intent_compilation"]["attempts"]) >= 1
    assert case.saved["intent_compilation"]["attempts"][0]["source_proposal"]["raw_response"] == (
        contextual_photo_wire(malformed("quote")) if photo else malformed("quote")
    ).model_dump(mode="json")
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_photo_source_correction_reuses_exact_retained_visual_and_text_context(
    tmp_path, monkeypatch
):
    case = scripted_intake(tmp_path, monkeypatch, [malformed("quote"), wire()], photo=True)
    assert len(case.calls) == 2 and case.saved["intent_compilation"]["status"] == "ready"
    first, corrected = case.calls
    assert first["image_png"] == corrected["image_png"]
    assert first["system"] == corrected["system"]
    first_context = json.loads(first["user"])
    corrected_context = json.loads(corrected["user"])
    assert corrected_context["source_context"].startswith(first_context["source_context"])
    assert corrected_context["author_workflow_context"] == first_context["author_workflow_context"]
    assert "Owned visual reference metadata:" in corrected["user"]
    assert "pixels" in corrected["system"]
    assert all(call["output_model"] is PhotoSourceContractDraftV2 for call in case.calls)
    assert case.saved["customer_reference"]["known_dimension"] is None
    assert case.saved["approval_required"] is True
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_photo_source_correction_timeout_does_not_retry_or_admit(tmp_path, monkeypatch):
    case = scripted_intake(
        tmp_path, monkeypatch, [malformed("quote"), LLMError("Model call timed out")], photo=True
    )
    assert len(case.calls) == 2 and case.state == "error"
    assert case.calls[0]["image_png"] == case.calls[1]["image_png"]
    assert len(case.saved["intent_compilation"]["attempts"]) == 1
    assert not any(event["kind"] == "model_retry" for event in case.saved["events"])
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_fresh_owner_source_change_during_correction_cannot_admit_old_proposal(
    tmp_path, monkeypatch
):
    def changed(service, sid, call):
        if call == 2:
            service.store.update(sid, prompt=SOURCE + " Make it wider.")

    case = scripted_intake(tmp_path, monkeypatch, [malformed("quote"), wire()], mutate=changed)
    assert len(case.calls) == 2
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_unrelated_genuine_question_survives_correction_and_cannot_be_removed(
    tmp_path, monkeypatch
):
    issue = {
        "id": "load_direction",
        "question": "Which load direction is required?",
        "reason": "The source does not specify load direction.",
        "requirement_ids": ["load_duty"],
    }
    initial = malformed("quote").model_dump(mode="json")
    initial["issues"] = [issue]
    corrected = wire().model_dump(mode="json")
    corrected["issues"] = [issue]
    case = run_source_intake(tmp_path, monkeypatch, SOURCE, [initial, corrected])
    assert case.state == "needs_input" and len(case.calls) == 2
    assert case.saved["intent_compilation"]["status"] == "needs_customer_input"
    assert case.saved["intent_attribution"]["attribution"]["issues"][0]["id"] == "load_direction"
    refused = preflight_source_delivery_draft(
        wire(),
        SOURCE,
        previous=preflight_source_delivery_draft(
            SourceDeliveryContractDraft.model_validate(initial), SOURCE
        ),
    )
    assert refused.compilation["status"] == "invalid_interpretation"
    assert any(
        item["code"] == "removed_customer_question" for item in refused.compilation["diagnostics"]
    )


@pytest.mark.parametrize("attack", ["source", "job", "raw", "hash", "contract"])
def test_corrected_authority_still_requires_exact_current_raw_replay(attack):
    corrected = preflight_source_delivery_draft(
        wire(), SOURCE, previous=preflight_source_delivery_draft(malformed("quote"), SOURCE)
    )
    canonical = corrected.draft
    document = {
        "intent_compilation": {
            "job_id": "job",
            "source_hash": digest(SOURCE),
            "status": "ready",
            "source_proposal": deepcopy(corrected.source_proposal),
        }
    }
    source = SOURCE
    if attack == "source":
        source += " Change it."
    elif attack == "job":
        document["intent_compilation"]["job_id"] = "another"
    elif attack == "raw":
        raw = document["intent_compilation"]["source_proposal"]["raw_response"]
        raw["nominal_targets"] = []
        document["intent_compilation"]["source_proposal"]["raw_response_hash"] = digest(raw)
    elif attack == "hash":
        document["intent_compilation"]["source_proposal"]["lowered_draft_hash"] = "a" * 64
    else:
        canonical = canonical.model_copy(deep=True)
        canonical.contract.requirements[0].critical = False
    with pytest.raises(ValueError):
        validate_server_interpretation_proposal(
            document, "job", source, canonical.contract, canonical.attribution
        )
