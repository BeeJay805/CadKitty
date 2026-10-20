"""Scripted extraction lifecycle only; native/model execution is forbidden."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.unit.workspace.compact_source_fixtures import compact_wire_fixture
from tests.unit.workspace.source_delivery_fixtures import (
    legacy_compact_profile,
    source_delivery_wire_fixture,
)
from tests.unit.workspace.test_intent_compilation import (
    compound_preparation_draft,
    invalid_fact_drafts,
    preparation_issue,
    valid_draft,
    wrong_family_draft,
    wrong_inventory_body_drafts,
)
from tests.unit.workspace.test_verification_repair import repair_payload, repair_slot

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.store import Store
from forge_mind.workspace.verification_repair import prepare_verification_repair


def run_intake(
    tmp_path,
    monkeypatch,
    responses,
    *,
    durations=None,
    cancel_after=None,
    source_text=None,
    legacy_v1=False,
    customer_v2=False,
):
    options = {
        "durations": durations,
        "cancel_after": cancel_after,
        "source_text": source_text,
        "customer_v2": customer_v2,
    }
    if legacy_v1:
        with legacy_compact_profile(monkeypatch):
            return _run_intake(tmp_path, monkeypatch, responses, **options)
    return _run_intake(tmp_path, monkeypatch, responses, **options)


def _run_intake(
    tmp_path,
    monkeypatch,
    responses,
    *,
    durations=None,
    cancel_after=None,
    source_text=None,
    customer_v2=False,
):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    valid, text = valid_draft()
    if source_text is not None:
        text = source_text
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", text, None, authoring_mode="native_operations")["id"]
    if customer_v2:
        service.store.update(
            sid,
            project_kind="workspace",
            creation_request_id="a" * 32,
            customer_request_version=2,
            customer_printable=True,
        )
    jid = service.store.enqueue(sid, "owner", {})
    job = service.store.claim()
    assert job is not None and job["id"] == jid
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock.now)
    calls = []

    class Scripted:
        model = "scripted-intake"
        timeout_s = None

        def generate_structured(self, **kwargs):
            # Explicitly encode the compact full canonical escape branch.
            # Preserve malformed source/predicate cases for ordinary admission.
            assert kwargs["output_model"].__name__ in {
                "CompactSourceContractDraft",
                "SourceDeliveryContractDraft",
                "ContractDraft",
                "VerificationRepair",
            }
            calls.append({**kwargs, "timeout": self.timeout_s})
            index = len(calls) - 1
            self.last_call_diagnostics = {"call_id": f"intake-{index}", "outcome": "success"}
            clock.now += (durations or [1.0] * len(responses))[index]
            if cancel_after == len(calls):
                service.stop.set()
            response = responses[index]
            if isinstance(response, Exception):
                raise response
            if callable(response):
                response = response(kwargs["output_model"])
            payload = (
                response.model_dump(mode="json") if hasattr(response, "model_dump") else response
            )
            if kwargs["output_model"].__name__ == "CompactSourceContractDraft":
                payload = compact_wire_fixture(payload)
            elif kwargs["output_model"].__name__ == "SourceDeliveryContractDraft":
                payload = source_delivery_wire_fixture(
                    payload, digital_requirement_ids=["native", "printing"]
                )
            return kwargs["output_model"].model_validate(payload), LLMUsage(self.model, 11, 7)

        def generate_structured_with_image(self, **kwargs):
            pytest.fail("Intake must not request rendered images")

    author = Scripted()
    service.client_factory = lambda **kwargs: author

    def forbid(*args, **kwargs):
        pytest.fail("Intake tests must not start CAD or native workers")

    monkeypatch.setattr(service, "build", forbid)
    monkeypatch.setattr(service.executor, "build", forbid)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbid)
    # End at the existing author iterator boundary after provisional admission.
    state = service._run_agent_job(
        {**job, "max_actions": 0, "author_deadline": 1100.0, "max_seconds": 100.0}
    )
    return SimpleNamespace(
        state=state,
        saved=service.store.get(sid, "owner"),
        calls=calls,
        valid=valid,
        text=text,
        job=job,
        clock=clock,
    )


def full_correction_draft():
    """A wire-valid invalid parent quote requires the existing full correction."""
    draft, text = valid_draft()
    draft.contract.requirements[0].source_quote = "Unquoted provisional source quote."
    draft.contract.requirements.append(
        Requirement(
            id="width",
            text="The plate has size 80 mm in local X.",
            source_quote="80 mm in local X",
            metric="extent_x",
            value=80,
        )
    )
    return draft, text + " Its size is 80 mm in local X."


def full_correction_candidate(initial):
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements[0].source_quote = corrected.contract.requirements[
        0
    ].verification.source_quote
    return corrected


def constrained_response(initial, corrected, source):
    preparation = prepare_verification_repair(
        initial, source, compile_intent_draft(initial, source)
    )
    assert preparation is not None
    payload = repair_payload(preparation, corrected)
    return lambda model: model.model_validate(payload)


def test_compound_machine_duty_reaches_customer_question_without_losing_critical_claim(
    tmp_path, monkeypatch
):
    initial, text = compound_preparation_draft()
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements[-1].verification = None
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, constrained_response(initial, corrected, text)],
        source_text=text,
        legacy_v1=True,
    )
    assert case.state == "needs_input" and len(case.calls) == 2
    assert case.saved["intent_compilation"]["status"] == "needs_customer_input"
    assert case.saved["intent_attribution"]["status"] == "needs_input"
    assert case.calls[1]["output_model"].__name__ == "VerificationRepair"
    assert case.saved["intent_compilation"]["attempts"][1]["raw_draft"][
        "attribution"
    ] == initial.attribution.model_dump(mode="json")
    retained = case.saved["intent_attribution"]["proposed_contract"]["requirements"][-1]
    assert retained["source_quote"] == initial.contract.requirements[-1].source_quote
    assert retained["text"] == initial.contract.requirements[-1].text and retained["critical"]
    assert retained.get("verification") is None
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_wrong_family_and_false_print_question_receive_one_recorded_correction(
    tmp_path, monkeypatch
):
    initial, _ = wrong_family_draft()
    initial.attribution.issues = [preparation_issue()]
    corrected, _ = valid_draft()
    case = run_intake(
        tmp_path, monkeypatch, [initial, corrected], durations=[10, 20], legacy_v1=True
    )
    assert len(case.calls) == 2
    assert [call["timeout"] for call in case.calls] == [100, 90]
    assert case.clock.now == 1030
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready"
    assert record["source_hash"] == digest(case.text)
    assert record["job_id"] == case.job["id"]
    assert record["attempts"][0]["raw_draft"] == initial.model_dump(mode="json")
    assert record["attempts"][0]["status"] == "invalid_interpretation"
    assert record["attempts"][1]["raw_draft"] == corrected.model_dump(mode="json")
    assert len(case.saved["usage"]) == 2
    assert len([event for event in case.saved["events"] if event["kind"] == "model_call"]) == 2
    assert case.saved["intent_attribution"]["status"] == "interpreted"
    assert case.saved["intent_attribution"]["proposed_contract"] == corrected.contract.model_dump(
        mode="json"
    )
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not case.saved.get("accepted_revision")


def test_valid_draft_needs_no_corrective_inference(tmp_path, monkeypatch):
    initial, _ = valid_draft()
    case = run_intake(tmp_path, monkeypatch, [initial])
    assert len(case.calls) == 1
    assert case.saved["intent_compilation"]["status"] == "ready"
    assert case.saved["intent_attribution"]["status"] == "interpreted"


def test_invalid_fact_removal_reaches_admission_with_source_duties_unchanged(tmp_path, monkeypatch):
    initial, corrected, text = invalid_fact_drafts()
    case = run_intake(tmp_path, monkeypatch, [initial, corrected], source_text=text, legacy_v1=True)
    assert len(case.calls) == 2
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready"
    assert record["attempts"][0]["status"] == "invalid_interpretation"
    assert record["attempts"][0]["diagnostics"][0]["field"] == "attribution.facts.broken.value"
    assert record["attempts"][1]["raw_draft"] == corrected.model_dump(mode="json")
    assert case.saved["intent_attribution"]["proposed_contract"] == initial.contract.model_dump(
        mode="json"
    )
    assert case.saved["intent_attribution"]["attribution"] == corrected.attribution.model_dump(
        mode="json"
    )
    assert not case.saved["revisions"] and not case.saved.get("accepted_revision")


@pytest.mark.parametrize("customer_v2", [False, True])
@pytest.mark.parametrize("repair", ["full", "constrained"])
def test_initial_extraction_and_full_correction_share_policy_outside_owner_source(
    tmp_path, monkeypatch, customer_v2, repair
):
    initial, corrected, text = invalid_fact_drafts()
    response = corrected
    if repair == "constrained":
        initial, text = wrong_family_draft()
        corrected, _ = valid_draft()
        response = constrained_response(initial, corrected, text)
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, response],
        source_text=text,
        legacy_v1=True,
        customer_v2=customer_v2,
    )
    assert len(case.calls) == 2 and case.saved["intent_compilation"]["status"] == "ready"
    for call in case.calls:
        if customer_v2:
            envelope = json.loads(call["user"])
            assert isinstance(envelope["author_workflow_context"], dict)
            source, separator, policy_context = envelope["source_context"].partition(
                "\nServer delivery policy (not owner source): "
            )
            assert separator and source == text
            policy, _ = json.JSONDecoder().raw_decode(policy_context)
            assert policy["version"] == 2 and policy["printable"] is True
            assert "not nominal acceptance authority" in call["system"]
        else:
            assert "Server delivery policy" not in call["user"]
            assert "Server delivery policy" not in call["system"]
    assert case.saved["intent_compilation"]["source_hash"] == digest(text)
    assert case.saved["intent_attribution"]["source_text"] == text
    assert (
        case.saved["intent_attribution"]["proposed_contract"]["requirements"]
        == corrected.contract.model_dump(mode="json")["requirements"]
    )


def test_same_family_quote_error_uses_one_protected_correction_without_certificate(
    tmp_path, monkeypatch
):
    corrected, text = valid_draft()
    initial = corrected.model_copy(deep=True)
    initial.contract.requirements[0].verification.source_quote = "editable native CAD"
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, constrained_response(initial, corrected, text)],
        durations=[10, 20],
    )
    assert len(case.calls) == 2
    assert [call["timeout"] for call in case.calls] == [100, 90]
    assert case.calls[1]["output_model"].__name__ == "VerificationRepair"
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready" and len(record["attempts"]) == 2
    assert record["attempts"][0]["raw_draft"] == initial.model_dump(mode="json")
    assert record["attempts"][1]["raw_draft"] == corrected.model_dump(mode="json")
    repair = record["attempts"][1]["verification_repair"]
    assert repair["protocol"] == "verification_repair_v3"
    assert repair["basis"]["slots"][0]["excluded_kinds"] == []
    assert repair["basis"]["slots"][0]["exclusion_certificates"] == []
    assert record["repair_request"] == repair
    assert len(case.saved["usage"]) == 2
    assert case.saved["intent_attribution"]["proposed_contract"] == corrected.contract.model_dump(
        mode="json"
    )
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_parent_source_error_uses_full_correction_and_retains_exact_scalar(tmp_path, monkeypatch):
    initial, text = full_correction_draft()
    corrected = full_correction_candidate(initial)
    case = run_intake(tmp_path, monkeypatch, [initial, corrected], source_text=text)
    assert len(case.calls) == 2 and case.calls[1]["output_model"] is ContractDraft
    assert case.saved["intent_compilation"]["status"] == "ready"
    assert "repair_request" not in case.saved["intent_compilation"]
    saved = next(
        row
        for row in case.saved["intent_attribution"]["proposed_contract"]["requirements"]
        if row["id"] == "width"
    )
    assert saved["metric"] == "extent_x" and saved["value"] == 80
    assert (
        saved["critical"]
        and saved["source_quote"] == initial.contract.requirements[-1].source_quote
    )
    assert saved.get("verification") is None
    assert case.saved["intent_compilation"]["attempts"][1]["raw_draft"] == corrected.model_dump(
        mode="json"
    )
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_one_correction_receives_both_independent_body_errors_before_authoring(
    tmp_path, monkeypatch
):
    initial, corrected, text = wrong_inventory_body_drafts()
    before = initial.model_dump(mode="json")
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, constrained_response(initial, corrected, text)],
        source_text=text,
        durations=[10, 15],
    )
    assert len(case.calls) == 2
    assert [call["timeout"] for call in case.calls] == [100, 90]
    attempts = case.saved["intent_compilation"]["attempts"]
    assert len(attempts) == 2 and attempts[0]["raw_draft"] == before
    first_errors = [
        item for item in attempts[0]["diagnostics"] if item.get("requirement_id") == "body"
    ]
    assert {"source_quote_mismatch", "incompatible_predicate_family"} <= {
        item["code"] for item in first_errors
    }
    assert "nominal_perforated_box" in case.calls[1]["user"]
    assert "source_quote_mismatch" in case.calls[1]["user"]
    assert "incompatible_predicate_family" in case.calls[1]["user"]
    assert case.saved["intent_compilation"]["status"] == "ready"
    assert case.saved["intent_attribution"]["proposed_contract"] == corrected.contract.model_dump(
        mode="json"
    )
    assert len(case.saved["usage"]) == 2
    assert case.calls[1]["output_model"].__name__ == "VerificationRepair"
    repair = attempts[1]["verification_repair"]
    preparation = prepare_verification_repair(initial, text, compile_intent_draft(initial, text))
    assert preparation is not None
    assert repair["raw_response"] == repair_payload(preparation, corrected)
    assert case.saved["intent_compilation"]["repair_request"] == repair
    assert initial.model_dump(mode="json") == before
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not case.saved.get("accepted_revision")


def test_two_invalid_drafts_stop_without_a_customer_question_or_contract(tmp_path, monkeypatch):
    initial, text = full_correction_draft()
    case = run_intake(tmp_path, monkeypatch, [initial, initial], source_text=text)
    assert len(case.calls) == 2
    assert case.calls[1]["output_model"] is ContractDraft
    assert case.state == "needs_review"
    assert case.saved["intent_compilation"]["status"] == "invalid_interpretation"
    assert len(case.saved["intent_compilation"]["attempts"]) == 2
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not any(event["kind"] == "intent_question" for event in case.saved["events"])


@pytest.mark.parametrize("cancel", [False, True])
def test_original_cutoff_or_cancellation_prevents_correction(tmp_path, monkeypatch, cancel):
    initial, _ = wrong_family_draft()
    corrected, _ = valid_draft()
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, corrected],
        durations=[101 if not cancel else 1, 1],
        cancel_after=1 if cancel else None,
        legacy_v1=True,
    )
    assert len(case.calls) == 1
    assert case.state == ("cancelled" if cancel else "budget_exhausted")
    assert len(case.saved["intent_compilation"]["attempts"]) == 1
    assert case.saved["contract"] is None


def test_correction_cannot_drop_requested_delivery(tmp_path, monkeypatch):
    initial, text = full_correction_draft()
    corrected = full_correction_candidate(initial)
    corrected.contract.requirements = [
        row for row in corrected.contract.requirements if row.id != "printing"
    ]
    case = run_intake(tmp_path, monkeypatch, [initial, corrected], source_text=text)
    assert case.state == "needs_review"
    assert len(case.calls) == 2 and case.calls[1]["output_model"] is ContractDraft
    assert any(
        item["code"] == "dropped_requirement"
        for item in case.saved["intent_compilation"]["attempts"][1]["diagnostics"]
    )
    assert case.saved["contract"] is None


def test_schema_validation_error_keeps_original_error_path_without_semantic_retry(
    tmp_path, monkeypatch
):
    with pytest.raises(ValidationError) as caught:
        ContractDraft.model_validate({"contract": {"requirements": []}})
    case = run_intake(tmp_path, monkeypatch, [caught.value])
    assert case.state == "error" and len(case.calls) == 1
    assert not case.saved.get("intent_compilation")
    assert not case.saved.get("intent_attribution")
    assert any(
        event["kind"] == "error" and "ValidationError" in event["message"]
        for event in case.saved["events"]
    )


def test_correction_diagnostics_preserve_original_source_and_do_not_add_owner_answers(
    tmp_path, monkeypatch
):
    initial, text = full_correction_draft()
    corrected = full_correction_candidate(initial)
    corrected.attribution.subjects[0].role = "reference"
    case = run_intake(tmp_path, monkeypatch, [initial, corrected], source_text=text)
    assert case.state == "needs_review"
    assert len(case.calls) == 2 and case.calls[1]["output_model"] is ContractDraft
    assert "Retain every prior requirement" in case.calls[1]["user"]
    assert case.text in case.calls[1]["user"]
    assert "owner_answers" not in case.saved["intent_compilation"]
    assert case.saved["contract"] is None


@pytest.mark.parametrize("cancel", [False, True])
def test_constrained_correction_cannot_admit_after_original_cutoff_or_cancellation(
    tmp_path, monkeypatch, cancel
):
    initial, text = wrong_family_draft()
    corrected, _ = valid_draft()
    case = run_intake(
        tmp_path,
        monkeypatch,
        [initial, constrained_response(initial, corrected, text)],
        durations=[10, 1 if cancel else 91],
        cancel_after=2 if cancel else None,
        legacy_v1=True,
    )
    assert len(case.calls) == 2
    assert [call["timeout"] for call in case.calls] == [100, 90]
    assert case.state == ("cancelled" if cancel else "budget_exhausted")
    record = case.saved["intent_compilation"]
    assert record["repair_request"]["protocol"] == "verification_repair_v3"
    assert record["attempts"][1]["verification_repair"]["raw_response"]
    assert len(case.saved["usage"]) == 2
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_invalid_constrained_wire_retains_request_without_a_third_call(tmp_path, monkeypatch):
    initial, _ = wrong_family_draft()
    case = run_intake(tmp_path, monkeypatch, [initial, {"wrong_slot": None}], legacy_v1=True)
    assert case.state == "error" and len(case.calls) == 2
    record = case.saved["intent_compilation"]
    assert len(record["attempts"]) == 1
    request = record["repair_request"]
    assert request["protocol"] == "verification_repair_v3"
    assert request["route"] == "verification_only"
    assert request["original_schema"]["title"] == "VerificationRepair"
    assert request["basis"]["slots"][0]["exclusion_certificates"]
    assert "raw_response" not in request
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]


@pytest.mark.parametrize("choice", ["multiple", "empty", "payload_and_reason", "blank_reason"])
def test_decoded_bad_family_choice_is_retained_before_local_rejection(
    tmp_path, monkeypatch, choice
):
    initial, text = wrong_family_draft()
    corrected, _ = valid_draft()
    preparation = prepare_verification_repair(initial, text, compile_intent_draft(initial, text))
    assert preparation is not None
    payload = repair_payload(preparation, corrected)
    item = payload["repair_0"]
    if choice == "multiple":
        item["nominal_delivery_scope"] = {
            "version": 1,
            "kind": "nominal_delivery_scope",
            "scope": "software_policy",
            "policy": "nominal_cad_and_digital_only",
            "source_quote": initial.contract.requirements[0].source_quote,
        }
    elif choice == "payload_and_reason":
        item["unverified_reason"] = "No supported predicate supplied."
    else:
        item["nominal_through_hole_pattern"] = None
        item["unverified_reason"] = None if choice == "empty" else "   "
    # Field decoding must succeed; cardinality is checked after raw persistence.
    response = preparation.response_model.model_validate(payload)
    case = run_intake(tmp_path, monkeypatch, [initial, response], legacy_v1=True)
    assert case.state == "error" and len(case.calls) == 2
    assert [call["timeout"] for call in case.calls] == [100, 99]
    record = case.saved["intent_compilation"]
    assert len(record["attempts"]) == 1
    assert record["repair_request"]["protocol"] == "verification_repair_v3"
    assert record["repair_request"]["raw_response"] == response.model_dump(mode="json")
    assert len(case.saved["usage"]) == 2
    assert len([event for event in case.saved["events"] if event["kind"] == "model_call"]) == 2
    assert any(
        event["kind"] == "error" and "exactly one family" in event["message"]
        for event in case.saved["events"]
    )
    assert not case.saved.get("intent_attribution")
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert case.saved.get("selected_revision") is None


def test_constrained_null_keeps_unverified_critical_duty(tmp_path, monkeypatch):
    initial, text = wrong_family_draft()
    case = run_intake(
        tmp_path,
        monkeypatch,
        [
            initial,
            {"repair_0": repair_slot(initial.contract.requirements[0].id, None)},
        ],
        legacy_v1=True,
    )
    assert len(case.calls) == 2
    proposed = case.saved["intent_attribution"]["proposed_contract"]["requirements"][0]
    assert proposed["critical"] and proposed.get("verification") is None
    assert (
        proposed["metric"] == "unmeasured"
        and proposed["source_quote"] == initial.contract.requirements[0].source_quote
    )
    assert case.saved["intent_compilation"]["source_hash"] == digest(text)
    assert case.state != "contract_satisfied"
    assert case.saved["contract"] is None and not case.saved["revisions"]
