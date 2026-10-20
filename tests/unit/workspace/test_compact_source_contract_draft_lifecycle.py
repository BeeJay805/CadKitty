"""Actual compact intake hook; no CAD, provider or native worker is permitted."""

from __future__ import annotations

from copy import deepcopy

import pytest
from tests.unit.workspace.compact_source_fixtures import compact_wire_fixture
from tests.unit.workspace.test_intent_compilation import valid_draft, wrong_family_draft
from tests.unit.workspace.test_source_contract_draft import deliveries
from tests.unit.workspace.test_source_contract_draft_lifecycle import run_source_intake, source_wire
from tests.unit.workspace.test_verification_repair import repair_payload, repair_slot

from forge_mind.workspace.compact_source_contract_draft import CompactSourceContractDraft
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import IntentIssue
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.source_delivery_contract_draft import SourceDeliveryContractDraft
from forge_mind.workspace.verification_repair import prepare_verification_repair


def compact_criterion_row(wire, identifier):
    """Select a named family explicitly, without modifying any model-authored value."""
    entry = next(
        row for row in wire["canonical_requirements"] if row["requirement"]["id"] == identifier
    )
    wire["canonical_requirements"].remove(entry)
    requirement = entry["requirement"]
    criterion = requirement["verification"]
    assert criterion["source_quote"] == requirement["source_quote"]
    reference = f"q_compact_{identifier}"
    wire["quotes"].append({"id": reference, "text": requirement["source_quote"]})
    criterion_id = f"v_compact_{identifier}"
    constants = {
        "version",
        "kind",
        "scope",
        "subject_role",
        "frame",
        "count_policy",
        "coverage",
        "policy",
    }
    wire[criterion["kind"]].append(
        {
            "id": criterion_id,
            "quote_ref": reference,
            **{
                name: deepcopy(value)
                for name, value in criterion.items()
                if name not in constants | {"source_quote"}
            },
        }
    )
    wire["requirements"].append(
        {
            "id": identifier,
            "text": requirement["text"],
            "source": requirement["source"],
            "critical": requirement["critical"],
            "quote_ref": reference,
            "subject_ids": entry["subject_ids"],
            "verification_id": criterion_id,
            "scalar": None,
        }
    )
    return wire


@pytest.mark.parametrize(
    "attack,legacy_v1",
    [
        ("dangling_quote", False),
        ("mismatched_quote", True),
        ("dangling_criterion", True),
        ("nonliteral_pool", False),
        ("duplicate_pool_id", False),
        ("missing_requirement_order", False),
        ("duplicate_family_id", True),
    ],
)
def test_wire_errors_retain_source_binding_with_one_correction_only_for_current_profile(
    tmp_path, monkeypatch, attack, legacy_v1
):
    draft, source = deliveries()
    # Three attacks require V1's independent digital criterion references.
    # Direct source-delivery rows deliberately have no such fields or pool.
    raw = (
        compact_wire_fixture(
            draft, source_only_requirement_ids=["native", "step", "stl", "printing"]
        )
        if legacy_v1
        else source_wire(draft)
    )
    if attack == "dangling_quote":
        raw["digital_requirements"][0]["quote_ref"] = "missing_quote"
    elif attack == "mismatched_quote":
        raw["digital_delivery"][0]["quote_ref"] = raw["requirements"][1]["quote_ref"]
    elif attack == "dangling_criterion":
        raw["requirements"][0]["verification_id"] = "missing_criterion"
    elif attack == "nonliteral_pool":
        raw["quotes"][0]["text"] = "Invented source text"
    elif attack == "duplicate_pool_id":
        raw["quotes"].append(deepcopy(raw["quotes"][0]))
    elif attack == "missing_requirement_order":
        raw["requirement_order"].pop(0)
    else:
        raw["nominal_delivery_scope"].append(
            {
                "id": raw["digital_delivery"][0]["id"],
                "quote_ref": raw["digital_delivery"][0]["quote_ref"],
            }
        )
    model = CompactSourceContractDraft if legacy_v1 else SourceDeliveryContractDraft
    expected = model.model_validate(raw).model_dump(mode="json")
    case = run_source_intake(
        tmp_path, monkeypatch, source, [raw] if legacy_v1 else [raw, raw], legacy_v1=legacy_v1
    )
    assert case.state == ("error" if legacy_v1 else "needs_review")
    assert len(case.calls) == (1 if legacy_v1 else 2)
    assert all(call["output_model"] is model for call in case.calls)
    record = case.saved["intent_compilation"]
    if legacy_v1:
        assert record["status"] == "lowering_rejected" and record["attempts"] == []
    else:
        assert record["status"] == "invalid_interpretation" and len(record["attempts"]) == 2
        assert all(
            attempt["source_proposal"]["raw_response"] == expected
            and attempt["status"] == "invalid_interpretation"
            for attempt in record["attempts"]
        )
    proposal = record["source_proposal"]
    assert proposal["protocol"] == (
        "compact_source_contract_draft_v1" if legacy_v1 else "source_delivery_contract_draft_v1"
    )
    assert proposal["raw_response"] == expected
    assert proposal["raw_response_hash"] == digest(expected)
    assert proposal["source_hash"] == record["source_hash"] == digest(source)
    assert proposal["source_only_requirement_ids"] == ["native", "step", "stl", "printing"]
    assert "lowered_draft_hash" not in proposal and "repair_request" not in record
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not case.saved.get("intent_attribution") and not case.saved.get("accepted_revision")


@pytest.mark.parametrize("supply_proof", [True, False])
def test_matching_quote_wrong_compact_family_uses_unchanged_one_repair_and_retains_duty(
    tmp_path, monkeypatch, supply_proof
):
    initial, source = wrong_family_draft()
    corrected, _ = valid_draft()
    requirement = initial.contract.requirements[0]
    # This is a genuine model-selected wrong family with matching source identity.
    # The host must retain it for canonical diagnosis, never relabel the collection.
    requirement.verification.source_quote = requirement.source_quote
    raw = compact_criterion_row(compact_wire_fixture(initial), requirement.id)
    before = initial.model_dump(mode="json")
    prepared = prepare_verification_repair(initial, source, compile_intent_draft(initial, source))
    assert prepared is not None
    response = (
        repair_payload(prepared, corrected)
        if supply_proof
        else {"repair_0": repair_slot(requirement.id, None)}
    )
    # This V1 digital-on-geometry shape is unavailable in the fresh profile.
    case = run_source_intake(tmp_path, monkeypatch, source, [raw, response], legacy_v1=True)
    assert len(case.calls) == 2
    assert case.calls[0]["output_model"] is CompactSourceContractDraft
    assert case.calls[1]["output_model"].__name__ == "VerificationRepair"
    record = case.saved["intent_compilation"]
    first, second = record["attempts"]
    assert first["raw_draft"] == before
    assert any(row["code"] == "incompatible_predicate_family" for row in first["diagnostics"])
    assert first["source_proposal"] == record["source_proposal"]
    assert record["source_proposal"]["source_only_requirement_ids"] == []
    assert record["source_proposal"]["lowered_draft_hash"] == digest(before)
    assert second["verification_repair"]["protocol"] == "verification_repair_v3"
    assert "source_proposal" not in second
    retained = second["raw_draft"]["contract"]["requirements"][0]
    for field, value in before["contract"]["requirements"][0].items():
        if field != "verification":
            assert retained[field] == value
    assert retained["critical"] and retained["metric"] == "unmeasured"
    if supply_proof:
        assert second["raw_draft"] == corrected.model_dump(mode="json")
        assert retained["verification"]["kind"] == "nominal_through_hole_pattern"
    else:
        assert retained.get("verification") is None
        assert case.state != "contract_satisfied"
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert len(case.saved["usage"]) == 2


def test_compact_file_clauses_cannot_hide_a_separate_explicit_physical_load(tmp_path, monkeypatch):
    draft, source = deliveries()
    source += " It must withstand 10 kg."
    draft.contract.requirements.append(
        Requirement(
            id="load",
            text="Retain the explicit physical load duty",
            source_quote="It must withstand 10 kg.",
        )
    )
    draft.attribution.issues.append(
        IntentIssue(
            id="load_application",
            question="Where is the load applied?",
            reason="The source does not state the load location or direction.",
            requirement_ids=["load"],
        )
    )
    case = run_source_intake(tmp_path, monkeypatch, source, [source_wire(draft)])
    assert case.state == "needs_input" and len(case.calls) == 1
    assert case.saved["intent_compilation"]["status"] == "needs_customer_input"
    proposed = case.saved["intent_attribution"]["proposed_contract"]["requirements"]
    assert [row["id"] for row in proposed] == [row.id for row in draft.contract.requirements]
    load = next(row for row in proposed if row["id"] == "load")
    assert load == draft.contract.requirements[-1].model_dump(mode="json")
    assert load["critical"] and load["metric"] == "unmeasured"
    assert load.get("verification") is None
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not case.saved.get("accepted_revision")
