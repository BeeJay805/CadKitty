"""Real fresh-profile requests and explicit legacy failure replay, without CAD."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture
from tests.unit.workspace.test_source_contract_draft import deliveries
from tests.unit.workspace.test_source_contract_draft_lifecycle import run_source_intake

from forge_mind.workspace import source_delivery_contract_draft as initial_profile
from forge_mind.workspace.compact_source_contract_draft import CompactSourceContractDraft
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution, IntentIssue
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.source_delivery_contract_draft import SourceDeliveryContractDraft

DIGITAL_IDS = ["native", "step", "stl", "printing"]
TRIAL003_DIGITAL_IDS = ["r5", "r6", "r7", "r8"]


def trial003():
    return json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("source-delivery-trial003.json")
        .read_text(encoding="utf-8-sig")
    )


def test_fixture_converter_requires_explicit_digital_selection_and_preserves_original():
    draft, _ = deliveries()
    before = draft.model_dump(mode="json")
    with pytest.raises(ValueError, match="explicitly"):
        source_delivery_wire_fixture(draft)
    with pytest.raises(ValueError, match="explicitly"):
        source_delivery_wire_fixture(draft, digital_requirement_ids=["native"])
    with pytest.raises(ValueError, match="existing"):
        source_delivery_wire_fixture(draft, digital_requirement_ids=[*DIGITAL_IDS, "missing"])
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=DIGITAL_IDS)
    assert draft.model_dump(mode="json") == before
    assert [row["id"] for row in raw["digital_requirements"]] == DIGITAL_IDS
    assert all(
        set(row) == {"id", "source", "critical", "quote_ref", "subject_ids", "deliverable"}
        for row in raw["digital_requirements"]
    )
    mismatched = draft.model_copy(deep=True)
    mismatched.contract.requirements[0].verification.source_quote = "STEP"
    with pytest.raises(ValueError, match="matching criterion quote"):
        source_delivery_wire_fixture(mismatched, digital_requirement_ids=DIGITAL_IDS)
    assert draft.model_dump(mode="json") == before


def test_legacy_seam_restores_real_new_profile_before_next_intake(tmp_path, monkeypatch):
    draft, source = deliveries()
    retained = trial003()
    original_model = initial_profile.SourceDeliveryContractDraft
    original_lower = initial_profile.lower_source_delivery_draft
    original_record = initial_profile.source_delivery_proposal_record
    legacy = run_source_intake(
        tmp_path / "legacy",
        monkeypatch,
        retained["source"],
        [retained["legacy_raw"], retained["legacy_repair_raw"]],
        legacy_v1=True,
    )
    assert legacy.state == "needs_review" and len(legacy.calls) == 2
    assert legacy.calls[0]["output_model"] is CompactSourceContractDraft
    assert legacy.saved["intent_compilation"]["source_proposal"]["protocol"] == (
        "compact_source_contract_draft_v1"
    )
    assert (
        initial_profile.SourceDeliveryContractDraft is original_model is SourceDeliveryContractDraft
    )
    assert initial_profile.lower_source_delivery_draft is original_lower
    assert initial_profile.source_delivery_proposal_record is original_record
    fresh = run_source_intake(
        tmp_path / "fresh",
        monkeypatch,
        source,
        [source_delivery_wire_fixture(draft, digital_requirement_ids=DIGITAL_IDS)],
    )
    assert fresh.calls[0]["output_model"] is SourceDeliveryContractDraft
    assert fresh.saved["intent_compilation"]["source_proposal"]["protocol"] == (
        "source_delivery_contract_draft_v1"
    )
    assert fresh.saved["intent_compilation"]["attempts"][0]["raw_draft"] == draft.model_dump(
        mode="json"
    )


def test_actual_trial003_generated_prose_failure_remains_a_legacy_failure(tmp_path, monkeypatch):
    retained = trial003()
    before = deepcopy(retained)
    case = run_source_intake(
        tmp_path,
        monkeypatch,
        retained["source"],
        [retained["legacy_raw"], retained["legacy_repair_raw"]],
        legacy_v1=True,
    )
    assert case.state == "needs_review" and len(case.calls) == 2
    assert case.calls[0]["output_model"] is CompactSourceContractDraft
    first, second = case.saved["intent_compilation"]["attempts"]
    assert first["raw_draft"] == retained["canonical_initial"]
    assert first["status"] == second["status"] == "invalid_interpretation"
    before_row = next(
        row for row in first["raw_draft"]["contract"]["requirements"] if row["id"] == "r8"
    )
    after_row = next(
        row for row in second["raw_draft"]["contract"]["requirements"] if row["id"] == "r8"
    )
    assert before_row == after_row
    assert before_row["text"] == "Deliver a prepared print package for the plate."
    assert before_row["source_quote"] == "a prepared print package"
    assert before_row["verification"]["deliverable"] == "prepared_print_package"
    assert len(first["raw_draft"]["attribution"]["facts"]) == 22
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert retained == before


def test_new_trial003_counterpart_has_source_text_and_identical_22_facts(tmp_path, monkeypatch):
    retained = trial003()
    original = ContractDraft.model_validate(retained["canonical_initial"])
    expected = original.model_dump(mode="json")
    for row in expected["contract"]["requirements"]:
        if row["id"] in TRIAL003_DIGITAL_IDS:
            # This explicit prospective counterpart expects source text; the
            # legacy generated text and its failure remain unchanged above.
            row["text"] = row["source_quote"]
    raw = source_delivery_wire_fixture(original, digital_requirement_ids=TRIAL003_DIGITAL_IDS)
    assert "digital_delivery" not in raw
    assert all("text" not in row for row in raw["digital_requirements"])
    case = run_source_intake(tmp_path, monkeypatch, retained["source"], [raw])
    assert len(case.calls) == 1 and case.calls[0]["output_model"] is SourceDeliveryContractDraft
    record = case.saved["intent_compilation"]
    assert record["status"] == "ready" and len(record["attempts"]) == 1
    assert record["attempts"][0]["raw_draft"] == expected
    assert record["source_proposal"]["source_only_requirement_ids"] == TRIAL003_DIGITAL_IDS
    assert record["source_proposal"]["lowered_draft_hash"] == digest(expected)
    assert expected["attribution"] == retained["canonical_initial"]["attribution"]
    assert len(expected["attribution"]["facts"]) == 22
    assert original.model_dump(mode="json") == retained["canonical_initial"]
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_all_five_digital_deliverables_use_actual_new_service(tmp_path, monkeypatch):
    draft, source = deliveries()
    source += " Provide bound instructions."
    draft.contract.requirements.append(
        Requirement.model_validate(
            {
                "id": "instructions",
                "text": "bound instructions",
                "source_quote": "bound instructions",
                "verification": {
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "coverage": "all_made",
                    "deliverable": "bound_instruction_plan",
                    "source_quote": "bound instructions",
                },
            }
        )
    )
    identifiers = [*DIGITAL_IDS, "instructions"]
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=identifiers)
    case = run_source_intake(tmp_path, monkeypatch, source, [raw])
    record = case.saved["intent_compilation"]
    assert len(case.calls) == 1 and record["status"] == "ready"
    assert record["source_proposal"]["source_only_requirement_ids"] == identifiers
    assert record["attempts"][0]["raw_draft"] == draft.model_dump(mode="json")
    assert case.saved["contract"] is None and not case.saved["revisions"]


@pytest.mark.parametrize(
    "attack", ["text", "scalar", "criterion_id", "canonical_escape", "old_pool"]
)
def test_fresh_schema_rejects_digital_prose_scalar_and_escape_smuggling_without_fallback(
    tmp_path, monkeypatch, attack
):
    draft, source = deliveries()
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=DIGITAL_IDS)
    if attack == "canonical_escape":
        raw["digital_requirements"].pop(0)
        raw["canonical_requirements"].append(
            {
                "requirement": draft.contract.requirements[0].model_dump(mode="json"),
                "subject_ids": [],
            }
        )
    elif attack == "old_pool":
        raw["digital_delivery"] = []
    elif attack == "criterion_id":
        raw["digital_requirements"][0]["verification_id"] = "v_old"
    else:
        raw["digital_requirements"][0][attack] = (
            "Deliver a prepared print package for the plate." if attack == "text" else {}
        )
    case = run_source_intake(tmp_path, monkeypatch, source, [raw])
    assert case.state == "error" and len(case.calls) == 1
    assert case.calls[0]["output_model"] is SourceDeliveryContractDraft
    assert not case.saved.get("intent_compilation")
    assert case.saved["contract"] is None and not case.saved["revisions"]


def test_ordinary_row_cannot_capture_an_internal_digital_criterion_id(tmp_path, monkeypatch):
    draft, source = deliveries()
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=DIGITAL_IDS)
    raw["requirements"].append(
        {
            "id": "smuggled",
            "text": "Retain a separate authored duty",
            "source": "request",
            "critical": True,
            "quote_ref": raw["digital_requirements"][0]["quote_ref"],
            "subject_ids": [],
            "verification_id": "source_delivery_0",
            "scalar": None,
        }
    )
    raw["requirement_order"].append("smuggled")
    expected = SourceDeliveryContractDraft.model_validate(raw).model_dump(mode="json")
    case = run_source_intake(tmp_path, monkeypatch, source, [raw, raw])
    assert case.state == "needs_review" and len(case.calls) == 2
    assert all(call["output_model"] is SourceDeliveryContractDraft for call in case.calls)
    record = case.saved["intent_compilation"]
    assert record["status"] == "invalid_interpretation" and len(record["attempts"]) == 2
    assert all(
        attempt["source_proposal"]["raw_response"] == expected for attempt in record["attempts"]
    )
    assert record["source_proposal"]["raw_response"] == expected
    assert record["source_proposal"]["source_only_requirement_ids"] == DIGITAL_IDS
    assert "lowered_draft_hash" not in record["source_proposal"]
    assert case.saved["contract"] is None and not case.saved["revisions"]


@pytest.mark.parametrize(
    "extra",
    [
        "withstand 10 kg",
        "be waterproof",
        "work on my Ender 3",
        "use heat-qualified material",
        "include G-code",
        "be 80 mm wide",
    ],
)
@pytest.mark.parametrize("retain_unknown", [False, True])
def test_legal_digital_wire_never_covers_extra_source_duties(
    tmp_path, monkeypatch, extra, retain_unknown
):
    source = f"Give me STL and {extra}."
    digital = Requirement.model_validate(
        {
            "id": "stl",
            "text": "STL",
            "source_quote": "STL",
            "verification": {
                "kind": "digital_delivery",
                "scope": "digital_artifacts",
                "coverage": "all_made",
                "deliverable": "project_stl",
                "source_quote": "STL",
            },
        }
    )
    rows = [digital]
    attribution = IntentAttribution()
    if retain_unknown:
        rows.append(Requirement(id="extra", text=extra, source_quote=extra))
        attribution.issues.append(
            IntentIssue(
                id="extra_input",
                question="Which source conditions must this extra duty meet?",
                reason="The explicit extra duty needs customer information.",
                requirement_ids=["extra"],
            )
        )
    draft = ContractDraft(contract=Contract(requirements=rows), attribution=attribution)
    raw = source_delivery_wire_fixture(draft, digital_requirement_ids=["stl"])
    responses = [raw] if retain_unknown else [raw, draft.model_dump(mode="json")]
    case = run_source_intake(tmp_path, monkeypatch, source, responses)
    record = case.saved["intent_compilation"]
    assert record["source_proposal"]["source_only_requirement_ids"] == ["stl"]
    assert case.calls[0]["output_model"] is SourceDeliveryContractDraft
    if retain_unknown:
        assert case.state == "needs_input" and len(case.calls) == 1
        retained = case.saved["intent_attribution"]["proposed_contract"]["requirements"][-1]
        assert retained == rows[-1].model_dump(mode="json")
        assert retained["critical"] and retained["metric"] == "unmeasured"
        assert retained.get("verification") is None
    else:
        assert case.state == "needs_review" and len(case.calls) == 2
        assert all(attempt["status"] == "invalid_interpretation" for attempt in record["attempts"])
        assert all(
            any(row["code"] == "incomplete_digital_source_scope" for row in attempt["diagnostics"])
            for attempt in record["attempts"]
        )
    assert case.saved["contract"] is None and not case.saved["revisions"]
    assert not case.saved.get("accepted_revision")
