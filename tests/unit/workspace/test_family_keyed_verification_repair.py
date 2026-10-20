"""Independent v3 wire/authority controls; no model, CAD or native worker."""

from copy import deepcopy
from typing import get_args

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_generic_verification_repair import offset003_drafts

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.requirement_verification import aggregate
from forge_mind.workspace.verification_models import VerificationRepairCriterionV3
from forge_mind.workspace.verification_repair import (
    apply_verification_repair,
    prepare_verification_repair,
)

KINDS = (
    "nominal_through_hole_pattern",
    "nominal_perforated_box",
    "digital_delivery",
    "nominal_subject_inventory",
    "nominal_delivery_scope",
)


def retained_offset004_drafts():
    """Exact failed first r1 and portable peer/fact context; no scratch dependency.

    Raw answer SHA256 1e8bd03745431cef617584faaf2cd870f0d957d267e9e82dee5a862f0ed9aad2.
    The corrected geometric payload is explicit test-author input from the
    shared fixture, never inferred by product code from the failed inventory.
    """
    initial, corrected, source = offset003_drafts()
    initial.contract.requirements[1].verification = deepcopy(
        initial.contract.requirements[0].verification
    )
    for draft in (initial, corrected):
        raw = draft.model_dump(mode="json")
        for row in raw["contract"]["requirements"]:
            if row["id"] == "r_plate_geometry":
                row["id"] = "r1"
                row["text"] = (
                    "The nominal CAD plate must be a rectangular body of local XYZ size "
                    "100 \u00d7 50 \u00d7 8 mm, centered on the local origin, with only the "
                    "specified four cylindrical through-holes parallel to local Z. Hole "
                    "diameters have symmetric ±0.1 mm tolerance, and hole centers have "
                    "radial position tolerance 0.1 mm."
                )
            criterion = row.get("verification") or {}
            if "subject_id" in criterion:
                criterion["subject_id"] = "offset_plate"
            for item in criterion.get("items", []):
                item["subject_id"] = "offset_plate"
        raw["attribution"]["subjects"][0]["id"] = "offset_plate"
        for fact in raw["attribution"]["facts"]:
            fact["subject_id"] = "offset_plate"
            fact["requirement_ids"] = [
                "r1" if key == "r_plate_geometry" else key for key in fact["requirement_ids"]
            ]
        raw["attribution"]["requirement_subjects"] = {
            "r1" if key == "r_plate_geometry" else key: ["offset_plate"]
            for key in raw["attribution"]["requirement_subjects"]
        }
        replacement = ContractDraft.model_validate(raw)
        draft.contract, draft.attribution = replacement.contract, replacement.attribution
    return initial, corrected, source


def prepared(draft, source):
    value = prepare_verification_repair(draft, source, compile_intent_draft(draft, source))
    assert value is not None
    return value


def slot_payload(preparation, criterion=None, reason=None):
    assert len(preparation.basis["slots"]) == 1
    slot = preparation.basis["slots"][0]
    value = dict.fromkeys(KINDS)
    value.update(requirement_id=slot["requirement_id"], unverified_reason=reason)
    if criterion is not None:
        value[criterion["kind"]] = criterion
    return {slot["slot"]: value}


def candidate(kind, initial, corrected):
    body = corrected.contract.requirements[1].verification.model_dump(mode="json")
    if kind == "nominal_perforated_box":
        return body
    if kind == "nominal_through_hole_pattern":
        return {
            **{key: value for key, value in body.items() if key not in {"size_mm", "center_mm"}},
            "kind": kind,
        }
    peer = {"digital_delivery": 4, "nominal_subject_inventory": 0, "nominal_delivery_scope": 3}
    return {
        **initial.contract.requirements[peer[kind]].verification.model_dump(mode="json"),
        "source_quote": initial.contract.requirements[1].source_quote,
    }


def test_saved004_body_replacement_preserves_peers_scalars_facts_and_complete_source():
    initial, corrected, source = retained_offset004_drafts()
    before = initial.model_dump(mode="json")
    preparation = prepared(initial, source)
    assert preparation.request_record()["protocol"] == "verification_repair_v3"
    assert preparation.basis["slots"][0]["excluded_kinds"] == ["nominal_subject_inventory"]
    payload = slot_payload(
        preparation, corrected.contract.requirements[1].verification.model_dump(mode="json")
    )
    response = preparation.response_model.model_validate(payload)
    retained = preparation.response_record(response)
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert retained["raw_response"] == response.model_dump(mode="json")
    assert lowered == corrected
    assert initial.model_dump(mode="json") == before
    assert lowered.attribution == initial.attribution
    for old, new in zip(initial.contract.requirements, lowered.contract.requirements, strict=True):
        assert old.model_dump(exclude={"verification"}) == new.model_dump(exclude={"verification"})
        if old.id != "r1":
            assert old == new
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"


@pytest.mark.parametrize("missing", ["requirement_id", *KINDS, "unverified_reason"])
def test_every_family_and_reason_key_is_locally_required(missing):
    initial, corrected, source = offset003_drafts()
    preparation = prepared(initial, source)
    payload = slot_payload(preparation, candidate("nominal_perforated_box", initial, corrected))
    del payload["repair_0"][missing]
    with pytest.raises(ValidationError):
        preparation.response_model.model_validate(payload)


def test_certified_exclusion_is_a_required_null_only_field():
    initial, corrected, source = retained_offset004_drafts()
    preparation = prepared(initial, source)
    wire = _strict_schema(preparation.original_schema)
    reference = wire["properties"]["repair_0"]["$ref"].split("/")[-1]
    fields = wire["$defs"][reference]
    assert "nominal_subject_inventory" in fields["required"]
    assert fields["properties"]["nominal_subject_inventory"]["type"] == "null"
    payload = slot_payload(preparation, candidate("nominal_perforated_box", initial, corrected))
    payload["repair_0"]["nominal_subject_inventory"] = candidate(
        "nominal_subject_inventory", initial, corrected
    )
    with pytest.raises(ValidationError):
        preparation.response_model.model_validate(payload)


@pytest.mark.parametrize("kind", KINDS)
def test_each_concrete_family_survives_actual_strict_conversion_and_restoration(kind):
    initial, corrected, source = offset003_drafts()
    preparation = prepared(initial, source)
    assert preparation.basis["slots"][0]["excluded_kinds"] == []
    assert (
        tuple(
            model.model_json_schema()["properties"]["kind"]["const"]
            for model in get_args(VerificationRepairCriterionV3)
        )
        == KINDS
    )
    criterion = candidate(kind, initial, corrected)
    response = preparation.response_model.model_validate(slot_payload(preparation, criterion))
    complete = response.model_dump(mode="json")
    wire = _strict_schema(preparation.original_schema)

    def inspect(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            if "anyOf" in node:
                assert len(node["anyOf"]) == 2
                assert sum(branch.get("type") == "null" for branch in node["anyOf"]) == 1
            assert "default" not in node and "prefixItems" not in node
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)

    inspect(wire)
    restored = _restore_maps(complete, wire, wire)
    assert restored == complete
    assert preparation.response_model.model_validate(restored) == response
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert (
        lowered.contract.requirements[1].verification.model_dump(mode="json")
        == complete["repair_0"][kind]
    )
    # Wire fidelity cannot certify applicability of another family to this duty.
    expected = "ready" if kind == "nominal_perforated_box" else "invalid_interpretation"
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == expected


@pytest.mark.parametrize(
    "choice", ["all_null", "blank_reason", "two_families", "payload_and_reason"]
)
def test_bad_choice_remains_recordable_then_apply_rejects(choice):
    initial, corrected, source = offset003_drafts()
    before = initial.model_dump(mode="json")
    preparation = prepared(initial, source)
    payload = slot_payload(preparation)
    if choice == "blank_reason":
        payload["repair_0"]["unverified_reason"] = " \t\n"
    elif choice == "two_families":
        for kind in ("nominal_perforated_box", "nominal_delivery_scope"):
            payload["repair_0"][kind] = candidate(kind, initial, corrected)
    elif choice == "payload_and_reason":
        payload["repair_0"]["nominal_perforated_box"] = candidate(
            "nominal_perforated_box", initial, corrected
        )
        payload["repair_0"]["unverified_reason"] = "Cannot supply a supported predicate"
    response = preparation.response_model.model_validate(payload)
    retained = preparation.response_record(response)
    assert retained["raw_response"] == response.model_dump(mode="json")
    with pytest.raises(ValueError):
        apply_verification_repair(initial, source, preparation, response)
    assert retained["raw_response"] == response.model_dump(mode="json")
    assert initial.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "change", ["wrong_key", "quote", "missing_target", "wrong_id", "old_union"]
)
def test_structural_fields_still_require_canonical_kind_quote_and_target(change):
    initial, corrected, source = offset003_drafts()
    preparation = prepared(initial, source)
    payload = slot_payload(preparation, candidate("nominal_perforated_box", initial, corrected))
    slot = payload["repair_0"]
    if change == "wrong_key":
        slot["nominal_delivery_scope"] = slot.pop("nominal_perforated_box")
        slot["nominal_perforated_box"] = None
    elif change == "quote":
        slot["nominal_perforated_box"]["source_quote"] = "A shortened quote"
    elif change == "missing_target":
        del slot["nominal_perforated_box"]["size_mm"]
    elif change == "wrong_id":
        slot["requirement_id"] = "r_inventory"
    else:
        payload["repair_0"] = {
            "requirement_id": slot["requirement_id"],
            "verification": slot["nominal_perforated_box"],
        }
    with pytest.raises(ValidationError):
        preparation.response_model.model_validate(payload)


@pytest.mark.parametrize(
    "change", ["size", "center", "diameter", "association", "diameter_band", "position_band"]
)
def test_populated_body_target_changes_fail_unchanged_compiler(change):
    initial, corrected, source = retained_offset004_drafts()
    preparation = prepared(initial, source)
    target = corrected.contract.requirements[1].verification.model_dump(mode="json")
    if change == "size":
        target["size_mm"][0] = 101
    elif change == "center":
        target["center_mm"][0] = 1
    elif change == "diameter":
        target["holes"][0]["diameter_mm"] = 9
    elif change == "association":
        target["holes"][0]["diameter_mm"], target["holes"][2]["diameter_mm"] = 6, 8
    elif change == "diameter_band":
        target["holes"][0]["diameter_tolerance_mm"] = 0.2
    else:
        target["holes"][0]["position_tolerance_mm"] = 0.2
    response = preparation.response_model.model_validate(slot_payload(preparation, target))
    preparation.response_record(response)
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert lowered.contract.requirements[1].verification.model_dump(mode="json") == target
    assert lowered.attribution == initial.attribution
    assert (
        compile_intent_draft(lowered, source, previous=initial)["status"]
        == "invalid_interpretation"
    )


def test_literal_geometry_name_can_be_repaired_within_inventory_family():
    quote = "Make one nominal CAD plate named 'rectangular plate'."
    initial = ContractDraft(
        contract=Contract(
            requirements=[
                Requirement.model_validate(
                    {
                        "id": "name",
                        "text": "Deliver the exact saved name rectangular plate.",
                        "source_quote": quote,
                        "verification": {
                            "kind": "nominal_subject_inventory",
                            "scope": "nominal_cad",
                            "coverage": "all_made",
                            "source_quote": quote,
                            "items": [
                                {
                                    "subject_id": "plate",
                                    "subject_role": "designed",
                                    "required_name": "wrong_name",
                                    "quantity": 1,
                                }
                            ],
                        },
                    }
                )
            ]
        ),
        attribution=IntentAttribution.model_validate(
            {
                "subjects": [
                    {"id": "plate", "label": "Plate", "role": "designed", "source_quote": quote}
                ],
                "requirement_subjects": {"name": ["plate"]},
            }
        ),
    )
    preparation = prepared(initial, quote)
    assert preparation.basis["slots"][0]["excluded_kinds"] == []
    target = initial.contract.requirements[0].verification.model_dump(mode="json")
    target["items"][0]["required_name"] = "rectangular plate"
    response = preparation.response_model.model_validate(slot_payload(preparation, target))
    lowered = apply_verification_repair(initial, quote, preparation, response)
    assert lowered.contract.requirements[0].verification.kind == "nominal_subject_inventory"
    assert compile_intent_draft(lowered, quote, previous=initial)["status"] == "ready"


def test_prepared_deliverable_can_be_repaired_without_excluding_digital_family():
    quote = "Give me a prepared print package."
    initial = ContractDraft(
        contract=Contract(
            requirements=[
                Requirement.model_validate(
                    {
                        "id": "printing",
                        "text": quote,
                        "source_quote": quote,
                        "verification": {
                            "kind": "digital_delivery",
                            "scope": "digital_artifacts",
                            "coverage": "all_made",
                            "deliverable": "project_stl",
                            "source_quote": quote,
                        },
                    }
                )
            ]
        ),
        attribution=IntentAttribution(),
    )
    preparation = prepared(initial, quote)
    assert preparation.basis["slots"][0]["excluded_kinds"] == []
    target = initial.contract.requirements[0].verification.model_dump(mode="json")
    target["deliverable"] = "prepared_print_package"
    response = preparation.response_model.model_validate(slot_payload(preparation, target))
    lowered = apply_verification_repair(initial, quote, preparation, response)
    assert compile_intent_draft(lowered, quote, previous=initial)["status"] == "ready"


def test_linked_question_stays_outside_writable_payload_and_blocks_ready():
    initial, corrected, source = offset003_drafts()
    raw = initial.attribution.model_dump(mode="json")
    raw["issues"] = [
        {
            "id": "datum",
            "question": "Which datum applies?",
            "reason": "Retained source ambiguity",
            "fact_ids": ["f_size_x"],
            "requirement_ids": [],
        }
    ]
    initial.attribution = IntentAttribution.model_validate(raw)
    preparation = prepared(initial, source)
    response = preparation.response_model.model_validate(
        slot_payload(preparation, candidate("nominal_perforated_box", initial, corrected))
    )
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert lowered.attribution == initial.attribution
    result = compile_intent_draft(lowered, source, previous=initial)
    assert result["status"] == "needs_customer_input" and result["blocking_issue_ids"] == ["datum"]


def test_explicit_physical_unknown_is_recorded_and_never_becomes_achieved_proof():
    quote = "The plate must withstand a 100 N load."
    initial = ContractDraft(
        contract=Contract(
            requirements=[
                Requirement.model_validate(
                    {
                        "id": "physical",
                        "text": quote,
                        "source_quote": quote,
                        "verification": {
                            "kind": "digital_delivery",
                            "scope": "digital_artifacts",
                            "coverage": "all_made",
                            "deliverable": "project_step",
                            "source_quote": quote,
                        },
                    }
                )
            ]
        ),
        attribution=IntentAttribution(),
    )
    before = initial.contract.requirements[0].model_dump(exclude={"verification"})
    preparation = prepared(initial, quote)
    reason = "No supported frozen predicate proves the retained physical load duty"
    response = preparation.response_model.model_validate(slot_payload(preparation, reason=reason))
    retained = preparation.response_record(response)
    wire = _strict_schema(preparation.original_schema)
    assert _restore_maps(retained["raw_response"], wire, wire) == retained["raw_response"]
    lowered = apply_verification_repair(initial, quote, preparation, response)
    row = lowered.contract.requirements[0]
    assert row.verification is None and row.metric == "unmeasured" and row.critical
    assert row.model_dump(exclude={"verification"}) == before
    assert retained["raw_response"]["repair_0"]["unverified_reason"] == reason
    assert aggregate(lowered.contract, [{"requirement_id": row.id, "status": "unverified"}]) == (
        "unverified",
        False,
    )
