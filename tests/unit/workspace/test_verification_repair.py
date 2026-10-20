"""Pure protected repair controls; no model or CAD execution."""

from copy import deepcopy
from typing import get_args

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_intent_compilation import (
    preparation_issue,
    valid_draft,
    wrong_family_draft,
    wrong_inventory_body_drafts,
)
from tests.unit.workspace.test_verification_admission import box_contract, pattern_contract

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.verification_models import VerificationRepairCriterionV3
from forge_mind.workspace.verification_repair import (
    apply_verification_repair,
    prepare_verification_repair,
)


def prepared(draft, source):
    return prepare_verification_repair(draft, source, compile_intent_draft(draft, source))


def repair_payload(preparation, corrected):
    rows = {row.id: row for row in corrected.contract.requirements}
    return {
        slot["slot"]: repair_slot(
            slot["requirement_id"],
            (
                rows[slot["requirement_id"]].verification.model_dump(mode="json")
                if rows[slot["requirement_id"]].verification is not None
                else None
            ),
        )
        for slot in preparation.basis["slots"]
    }


def repair_slot(identifier, verification, *, unverified_reason=None):
    payload = {
        model.model_json_schema()["properties"]["kind"]["const"]: None
        for model in get_args(VerificationRepairCriterionV3)
    }
    payload.update(requirement_id=identifier, unverified_reason=unverified_reason)
    if verification is not None:
        payload[verification["kind"]] = verification
    elif unverified_reason is None:
        payload["unverified_reason"] = "No supported source-backed criterion supplied."
    return payload


def test_whole_kind_certificate_removes_only_bad_kind_and_preserves_peers():
    initial, source = wrong_family_draft()
    corrected, _ = valid_draft()
    before = initial.model_dump(mode="json")
    preparation = prepared(initial, source)
    assert preparation is not None
    assert preparation.response_model.__name__ == "VerificationRepair"
    assert preparation.basis["slots"][0]["excluded_kinds"] == ["digital_delivery"]
    response = preparation.response_model.model_validate(repair_payload(preparation, corrected))
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert lowered.model_dump(mode="json") == corrected.model_dump(mode="json")
    assert initial.model_dump(mode="json") == before
    assert lowered.attribution == initial.attribution
    assert lowered.contract.requirements[1:] == initial.contract.requirements[1:]
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"
    record = preparation.response_record(response)
    assert record["raw_response"] == response.model_dump(mode="json")
    assert record["original_schema"] == preparation.original_schema
    assert "raw_response" not in preparation.request_record()


def test_raw_short_inventory_body_can_receive_source_backed_body_replacement():
    initial, corrected, source = wrong_inventory_body_drafts()
    preparation = prepared(initial, source)
    assert preparation is not None
    assert preparation.basis["slots"][0]["excluded_kinds"] == ["nominal_subject_inventory"]
    response = preparation.response_model.model_validate(repair_payload(preparation, corrected))
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert lowered.model_dump(mode="json") == corrected.model_dump(mode="json")
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"


@pytest.mark.parametrize(
    "change", ["excluded_kind", "wrong_id", "missing_slot", "extra_slot", "quote", "peer"]
)
def test_wire_schema_rejects_bad_kind_slot_quote_and_protected_peer(change):
    initial, source = wrong_family_draft()
    corrected, _ = valid_draft()
    preparation = prepared(initial, source)
    assert preparation is not None
    payload = repair_payload(preparation, corrected)
    slot = next(iter(payload))
    if change == "excluded_kind":
        payload[slot]["digital_delivery"] = initial.contract.requirements[
            0
        ].verification.model_dump(mode="json")
    elif change == "wrong_id":
        payload[slot]["requirement_id"] = "native"
    elif change == "missing_slot":
        payload.pop(slot)
    elif change == "extra_slot":
        payload["repair_999"] = deepcopy(payload[slot])
    elif change == "quote":
        payload[slot]["nominal_through_hole_pattern"]["source_quote"] = "editable native CAD"
    else:
        payload["contract"] = corrected.contract.model_dump(mode="json")
    with pytest.raises(ValidationError):
        preparation.response_model.model_validate(payload)


def test_null_retains_critical_unknown_without_author_targets_or_pass():
    initial, source = wrong_family_draft()
    preparation = prepared(initial, source)
    assert preparation is not None
    response = preparation.response_model.model_validate(
        {
            slot["slot"]: repair_slot(slot["requirement_id"], None)
            for slot in preparation.basis["slots"]
        }
    )
    lowered = apply_verification_repair(initial, source, preparation, response)
    original = initial.contract.requirements[0]
    result = lowered.contract.requirements[0]
    assert result.verification is None and result.critical
    assert result.model_dump(exclude={"verification"}) == original.model_dump(
        exclude={"verification"}
    )
    assert result.metric == "unmeasured" and result.value is None
    assert lowered.contract.requirements[1:] == initial.contract.requirements[1:]


@pytest.mark.parametrize("change", ["source", "draft", "certificate", "schema"])
def test_fresh_basis_and_certificate_recomputation_rejects_changed_input(change):
    initial, source = wrong_family_draft()
    corrected, _ = valid_draft()
    preparation = prepared(initial, source)
    assert preparation is not None
    response = preparation.response_model.model_validate(repair_payload(preparation, corrected))
    if change == "source":
        source += " Extra source duty."
    elif change == "draft":
        initial.contract.requirements[1].critical = False
    elif change == "certificate":
        preparation.basis["slots"][0]["exclusion_certificates"][0]["rule"] = "forged"
    else:
        preparation.original_schema["title"] = "Changed"
    with pytest.raises(ValueError, match="basis or shared exclusion"):
        apply_verification_repair(initial, source, preparation, response)


@pytest.mark.parametrize("change", ["blank", "whitespace", "question", "valid"])
def test_ineligible_draft_keeps_existing_full_correction(change):
    initial, source = wrong_family_draft()
    if change in {"blank", "whitespace"}:
        initial.contract.requirements[0].source_quote = "" if change == "blank" else " "
    elif change == "question":
        initial.attribution.issues = [preparation_issue()]
    else:
        initial, source = valid_draft()
    assert prepared(initial, source) is None


def test_no_certificate_same_family_quote_repair_uses_protected_slots():
    initial, source = valid_draft()
    corrected = initial.model_copy(deep=True)
    initial.contract.requirements[0].verification.source_quote = "editable native CAD"
    before = initial.model_dump(mode="json")
    preparation = prepared(initial, source)
    assert preparation is not None
    assert preparation.request_record()["protocol"] == "verification_repair_v3"
    assert preparation.basis["slots"][0]["excluded_kinds"] == []
    assert preparation.basis["slots"][0]["exclusion_certificates"] == []
    response = preparation.response_model.model_validate(repair_payload(preparation, corrected))
    lowered = apply_verification_repair(initial, source, preparation, response)
    assert lowered.model_dump(mode="json") == corrected.model_dump(mode="json")
    assert initial.model_dump(mode="json") == before
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"


def test_schema_does_not_fill_required_geometry_targets():
    initial, source = wrong_family_draft()
    preparation = prepared(initial, source)
    assert preparation is not None
    quote = initial.contract.requirements[0].source_quote
    with pytest.raises(ValidationError):
        preparation.response_model.model_validate(
            {
                "repair_0": repair_slot(
                    initial.contract.requirements[0].id,
                    {
                        "kind": "nominal_perforated_box",
                        "scope": "nominal_cad",
                        "source_quote": quote,
                    },
                )
            }
        )


@pytest.mark.parametrize(
    "kind",
    [
        "nominal_through_hole_pattern",
        "nominal_perforated_box",
        "nominal_subject_inventory",
        "nominal_delivery_scope",
        None,
    ],
)
def test_actual_codex_converter_preserves_each_remaining_family_and_literal_slot(kind):
    initial, source = wrong_family_draft()
    preparation = prepared(initial, source)
    assert preparation is not None
    original = preparation.original_schema
    wire = _strict_schema(original)
    constants = []

    def inspect(node):
        if isinstance(node, dict):
            if "const" in node:
                constants.append(node["const"])
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "prefixItems" not in node and "default" not in node
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)

    inspect(wire)
    assert "digital_delivery" not in constants
    assert {
        "nominal_through_hole_pattern",
        "nominal_perforated_box",
        "nominal_subject_inventory",
        "nominal_delivery_scope",
    } <= set(constants)
    requirement = initial.contract.requirements[0]
    assert requirement.id in constants and requirement.source_quote in constants
    if kind == "nominal_through_hole_pattern":
        criterion = pattern_contract().requirements[0].verification.model_dump(mode="json")
    elif kind == "nominal_perforated_box":
        criterion = box_contract().requirements[0].verification.model_dump(mode="json")
    elif kind == "nominal_subject_inventory":
        criterion = {
            "kind": kind,
            "scope": "nominal_cad",
            "coverage": "all_made",
            "items": [
                {
                    "subject_id": "plate",
                    "subject_role": "designed",
                    "required_name": "plate",
                    "quantity": 1,
                }
            ],
        }
    elif kind == "nominal_delivery_scope":
        criterion = {
            "kind": kind,
            "scope": "software_policy",
            "policy": "nominal_cad_and_digital_only",
        }
    else:
        criterion = None
    if criterion is not None:
        criterion["source_quote"] = requirement.source_quote
    payload = {"repair_0": repair_slot(requirement.id, criterion)}
    # Transport preserves payloads. Only later canonical admission can certify scope/targets.
    response = preparation.response_model.model_validate(payload)
    complete_payload = response.model_dump(mode="json")
    restored = _restore_maps(complete_payload, original, original)
    assert restored == complete_payload
    assert preparation.response_model.model_validate(restored) == response


def test_offset_trial_shaped_inventory_body_repairs_without_changing_valid_peer_or_fact():
    first = "Make one nominal CAD rectangular mounting plate named offset_plate."
    quote = (
        first + " Its overall size is 100 mm in local X, 50 mm in local Y and 8 mm "
        "in local Z, centered on the local origin. It must be one solid. "
        + pattern_contract(offset=True).requirements[0].source_quote
    )
    text = (
        "Provide a rectangular nominal CAD mounting plate with local XYZ size "
        "100 \u00d7 50 \u00d7 8 mm and outer-body center at the local origin. Include only the four "
        "specified uniform cylindrical through-holes parallel to local Z. Apply the stated "
        "symmetric diameter and radial position allowances to each hole."
    )
    inventory = {
        "kind": "nominal_subject_inventory",
        "scope": "nominal_cad",
        "coverage": "all_made",
        "source_quote": first,
        "items": [
            {
                "subject_id": "offset_plate",
                "subject_role": "designed",
                "required_name": "offset_plate",
                "quantity": 1,
            }
        ],
    }
    body = Requirement(id="r_geometry", text=text, source_quote=quote, verification=inventory)
    peer = Requirement(
        id="r_inventory",
        text="Provide one designed mounting plate with the saved PartDefinition.name offset_plate.",
        source_quote=first,
        verification=inventory,
    )
    attrs = IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "offset_plate",
                    "label": "offset_plate mounting plate",
                    "role": "designed",
                    "source_quote": first,
                }
            ],
            "facts": [
                {
                    "id": "f_size_x",
                    "subject_id": "offset_plate",
                    "property": "local_x_size",
                    "value": 100,
                    "unit": "mm",
                    "source_quote": "100 mm in local X",
                    "requirement_ids": ["r_geometry"],
                }
            ],
            "requirement_subjects": {
                "r_geometry": ["offset_plate"],
                "r_inventory": ["offset_plate"],
            },
        }
    )
    initial = ContractDraft(contract=Contract(requirements=[peer, body]), attribution=attrs)
    corrected = initial.model_copy(deep=True)
    criterion = pattern_contract(offset=True).requirements[0].verification.model_dump(mode="json")
    criterion.update(
        kind="nominal_perforated_box",
        size_mm=[100, 50, 8],
        center_mm=[0, 0, 0],
        subject_id="offset_plate",
        source_quote=quote,
    )
    corrected.contract.requirements[1] = Requirement.model_validate(
        {**body.model_dump(mode="json"), "verification": criterion}
    )
    preparation = prepared(initial, quote)
    assert preparation is not None
    assert [slot["requirement_id"] for slot in preparation.basis["slots"]] == ["r_geometry"]
    assert preparation.basis["slots"][0]["excluded_kinds"] == ["nominal_subject_inventory"]
    response = preparation.response_model.model_validate(repair_payload(preparation, corrected))
    result = apply_verification_repair(initial, quote, preparation, response)
    assert result.model_dump(mode="json") == corrected.model_dump(mode="json")
    assert result.contract.requirements[0] == peer and result.attribution == attrs
    assert compile_intent_draft(result, quote, previous=initial)["status"] == "ready"
