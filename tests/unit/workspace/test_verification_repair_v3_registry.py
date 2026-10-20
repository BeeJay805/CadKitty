"""The retained V3 response catalog cannot mint source-authenticated photo claims."""

from typing import get_args

import pytest
from pydantic import ValidationError

from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.verification_models import (
    VerificationCriterion,
    VerificationRepairCriterionV3,
)
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


def test_v3_catalog_stays_exactly_five_while_canonical_can_persist_photo():
    def kinds(union):
        return tuple(
            model.model_json_schema()["properties"]["kind"]["const"] for model in get_args(union)
        )

    assert kinds(VerificationRepairCriterionV3) == KINDS
    assert kinds(VerificationCriterion) == (*KINDS, "nominal_photo_solid")


def test_v3_response_has_no_photo_slot_and_rejects_extra_even_null():
    source = "Give me a prepared print package."
    original = ContractDraft.model_validate(
        {
            "contract": {
                "requirements": [
                    {
                        "id": "printing",
                        "text": source,
                        "source_quote": source,
                        "verification": {
                            "kind": "digital_delivery",
                            "scope": "digital_artifacts",
                            "deliverable": "project_stl",
                            "coverage": "all_made",
                            "source_quote": source,
                        },
                    }
                ]
            },
            "attribution": {},
        }
    )
    prepared = prepare_verification_repair(original, source, compile_intent_draft(original, source))
    assert prepared is not None
    slot = {
        **dict.fromkeys(KINDS),
        "requirement_id": "printing",
        "unverified_reason": "Unsupported duty retained",
    }
    response = prepared.response_model.model_validate({"repair_0": slot})
    assert set(response.repair_0.model_dump()) == {*KINDS, "requirement_id", "unverified_reason"}
    with pytest.raises(ValidationError):
        prepared.response_model.model_validate({"repair_0": {**slot, "nominal_photo_solid": None}})
    retained = apply_verification_repair(original, source, prepared, response)
    assert (
        retained.contract.requirements[0].critical
        and retained.contract.requirements[0].verification is None
    )


def test_invalid_photo_slot_requires_original_source_wire_not_v3():
    from tests.unit.workspace.test_photo_nominal_claims import SOURCE, claim, draft

    from forge_mind.workspace.photo_nominal import attach_photo_nominal_claims

    original = attach_photo_nominal_claims(draft(SOURCE), SOURCE, [claim()])
    original.contract.requirements[0].verification.size_mm = (51.0, 30.0, 5.0)
    compilation = compile_intent_draft(original, SOURCE)
    assert compilation["status"] == "invalid_interpretation"
    assert prepare_verification_repair(original, SOURCE, compilation) is None
