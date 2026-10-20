"""Generic protected correction keeps source duties and author-supplied targets."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_verification_repair import repair_payload

from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.verification_admission import verification_repair_exclusions
from forge_mind.workspace.verification_repair import (
    apply_verification_repair,
    prepare_verification_repair,
)


def offset003_drafts():
    """Exact failed body row plus compact retained peer/fact context from trial003.

    The corrected body is explicit test-author input. Production must not build
    it from the failed predicate, prose, or attribution facts.
    """
    subject = "s_offset_plate"
    naming = "Make one nominal CAD rectangular mounting plate named offset_plate."
    quote = (
        naming + " Its overall size is 100 mm in local X, 50 mm in local Y and 8 mm in local Z, "
        "centered on the local origin. It must be one solid. It has exactly four uniform "
        "cylindrical through-holes parallel to local Z: two 8 mm diameter holes at local "
        "XY (-35, -12) and (-35, 12) mm, and two 6 mm diameter holes at (30, -15) and "
        "(30, 15) mm. Use a symmetric diameter tolerance of 0.1 mm and a radial position "
        "tolerance of 0.1 mm for each hole."
    )
    scope_quote = "I need nominal CAD and digital files only."
    scope = {
        "version": 1,
        "kind": "nominal_delivery_scope",
        "scope": "software_policy",
        "policy": "nominal_cad_and_digital_only",
        "source_quote": scope_quote,
    }
    inventory = {
        "version": 1,
        "kind": "nominal_subject_inventory",
        "scope": "nominal_cad",
        "coverage": "all_made",
        "source_quote": naming,
        "items": [
            {
                "subject_id": subject,
                "subject_role": "designed",
                "required_name": "offset_plate",
                "quantity": 1,
            }
        ],
    }
    rows = [
        {
            "id": "r_inventory",
            "text": "Make one designed mounting plate with the saved definition name offset_plate.",
            "source_quote": naming,
            "verification": inventory,
        },
        {
            "id": "r_plate_geometry",
            "text": (
                "The nominal CAD plate is a rectangular body of local XYZ size "
                "100 \u00d7 50 \u00d7 8 mm, with its outer-body center at the local origin. "
                "Its only material removals are four uniform cylindrical through-holes "
                "parallel to local Z: diameter 8 mm at local XY (-35, -12) and (-35, 12) mm, "
                "and diameter 6 mm at (30, -15) and (30, 15) mm. Each hole has symmetric "
                "diameter tolerance 0.1 mm and radial position tolerance 0.1 mm."
            ),
            "source_quote": quote,
            "source": "request",
            "metric": "unmeasured",
            "comparator": "eq",
            "value": None,
            "tolerance": 0.0,
            "part_id": None,
            "axis": "z",
            "center": None,
            "position_tolerance": 0.1,
            "critical": True,
            "verification": deepcopy(scope),
        },
        {
            "id": "r_solid_count",
            "text": "The plate must be one solid.",
            "source_quote": "It must be one solid.",
            "metric": "solid_count",
            "value": 1.0,
        },
        {
            "id": "r_delivery_scope",
            "text": "Delivery is restricted to nominal CAD and digital files.",
            "source_quote": scope_quote,
            "verification": scope,
        },
    ]
    for identifier, text, clause, deliverable in (
        (
            "r_native",
            "Deliver editable native CAD for all made parts.",
            "editable native CAD",
            "editable_native",
        ),
        ("r_step", "Deliver STEP for all made parts.", "STEP", "project_step"),
        ("r_stl", "Deliver STL for all made parts.", "STL", "project_stl"),
        (
            "r_print_package",
            "Deliver a prepared print package for all made parts.",
            "a prepared print package",
            "prepared_print_package",
        ),
    ):
        rows.append(
            {
                "id": identifier,
                "text": text,
                "source_quote": clause,
                "verification": {
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "coverage": "all_made",
                    "deliverable": deliverable,
                    "source_quote": clause,
                },
            }
        )
    initial = ContractDraft(
        contract=Contract.model_validate({"requirements": rows}),
        attribution=IntentAttribution.model_validate(
            {
                "subjects": [
                    {
                        "id": subject,
                        "label": "offset_plate mounting plate",
                        "role": "designed",
                        "source_quote": naming,
                    }
                ],
                "requirement_subjects": {row["id"]: [subject] for row in rows},
                "facts": [
                    {
                        "id": "f_plate_quantity",
                        "subject_id": subject,
                        "property": "occurrence_quantity",
                        "value": 1.0,
                        "unit": "plate",
                        "source_quote": naming,
                        "requirement_ids": ["r_inventory"],
                    },
                    {
                        "id": "f_size_x",
                        "subject_id": subject,
                        "property": "overall_size_local_x",
                        "value": 100.0,
                        "unit": "mm",
                        "source_quote": "100 mm in local X",
                        "requirement_ids": ["r_plate_geometry"],
                    },
                    {
                        "id": "f_size_y",
                        "subject_id": subject,
                        "property": "overall_size_local_y",
                        "value": 50.0,
                        "unit": "mm",
                        "source_quote": "50 mm in local Y",
                        "requirement_ids": ["r_plate_geometry"],
                    },
                ],
            }
        ),
    )
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements[1] = Requirement.model_validate(
        {
            **initial.contract.requirements[1].model_dump(mode="json"),
            "verification": {
                "version": 1,
                "kind": "nominal_perforated_box",
                "scope": "nominal_cad",
                "subject_id": subject,
                "subject_role": "designed",
                "frame": "definition_local",
                "axis": "z",
                "count_policy": "exact",
                "size_mm": [100, 50, 8],
                "center_mm": [0, 0, 0],
                "source_quote": quote,
                "holes": [
                    {
                        "center_mm": center,
                        "diameter_mm": diameter,
                        "diameter_tolerance_mm": 0.1,
                        "position_tolerance_mm": 0.1,
                    }
                    for center, diameter in [
                        ([-35, -12], 8),
                        ([-35, 12], 8),
                        ([30, -15], 6),
                        ([30, 15], 6),
                    ]
                ],
            },
        }
    )
    source = (
        quote
        + " "
        + scope_quote
        + " Give me editable native CAD, STEP, STL and a prepared print package."
    )
    return initial, corrected, source


def prepared(draft, source):
    return prepare_verification_repair(draft, source, compile_intent_draft(draft, source))


def authored_response(preparation, corrected):
    return preparation.response_model.model_validate(repair_payload(preparation, corrected))


def test_retained_offset003_wrong_scope_enters_protected_repair_without_certificate():
    initial, corrected, source = offset003_drafts()
    before = initial.model_dump(mode="json")
    body = initial.contract.requirements[1]
    assert body.verification.kind == "nominal_delivery_scope"
    assert verification_repair_exclusions(body, source, initial.attribution) == []
    compilation = compile_intent_draft(initial, source)
    assert compilation["status"] == "invalid_interpretation"
    assert {row["requirement_id"] for row in compilation["diagnostics"]} == {body.id}
    preparation = prepared(initial, source)
    assert preparation is not None
    assert preparation.request_record()["protocol"] == "verification_repair_v3"
    assert [slot["requirement_id"] for slot in preparation.basis["slots"]] == [body.id]
    assert preparation.basis["slots"][0]["excluded_kinds"] == []
    assert preparation.basis["slots"][0]["exclusion_certificates"] == []
    lowered = apply_verification_repair(
        initial, source, preparation, authored_response(preparation, corrected)
    )
    assert lowered.model_dump(mode="json") == corrected.model_dump(mode="json")
    assert initial.model_dump(mode="json") == before
    assert lowered.attribution == initial.attribution
    for old, new in zip(initial.contract.requirements, lowered.contract.requirements, strict=True):
        assert old.model_dump(exclude={"verification"}) == new.model_dump(exclude={"verification"})
        if old.id != body.id:
            assert old == new
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"


@pytest.mark.parametrize(
    "kind", ["digital_delivery", "nominal_subject_inventory", "nominal_delivery_scope"]
)
def test_protected_route_does_not_depend_on_current_wrong_family(kind):
    initial, corrected, source = offset003_drafts()
    peer_index = {
        "digital_delivery": 4,
        "nominal_subject_inventory": 0,
        "nominal_delivery_scope": 3,
    }[kind]
    initial.contract.requirements[1].verification = deepcopy(
        initial.contract.requirements[peer_index].verification
    )
    preparation = prepared(initial, source)
    assert preparation is not None
    lowered = apply_verification_repair(
        initial, source, preparation, authored_response(preparation, corrected)
    )
    assert lowered == corrected
    assert compile_intent_draft(lowered, source, previous=initial)["status"] == "ready"


@pytest.mark.parametrize(
    "change", ["size", "center", "diameter", "diameter_band", "position_band", "hole_count"]
)
def test_author_replacement_targets_are_not_filled_or_accepted_when_changed(change):
    initial, corrected, source = offset003_drafts()
    body = corrected.contract.requirements[1].verification
    if change == "size":
        body.size_mm = (101, 50, 8)
    elif change == "center":
        body.center_mm = (1, 0, 0)
    elif change == "diameter":
        body.holes[0].diameter_mm = 9
    elif change == "diameter_band":
        body.holes[0].diameter_tolerance_mm = 0.2
    elif change == "position_band":
        body.holes[0].position_tolerance_mm = 0.2
    else:
        body.holes.pop()
    preparation = prepared(initial, source)
    assert preparation is not None
    lowered = apply_verification_repair(
        initial, source, preparation, authored_response(preparation, corrected)
    )
    assert lowered.contract.requirements[1].verification == body
    assert lowered.attribution == initial.attribution
    assert (
        lowered.contract.requirements[1].source_quote
        == initial.contract.requirements[1].source_quote
    )
    assert (
        compile_intent_draft(lowered, source, previous=initial)["status"]
        == "invalid_interpretation"
    )


@pytest.mark.parametrize("family", ["literal_inventory_name", "prepared_deliverable"])
def test_valid_same_family_repair_is_available_without_kind_exclusion(family):
    if family == "literal_inventory_name":
        quote = "Make one nominal CAD plate named 'rectangular plate'."
        row = {
            "id": "identity",
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
        attrs = IntentAttribution.model_validate(
            {
                "subjects": [
                    {"id": "plate", "label": "Plate", "role": "designed", "source_quote": quote}
                ],
                "requirement_subjects": {"identity": ["plate"]},
            }
        )
    else:
        quote = "Give me a prepared print package."
        row = {
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
        attrs = IntentAttribution()
    initial = ContractDraft(
        contract=Contract(requirements=[Requirement.model_validate(row)]), attribution=attrs
    )
    corrected = initial.model_copy(deep=True)
    if family == "literal_inventory_name":
        corrected.contract.requirements[0].verification.items[0].required_name = "rectangular plate"
    else:
        corrected.contract.requirements[0].verification.deliverable = "prepared_print_package"
    assert verification_repair_exclusions(initial.contract.requirements[0], quote, attrs) == []
    preparation = prepared(initial, quote)
    assert preparation is not None
    lowered = apply_verification_repair(
        initial, quote, preparation, authored_response(preparation, corrected)
    )
    assert (
        lowered.contract.requirements[0].verification.kind
        == initial.contract.requirements[0].verification.kind
    )
    assert compile_intent_draft(lowered, quote, previous=initial)["status"] == "ready"


@pytest.mark.parametrize("change", ["source", "scalar", "assumption", "attribution"])
def test_nonverification_authority_errors_keep_full_correction(change):
    initial, _, source = offset003_drafts()
    body = initial.contract.requirements[1]
    if change == "source":
        body.source_quote = "This source is not in the customer request."
    elif change == "scalar":
        body.metric, body.value = "extent_x", 100.0
    elif change == "assumption":
        body.source = "assumption"
    else:
        initial.attribution.subjects[0].source_quote = "Invented subject source"
    assert compile_intent_draft(initial, source)["status"] == "invalid_interpretation"
    assert prepared(initial, source) is None


def test_linked_fact_question_survives_protected_repair_and_blocks_ready():
    initial, corrected, source = offset003_drafts()
    raw = initial.attribution.model_dump(mode="json")
    raw["issues"] = [
        {
            "id": "datum_question",
            "question": "Which source datum applies?",
            "reason": "Retained unresolved source fact",
            "fact_ids": ["f_size_x"],
            "requirement_ids": [],
        }
    ]
    initial.attribution = IntentAttribution.model_validate(raw)
    corrected.attribution = initial.attribution.model_copy(deep=True)
    before = initial.attribution.model_dump(mode="json")
    assert (
        verification_repair_exclusions(
            initial.contract.requirements[1], source, initial.attribution
        )
        == []
    )
    preparation = prepared(initial, source)
    assert preparation is not None
    lowered = apply_verification_repair(
        initial, source, preparation, authored_response(preparation, corrected)
    )
    assert lowered.attribution.model_dump(mode="json") == before
    result = compile_intent_draft(lowered, source, previous=initial)
    assert result["status"] == "needs_customer_input"
    assert result["blocking_issue_ids"] == ["datum_question"]
    lowered.attribution.issues = []
    assert any(
        row["code"] == "removed_customer_question"
        for row in compile_intent_draft(lowered, source, previous=initial)["diagnostics"]
    )


def test_prepared_machine_subcase_retains_question_and_unknown_critical_duty():
    quote = "Give me a prepared print package for my Ender 3."
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
                            "deliverable": "prepared_print_package",
                            "source_quote": quote,
                        },
                    }
                )
            ]
        ),
        attribution=IntentAttribution.model_validate(
            {
                "issues": [
                    {
                        "id": "machine",
                        "question": "Which printer configuration applies?",
                        "reason": "Explicit machine-specific source scope",
                        "requirement_ids": ["printing"],
                        "input_dependency": {
                            "kind": "source_fact",
                            "input_name": "printer_configuration",
                        },
                    }
                ]
            }
        ),
    )
    assert (
        verification_repair_exclusions(initial.contract.requirements[0], quote, initial.attribution)
        == []
    )
    preparation = prepared(initial, quote)
    assert preparation is not None
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements[0].verification = None
    lowered = apply_verification_repair(
        initial, quote, preparation, authored_response(preparation, corrected)
    )
    assert lowered.contract.requirements[0].critical
    assert lowered.contract.requirements[0].metric == "unmeasured"
    assert lowered.contract.requirements[0].source_quote == quote
    assert lowered.attribution == initial.attribution
    assert (
        compile_intent_draft(lowered, quote, previous=initial)["status"] == "needs_customer_input"
    )
