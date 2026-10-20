"""Source-retaining intake classification; no model, CAD or worker execution."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_verification_admission import (
    attribution,
    box_attribution,
    box_contract,
    pattern_contract,
    source,
)

from forge_mind.workspace.intent_attribution import (
    ContractDraft,
    IntentAttribution,
    IntentIssue,
)
from forge_mind.workspace.intent_capabilities import (
    preparation_capability,
    validate_preparation_source_scope,
)
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.native_authoring import native_bootstrap
from forge_mind.workspace.print_setup_models import SCOPE, PrintSetupRequest


def valid_draft():
    target = pattern_contract()
    target.requirements.extend(
        [
            Requirement.model_validate(
                {
                    "id": requirement_id,
                    "text": quote,
                    "source_quote": quote,
                    "verification": {
                        "kind": "digital_delivery",
                        "scope": "digital_artifacts",
                        "deliverable": deliverable,
                        "coverage": "all_made",
                        "source_quote": quote,
                    },
                }
            )
            for requirement_id, quote, deliverable in (
                ("native", "editable native CAD", "editable_native"),
                ("printing", "a prepared print package", "prepared_print_package"),
            )
        ]
    )
    text = source(target) + " Give me editable native CAD and a prepared print package."
    mapped = attribution()
    return ContractDraft(contract=target, attribution=mapped), text


def wrong_family_draft():
    draft, text = valid_draft()
    draft.contract.requirements[0].verification = deepcopy(
        draft.contract.requirements[1].verification
    )
    return draft, text


def wrong_inventory_body_drafts():
    """Retain the trial's full body duty with an unrelated short-quote inventory."""
    corrected, _ = valid_draft()
    body = box_contract().requirements[0]
    body.text = (
        "The nominal CAD motor adapter plate must be a rectangular prism measuring "
        "80 \u00d7 60 \u00d7 6 mm in definition-local XYZ, with its outer-body center at the local "
        "origin. It must contain only the specified five uniform cylindrical through-holes "
        "parallel to local Z, with the stated diameter and radial position tolerances."
    )
    body.position_tolerance = 0.05
    corrected.contract.requirements[0] = body
    corrected.attribution = box_attribution()
    first_sentence = body.source_quote.split(". ", 1)[0] + "."
    corrected.attribution.subjects[0].source_quote = first_sentence
    previous = corrected.model_copy(deep=True)
    previous.contract.requirements[0] = Requirement.model_validate(
        {
            **body.model_dump(mode="json"),
            "verification": {
                "version": 1,
                "kind": "nominal_subject_inventory",
                "scope": "nominal_cad",
                "coverage": "all_made",
                "items": [
                    {
                        "subject_id": "plate",
                        "subject_role": "designed",
                        "required_name": "motor_plate",
                        "quantity": 1,
                    }
                ],
                "source_quote": first_sentence,
            },
        }
    )
    text = body.source_quote + " Give me editable native CAD and a prepared print package."
    return previous, corrected, text


def test_first_diagnostic_packet_reports_short_quote_and_wrong_inventory_body_family():
    previous, corrected, text = wrong_inventory_body_drafts()
    before = previous.model_dump(mode="json")
    result = compile_intent_draft(previous, text)
    body_errors = [item for item in result["diagnostics"] if item.get("requirement_id") == "body"]
    assert result["status"] == "invalid_interpretation"
    assert {"source_quote_mismatch", "incompatible_predicate_family"} <= {
        item["code"] for item in body_errors
    }
    assert any("nominal_perforated_box" in item.get("supported_kinds", {}) for item in body_errors)
    assert previous.model_dump(mode="json") == before
    assert compile_intent_draft(corrected, text, previous=previous)["status"] == "ready"
    assert previous.model_dump(mode="json") == before


@pytest.mark.parametrize("change", ["critical", "source_quote", "dimension", "hole_allowance"])
def test_body_family_correction_cannot_weaken_original_source_duties(change):
    previous, corrected, text = wrong_inventory_body_drafts()
    body = corrected.contract.requirements[0]
    if change == "critical":
        body.critical = False
    elif change == "source_quote":
        body.source_quote = body.verification.source_quote = previous.contract.requirements[
            0
        ].verification.source_quote
    else:
        raw = body.model_dump(mode="json")
        if change == "dimension":
            raw["verification"]["size_mm"][0] = 81.0
        else:
            for hole in raw["verification"]["holes"]:
                hole["diameter_tolerance_mm"] = 0.1
        corrected.contract.requirements[0] = Requirement.model_validate(raw)
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"]
        == "invalid_interpretation"
    )


def preparation_issue(*, dependency=None):
    return IntentIssue.model_validate(
        {
            "id": "print_target",
            "question": "Which printer, material and slicer should this target?",
            "reason": "Proposed operation prerequisite",
            "requirement_ids": ["printing"],
            **({"input_dependency": dependency} if dependency is not None else {}),
        }
    )


def compound_preparation_draft(
    quote="Provide a prepared print package including G-code for my printer.",
):
    draft, _ = valid_draft()
    requirement = draft.contract.requirements[-1]
    requirement.text = requirement.source_quote = requirement.verification.source_quote = quote
    draft.attribution.issues = [preparation_issue()]
    return draft, source(draft.contract) + " Give me editable native CAD. " + quote


@pytest.mark.parametrize(
    "quote",
    [
        "Provide a prepared print package including G-code for my printer.",
        "Give me a prepared print package for my Ender 3.",
        "A prepared print package including heat-qualified material.",
    ],
)
def test_compound_source_duty_and_question_survive_explicit_correction(quote):
    previous, text = compound_preparation_draft(quote)
    result = compile_intent_draft(previous, text)
    assert result["status"] == "invalid_interpretation"
    assert result["blocking_issue_ids"] == ["print_target"]
    assert not any(item["code"] == "invalid_dependency" for item in result["diagnostics"])
    corrected = previous.model_copy(deep=True)
    corrected.contract.requirements[-1].verification = None
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"] == "needs_customer_input"
    )
    assert corrected.contract.requirements[-1].source_quote == quote
    assert corrected.contract.requirements[-1].critical
    corrected.attribution.issues = []
    result = compile_intent_draft(corrected, text, previous=previous)
    assert any(item["code"] == "removed_customer_question" for item in result["diagnostics"])


def test_compound_operation_dependency_can_be_explicitly_reclassified_without_answering_it():
    previous, text = compound_preparation_draft()
    previous.attribution.issues = [
        preparation_issue(
            dependency={
                "kind": "operation_input",
                "input_name": "printer",
                "operation": "prepared_print_package",
            }
        )
    ]
    corrected = previous.model_copy(deep=True)
    corrected.contract.requirements[-1].verification = None
    corrected.attribution.issues[0].input_dependency = type(
        previous.attribution.issues[0].input_dependency
    ).model_validate({"kind": "source_fact", "input_name": "printer"})
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"] == "needs_customer_input"
    )


def test_text_only_extra_machine_duty_preserves_the_real_question():
    previous, text = valid_draft()
    previous.contract.requirements[-1].text = "Provide a prepared print package for my Ender 3."
    previous.attribution.issues = [preparation_issue()]
    result = compile_intent_draft(previous, text)
    assert result["status"] == "invalid_interpretation"
    assert result["blocking_issue_ids"] == ["print_target"]
    assert not any(item["code"] == "invalid_dependency" for item in result["diagnostics"])


def test_closed_structured_preparation_source_preserves_explicit_alternative():
    draft, text = valid_draft()
    requirement = draft.contract.requirements[-1]
    quote = json.dumps(requirement.verification.model_dump(exclude={"source_quote"}))
    requirement.source_quote = requirement.verification.source_quote = quote
    validate_preparation_source_scope(quote)
    assert compile_intent_draft(draft, text + " " + quote)["status"] == "ready"


@pytest.mark.parametrize(
    "quote",
    [
        "a prepared print package",
        "Provide an unsliced prepared print package.",
        "Give me a prepared print package for all made parts.",
        "Deliver a prepared print package covering everything made.",
    ],
)
def test_closed_preparation_source_accepts_artifact_only_clauses(quote):
    validate_preparation_source_scope(quote)


@pytest.mark.parametrize(
    "quote",
    [
        "a prepared print package including G-code",
        "a prepared print package for my Ender 3",
        "a prepared print package that withstands heat",
        "a prepared print package and test its strength",
    ],
)
def test_closed_preparation_source_rejects_unparsed_extra_duties(quote):
    with pytest.raises(ValueError, match="artifact-only"):
        validate_preparation_source_scope(quote)


def test_correct_supported_draft_is_ready_without_mutation():
    draft, text = valid_draft()
    before = draft.model_dump(mode="json")
    assert compile_intent_draft(draft, text) == {
        "status": "ready",
        "diagnostics": [],
        "blocking_issue_ids": [],
    }
    assert draft.model_dump(mode="json") == before


def test_wrong_typed_family_is_diagnosed_before_safe_grounding_can_demote_it():
    draft, text = wrong_family_draft()
    result = compile_intent_draft(draft, text)
    assert result["status"] == "invalid_interpretation"
    assert any(item.get("requirement_id") == "pattern" for item in result["diagnostics"])
    assert draft.contract.requirements[0].verification.kind == "digital_delivery"
    corrected, _ = valid_draft()
    assert compile_intent_draft(corrected, text, previous=draft)["status"] == "ready"


@pytest.mark.parametrize(
    "dependency",
    [
        None,
        {"kind": "operation_input", "operation": "prepared_print_package", "input_name": "printer"},
        {
            "kind": "operation_input",
            "operation": "prepared_print_package",
            "input_name": "revision_id",
        },
    ],
)
def test_unsliced_export_does_not_invent_missing_customer_inputs(dependency):
    draft, text = valid_draft()
    draft.attribution.issues = [preparation_issue(dependency=dependency)]
    result = compile_intent_draft(draft, text)
    assert result["status"] == "invalid_interpretation"
    assert result["blocking_issue_ids"] == []
    assert any(item["code"] == "invalid_dependency" for item in result["diagnostics"])
    corrected = draft.model_copy(deep=True)
    corrected.attribution.issues = []
    assert compile_intent_draft(corrected, text, previous=draft)["status"] == "ready"


def test_explicit_machine_duty_stays_a_real_question_with_legacy_absence():
    draft, text = valid_draft()
    quote = "Provide G-code for my printer."
    text += " " + quote
    draft.contract.requirements.append(Requirement(id="machine", text=quote, source_quote=quote))
    issue = preparation_issue()
    issue.requirement_ids = ["machine"]
    draft.attribution.issues = [issue]
    assert "input_dependency" not in issue.model_dump()
    assert compile_intent_draft(draft, text)["status"] == "needs_customer_input"
    corrected = draft.model_copy(deep=True)
    corrected.attribution.issues = []
    result = compile_intent_draft(corrected, text, previous=draft)
    assert any(item["code"] == "removed_customer_question" for item in result["diagnostics"])


@pytest.mark.parametrize(
    "change", ["drop", "critical", "quote", "text", "valid_predicate", "scalar"]
)
def test_correction_cannot_remove_or_widen_original_duties(change):
    previous, text = wrong_family_draft()
    previous.contract.requirements.append(
        Requirement(id="count", text="One solid", source_quote="one", metric="solid_count", value=1)
    )
    corrected = previous.model_copy(deep=True)
    corrected.contract.requirements[0] = valid_draft()[0].contract.requirements[0]
    if change == "drop":
        corrected.contract.requirements.pop()
    elif change == "critical":
        corrected.contract.requirements[-1].critical = False
    elif change == "quote":
        corrected.contract.requirements[0].source_quote = "through-holes"
    elif change == "text":
        corrected.contract.requirements[0].text = "An arbitrary hole"
    elif change == "valid_predicate":
        corrected.contract.requirements[1].verification = None
    else:
        corrected.contract.requirements[-1].value = 2
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"]
        == "invalid_interpretation"
    )


def test_legacy_physical_uncertainty_stays_unknown_and_asks_for_real_dimensions():
    text = "Make a carrier for a sample. The body is 45 mm wide."
    draft = ContractDraft(
        contract=Contract(
            requirements=[Requirement(id="fit", text="Fit the sample", source_quote="45 mm wide")]
        ),
        attribution=IntentAttribution.model_validate(
            {
                "subjects": [
                    {"id": key, "label": key, "role": role, "source_quote": key}
                    for key, role in (("carrier", "designed"), ("sample", "reference"))
                ],
                "facts": [
                    {
                        "id": "width",
                        "candidate_subject_ids": ["carrier", "sample"],
                        "property": "width",
                        "value": 45.0,
                        "unit": "mm",
                        "source_quote": "45 mm wide",
                        "requirement_ids": ["fit"],
                    }
                ],
                "issues": [
                    {
                        "id": "owner",
                        "question": "Which body is 45 mm wide?",
                        "reason": "Two possible subjects",
                        "fact_ids": ["width"],
                        "requirement_ids": ["fit"],
                        "input_dependency": {
                            "kind": "subject_assignment",
                            "input_name": "width.subject_id",
                        },
                    }
                ],
            }
        ),
    )
    result = compile_intent_draft(draft, text)
    assert result["status"] == "needs_customer_input"
    assert result["blocking_issue_ids"] == ["owner"]
    assert draft.contract.requirements[0].verification is None


@pytest.mark.parametrize("invalid", ["quote", "value", "subject_reference"])
def test_invalid_attribution_can_be_corrected_without_changing_valid_source_facts(invalid):
    previous, text = wrong_family_draft()
    fact = {
        "id": "diameter",
        "subject_id": "plate",
        "property": "diameter",
        "value": 22.0,
        "unit": "mm",
        "source_quote": "22 mm diameter",
        "requirement_ids": ["pattern"],
    }
    previous.attribution.facts = [
        type(previous.attribution).model_validate({"facts": [fact]}).facts[0]
    ]
    corrected = previous.model_copy(deep=True)
    if invalid == "quote":
        previous.attribution.facts[0].source_quote = "22 mm impossible source"
    elif invalid == "value":
        previous.attribution.facts[0].value = 99
    else:
        previous.attribution.facts[0].subject_id = "dangling"
    corrected.contract.requirements[0] = valid_draft()[0].contract.requirements[0]
    assert compile_intent_draft(previous, text)["status"] == "invalid_interpretation"
    assert compile_intent_draft(corrected, text, previous=previous)["status"] == "ready"
    corrected.attribution.facts[0].value = 5.2
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"]
        == "invalid_interpretation"
    )


def invalid_fact_drafts(invalid="value"):
    previous, text = valid_draft()
    fact = {
        "id": "retained_diameter",
        "subject_id": "plate",
        "property": "diameter",
        "value": 22.0,
        "unit": "mm",
        "source_quote": "22 mm diameter",
        "requirement_ids": ["pattern"],
    }
    previous.attribution.facts = IntentAttribution.model_validate({"facts": [fact]}).facts
    broken = previous.attribution.facts[0].model_copy(deep=True, update={"id": "broken"})
    if invalid == "value":
        broken.value = 99.0
    elif invalid == "unit":
        broken.unit = "inches"
    elif invalid == "source_quote":
        broken.source_quote = "22 mm imagined diameter"
    elif invalid == "requirement_ids":
        broken.requirement_ids = ["unknown"]
    elif invalid == "candidate_subject_ids":
        broken.subject_id = None
        broken.candidate_subject_ids = ["unknown"]
    elif invalid == "subject_id":
        broken.subject_id = "unknown"
    elif invalid == "subject_assignment":
        broken.candidate_subject_ids = ["plate"]
    else:
        broken.subject_id = None
    previous.attribution.facts.append(broken)
    corrected = previous.model_copy(deep=True)
    corrected.attribution.facts.pop()
    return previous, corrected, text


@pytest.mark.parametrize(
    "invalid,field",
    [
        ("value", "value"),
        ("unit", "unit"),
        ("source_quote", "source_quote"),
        ("requirement_ids", "requirement_ids"),
        ("candidate_subject_ids", "candidate_subject_ids"),
        ("subject_id", "subject_id"),
        ("subject_assignment", "subject_id"),
        ("unresolved", "subject_id"),
    ],
)
def test_invalid_fact_can_be_removed_with_exact_field_diagnostic(invalid, field):
    previous, corrected, text = invalid_fact_drafts(invalid)
    first = compile_intent_draft(previous, text)
    assert first["status"] == "invalid_interpretation"
    assert any(
        item["field"] == f"attribution.facts.broken.{field}"
        and item["code"] == "invalid_attribution"
        for item in first["diagnostics"]
    )
    assert compile_intent_draft(corrected, text, previous=previous)["status"] == "ready"
    assert corrected.contract == previous.contract
    assert corrected.attribution.facts == previous.attribution.facts[:1]


@pytest.mark.parametrize("change", ["remove_valid_fact", "change_valid_fact", "drop_duty"])
def test_invalid_fact_removal_does_not_release_valid_facts_or_source_duties(change):
    previous, corrected, text = invalid_fact_drafts()
    if change == "remove_valid_fact":
        corrected.attribution.facts = []
    elif change == "change_valid_fact":
        corrected.attribution.facts[0].property = "unrelated property"
    else:
        corrected.contract.requirements.pop()
    result = compile_intent_draft(corrected, text, previous=previous)
    assert result["status"] == "invalid_interpretation"
    assert any(
        item["code"]
        == ("dropped_requirement" if change == "drop_duty" else "changed_source_attribution")
        for item in result["diagnostics"]
    )


def test_invalid_fact_removal_does_not_answer_its_real_customer_question():
    previous, corrected, text = invalid_fact_drafts()
    previous.attribution.issues = [
        IntentIssue(
            id="owner_question",
            question="Which diameter applies?",
            reason="Owner scope remains uncertain",
            fact_ids=["broken"],
            requirement_ids=["pattern"],
        )
    ]
    result = compile_intent_draft(corrected, text, previous=previous)
    assert result["status"] == "invalid_interpretation"
    assert any(item["code"] == "removed_customer_question" for item in result["diagnostics"])


def incomplete_unit_quote_drafts():
    text = "Make a spacer plate with four 5 mm holes going through its thickness, one in each corner, 8 mm from each adjacent edge."
    previous = ContractDraft.model_validate(
        {
            "contract": {
                "requirements": [
                    {
                        "id": "r7",
                        "text": "One hole per corner",
                        "source_quote": "one in each corner",
                    },
                    {
                        "id": "r8",
                        "text": "8 mm edge offset",
                        "source_quote": "8 mm from each adjacent edge",
                    },
                ]
            },
            "attribution": {
                "subjects": [
                    {
                        "id": "plate",
                        "label": "plate",
                        "role": "designed",
                        "source_quote": "spacer plate",
                    }
                ],
                "facts": [
                    {
                        "id": "f6",
                        "subject_id": "plate",
                        "property": "holes per corner",
                        "value": 1.0,
                        "unit": "holes",
                        "source_quote": "one in each corner",
                        "requirement_ids": ["r7"],
                    },
                    {
                        "id": "f7",
                        "subject_id": "plate",
                        "property": "edge offset",
                        "value": 8.0,
                        "unit": "mm",
                        "source_quote": "8 mm from each adjacent edge",
                        "requirement_ids": ["r8"],
                    },
                ],
                "issues": [
                    {
                        "id": "offset_reference",
                        "question": "Is the offset to the hole centre or boundary?",
                        "reason": "The offset reference changes hole positions",
                        "fact_ids": ["f7"],
                        "requirement_ids": ["r8"],
                    }
                ],
            },
        }
    )
    corrected = previous.model_copy(deep=True)
    corrected.attribution.facts[
        0
    ].source_quote = "four 5 mm holes going through its thickness, one in each corner"
    return previous, corrected, text


def test_incomplete_unit_quote_can_expand_without_changing_fact_or_answering_question():
    previous, corrected, text = incomplete_unit_quote_drafts()
    assert (
        compile_intent_draft(previous, text)["diagnostics"][0]["field"]
        == "attribution.facts.f6.unit"
    )
    result = compile_intent_draft(corrected, text, previous=previous)
    assert result == {
        "status": "needs_customer_input",
        "diagnostics": [],
        "blocking_issue_ids": ["offset_reference"],
    }
    assert corrected.contract == previous.contract
    assert corrected.attribution.issues == previous.attribution.issues


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "unit",
        "property",
        "subject",
        "requirements",
        "anchor",
        "noncontiguous",
        "metadata",
        "unit_missing",
        "invalid_old_value",
        "valid_old_fact",
        "remove_question",
        "reword_question",
    ],
)
def test_unit_quote_expansion_cannot_change_other_authority(change):
    previous, corrected, text = incomplete_unit_quote_drafts()
    fact = corrected.attribution.facts[0]
    if change == "value":
        fact.value = 4.0
    elif change == "unit":
        fact.unit = "mm"
    elif change == "property":
        fact.property = "plate thickness"
    elif change == "subject":
        subject = previous.attribution.subjects[0].model_copy(update={"id": "other"})
        previous.attribution.subjects.append(subject)
        corrected.attribution.subjects.append(subject)
        fact.subject_id = "other"
    elif change == "requirements":
        fact.requirement_ids = ["r8"]
    elif change == "anchor":
        text += " There is one of the holes near an edge."
        fact.source_quote = "one of the holes near an edge"
    elif change == "noncontiguous":
        fact.source_quote = "four 5 mm holes, one in each corner"
    elif change == "metadata":
        fact.source_quote = "Server policy: one in each corner with holes"
    elif change == "unit_missing":
        fact.source_quote = "one in each corner, 8 mm from each adjacent edge"
    elif change == "invalid_old_value":
        text += " There are 99 spare holes."
        previous.attribution.facts[0].value = fact.value = 99.0
        fact.source_quote = text
    elif change == "valid_old_fact":
        previous.attribution.facts[0].source_quote = fact.source_quote
        fact.source_quote = text
    elif change == "remove_question":
        corrected.attribution.issues = []
    else:
        corrected.attribution.issues[0].question = "A different question"
    assert (
        compile_intent_draft(corrected, text, previous=previous)["status"]
        == "invalid_interpretation"
    )


def test_shared_operation_descriptor_tracks_actual_request_and_scope():
    descriptor = preparation_capability()
    assert descriptor["required_saved_inputs"] == [
        name
        for name, field in PrintSetupRequest.model_fields.items()
        if field.is_required() and name != "operation"
    ]
    assert descriptor["required_customer_inputs"] == []
    assert descriptor["scope"] == SCOPE
    assert native_bootstrap()["discovery"]["prepared_print_capability"] == descriptor
    descriptor["required_saved_inputs"].append("invented")
    assert "invented" not in preparation_capability()["required_saved_inputs"]
