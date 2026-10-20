"""Source-bound identity/policy criteria remain separate from physical proof."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.intent_attribution import IntentAttribution, validate_attribution
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.verification_admission import (
    criterion_hash,
    resolve_verification_inventory,
    validate_verification_bindings,
    validate_verification_contract,
    verification_diagnostics,
    verification_repair_exclusions,
)

INVENTORY_QUOTE = "Make two nominal CAD plates named clamp_pair and three spacers named stand_off."
SCOPE_QUOTE = "I need nominal CAD and digital files only."


def retained_offset002_r2():
    """Exact failed first r2, with its subject projection, from offset trial002.

    This records raw failed interpretation, not an authored geometric answer.
    The original full trial and all other requirements remain immutable.
    """
    quote = (
        "Make one nominal CAD rectangular mounting plate named offset_plate. "
        "Its overall size is 100 mm in local X, 50 mm in local Y and 8 mm in local Z, "
        "centered on the local origin. It must be one solid. It has exactly four uniform "
        "cylindrical through-holes parallel to local Z: two 8 mm diameter holes at local "
        "XY (-35, -12) and (-35, 12) mm, and two 6 mm diameter holes at (30, -15) and "
        "(30, 15) mm. Use a symmetric diameter tolerance of 0.1 mm and a radial position "
        "tolerance of 0.1 mm for each hole."
    )
    subject_quote = "Make one nominal CAD rectangular mounting plate named offset_plate."
    requirement = Requirement.model_validate(
        {
            "id": "r2",
            "text": (
                "The nominal CAD plate must be a rectangular prism measuring 100 \u00d7 50 \u00d7 8 mm "
                "in local XYZ, with its outer-body center at the local origin. It must contain "
                "exactly four uniform cylindrical through-holes parallel to local Z: diameter "
                "8 mm at XY (-35, -12) and (-35, 12) mm, and diameter 6 mm at XY (30, -15) and "
                "(30, 15) mm. Each hole has symmetric diameter tolerance ±0.1 mm and radial "
                "position tolerance 0.1 mm. The body has no other material additions or "
                "removals under this closed nominal predicate."
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
            "verification": {
                "version": 1,
                "kind": "nominal_subject_inventory",
                "scope": "nominal_cad",
                "coverage": "all_made",
                "items": [
                    {
                        "subject_id": "offset_plate",
                        "subject_role": "designed",
                        "required_name": "offset_plate",
                        "quantity": 1,
                    }
                ],
                "source_quote": subject_quote,
            },
        }
    )
    attrs = IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "offset_plate",
                    "label": "Mounting plate named offset_plate",
                    "role": "designed",
                    "source_quote": subject_quote,
                }
            ],
            "requirement_subjects": {"r2": ["offset_plate"]},
            "issues": [],
        }
    )
    return requirement, quote, attrs


def test_saved_offset002_generated_disjunction_does_not_create_source_ambiguity():
    requirement, source, attrs = retained_offset002_r2()
    before = requirement.model_dump(mode="json")
    assert "no other material additions or removals" in requirement.text
    evidence = verification_repair_exclusions(requirement, source, attrs)
    assert [row["kind"] for row in evidence] == ["nominal_subject_inventory"]
    assert evidence[0]["rule"] == "inventory_fixed_names_geometry_conflict_v1"
    assert evidence[0]["requirement_sha256"] == criterion_hash(before)
    assert (
        evidence[0]["requirement_text_sha256"]
        == hashlib.sha256(requirement.text.encode()).hexdigest()
    )
    assert requirement.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="complete requirement source quote"):
        validate_verification_contract(Contract(requirements=[requirement]), source, attrs)


@pytest.mark.parametrize(
    "source_prefix", ["If available, ", "Either one alternative or the other: "]
)
def test_saved_offset002_with_real_source_identity_ambiguity_has_no_exclusion(source_prefix):
    requirement, source, attrs = retained_offset002_r2()
    original_text = requirement.text
    requirement.source_quote = source_prefix + source
    assert verification_repair_exclusions(requirement, requirement.source_quote, attrs) == []
    assert requirement.text == original_text


def body_inventory_request():
    from tests.unit.workspace.test_verification_admission import box_contract

    body = box_contract().requirements[0]
    raw = target().requirements[0].model_dump(mode="json")
    raw.update(id=body.id, text=body.text, source_quote=body.source_quote)
    raw["verification"]["items"] = [
        {
            "subject_id": "plate",
            "subject_role": "designed",
            "required_name": "motor_plate",
            "quantity": 1,
        }
    ]
    raw["verification"]["source_quote"] = body.source_quote.split(". ", 1)[0] + "."
    return Requirement.model_validate(raw)


def test_fixed_source_names_certify_inventory_kind_without_repairing_targets():
    requirement = body_inventory_request()
    before = requirement.model_dump(mode="json")
    certificates = verification_repair_exclusions(requirement, requirement.source_quote, None)
    assert len(certificates) == 1
    evidence = certificates[0]
    assert evidence["kind"] == "nominal_subject_inventory"
    assert evidence["rule"] == "inventory_fixed_names_geometry_conflict_v1"
    assert evidence["requirement_sha256"] == criterion_hash(before)
    assert evidence["criterion_sha256"] == criterion_hash(before["verification"])
    assert requirement.model_dump(mode="json") == before
    with pytest.raises(ValueError):
        validate_verification_contract(
            Contract(requirements=[requirement]), requirement.source_quote, None
        )
    changed_source = verification_repair_exclusions(
        requirement, requirement.source_quote + " Extra source.", None
    )
    assert changed_source[0]["source_sha256"] != evidence["source_sha256"]


@pytest.mark.parametrize("field,value", [("required_name", "wrong_name"), ("quantity", 2)])
def test_wrong_inventory_payload_cannot_certify_a_whole_kind(field, value):
    requirement = body_inventory_request()
    setattr(requirement.verification.items[0], field, value)
    assert verification_repair_exclusions(requirement, requirement.source_quote, None) == []


@pytest.mark.parametrize("kind", ["nominal_subject_inventory", "digital_delivery"])
def test_fact_only_issue_withholds_kind_evidence_without_erasing_uncertainty(kind):
    from tests.unit.workspace.test_verification_admission import box_attribution

    requirement = body_inventory_request()
    if kind == "digital_delivery":
        raw = requirement.model_dump(mode="json")
        raw["verification"] = {
            "kind": kind,
            "scope": "digital_artifacts",
            "deliverable": "editable_native",
            "coverage": "all_made",
            "source_quote": requirement.source_quote,
        }
        requirement = Requirement.model_validate(raw)
    attrs = box_attribution()
    attrs.subjects[0].source_quote = requirement.source_quote.split(". ", 1)[0] + "."
    assert verification_repair_exclusions(requirement, requirement.source_quote, attrs)
    raw_attrs = attrs.model_dump(mode="json")
    raw_attrs["facts"] = [
        {
            "id": "quantity",
            "subject_id": "plate",
            "property": "occurrence_quantity",
            "value": 1,
            "unit": "plate",
            "source_quote": attrs.subjects[0].source_quote,
            "requirement_ids": [requirement.id],
        }
    ]
    raw_attrs["issues"] = [
        {
            "id": "quantity_question",
            "question": "Which quantity applies?",
            "reason": "Unresolved quantity fact",
            "fact_ids": ["quantity"],
            "requirement_ids": [],
        }
    ]
    linked = IntentAttribution.model_validate(raw_attrs)
    validate_attribution(
        linked,
        Contract(requirements=[requirement]),
        requirement.source_quote,
        validate_predicates=False,
    )
    before = linked.model_dump(mode="json")
    assert verification_repair_exclusions(requirement, requirement.source_quote, linked) == []
    assert linked.model_dump(mode="json") == before
    assert linked.issues[0].requirement_ids == []
    assert linked.issues[0].fact_ids == ["quantity"]


@pytest.mark.parametrize(
    "change",
    [
        "conditional_label",
        "conditional_prefix",
        "disjunction",
        "approximate",
        "grouped",
        "unrecognized_prefix",
        "assumption",
        "source_missing",
        "linked_issue",
    ],
)
def test_ambiguous_or_unverified_inventory_identity_has_no_exclusion(change):
    from tests.unit.workspace.test_verification_admission import box_attribution

    requirement = body_inventory_request()
    attrs = box_attribution()
    attrs.subjects[0].source_quote = requirement.verification.source_quote
    if change == "conditional_label":
        requirement.source_quote = requirement.source_quote.replace(
            "plate named", "plate if available named"
        )
    elif change == "conditional_prefix":
        requirement.source_quote = "If available, " + requirement.source_quote
    elif change == "disjunction":
        requirement.source_quote = requirement.source_quote.replace(
            "plate named", "plate or bracket named"
        )
    elif change == "approximate":
        requirement.source_quote = requirement.source_quote.replace(
            "Make one", "Make approximately one"
        )
    elif change == "grouped":
        requirement.source_quote = requirement.source_quote.replace("Make one", "Make one pair of")
    elif change == "unrecognized_prefix":
        requirement.source_quote = "Potentially " + requirement.source_quote
    elif change == "assumption":
        requirement.source = "assumption"
    elif change == "linked_issue":
        raw = attrs.model_dump(mode="json")
        raw["issues"] = [
            {
                "id": "uncertain",
                "question": "Which plate?",
                "reason": "Unresolved source assignment",
                "requirement_ids": [requirement.id],
            }
        ]
        attrs = IntentAttribution.model_validate(raw)
    source = "Unrelated source" if change == "source_missing" else requirement.source_quote
    assert verification_repair_exclusions(requirement, source, attrs) == []


def test_literal_geometry_name_allows_valid_same_inventory_family_repair():
    quote = "Make one nominal CAD plate named 'rectangular plate'."
    raw = target().requirements[0].model_dump(mode="json")
    raw.update(text="Deliver the exact saved name rectangular plate.", source_quote=quote)
    raw["verification"].update(
        source_quote=quote,
        items=[
            {
                "subject_id": "plate",
                "subject_role": "designed",
                "required_name": "wrong_name",
                "quantity": 1,
            }
        ],
    )
    malformed = Requirement.model_validate(raw)
    diagnostics = verification_diagnostics(Contract(requirements=[malformed]), quote, None)
    assert "incompatible_predicate_family" in {row["code"] for row in diagnostics}
    assert verification_repair_exclusions(malformed, quote, None) == []
    corrected = malformed.model_copy(deep=True)
    corrected.verification.items[0].required_name = "rectangular plate"
    validate_verification_contract(Contract(requirements=[corrected]), quote, None)
    assert corrected.verification.kind == malformed.verification.kind
    assert corrected.source_quote == malformed.source_quote == quote
    assert verification_repair_exclusions(corrected, quote, None) == []


def test_inventory_body_mismatch_reports_quote_and_geometry_family_together():
    from tests.unit.workspace.test_verification_admission import box_contract

    body_contract = box_contract()
    body = body_contract.requirements[0]
    short_quote = body.source_quote.split(". ", 1)[0] + "."
    raw = body_contract.model_dump(mode="json")
    raw["requirements"][0]["verification"] = {
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
        "source_quote": short_quote,
    }
    malformed = Contract.model_validate(raw)
    before = malformed.model_dump(mode="json")
    first = verification_diagnostics(malformed, body.source_quote, None)
    assert {"source_quote_mismatch", "incompatible_predicate_family"} <= {
        row["code"] for row in first
    }
    family = next(row for row in first if row["code"] == "incompatible_predicate_family")
    assert family["field"] == "contract.requirements.body.verification.kind"
    assert "names and occurrence counts only" in family["message"]
    assert "nominal_perforated_box" in family["supported_kinds"]
    assert malformed.model_dump(mode="json") == before
    # Copying the parent quote cannot turn an identity predicate into body proof.
    quote_only = malformed.model_copy(deep=True)
    quote_only.requirements[0].verification.source_quote = body.source_quote
    with pytest.raises(ValueError, match="cannot replace a geometry requirement"):
        validate_verification_contract(quote_only, body.source_quote, None)
    grounded = ground_contract(quote_only, body.source_quote)
    assert grounded.requirements[0].critical
    assert grounded.requirements[0].verification is None
    assert grounded.requirements[0].source_quote == body.source_quote
    assert verification_diagnostics(body_contract, body.source_quote, None) == []


def target():
    return Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "inventory",
                    "text": "Exact named made inventory and occurrence quantities",
                    "source_quote": INVENTORY_QUOTE,
                    "verification": {
                        "version": 1,
                        "kind": "nominal_subject_inventory",
                        "scope": "nominal_cad",
                        "coverage": "all_made",
                        "source_quote": INVENTORY_QUOTE,
                        "items": [
                            {
                                "subject_id": "plates",
                                "subject_role": "designed",
                                "required_name": "clamp_pair",
                                "quantity": 2,
                            },
                            {
                                "subject_id": "spacers",
                                "subject_role": "designed",
                                "required_name": "stand_off",
                                "quantity": 3,
                            },
                        ],
                    },
                },
                {
                    "id": "scope",
                    "text": "Limit delivery to nominal CAD and digital files",
                    "source_quote": SCOPE_QUOTE,
                    "verification": {
                        "version": 1,
                        "kind": "nominal_delivery_scope",
                        "scope": "software_policy",
                        "policy": "nominal_cad_and_digital_only",
                        "source_quote": SCOPE_QUOTE,
                    },
                },
            ]
        }
    )


def attribution():
    return IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "plates",
                    "label": "Plates",
                    "role": "designed",
                    "source_quote": "two nominal CAD plates named clamp_pair",
                },
                {
                    "id": "spacers",
                    "label": "Spacers",
                    "role": "designed",
                    "source_quote": "three spacers named stand_off",
                },
            ],
            "requirement_subjects": {
                "inventory": ["plates", "spacers"],
                "scope": ["plates", "spacers"],
            },
        }
    )


def plan():
    return DesignPlan.model_validate(
        {
            "objective": "Construct the requested named definitions",
            "verification_bindings": [
                {"subject_id": "plates", "part_id": "opaque_a", "frame": "definition_local"},
                {"subject_id": "spacers", "part_id": "opaque_b", "frame": "definition_local"},
            ],
        }
    )


def document():
    parts, roots = [], []
    for part_id, name, quantity in (("opaque_a", "clamp_pair", 2), ("opaque_b", "stand_off", 3)):
        parts.append(
            {
                "id": part_id,
                "name": name,
                "production": {"role": "make"},
                "features": [
                    {
                        "id": "blank",
                        "name": "Blank",
                        "kind": "box",
                        "params": {"x": 4, "y": 3, "z": 2},
                    }
                ],
                "output": "blank",
            }
        )
        roots.extend(
            {"id": f"{part_id}_{index}", "definition": part_id} for index in range(quantity)
        )
    return DesignDocument.model_validate({"assembly": {"parts": parts, "roots": roots}})


def test_complete_multi_subject_source_survives_per_requirement_grounding():
    contract = target()
    source = INVENTORY_QUOTE + " " + SCOPE_QUOTE
    validate_verification_contract(contract, source, attribution())
    grounded = ground_contract(contract, source, attribution=attribution())
    assert grounded.model_dump(mode="json") == contract.model_dump(mode="json")
    assert all(row.critical for row in grounded.requirements)


def test_inventory_resolves_target_free_bindings_without_claiming_final_proof():
    contract, design, saved = target(), plan(), document()
    validate_verification_bindings(design, contract, saved)
    resolved = resolve_verification_inventory(contract.requirements[0], design, saved)
    assert [row["part_id"] for row in resolved] == ["opaque_a", "opaque_b"]
    assert [row["step_artifact"] for row in resolved] == [
        "definition_opaque_a.step",
        "definition_opaque_b.step",
    ]
    assert all(
        "status" not in row and "quantity" not in row and "required_name" not in row
        for row in resolved
    )


@pytest.mark.parametrize(
    "change",
    [
        "name",
        "case",
        "quantity",
        "solid_count",
        "partial_quote",
        "wrong_role",
        "missing_subject",
        "subject_quote",
        "issue",
        "mixed_geometry",
        "mixed_physical",
    ],
)
def test_inventory_never_promotes_untraced_or_mixed_source(change):
    raw, attrs = target().model_dump(mode="json"), attribution().model_dump(mode="json")
    req = raw["requirements"][0]
    source = INVENTORY_QUOTE + " " + SCOPE_QUOTE
    if change == "name":
        req["verification"]["items"][0]["required_name"] = "wrong_name"
    elif change == "case":
        req["verification"]["items"][0]["required_name"] = "Clamp_pair"
    elif change == "quantity":
        req["verification"]["items"][0]["quantity"] = 3
    elif change == "solid_count":
        req["source_quote"] = req["verification"]["source_quote"] = (
            "Make two nominal CAD solids named clamp_pair and three spacers named stand_off."
        )
        source = req["source_quote"] + " " + SCOPE_QUOTE
    elif change == "partial_quote":
        req["verification"]["source_quote"] = "two nominal CAD plates named clamp_pair"
    elif change == "wrong_role":
        attrs["subjects"][0]["role"] = "reference"
    elif change == "missing_subject":
        attrs["requirement_subjects"]["inventory"] = ["plates"]
    elif change == "subject_quote":
        attrs["subjects"][0]["source_quote"] = "plates"
    elif change == "issue":
        attrs["issues"] = [
            {
                "id": "uncertain",
                "question": "Which subject?",
                "reason": "Source subject is ambiguous",
                "requirement_ids": ["inventory"],
            }
        ]
    elif change == "mixed_geometry":
        req["text"] += " and prove its 5 mm hole diameter"
    else:
        req["text"] += " and hold a 20 kg load"
    with pytest.raises(ValueError):
        validate_verification_contract(
            Contract.model_validate(raw), source, IntentAttribution.model_validate(attrs)
        )


@pytest.mark.parametrize(
    "quote",
    [
        "I need nominal CAD and digital files.",
        "I need nominal CAD and digital files only, and it must support 5 kg.",
        "I need nominal CAD and digital files only. Add three holes.",
        "I need digital files only.",
    ],
)
def test_policy_accepts_no_missing_boundary_or_additional_duty(quote):
    raw = target().model_dump(mode="json")
    raw["requirements"][1]["source_quote"] = raw["requirements"][1]["verification"][
        "source_quote"
    ] = quote
    with pytest.raises(ValueError):
        validate_verification_contract(
            Contract.model_validate(raw), INVENTORY_QUOTE + " " + quote, None
        )


def test_physical_peer_requirement_stays_critical_and_unverified():
    raw = target().model_dump(mode="json")
    raw["requirements"].append(
        {
            "id": "strength",
            "text": "Hold 20 kg safely",
            "source_quote": "Hold 20 kg safely.",
            "critical": True,
        }
    )
    contract = Contract.model_validate(raw)
    result = ground_contract(contract, INVENTORY_QUOTE + " " + SCOPE_QUOTE + " Hold 20 kg safely.")
    physical = result.requirements[-1]
    assert physical.critical and physical.metric == "unmeasured" and physical.verification is None


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "duplicate_subject",
        "duplicate_name",
        "bool_quantity",
        "zero_quantity",
        "reference_role",
        "author_part_id",
    ],
)
def test_inventory_schema_has_no_ambiguous_or_authored_targets(change):
    raw = target().model_dump(mode="json")
    criterion = raw["requirements"][0]["verification"]
    if change == "empty":
        criterion["items"] = []
    elif change == "duplicate_subject":
        criterion["items"][1]["subject_id"] = "plates"
    elif change == "duplicate_name":
        criterion["items"][1]["required_name"] = "clamp_pair"
    elif change == "bool_quantity":
        criterion["items"][0]["quantity"] = True
    elif change == "zero_quantity":
        criterion["items"][0]["quantity"] = 0
    elif change == "reference_role":
        criterion["items"][0]["subject_role"] = "reference"
    else:
        criterion["items"][0]["part_id"] = "author_selected"
    with pytest.raises(ValidationError):
        Contract.model_validate(raw)


def test_conflicting_all_made_inventories_and_missing_binding_cannot_pass():
    raw = target().model_dump(mode="json")
    duplicate = deepcopy(raw["requirements"][0])
    duplicate["id"] = "other_inventory"
    duplicate["source_quote"] = duplicate["verification"]["source_quote"] = (
        "Make four nominal CAD plates named clamp_pair and three spacers named stand_off."
    )
    duplicate["verification"]["items"][0]["quantity"] = 4
    raw["requirements"].append(duplicate)
    with pytest.raises(ValueError, match="Conflicting frozen all-made"):
        validate_verification_contract(
            Contract.model_validate(raw),
            INVENTORY_QUOTE + " " + SCOPE_QUOTE + " " + duplicate["source_quote"],
            None,
        )
    partial = plan().model_copy(update={"verification_bindings": plan().verification_bindings[:1]})
    with pytest.raises(ValueError, match="explicit verification subject binding"):
        resolve_verification_inventory(target().requirements[0], partial, document())


def test_large_explicit_occurrence_quantity_has_no_inventory_specific_cap():
    raw = target().model_dump(mode="json")
    quote = INVENTORY_QUOTE.replace("two nominal", "10000 nominal")
    raw["requirements"][0]["source_quote"] = raw["requirements"][0]["verification"][
        "source_quote"
    ] = quote
    raw["requirements"][0]["verification"]["items"][0]["quantity"] = 10000
    validate_verification_contract(Contract.model_validate(raw), quote + " " + SCOPE_QUOTE, None)


@pytest.mark.parametrize("change", ["omitted_item", "duplicate_naming", "wrong_literal_case"])
def test_complete_named_source_inventory_cannot_silently_drop_or_alias_items(change):
    raw = target().model_dump(mode="json")
    criterion = raw["requirements"][0]["verification"]
    quote = INVENTORY_QUOTE
    if change == "omitted_item":
        criterion["items"].pop()
    elif change == "duplicate_naming":
        quote += " Make two plates named clamp_pair."
        raw["requirements"][0]["source_quote"] = criterion["source_quote"] = quote
    else:
        criterion["items"][0]["required_name"] = "Clamp_pair"
    with pytest.raises(ValueError, match="complete explicit named inventory"):
        validate_verification_contract(
            Contract.model_validate(raw), quote + " " + SCOPE_QUOTE, None
        )


@pytest.mark.parametrize(
    "additional", ["and one unnamed plate", "and manufacture it", "and inspect its appearance"]
)
def test_unmatched_source_duties_are_not_inventory_success(additional):
    raw = target().model_dump(mode="json")
    quote = INVENTORY_QUOTE.removesuffix(".") + " " + additional + "."
    raw["requirements"][0]["source_quote"] = raw["requirements"][0]["verification"][
        "source_quote"
    ] = quote
    with pytest.raises(ValueError, match="Additional source duties"):
        validate_verification_contract(
            Contract.model_validate(raw), quote + " " + SCOPE_QUOTE, None
        )


@pytest.mark.parametrize(
    "group", ["pair of", "set of", "pack of", "dozen", "hundred", "bundle of", "box of"]
)
def test_grouping_modifiers_cannot_become_occurrence_quantities(group):
    raw = target().model_dump(mode="json")
    quote = INVENTORY_QUOTE.replace("two nominal CAD plates", f"one {group} nominal CAD plates")
    raw["requirements"][0]["source_quote"] = raw["requirements"][0]["verification"][
        "source_quote"
    ] = quote
    raw["requirements"][0]["verification"]["items"][0]["quantity"] = 1
    with pytest.raises(ValueError, match="occurrence quantity must trace"):
        validate_verification_contract(
            Contract.model_validate(raw), quote + " " + SCOPE_QUOTE, None
        )


@pytest.mark.parametrize(
    "extra",
    ["containing G-code", "for my Ender 3", "with printer settings", "with sliced toolpaths"],
)
def test_unsliced_preparation_cannot_cover_compound_machine_duty(extra):
    quote = "Give me a prepared print package " + extra + "."
    contract = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "prepared",
                    "text": quote,
                    "source_quote": quote,
                    "critical": True,
                    "verification": {
                        "version": 1,
                        "kind": "digital_delivery",
                        "scope": "digital_artifacts",
                        "deliverable": "prepared_print_package",
                        "coverage": "all_made",
                        "source_quote": quote,
                    },
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        validate_verification_contract(contract, quote, None)
    grounded = ground_contract(contract, quote)
    assert len(grounded.requirements) == 1
    requirement = grounded.requirements[0]
    assert (
        requirement.critical
        and requirement.verification is None
        and requirement.metric == "unmeasured"
    )
    assert requirement.source_quote == quote
    assert requirement.text.endswith(quote)


def test_explicit_preparation_data_keeps_structured_source_alternative():
    value = {
        "version": 1,
        "kind": "digital_delivery",
        "scope": "digital_artifacts",
        "deliverable": "prepared_print_package",
        "coverage": "all_made",
    }
    quote = json.dumps(value)
    contract = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "prepared",
                    "text": "Give me a prepared print package.",
                    "source_quote": quote,
                    "verification": {**value, "source_quote": quote},
                }
            ]
        }
    )
    validate_verification_contract(contract, quote, None)
    assert ground_contract(contract, quote) == contract

    for extra in ("containing G-code", "for my Ender 3", "with sliced toolpaths"):
        changed = contract.model_copy(deep=True)
        changed.requirements[0].text = "Give me a prepared print package " + extra + "."
        with pytest.raises(ValueError):
            validate_verification_contract(changed, quote, None)
        grounded = ground_contract(changed, quote)
        assert grounded.requirements[0].verification is None
        assert grounded.requirements[0].critical
        assert grounded.requirements[0].source_quote == quote
        assert grounded.requirements[0].text.endswith(changed.requirements[0].text)
