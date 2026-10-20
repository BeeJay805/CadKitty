"""Frozen nominal targets, legacy identities and subject bindings; no CAD/model work."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.intent_attribution import IntentAttribution, validate_attribution
from forge_mind.workspace.measure import ground_contract, merge_contract_edit
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.operation_plan import snapshot, validate_plan
from forge_mind.workspace.plan_draft import (
    PlanDraftCommit,
    PlanDraftPatch,
    PlanDraftRead,
    commit_draft,
    patch_draft,
    read_draft,
)
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import Store
from forge_mind.workspace.verification_admission import (
    criterion_hash,
    resolve_verification_subject,
    validate_verification_contract,
    verification_diagnostics,
    verification_repair_exclusions,
)

MOTOR_QUOTE = (
    "It has exactly five uniform cylindrical through-holes parallel to local Z: one 22 mm "
    "diameter hole at local XY (0, 0), and four 5.2 mm diameter holes at (-30, -20), "
    "(-30, 20), (30, -20) and (30, 20) mm. Use a symmetric diameter tolerance of "
    "0.05 mm and a radial position tolerance of 0.05 mm for each hole."
)
OFFSET_QUOTE = (
    "It has exactly four uniform cylindrical through-holes parallel to local Z: two 8 mm "
    "diameter holes at local XY (-35, -12) and (-35, 12) mm, and two 6 mm diameter holes "
    "at (30, -15) and (30, 15) mm. Use a symmetric diameter tolerance of 0.1 mm and a "
    "radial position tolerance of 0.1 mm for each hole."
)


def pattern_contract(offset=False):
    rows = (
        [(-35, -12, 8), (-35, 12, 8), (30, -15, 6), (30, 15, 6)]
        if offset
        else [(0, 0, 22), (-30, -20, 5.2), (-30, 20, 5.2), (30, -20, 5.2), (30, 20, 5.2)]
    )
    quote = OFFSET_QUOTE if offset else MOTOR_QUOTE
    return Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "pattern",
                    "text": "Nominal through-hole pattern",
                    "source_quote": quote,
                    "verification": {
                        "version": 1,
                        "kind": "nominal_through_hole_pattern",
                        "scope": "nominal_cad",
                        "subject_id": "plate",
                        "subject_role": "designed",
                        "frame": "definition_local",
                        "axis": "z",
                        "count_policy": "exact",
                        "source_quote": quote,
                        "holes": [
                            {
                                "center_mm": [x, y],
                                "diameter_mm": diameter,
                                "diameter_tolerance_mm": 0.1 if offset else 0.05,
                                "position_tolerance_mm": 0.1 if offset else 0.05,
                            }
                            for x, y, diameter in rows
                        ],
                    },
                }
            ]
        }
    )


def attribution(role="designed"):
    return IntentAttribution.model_validate(
        {
            "subjects": [{"id": "plate", "label": "Plate", "role": role, "source_quote": "plate"}],
            "requirement_subjects": {"pattern": ["plate"]},
        }
    )


def source(contract):
    return "Make one nominal CAD plate. " + contract.requirements[0].source_quote


def bound_plan(part_id="plate_body"):
    return DesignPlan.model_validate(
        {
            "objective": "Build the nominal plate",
            "verification_bindings": [
                {"subject_id": "plate", "part_id": part_id, "frame": "definition_local"},
            ],
        }
    )


def document(*, assembly=True, role="make"):
    feature = {
        "id": "body",
        "name": "Plate blank",
        "kind": "box",
        "params": {"x": 80, "y": 60, "z": 6},
    }
    if not assembly:
        return DesignDocument.model_validate(
            {"features": [feature], "outputs": ["body"], "production": {"role": role}}
        )
    return DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "plate_body",
                        "name": "Plate",
                        "features": [feature],
                        "output": "body",
                        "production": {"role": role},
                    }
                ],
                "roots": [{"id": "plate_occ", "definition": "plate_body", "position": [12, 9, 7]}],
            }
        }
    )


def test_legacy_requirement_and_plan_hashes_keep_exact_absent_fields():
    legacy = {
        "id": "width",
        "text": "10 mm wide",
        "source_quote": "10 mm wide",
        "source": "request",
        "metric": "extent_x",
        "comparator": "eq",
        "value": 10.0,
        "tolerance": 0.0,
        "part_id": None,
        "axis": "z",
        "center": None,
        "position_tolerance": 0.1,
        "critical": True,
    }
    expected_contract = {"requirements": [legacy], "coverage_notes": []}
    target = Contract.model_validate(expected_contract)
    assert target.model_dump() == expected_contract
    assert (
        target.fingerprint()
        == hashlib.sha256(json.dumps(expected_contract, sort_keys=True).encode()).hexdigest()
    )
    expected_plan = {
        "objective": "Legacy",
        "parts": [],
        "interfaces": [],
        "assembly_sequence": [],
        "verification": [],
        "unresolved": [],
        "subsystems": [],
        "requirement_dispositions": [],
        "inspection": [],
    }
    plan = DesignPlan.model_validate(expected_plan)
    assert plan.model_dump() == expected_plan
    assert (
        snapshot(plan, 1, target)["plan_hash"]
        == hashlib.sha256(json.dumps(expected_plan, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize("offset", [False, True])
def test_prospective_grouped_pattern_targets_retain_exact_source_and_identity(offset):
    target = pattern_contract(offset)
    before = target.model_dump(mode="json")
    grounded = ground_contract(target, source(target), attribution=attribution())
    assert grounded.model_dump(mode="json") == before
    validate_attribution(attribution(), grounded, source(target))
    criterion = grounded.requirements[0].verification
    assert criterion is not None
    assert (
        criterion_hash(criterion.model_dump(mode="json"))
        == hashlib.sha256(
            json.dumps(criterion.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "change", ["diameter", "position", "allowance", "count", "scope", "physical", "role"]
)
def test_ungrounded_or_mixed_targets_keep_critical_legacy_uncertainty(change):
    target = pattern_contract()
    raw = target.model_dump(mode="json")
    req = raw["requirements"][0]
    attrs = attribution()
    text = source(target)
    if change == "diameter":
        req["verification"]["holes"][0]["diameter_mm"] = (
            5.2  # Present elsewhere; wrong source group.
        )
    elif change == "position":
        req["verification"]["holes"][0]["center_mm"] = [30, 0]
    elif change == "allowance":
        for hole in req["verification"]["holes"]:
            hole["diameter_tolerance_mm"] = 0.5
    elif change == "count":
        req["verification"]["holes"].pop()
    elif change == "scope":
        text = text.replace("nominal CAD", "printed")
    elif change == "physical":
        req["text"] += " and support a 220 g camera"
    elif change == "role":
        attrs = attribution("reference")
    result = ground_contract(Contract.model_validate(raw), text, attribution=attrs)
    actual = result.requirements[0]
    assert actual.verification is None and actual.metric == "unmeasured" and actual.critical
    assert actual.source_quote == target.requirements[0].source_quote
    assert "Typed verification unavailable" in actual.text


def test_authoritative_attribution_rejects_wrong_subject_even_with_valid_source_numbers():
    target = pattern_contract()
    with pytest.raises(ValueError, match="designed subject"):
        validate_attribution(attribution("environment"), target, source(target))
    attrs = attribution().model_copy(update={"requirement_subjects": {"pattern": []}})
    with pytest.raises(ValueError, match="one exact requirement subject"):
        validate_verification_contract(target, source(target), attrs)


def test_invalid_typed_edit_cannot_replace_old_frozen_pattern_or_drop_other_claims():
    target = pattern_contract()
    target.requirements.append(
        Requirement(id="physical", text="Support the camera", source_quote="Support the camera")
    )
    raw = pattern_contract().model_dump(mode="json")
    raw["requirements"][0]["verification"]["holes"][0]["diameter_mm"] = 5.2
    changed = merge_contract_edit(target, Contract.model_validate(raw), source(target))
    assert changed == target
    assert changed.requirements[1].metric == "unmeasured" and changed.requirements[1].critical


@pytest.mark.parametrize(
    "kind,quote",
    [
        ("project_stl", "STL"),
        ("project_step", "STEP"),
        ("editable_native", "editable native CAD"),
        ("prepared_print_package", "a prepared print package"),
        ("bound_instruction_plan", "current bound instruction steps"),
    ],
)
def test_digital_claims_are_separate_frozen_obligations(kind, quote):
    target = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "delivery",
                    "text": quote,
                    "source_quote": quote,
                    "verification": {
                        "version": 1,
                        "kind": "digital_delivery",
                        "scope": "digital_artifacts",
                        "deliverable": kind,
                        "coverage": "all_made",
                        "source_quote": quote,
                    },
                }
            ]
        }
    )
    result = ground_contract(target, "Provide " + quote + ".")
    assert result == target and result.requirements[0].verification is not None


def test_one_digital_predicate_cannot_cover_compound_file_request():
    quote = "Give me editable native CAD, STEP, STL and a prepared print package"
    target = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "delivery",
                    "text": quote,
                    "source_quote": quote,
                    "verification": {
                        "kind": "digital_delivery",
                        "scope": "digital_artifacts",
                        "deliverable": "project_stl",
                        "coverage": "all_made",
                        "source_quote": quote,
                    },
                }
            ]
        }
    )
    assert ground_contract(target, quote).requirements[0].verification is None


def test_explicit_structured_nominal_data_is_supported_without_special_english_phrase():
    raw = pattern_contract().model_dump(mode="json")
    criterion = raw["requirements"][0]["verification"]
    quote = json.dumps({key: value for key, value in criterion.items() if key != "source_quote"})
    criterion["source_quote"] = quote
    raw["requirements"][0]["source_quote"] = quote
    target = Contract.model_validate(raw)
    assert ground_contract(target, quote, attribution=attribution()) == target


@pytest.mark.parametrize(
    "field,value", [("value", 2), ("part_id", "another"), ("tolerance", 0.5), ("center", [0, 0])]
)
def test_typed_and_scalar_selector_fields_cannot_mix(field, value):
    raw = pattern_contract().model_dump(mode="json")
    raw["requirements"][0][field] = value
    with pytest.raises(ValidationError):
        Contract.model_validate(raw)


def test_binding_is_target_free_and_resolves_exact_saved_definition_frame():
    plan, target, doc = bound_plan(), pattern_contract(), document()
    validate_plan(plan, target, document=doc.model_dump(mode="json"))
    assert resolve_verification_subject(target.requirements[0], plan, doc) == {
        "subject_id": "plate",
        "part_id": "plate_body",
        "frame": "definition_local",
        "step_artifact": "definition_plate_body.step",
    }
    raw = plan.model_dump(mode="json")
    raw["verification_bindings"][0]["diameter_tolerance_mm"] = 5
    with pytest.raises(ValidationError):
        DesignPlan.model_validate(raw)
    raw["verification_bindings"][0].pop("diameter_tolerance_mm")
    raw["verification_bindings"][0]["frame"] = "shifted_datum"
    with pytest.raises(ValidationError):
        DesignPlan.model_validate(raw)


@pytest.mark.parametrize("role", ["buy", "reference", "unspecified"])
def test_non_make_target_never_falls_back_to_the_assembly(role):
    with pytest.raises(ValueError, match="make definition"):
        resolve_verification_subject(
            pattern_contract().requirements[0], bound_plan(), document(role=role)
        )


def test_missing_definition_and_unbound_subject_remain_unavailable():
    req = pattern_contract().requirements[0]
    with pytest.raises(ValueError, match="explicit verification subject binding"):
        resolve_verification_subject(req, DesignPlan(objective="Pending"), document())
    with pytest.raises(ValueError, match="named make definition"):
        resolve_verification_subject(req, bound_plan("missing"), document())


def test_flat_document_uses_explicit_canonical_part_only():
    req, doc = pattern_contract().requirements[0], document(assembly=False)
    assert (
        resolve_verification_subject(req, bound_plan("part"), doc)["step_artifact"] == "part.step"
    )
    with pytest.raises(ValueError, match="canonical made part"):
        resolve_verification_subject(req, bound_plan("plate_body"), doc)


def test_unused_custom_feature_cannot_disguise_transformed_import_output():
    raw = document().model_dump(mode="json")
    part = raw["assembly"]["parts"][0]
    part["features"].extend(
        [
            {
                "id": "imported",
                "name": "Imported reference",
                "kind": "import",
                "asset_id": "1" * 32,
            },
            {
                "id": "placed",
                "name": "Placed reference",
                "kind": "transform",
                "inputs": ["imported"],
            },
        ]
    )
    part["output"] = "placed"
    # The unused authored box still exists, but it contributes no output material.
    with pytest.raises(ValueError, match="import-only"):
        resolve_verification_subject(
            pattern_contract().requirements[0], bound_plan(), DesignDocument.model_validate(raw)
        )


def test_binding_patch_retains_existing_binding_and_cannot_change_target_fields():
    target, plan = pattern_contract(), bound_plan()
    basis = EditBasis(
        plan_version=1,
        plan_hash=snapshot(plan, 1, target)["plan_hash"],
        contract_hash=target.fingerprint(),
        input_version=0,
        source_fingerprint="0" * 64,
    )
    result = apply_plan_patch(
        plan, PlanPatch(basis=basis, unresolved=["Print not physically tested"]), target
    )
    assert result.verification_bindings == plan.verification_bindings
    with pytest.raises(ValueError, match="Retain existing verification"):
        validate_plan(DesignPlan(objective="Lost binding"), target, previous=plan)


def test_plan_draft_reads_and_admits_subject_keyed_binding(tmp_path):
    target = pattern_contract()
    store = Store(tmp_path)
    sid = store.create(
        "owner", source(target), target.model_dump(mode="json"), authoring_mode="native_operations"
    )["id"]
    basis = store.edit_basis(sid, "owner")
    meta = patch_draft(
        store,
        sid,
        "owner",
        PlanDraftPatch.model_validate(
            {
                "basis": basis.model_dump(),
                "draft_version": 0,
                "changes": {
                    "objective": "Build the nominal plate",
                    "part_ids": ["plate_body"],
                    "operation_steps": [
                        {
                            "id": "construct_plate",
                            "part_id": "plate_body",
                            "feature_ids": ["body"],
                            "requirement_ids": ["pattern"],
                            "expected_effect": "Construct the owned plate body and requested through pattern",
                        }
                    ],
                    "subsystems": [
                        {
                            "id": "plate_work",
                            "name": "Plate construction",
                            "part_ids": ["plate_body"],
                            "requirement_ids": ["pattern"],
                        }
                    ],
                    "verification_bindings": [
                        {
                            "subject_id": "plate",
                            "part_id": "plate_body",
                            "frame": "definition_local",
                        }
                    ],
                },
            }
        ),
    )
    page = read_draft(
        store, sid, "owner", PlanDraftRead(collection="verification_bindings", record_id="plate")
    )
    assert page["total"] == 1 and page["records"][0]["part_id"] == "plate_body"
    request = PlanDraftCommit.model_validate(
        {key: meta[key] for key in ("basis", "draft_version", "draft_hash")}
    )
    result = commit_draft(store, sid, "owner", request)
    assert result["plan_applied"] is True
    assert (
        store.session_record(sid, "owner")["design_plan"]["verification_bindings"]
        == page["records"]
    )


def test_pattern_has_no_arbitrary_complexity_cap_and_rejects_nonfinite_targets():
    raw = pattern_contract().model_dump(mode="json")
    raw["requirements"][0]["verification"]["holes"] = [
        {
            "center_mm": [index, 0],
            "diameter_mm": 1,
            "diameter_tolerance_mm": 0,
            "position_tolerance_mm": 0,
        }
        for index in range(40)
    ]
    assert len(Contract.model_validate(raw).requirements[0].verification.holes) == 40
    broken = deepcopy(raw)
    broken["requirements"][0]["verification"]["holes"][0]["center_mm"][0] = float("nan")
    with pytest.raises(ValidationError):
        Contract.model_validate(broken)


def box_contract():
    raw = pattern_contract().model_dump(mode="json")
    requirement = raw["requirements"][0]
    quote = (
        "Make one nominal CAD rectangular motor adapter plate named motor_plate. "
        "Its overall size is 80 mm in local X, 60 mm in local Y and 6 mm in local Z, "
        "centered on the local origin. It must be one solid. " + MOTOR_QUOTE
    )
    requirement.update(
        id="body", text="Nominal rectangular body and through-hole pattern", source_quote=quote
    )
    requirement["verification"].update(
        kind="nominal_perforated_box", size_mm=[80, 60, 6], center_mm=[0, 0, 0], source_quote=quote
    )
    return Contract.model_validate(raw)


def box_attribution():
    attrs = attribution().model_dump(mode="json")
    attrs["requirement_subjects"] = {"body": ["plate"]}
    return IntentAttribution.model_validate(attrs)


def test_first_diagnostics_show_digital_quote_and_complete_body_family_errors():
    target = box_contract()
    body = target.requirements[0]
    original_body = body.model_dump(mode="json")
    short_quote = "editable native CAD"
    raw = target.model_dump(mode="json")
    raw["requirements"][0]["verification"] = {
        "kind": "digital_delivery",
        "scope": "digital_artifacts",
        "deliverable": "editable_native",
        "coverage": "all_made",
        "source_quote": short_quote,
    }
    malformed = Contract.model_validate(raw)
    request = body.source_quote + " Give me " + short_quote + "."
    before = malformed.model_dump(mode="json")
    attrs = box_attribution()
    attrs_before = attrs.model_dump(mode="json")
    diagnostics = verification_diagnostics(malformed, request, attrs)
    errors = {row["code"]: row for row in diagnostics}
    assert {"source_quote_mismatch", "incompatible_predicate_family"} <= errors.keys()
    assert errors["source_quote_mismatch"]["field"].endswith(".verification.source_quote")
    family = errors["incompatible_predicate_family"]
    assert family["field"] == "contract.requirements.body.verification.kind"
    assert "size_mm" in family["supported_kinds"]["nominal_perforated_box"]["required_fields"]
    assert family["supported_kinds"]["nominal_perforated_box"]["scope"] == "nominal_cad"
    assert "nominal_through_hole_pattern" in family["message"]
    assert malformed.model_dump(mode="json") == before
    assert attrs.model_dump(mode="json") == attrs_before
    with pytest.raises(ValueError, match="complete requirement source quote"):
        validate_verification_contract(malformed, request, attrs)
    grounded = ground_contract(malformed, request, attribution=attrs)
    assert grounded.requirements[0].verification is None
    assert grounded.requirements[0].critical
    assert grounded.requirements[0].source_quote == body.source_quote
    # Only the author can supply the actual source-backed body targets.
    assert target.requirements[0].model_dump(mode="json") == original_body
    assert verification_diagnostics(target, request, attrs) == []


def test_digital_whole_geometry_exclusion_binds_exact_unchanged_source_and_row():
    target = box_contract()
    raw = target.model_dump(mode="json")
    raw["requirements"][0]["verification"] = {
        "kind": "digital_delivery",
        "scope": "digital_artifacts",
        "deliverable": "editable_native",
        "coverage": "all_made",
        "source_quote": "editable native CAD",
    }
    requirement = Contract.model_validate(raw).requirements[0]
    source_text = requirement.source_quote + " Give me editable native CAD."
    attrs = box_attribution()
    before = requirement.model_dump(mode="json")
    evidence = verification_repair_exclusions(requirement, source_text, attrs)
    assert len(evidence) == 1
    assert evidence[0]["kind"] == "digital_delivery"
    assert evidence[0]["rule"] == "digital_fixed_geometry_conflict_v1"
    assert evidence[0]["source_sha256"] == hashlib.sha256(source_text.encode()).hexdigest()
    assert (
        evidence[0]["parent_quote_sha256"]
        == hashlib.sha256(requirement.source_quote.encode()).hexdigest()
    )
    assert evidence[0]["requirement_sha256"] == criterion_hash(before)
    assert evidence[0]["criterion_sha256"] == criterion_hash(before["verification"])
    assert requirement.model_dump(mode="json") == before
    changed = requirement.model_copy(deep=True)
    changed.text += " Retain the same geometry."
    refreshed = verification_repair_exclusions(changed, source_text, attrs)
    assert refreshed[0]["requirement_sha256"] != evidence[0]["requirement_sha256"]
    attrs.subjects[0].label = "A changed source interpretation"
    assert (
        verification_repair_exclusions(requirement, source_text, attrs)[0]["attribution_sha256"]
        != evidence[0]["attribution_sha256"]
    )


@pytest.mark.parametrize("extra", ["for my Ender 3", "containing G-code", "with sliced toolpaths"])
def test_prepared_specific_family_error_never_excludes_whole_digital_kind(extra):
    quote = "Give me a prepared print package " + extra + "."
    contract = Contract(
        requirements=[
            Requirement(
                id="printing",
                text=quote,
                source_quote=quote,
                verification={
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "deliverable": "prepared_print_package",
                    "coverage": "all_made",
                    "source_quote": quote,
                },
            )
        ]
    )
    assert "incompatible_predicate_family" in {
        row["code"] for row in verification_diagnostics(contract, quote, None)
    }
    assert verification_repair_exclusions(contract.requirements[0], quote, None) == []


def test_diagnostics_collect_independent_requirement_source_and_attribution_errors():
    target = pattern_contract()
    raw = target.model_dump(mode="json")
    raw["requirements"][0]["verification"]["source_quote"] = "invented source"
    raw["requirements"][0]["verification"]["holes"][0]["diameter_mm"] = 9
    malformed = Contract.model_validate(raw)
    diagnostics = verification_diagnostics(malformed, source(target), attribution("reference"))
    assert {
        "source_quote_mismatch",
        "source_quote_not_in_request",
        "unsupported_source_targets",
        "invalid_subject_attribution",
    } <= {row["code"] for row in diagnostics}
    assert all(row["requirement_id"] == "pattern" for row in diagnostics)


def test_complete_rectangular_body_and_embedded_pattern_are_frozen_together():
    target = box_contract()
    frozen = target.model_dump(mode="json")
    result = ground_contract(
        target, target.requirements[0].source_quote, attribution=box_attribution()
    )
    assert result.model_dump(mode="json") == frozen
    validate_attribution(box_attribution(), result, target.requirements[0].source_quote)
    resolved = resolve_verification_subject(result.requirements[0], bound_plan(), document())
    assert resolved["step_artifact"] == "definition_plate_body.step"
    criterion = result.requirements[0].verification
    assert criterion.kind == "nominal_perforated_box"
    assert criterion.size_mm == (80, 60, 6) and criterion.center_mm == (0, 0, 0)
    assert len(criterion.holes) == 5


@pytest.mark.parametrize("change", ["shape", "origin", "size", "digital", "physical"])
def test_body_admission_cannot_drop_requested_shape_center_or_other_duties(change):
    target = box_contract()
    raw = target.model_dump(mode="json")
    req = raw["requirements"][0]
    if change == "shape":
        req["source_quote"] = req["source_quote"].replace("rectangular ", "")
    elif change == "origin":
        req["verification"]["center_mm"] = [1, 0, 0]
    elif change == "size":
        req["verification"]["size_mm"] = [60, 80, 6]  # Both values exist; wrong source axes.
    elif change == "digital":
        req["source_quote"] += " Provide STL."
    elif change == "physical":
        req["source_quote"] += " It must hold the motor safely."
    req["verification"]["source_quote"] = req["source_quote"]
    target = Contract.model_validate(raw)
    result = ground_contract(target, req["source_quote"], attribution=box_attribution())
    assert result.requirements[0].verification is None and result.requirements[0].critical


def test_body_accepts_an_explicit_local_center_without_changing_its_definition():
    raw = box_contract().model_dump(mode="json")
    req = raw["requirements"][0]
    quote = req["source_quote"].replace("centered on the local origin", "center at local (5, 0, 0)")
    req["source_quote"] = req["verification"]["source_quote"] = quote
    req["verification"]["center_mm"] = [5, 0, 0]
    target = Contract.model_validate(raw)
    assert ground_contract(target, quote, attribution=box_attribution()) == target


def test_embedded_body_preserves_separate_scalar_and_pattern_requirements():
    body = box_contract()
    pattern = pattern_contract()
    body.requirements.extend(pattern.requirements)
    body.requirements.append(
        Requirement(
            id="width",
            text="80 mm wide",
            source_quote="80 mm",
            metric="extent_x",
            value=80,
            part_id="plate_body",
        )
    )
    attrs = box_attribution().model_copy(
        update={"requirement_subjects": {"body": ["plate"], "pattern": ["plate"]}}
    )
    validate_verification_contract(body, body.requirements[0].source_quote, attrs)
    validate_plan(bound_plan(), body)
    assert [req.id for req in body.requirements] == ["body", "pattern", "width"]
    contradictory = body.model_copy(deep=True)
    contradictory.requirements[-1].value = 81
    with pytest.raises(ValueError, match="scalar target conflicts"):
        validate_plan(bound_plan(), contradictory)


def test_reliably_disjoint_embedded_and_standalone_patterns_are_rejected():
    target = box_contract()
    conflicting = pattern_contract().model_dump(mode="json")["requirements"][0]
    quote = conflicting["source_quote"].replace("one 22 mm", "one 10 mm")
    conflicting["source_quote"] = conflicting["verification"]["source_quote"] = quote
    conflicting["verification"]["holes"][0]["diameter_mm"] = 10
    target.requirements.append(Requirement.model_validate(conflicting))
    text = target.requirements[0].source_quote + " " + quote
    with pytest.raises(ValueError, match="Conflicting frozen patterns"):
        validate_verification_contract(target, text, None)


@pytest.mark.parametrize(
    "field,value", [("size_mm", [80, 0, 6]), ("center_mm", [0, float("inf"), 0])]
)
def test_body_vector_targets_are_positive_and_finite(field, value):
    raw = box_contract().model_dump(mode="json")
    raw["requirements"][0]["verification"][field] = value
    with pytest.raises(ValidationError):
        Contract.model_validate(raw)
