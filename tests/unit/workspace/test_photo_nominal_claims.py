"""Prospective source proof keeps image estimates outside literal facts."""

import pytest

from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.photo_nominal import PhotoNominalClaim, attach_photo_nominal_claims

SOURCE = "Copy this 50 by 30 mm rectangular plate, 5 mm thick. Both holes go straight through."


def draft(text="Copy this rectangular plate", quote=SOURCE):
    return ContractDraft.model_validate(
        {
            "contract": {"requirements": [{"id": "r1", "text": text, "source_quote": quote}]},
            "attribution": {
                "subjects": [
                    {"id": "plate", "label": "plate", "role": "designed", "source_quote": "plate"}
                ],
                "requirement_subjects": {"r1": ["plate"]},
                "facts": [
                    {
                        "id": key,
                        "property": key,
                        "value": value,
                        "unit": unit,
                        "source_quote": source,
                        "subject_id": "plate",
                        "requirement_ids": ["r1"],
                    }
                    for key, value, unit, source in [
                        ("width", 50.0, "mm", "50 by 30 mm"),
                        ("height", 30.0, "mm", "50 by 30 mm"),
                        ("depth", 5.0, "mm", "5 mm thick"),
                        ("count", 2.0, "holes", "Both holes go straight through."),
                    ]
                ],
            },
        }
    )


def claim(**changes):
    return PhotoNominalClaim.model_validate(
        {
            "requirement_id": "r1",
            "subject_id": "plate",
            "claims": [
                "visible_profile",
                "rectangular_outline",
                "extent_u",
                "extent_v",
                "thickness",
                "opening_count",
                "straight_through",
            ],
            "size_fact_ids": ["width", "height", "depth"],
            "hole_count_fact_id": "count",
            "frame_policy": "image_plane_xy",
            **changes,
        }
    )


def test_fresh_claim_preserves_original_duty_and_literal_facts():
    original = draft()
    result = attach_photo_nominal_claims(original, SOURCE, [claim()])
    row = result.contract.requirements[0]
    assert row.model_dump(exclude={"verification"}) == original.contract.requirements[0].model_dump(
        exclude={"verification"}
    )
    assert result.attribution == original.attribution
    assert row.critical and row.verification.size_mm == (50.0, 30.0, 5.0)
    assert row.verification.hole_count == 2
    assert original.contract.requirements[0].verification is None


@pytest.mark.parametrize(
    "extra",
    [
        " and withstand 100 N",
        " and fit a shaft",
        " and engrave a name",
        " with rounded corners",
        " and seal water",
    ],
)
def test_physical_or_unsupported_conjunct_cannot_close(extra):
    source = SOURCE + extra
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(draft(SOURCE + extra, source), source, [claim()])


def test_wrong_count_unit_cannot_become_owner_hole_count():
    original = draft()
    original.attribution.facts[-1].unit = "mm"
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(original, SOURCE, [claim()])


def test_unknown_subject_and_ambiguous_subject_refuse():
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(draft(), SOURCE, [claim(subject_id="other")])
    original = draft()
    original.attribution.facts[0].candidate_subject_ids = ["plate"]
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(original, SOURCE, [claim()])


def test_fabricated_dimension_refuses():
    original = draft()
    original.attribution.facts[0].value = 51.0
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(original, SOURCE, [claim()])


def test_absent_claims_leave_legacy_identical():
    original = draft()
    assert attach_photo_nominal_claims(original, SOURCE, []) is original


@pytest.mark.parametrize(
    "extra",
    [
        " and add counterbores to both holes",
        " and apply a blue coating",
        " and add hidden recesses",
        ". Both holes must have width 5 mm.",
        ". Add a 12 by 8 mm rectangular panel.",
    ],
)
def test_complete_claim_list_cannot_hide_residual_duty(extra):
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(
            draft(SOURCE + extra, SOURCE + extra), SOURCE + extra, [claim()]
        )


def test_negated_throughness_refuses():
    source = SOURCE.replace("go straight through", "do not go straight through")
    original = draft("Both holes do not go straight through.", source)
    original.attribution.facts[-1].source_quote = "Both holes do not go straight through."
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(original, source, [claim()])


def wire(original=None, proposals=None):
    from forge_mind.workspace.source_delivery_contract_draft import PhotoSourceDeliveryContractDraft

    original = original or draft(SOURCE)
    quotes = []

    def quote(text):
        key = f"q{len(quotes)}"
        quotes.append({"id": key, "text": text})
        return key

    rows = [
        {
            "id": row.id,
            "text": row.text,
            "quote_ref": quote(row.source_quote),
            "subject_ids": ["plate"],
        }
        for row in original.contract.requirements
    ]
    subject_quote = quote("plate")
    groups = [
        {
            "subject_id": fact.subject_id,
            "unit": fact.unit,
            "quote_ref": quote(fact.source_quote),
            "requirement_ids": fact.requirement_ids,
            "values": [{"id": fact.id, "property": fact.property, "value": fact.value}],
        }
        for fact in original.attribution.facts
    ]
    return PhotoSourceDeliveryContractDraft.model_validate(
        {
            "quotes": quotes,
            "requirements": rows,
            "requirement_order": [row.id for row in original.contract.requirements],
            "subjects": [
                {"id": "plate", "label": "plate", "role": "designed", "quote_ref": subject_quote}
            ],
            "fact_groups": groups,
            "fact_order": [fact.id for fact in original.attribution.facts],
            "photo_nominal_claims": [
                proposal.model_dump() for proposal in (proposals or [claim()])
            ],
            "photo_reference": {"role": "copy", "source_quote": SOURCE},
        }
    )


def test_actual_source_wire_preflight_and_exact_replay():
    from forge_mind.workspace.document import digest
    from forge_mind.workspace.photo_nominal import validate_photo_source_replay
    from forge_mind.workspace.source_delivery_contract_draft import preflight_source_delivery_draft

    preflight = preflight_source_delivery_draft(wire(), SOURCE)
    assert preflight.compilation["status"] == "ready", preflight.compilation
    lowered = preflight.draft
    document = {
        "intent_compilation": {
            **preflight.compilation,
            "job_id": "job",
            "source_hash": digest(SOURCE),
            "source_proposal": preflight.source_proposal,
        }
    }
    validate_photo_source_replay(document, "job", SOURCE, lowered.contract, lowered.attribution)
    for key, value in [("job_id", "other"), ("source_hash", "0" * 64)]:
        changed = {"intent_compilation": {**document["intent_compilation"], key: value}}
        with pytest.raises(ValueError):
            validate_photo_source_replay(
                changed, "job", SOURCE, lowered.contract, lowered.attribution
            )
    changed = lowered.model_copy(deep=True)
    changed.contract.requirements[0].text += " and add counterbores"
    with pytest.raises(ValueError):
        validate_photo_source_replay(document, "job", SOURCE, changed.contract, changed.attribution)


def test_actual_seven_atomic_rows_with_shared_quotes_admit():
    original = draft()
    templates = [
        (
            "Copy the plate shown in the supplied visual reference, including its visible hole shapes, relative sizes and positions.",
            "Copy this 50 by 30 mm rectangular plate, 5 mm thick.",
            ["visible_profile"],
        ),
        (
            "The plate must have a rectangular outline.",
            "rectangular plate",
            ["rectangular_outline"],
        ),
        ("The plate must have a 50 mm side dimension.", "50 by 30 mm", ["extent_u"]),
        ("The plate must have a 30 mm side dimension.", "50 by 30 mm", ["extent_v"]),
        ("The plate must be 5 mm thick.", "5 mm thick", ["thickness"]),
        (
            "Reproduce both visible holes.",
            "Both holes go straight through.",
            ["visible_profile", "opening_count"],
        ),
        (
            "Both holes must go straight through the plate.",
            "Both holes go straight through.",
            ["opening_count", "straight_through"],
        ),
    ]
    from forge_mind.workspace.models import Requirement

    original.contract.requirements = [
        Requirement(id=f"r{i}", text=text, source_quote=quote)
        for i, (text, quote, _) in enumerate(templates, 1)
    ]
    original.attribution.requirement_subjects = {
        row.id: ["plate"] for row in original.contract.requirements
    }
    for fact, ids in zip(
        original.attribution.facts, [["r3", "r4"], ["r3", "r4"], ["r5"], ["r6", "r7"]], strict=True
    ):
        fact.requirement_ids = ids
    proposals = [
        claim(requirement_id=f"r{i}", claims=claims)
        for i, (_, _, claims) in enumerate(templates, 1)
    ]
    result = attach_photo_nominal_claims(original, SOURCE, proposals)
    assert [row.id for row in result.contract.requirements] == [f"r{i}" for i in range(1, 8)]
    assert all(row.critical and row.verification for row in result.contract.requirements)
    from forge_mind.workspace.source_delivery_contract_draft import preflight_source_delivery_draft

    preflight = preflight_source_delivery_draft(wire(original, proposals), SOURCE)
    assert preflight.compilation["status"] == "ready", preflight.compilation


def test_missing_disposition_refuses_and_unsupported_stays_unknown():
    original = draft()
    original.contract.requirements.append(
        original.contract.requirements[0].model_copy(
            update={"id": "r2", "text": "Withstand 100 N", "source_quote": "Withstand 100 N"}
        )
    )
    original.attribution.requirement_subjects["r2"] = ["plate"]
    source = SOURCE + " Withstand 100 N"
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(original, source, [claim()])
    result = attach_photo_nominal_claims(
        original,
        source,
        [
            claim(),
            PhotoNominalClaim(requirement_id="r2", unsupported_reason="No supported load proof"),
        ],
    )
    assert result.contract.requirements[1].critical
    assert result.contract.requirements[1].verification is None


def test_explicit_zero_holes_and_derived_count_have_distinct_authority():
    from forge_mind.workspace.photo_nominal import (
        photo_source_constraints,
        validate_photo_target_constraints,
    )

    zero_source = SOURCE.replace("Both holes go straight through.", "No holes.")
    original = draft("Copy this 50 by 30 mm rectangular plate, 5 mm thick. No holes.", zero_source)
    original.attribution.facts[-1].value = 0.0
    original.attribution.facts[-1].source_quote = "No holes."
    result = attach_photo_nominal_claims(
        original,
        zero_source,
        [
            claim(
                claims=[
                    "visible_profile",
                    "rectangular_outline",
                    "extent_u",
                    "extent_v",
                    "thickness",
                    "opening_count",
                ]
            )
        ],
    )
    criterion = result.contract.requirements[0].verification
    assert criterion.hole_count == 0 and criterion.hole_count_fact_id == "count"
    constraints = photo_source_constraints(criterion)
    with pytest.raises(ValueError):
        validate_photo_target_constraints(criterion, {**constraints, "hole_count": 1})
    source = SOURCE.replace("Both holes go straight through.", "Visible holes go straight through.")
    original = draft("Copy this rectangular plate", source)
    original.attribution.facts.pop()
    criterion = (
        attach_photo_nominal_claims(
            original,
            source,
            [
                claim(
                    claims=[
                        "visible_profile",
                        "rectangular_outline",
                        "extent_u",
                        "extent_v",
                        "thickness",
                        "straight_through",
                    ],
                    hole_count_fact_id=None,
                )
            ],
        )
        .contract.requirements[0]
        .verification
    )
    assert criterion.hole_count is None and criterion.hole_count_fact_id is None
    validate_photo_target_constraints(
        criterion, {**photo_source_constraints(criterion), "hole_count": 3}
    )
    with pytest.raises(ValueError):
        validate_photo_target_constraints(
            criterion, {**photo_source_constraints(criterion), "hole_count": -1}
        )


def test_explicit_count_cannot_be_replaced_by_derived_inventory():
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(draft(), SOURCE, [claim(hole_count_fact_id=None)])


def test_photo_schema_requires_role_and_dispositions_but_legacy_does_not():
    from pydantic import ValidationError

    from forge_mind.workspace.source_delivery_contract_draft import (
        PhotoSourceDeliveryContractDraft,
        SourceDeliveryContractDraft,
    )

    assert SourceDeliveryContractDraft().photo_nominal_claims == []
    with pytest.raises(ValidationError):
        PhotoSourceDeliveryContractDraft()
    from forge_mind.workspace.photo_nominal import (
        PhotoReferenceDisposition,
        validate_photo_reference,
    )

    with pytest.raises(ValueError):
        validate_photo_reference(
            PhotoReferenceDisposition(role="context", source_quote=SOURCE), SOURCE
        )
    context = "Make a bracket to hold this object."
    with pytest.raises(ValueError):
        validate_photo_reference(
            PhotoReferenceDisposition(role="context", source_quote=context), context
        )
    validate_photo_reference(
        PhotoReferenceDisposition(role="ambiguous", source_quote=context), context
    )
    context = "Use this photo only as context."
    validate_photo_reference(
        PhotoReferenceDisposition(role="context", source_quote=context), context
    )


@pytest.mark.parametrize(
    "source",
    [
        "Use this photo to create an exact replica.",
        "Make an exact duplicate from this image.",
        "Make a bracket to hold this object.",
    ],
)
def test_context_exemption_needs_positive_source_scope(source):
    from forge_mind.workspace.photo_nominal import (
        PhotoReferenceDisposition,
        validate_photo_reference,
    )

    with pytest.raises(ValueError):
        validate_photo_reference(
            PhotoReferenceDisposition(role="context", source_quote=source), source
        )


@pytest.mark.parametrize(
    "extra",
    [
        " Both holes must be 50 by 30 mm.",
        " Both holes must be 5 mm thick.",
        " Both bores must have thickness 5 mm.",
    ],
)
def test_pair_and_thickness_keep_their_actual_referent(extra):
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(
            draft(SOURCE + extra, SOURCE + extra), SOURCE + extra, [claim()]
        )


@pytest.mark.parametrize(
    "extra",
    [
        " Both holes must have width 5 mm.",
        " Both holes must have diameter 5 mm.",
        " Copy a 12 by 8 mm rectangular panel.",
        " The plate must be 6 mm thick.",
    ],
)
def test_numeric_roles_cannot_acquire_an_unrelated_source_value(extra):
    with pytest.raises(ValueError):
        attach_photo_nominal_claims(
            draft(SOURCE + extra, SOURCE + extra), SOURCE + extra, [claim()]
        )


def test_photo_geometric_mapping_needs_binding_and_valid_source():
    from forge_mind.workspace.completion_coverage import _typed_mappings
    from forge_mind.workspace.models import DesignPlan

    lowered = attach_photo_nominal_claims(draft(SOURCE), SOURCE, [claim()])
    plan = DesignPlan(
        objective="Copy the source profile",
        verification_bindings=[
            {"subject_id": "plate", "part_id": "part", "frame": "definition_local"}
        ],
    ).model_dump(mode="json")
    rows = _typed_mappings(
        plan, lowered.contract.model_dump(mode="json"), SOURCE, lowered.attribution
    )
    assert rows[0]["mapping_status"] == "subject_binding_declared"
    assert rows[0]["status"] == "unverified"
    plan["verification_bindings"] = []
    rows = _typed_mappings(
        plan, lowered.contract.model_dump(mode="json"), SOURCE, lowered.attribution
    )
    assert rows[0]["mapping_status"] == "missing_subject_binding"
    assert (
        _typed_mappings(
            plan, lowered.contract.model_dump(mode="json"), "changed source", lowered.attribution
        )
        == []
    )


def test_ambiguous_reference_uses_existing_question_before_construction():
    from forge_mind.workspace.source_delivery_contract_draft import preflight_source_delivery_draft

    value = wire()
    value.photo_reference.role = "ambiguous"
    value.photo_nominal_claims = [
        PhotoNominalClaim(
            requirement_id="r1",
            unsupported_reason="Clarify whether the image is a copy target or context",
        )
    ]
    first = preflight_source_delivery_draft(value, SOURCE)
    assert first.compilation["status"] == "invalid_interpretation"
    from forge_mind.workspace.intent_attribution import IntentIssue

    value.issues = [
        IntentIssue(
            id="reference_role",
            question="Should I copy the pictured object or use it as context?",
            reason="The reference role is unresolved",
            requirement_ids=["r1"],
            input_dependency={"kind": "source_fact", "input_name": "reference_role"},
        )
    ]
    result = preflight_source_delivery_draft(value, SOURCE)
    assert result.compilation["status"] == "needs_customer_input"
    assert result.compilation["blocking_issue_ids"] == ["reference_role"]
    assert result.draft.contract.requirements[0].verification is None


def test_context_role_survives_legitimate_legacy_canonical_correction_only():
    from copy import deepcopy

    from forge_mind.workspace.document import digest
    from forge_mind.workspace.models import Requirement
    from forge_mind.workspace.photo_nominal import validate_photo_source_replay
    from forge_mind.workspace.source_delivery_contract_draft import (
        SourceDeliveryContractDraft,
        lower_source_delivery_draft,
        source_delivery_proposal_record,
    )

    source = "Make a bracket for my Ender 3. Use this photo only as context."
    proposal = SourceDeliveryContractDraft.model_validate(
        {
            "quotes": [{"id": "q1", "text": "Make a bracket"}, {"id": "q2", "text": "bracket"}],
            "requirements": [
                {
                    "id": "r1",
                    "text": "Make a bracket",
                    "quote_ref": "q1",
                    "subject_ids": ["bracket"],
                }
            ],
            "requirement_order": ["r1"],
            "subjects": [
                {"id": "bracket", "label": "bracket", "role": "designed", "quote_ref": "q2"}
            ],
            "photo_reference": {
                "role": "context",
                "source_quote": "Use this photo only as context.",
            },
        }
    )
    lowered = lower_source_delivery_draft(proposal, source)
    corrected = lowered.model_copy(deep=True)
    corrected.contract.requirements.append(
        Requirement(id="machine", text="Retain the Ender 3 duty", source_quote="for my Ender 3")
    )
    record = source_delivery_proposal_record(proposal, source, lowered)
    document = {
        "intent_compilation": {
            "job_id": "job",
            "status": "ready",
            "source_hash": digest(source),
            "source_proposal": record,
        }
    }
    validate_photo_source_replay(
        document, "job", source, corrected.contract, corrected.attribution, require_photo=True
    )
    for changed in [
        {**document["intent_compilation"], "job_id": "other"},
        {**document["intent_compilation"], "source_hash": "0" * 64},
        {
            **document["intent_compilation"],
            "source_proposal": {**record, "raw_response_hash": "0" * 64},
        },
    ]:
        with pytest.raises(ValueError):
            validate_photo_source_replay(
                {"intent_compilation": changed},
                "job",
                source,
                corrected.contract,
                corrected.attribution,
                require_photo=True,
            )
    # The context exception cannot install even one supported geometric photo row.
    copied = attach_photo_nominal_claims(draft(SOURCE), SOURCE, [claim()])
    with pytest.raises(ValueError):
        validate_photo_source_replay(
            document, "job", source, copied.contract, copied.attribution, require_photo=True
        )
    forged = deepcopy(document)
    forged["intent_compilation"]["source_proposal"]["raw_response"]["photo_reference"][
        "source_quote"
    ] = "Make a bracket"
    forged["intent_compilation"]["source_proposal"]["raw_response_hash"] = digest(
        forged["intent_compilation"]["source_proposal"]["raw_response"]
    )
    with pytest.raises(ValueError):
        validate_photo_source_replay(
            forged, "job", source, corrected.contract, corrected.attribution, require_photo=True
        )
