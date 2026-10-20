"""Pure intent provenance and identity controls; no model or CAD execution."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.intent_attribution import (
    ContractDraft,
    IntentAnswer,
    IntentAttribution,
    IntentConflict,
    IntentResolution,
    validate_attribution,
)
from forge_mind.workspace.models import Contract

SOURCE = "Make a carrier for the sensor. The sensor is 45 mm wide. Use two screws."


def contract() -> Contract:
    return Contract.model_validate(
        {
            "requirements": [
                {"id": "fit", "text": "Fit the sensor", "source_quote": "for the sensor"},
                {"id": "fasteners", "text": "Use two screws", "source_quote": "Use two screws"},
            ]
        }
    )


def payload() -> dict:
    return {
        "subjects": [
            {"id": "carrier", "label": "Carrier", "role": "designed", "source_quote": "carrier"},
            {"id": "sensor", "label": "Sensor", "role": "reference", "source_quote": "sensor"},
        ],
        "facts": [
            {
                "id": "sensor_width",
                "subject_id": "sensor",
                "property": "body width",
                "value": 45,
                "unit": "mm",
                "source_quote": "The sensor is 45 mm wide",
                "requirement_ids": ["fit"],
            },
            {
                "id": "screw_count",
                "subject_id": "carrier",
                "property": "screw count",
                "value": 2,
                "unit": "screws",
                "source_quote": "Use two screws",
                "requirement_ids": ["fasteners"],
            },
        ],
        "requirement_subjects": {"fit": ["carrier", "sensor"], "fasteners": ["carrier"]},
    }


def ambiguous_payload() -> dict:
    data = payload()
    data["facts"][0].pop("subject_id")
    data["facts"][0]["candidate_subject_ids"] = ["carrier", "sensor"]
    data["issues"] = [
        {
            "id": "width_subject",
            "question": "Which object does the quoted width describe?",
            "reason": "The subject changes the fit geometry.",
            "fact_ids": ["sensor_width"],
            "requirement_ids": ["fit"],
        }
    ]
    return data


def test_clear_attribution_retains_literal_facts_and_canonical_contract():
    data, target = payload(), contract()
    before = deepcopy(data)
    canonical, fingerprint = target.model_dump(), target.fingerprint()
    draft = ContractDraft(contract=target, attribution=IntentAttribution.model_validate(data))
    assert validate_attribution(draft.attribution, draft.contract, SOURCE) is None
    assert data == before
    assert target.model_dump() == draft.contract.model_dump() == canonical
    assert target.fingerprint() == draft.contract.fingerprint() == fingerprint
    assert draft.attribution.facts[0].subject_id == "sensor"
    assert draft.attribution.facts[1].value == 2
    assert not draft.attribution.issues
    assert ContractDraft.model_validate_json(draft.model_dump_json()) == draft


def test_equal_values_do_not_merge_reference_and_designed_subjects():
    data = payload()
    source = SOURCE + " The carrier is 45 mm wide."
    data["facts"].append(
        {
            **data["facts"][0],
            "id": "carrier_width",
            "subject_id": "carrier",
            "source_quote": "The carrier is 45 mm wide",
        }
    )
    attribution = IntentAttribution.model_validate(data)
    validate_attribution(attribution, contract(), source)
    assert [(fact.id, fact.subject_id, fact.value) for fact in attribution.facts[::2]] == [
        ("sensor_width", "sensor", 45),
        ("carrier_width", "carrier", 45),
    ]


def test_ambiguity_is_retained_without_asserting_resolution_or_proof():
    attribution = IntentAttribution.model_validate(ambiguous_payload())
    validate_attribution(attribution, contract(), SOURCE)
    assert attribution.facts[0].subject_id is None
    assert attribution.issues[0].fact_ids == ["sensor_width"]
    assert set(attribution.model_dump()) == {"subjects", "facts", "requirement_subjects", "issues"}


def test_single_candidate_can_still_need_an_answer():
    data = ambiguous_payload()
    data["facts"][0]["candidate_subject_ids"] = ["sensor"]
    validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize("missing", ["candidates", "issue"])
def test_unresolved_subject_needs_candidates_and_a_linked_issue(missing):
    data = ambiguous_payload()
    if missing == "candidates":
        data["facts"][0]["candidate_subject_ids"] = []
    else:
        data["issues"] = []
    with pytest.raises(ValueError, match="unresolved fact"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


def test_resolved_subject_cannot_also_claim_unresolved_candidates():
    data = payload()
    data["facts"][0]["candidate_subject_ids"] = ["sensor"]
    with pytest.raises(ValueError, match="subject and unresolved candidates"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("fact", "subject_id", "unknown"),
        ("fact", "candidate_subject_ids", ["unknown"]),
        ("fact", "requirement_ids", ["unknown"]),
        ("issue", "fact_ids", ["unknown"]),
        ("issue", "requirement_ids", ["unknown"]),
        ("map", "unknown", ["sensor"]),
        ("map", "fit", ["unknown"]),
    ],
)
def test_unknown_references_are_rejected(target, field, value):
    data = ambiguous_payload()
    if target == "fact":
        data["facts"][0][field] = value
    elif target == "issue":
        data["issues"][0][field] = value
    else:
        data["requirement_subjects"][field] = value
    with pytest.raises(ValueError, match="unknown"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize("collection", ["subjects", "facts", "issues"])
def test_record_ids_are_unique(collection):
    data = ambiguous_payload()
    data[collection].append(deepcopy(data[collection][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("fact", "candidate_subject_ids", ["sensor", "sensor"]),
        ("fact", "requirement_ids", ["fit", "fit"]),
        ("issue", "fact_ids", ["sensor_width", "sensor_width"]),
        ("issue", "requirement_ids", ["fit", "fit"]),
        ("map", "fit", ["sensor", "sensor"]),
    ],
)
def test_reference_lists_do_not_repeat_ids(target, field, value):
    data = ambiguous_payload()
    if target == "fact":
        data["facts"][0][field] = value
    elif target == "issue":
        data["issues"][0][field] = value
    else:
        data["requirement_subjects"][field] = value
    with pytest.raises(ValueError, match="unique"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize("target", ["subjects", "facts"])
def test_source_quotes_are_exact_not_normalized(target):
    data = payload()
    data[target][0]["source_quote"] = data[target][0]["source_quote"].upper()
    with pytest.raises(ValueError, match="quote must occur exactly"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize(
    "field,value", [("value", 4.5), ("value", 45000), ("unit", "cm"), ("unit", "m")]
)
def test_facts_cannot_convert_values_or_invent_units(field, value):
    data = payload()
    data["facts"][0][field] = value
    with pytest.raises(ValueError, match="must occur literally"):
        validate_attribution(IntentAttribution.model_validate(data), contract(), SOURCE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "45", True])
def test_fact_values_are_finite_numeric_literals(value):
    data = payload()
    data["facts"][0]["value"] = value
    with pytest.raises(ValidationError):
        IntentAttribution.model_validate(data)


def test_legacy_conflict_can_reference_only_frozen_requirements():
    attribution = IntentAttribution.model_validate(
        {
            "issues": [
                {
                    "id": "legacy",
                    "question": "Which object?",
                    "reason": "Fit changes.",
                    "requirement_ids": ["fit"],
                }
            ]
        }
    )
    validate_attribution(attribution, contract(), SOURCE)
    assert not attribution.facts and not attribution.subjects


def test_issue_without_any_affected_record_is_not_an_attribution_conflict():
    attribution = IntentAttribution.model_validate(
        {
            "issues": [
                {"id": "generic", "question": "Anything else?", "reason": "No affected intent."}
            ]
        }
    )
    with pytest.raises(ValueError, match="reference a fact or requirement"):
        validate_attribution(attribution, contract(), SOURCE)


@pytest.mark.parametrize("target", ["attribution", "subject", "fact", "issue"])
def test_model_fields_cannot_assert_confirmation_or_proof(target):
    data = ambiguous_payload()
    collection = {"subject": "subjects", "fact": "facts", "issue": "issues"}
    item = data if target == "attribution" else data[collection[target]][0]
    item["owner_confirmed"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        IntentAttribution.model_validate(data)


def test_raw_owner_answer_is_preserved_and_bound_to_an_exact_draft():
    answer = "  The quoted size belongs to the existing sensor.\nKeep its buttons clear.  "
    resolution = IntentResolution(
        expected_version=3,
        draft_hash="a" * 64,
        answers=[IntentAnswer(issue_id="width_subject", answer=answer)],
    )
    assert resolution.answers[0].answer == answer
    assert IntentResolution.model_validate_json(resolution.model_dump_json()) == resolution


@pytest.mark.parametrize(
    "change",
    [
        {"expected_version": 0},
        {"expected_version": True},
        {"expected_version": "1"},
        {"draft_hash": "A" * 64},
        {"draft_hash": "a" * 63},
        {"answers": []},
        {"answers": [{"issue_id": "width_subject", "answer": "  \n"}]},
        {"answers": [{"issue_id": "width_subject", "answer": "sensor", "confirmed": True}]},
        {"approved": True},
    ],
)
def test_resolution_rejects_malformed_identity_answers_and_approval_fields(change):
    data = {
        "expected_version": 1,
        "draft_hash": "a" * 64,
        "answers": [{"issue_id": "width_subject", "answer": "sensor"}],
    }
    data.update(change)
    with pytest.raises(ValidationError):
        IntentResolution.model_validate(data)


def test_resolution_rejects_two_answers_for_the_same_issue():
    with pytest.raises(ValidationError, match="unique"):
        IntentResolution(
            expected_version=1,
            draft_hash="a" * 64,
            answers=[
                IntentAnswer(issue_id="same", answer="first"),
                IntentAnswer(issue_id="same", answer="second"),
            ],
        )


def test_conflict_wrapper_requires_issues_and_cannot_contain_owner_answers():
    conflict = IntentConflict.model_validate({"issues": ambiguous_payload()["issues"]})
    assert conflict.issues[0].id == "width_subject"
    with pytest.raises(ValidationError):
        IntentConflict(issues=[])
    with pytest.raises(ValidationError, match="Extra inputs"):
        IntentConflict.model_validate(
            {"issues": ambiguous_payload()["issues"], "answers": [{"answer": "sensor"}]}
        )


def test_no_subject_or_fact_count_cap_is_added():
    source = " ".join(f"Item {index} has width 45 mm." for index in range(150))
    attribution = IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": f"item_{index}",
                    "label": f"Item {index}",
                    "role": "reference",
                    "source_quote": f"Item {index}",
                }
                for index in range(150)
            ],
            "facts": [
                {
                    "id": f"width_{index}",
                    "subject_id": f"item_{index}",
                    "property": "width",
                    "value": 45,
                    "unit": "mm",
                    "source_quote": f"Item {index} has width 45 mm.",
                }
                for index in range(150)
            ],
        }
    )
    validate_attribution(attribution, contract(), source)
    assert len(attribution.subjects) == len(attribution.facts) == 150
