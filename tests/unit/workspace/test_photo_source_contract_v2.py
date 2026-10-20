"""Source-only photo duties retain source authority without authored display prose."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from forge_mind.workspace.document import digest
from forge_mind.workspace.photo_nominal import validate_photo_source_replay
from forge_mind.workspace.source_delivery_contract_draft import (
    PhotoSourceContractDraftV2,
    PhotoSourceDeliveryContractDraft,
    lower_source_delivery_draft,
    preflight_source_delivery_draft,
)

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "photo_source_actual002.json").read_text(
        encoding="utf-8"
    )
)
SOURCE = FIXTURE["source"]


def source_wire(raw=None):
    old = deepcopy(raw or FIXTURE["raw_attempts"][0])
    claims = {row["requirement_id"]: row for row in old.pop("photo_nominal_claims")}
    old["photo_requirements"] = [
        {
            **{key: row[key] for key in ("id", "source", "critical", "quote_ref", "subject_ids")},
            "facets": claims[row["id"]]["claims"],
            "size_fact_ids": claims[row["id"]]["size_fact_ids"],
            "hole_count_fact_id": claims[row["id"]]["hole_count_fact_id"],
            "frame_policy": claims[row["id"]]["frame_policy"],
        }
        for row in old.pop("requirements")
    ]
    return PhotoSourceContractDraftV2.model_validate(old)


def document(preflight):
    return {
        "intent_compilation": {
            **preflight.compilation,
            "job_id": "job",
            "source_hash": digest(SOURCE),
            "source_proposal": preflight.source_proposal,
        }
    }


def test_retained_actual002_prose_still_rejects_both_historical_v1_attempts():
    for raw in FIXTURE["raw_attempts"]:
        original = PhotoSourceDeliveryContractDraft.model_validate(raw)
        rejected = preflight_source_delivery_draft(original, SOURCE)
        assert rejected.compilation["status"] == "invalid_interpretation"
        assert "Unrepresented source clause" in rejected.compilation["diagnostics"][0]["message"]
        assert rejected.source_proposal["raw_response"] == raw


def test_source_only_actual002_preserves_seven_ids_quotes_flags_facts_and_one_replay():
    proposal = source_wire()
    preflight = preflight_source_delivery_draft(proposal, SOURCE)
    assert preflight.compilation["status"] == "ready", preflight.compilation
    assert preflight.source_proposal["protocol"] == "photo_source_contract_draft_v2"
    assert preflight.source_proposal["source_only_photo_requirement_ids"] == [
        f"r{i}" for i in range(1, 8)
    ]
    lowered = preflight.draft
    assert lowered is not None
    quotes = {quote["id"]: quote["text"] for quote in FIXTURE["raw_attempts"][0]["quotes"]}
    for row, original in zip(
        lowered.contract.requirements, FIXTURE["raw_attempts"][0]["requirements"], strict=True
    ):
        assert (row.id, row.source, row.critical, row.source_quote) == (
            original["id"],
            original["source"],
            original["critical"],
            quotes[original["quote_ref"]],
        )
        assert row.verification.source_text_policy == "source_only_v2"
    assert [fact.id for fact in lowered.attribution.facts] == ["f1", "f2", "f3", "f4"]
    assert [fact.value for fact in lowered.attribution.facts] == [50.0, 30.0, 5.0, 2.0]
    assert lowered.contract.requirements[0].text == "Copy the visible profile."
    validate_photo_source_replay(
        document(preflight), "job", SOURCE, lowered.contract, lowered.attribution
    )


@pytest.mark.parametrize(
    "field,value",
    [("text", "Invented prose"), ("scalar", None), ("unsupported_reason", "Ignore this")],
)
def test_source_only_row_refuses_parallel_authored_acceptance_fields(field, value):
    raw = source_wire().model_dump(mode="json")
    raw["photo_requirements"][0][field] = value
    with pytest.raises(ValueError):
        PhotoSourceContractDraftV2.model_validate(raw)


@pytest.mark.parametrize(
    "addition",
    [
        " and withstand 100 N",
        " and apply a blue coating",
        " and add counterbores",
        " and add hidden recesses",
    ],
)
def test_truncated_physical_conjunct_cannot_be_covered_by_an_unsupported_label(addition):
    proposed = source_wire().model_dump(mode="json")
    next(quote for quote in proposed["quotes"] if quote["id"] == "q1")["text"] = (
        "Copy this 50 by 30 mm rectangular plate, 5 mm thick"
    )
    source = SOURCE.replace("thick.", "thick" + addition + ".")
    proposed["photo_reference"]["source_quote"] = source
    proposal = PhotoSourceContractDraftV2.model_validate(proposed)
    bad = preflight_source_delivery_draft(proposal, source)
    assert bad.compilation["status"] == "invalid_interpretation"
    assert bad.compilation["diagnostics"][0]["field"] == "photo_requirements.r1.quote_ref"
    raw = proposal.model_dump(mode="json")
    raw["quotes"].append({"id": "extra", "text": addition.strip()})
    raw["requirements"].append(
        {"id": "physical", "text": addition.strip(), "quote_ref": "extra", "critical": True}
    )
    raw["requirement_order"].append("physical")
    assert (
        preflight_source_delivery_draft(
            PhotoSourceContractDraftV2.model_validate(raw), source
        ).compilation["status"]
        == "invalid_interpretation"
    )


def test_separate_physical_clause_remains_critical_unverified():
    raw = source_wire().model_dump(mode="json")
    source = SOURCE + " It must withstand 100 N."
    raw["quotes"].append({"id": "physical", "text": "It must withstand 100 N."})
    raw["requirements"].append(
        {
            "id": "physical",
            "text": "It must withstand 100 N.",
            "quote_ref": "physical",
            "critical": True,
        }
    )
    raw["requirement_order"].append("physical")
    value = preflight_source_delivery_draft(PhotoSourceContractDraftV2.model_validate(raw), source)
    assert value.compilation["status"] == "ready", value.compilation
    physical = value.draft.contract.requirements[-1]
    assert physical.critical and physical.metric == "unmeasured" and physical.verification is None


def test_repeated_source_occurrence_refuses_ambiguous_quote():
    value = preflight_source_delivery_draft(source_wire(), SOURCE + " " + SOURCE)
    assert value.compilation["status"] == "invalid_interpretation"
    assert "unique" in value.compilation["diagnostics"][0]["message"]


@pytest.mark.parametrize("extra", ["It must withstand 100 N.", "Add a blind pocket underneath."])
def test_unchanged_seven_rows_cannot_omit_an_extra_complete_owner_clause(extra):
    rejected = preflight_source_delivery_draft(source_wire(), SOURCE + " " + extra)
    assert rejected.compilation["status"] == "invalid_interpretation"
    diagnostic = rejected.compilation["diagnostics"][0]
    assert diagnostic["field"] == "photo_requirements.*.source_coverage"
    assert extra in diagnostic["message"]


@pytest.mark.parametrize(
    "substitute", ["note", "subject_label", "noncritical", "fragment", "residual_label"]
)
def test_source_metadata_or_residual_label_cannot_cover_an_omitted_clause(substitute):
    extra = "It must withstand 100 N."
    source = SOURCE + " " + extra
    raw = source_wire().model_dump(mode="json")
    if substitute == "note":
        raw["coverage_notes"].append(extra)
    elif substitute == "subject_label":
        raw["subjects"][0]["label"] += " " + extra
    else:
        quote = "100 N" if substitute == "fragment" else extra
        raw["quotes"].append({"id": "extra", "text": quote})
        raw["requirements"].append(
            {
                "id": "extra",
                "quote_ref": "extra",
                "text": "Unverified residual duty" if substitute == "residual_label" else quote,
                "critical": substitute != "noncritical",
            }
        )
        raw["requirement_order"].append("extra")
    rejected = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(raw), source
    )
    assert rejected.compilation["status"] == "invalid_interpretation"
    assert any(
        row["field"] == "photo_requirements.*.source_coverage"
        for row in rejected.compilation["diagnostics"]
    )


def test_fake_context_directive_cannot_exempt_a_copy_source_or_its_predicates():
    source = SOURCE + " Use this photo only as context."
    raw = source_wire().model_dump(mode="json")
    raw["photo_reference"] = {"role": "context", "source_quote": "Use this photo only as context."}
    rejected = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(raw), source
    )
    assert rejected.compilation["status"] == "invalid_interpretation"
    assert "copy clause cannot be skipped" in rejected.compilation["diagnostics"][0]["message"]


def test_facet_cannot_move_to_a_different_quote_or_disappear():
    raw = source_wire().model_dump(mode="json")
    raw["photo_requirements"][0]["quote_ref"] = "q2"
    value = preflight_source_delivery_draft(PhotoSourceContractDraftV2.model_validate(raw), SOURCE)
    assert value.compilation["status"] == "invalid_interpretation"
    raw = source_wire().model_dump(mode="json")
    raw["photo_requirements"][1]["facets"] = ["visible_profile"]
    value = preflight_source_delivery_draft(PhotoSourceContractDraftV2.model_validate(raw), SOURCE)
    assert value.compilation["status"] == "invalid_interpretation"


def test_invalid_binding_can_correct_without_freezing_provisional_display():
    raw = source_wire().model_dump(mode="json")
    raw["photo_requirements"][-1]["size_fact_ids"][0] = "missing_fact"
    previous = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(raw), SOURCE
    )
    assert previous.compilation["status"] == "invalid_interpretation"
    assert previous.compilation["diagnostics"][0]["field"] == "photo_requirements.r7.size_fact_ids"
    fixed = preflight_source_delivery_draft(source_wire(), SOURCE, previous=previous)
    assert fixed.compilation["status"] == "ready", fixed.compilation
    swapped = source_wire().model_dump(mode="json")
    swapped["photo_requirements"][0]["facets"], swapped["photo_requirements"][1]["facets"] = (
        swapped["photo_requirements"][1]["facets"],
        swapped["photo_requirements"][0]["facets"],
    )
    bad = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(swapped), SOURCE, previous=previous
    )
    assert bad.compilation["status"] == "invalid_interpretation"
    assert any(row["code"] == "changed_source_proposal" for row in bad.compilation["diagnostics"])


@pytest.mark.parametrize(
    "mutation",
    ["display", "missing_record", "wrong_job", "v1_protocol", "raw_forgery", "removed", "demoted"],
)
def test_current_replay_refuses_forged_marker_or_changed_source_authority(mutation):
    value = preflight_source_delivery_draft(source_wire(), SOURCE)
    lowered = value.draft.model_copy(deep=True)
    record = deepcopy(document(value))
    if mutation == "display":
        lowered.contract.requirements[0].text += " and coat it blue"
    elif mutation == "missing_record":
        record = {}
    elif mutation == "wrong_job":
        record["intent_compilation"]["job_id"] = "other"
    elif mutation == "v1_protocol":
        record["intent_compilation"]["source_proposal"]["protocol"] = (
            "source_delivery_contract_draft_v1"
        )
    elif mutation == "raw_forgery":
        record["intent_compilation"]["source_proposal"]["raw_response"]["photo_requirements"][0][
            "facets"
        ] = ["thickness"]
    elif mutation == "removed":
        lowered.contract.requirements.pop()
    else:
        lowered.contract.requirements[0].critical = False
    with pytest.raises(ValueError):
        validate_photo_source_replay(record, "job", SOURCE, lowered.contract, lowered.attribution)


def test_legacy_criterion_serialization_has_no_new_policy_field():
    from forge_mind.workspace.verification_models import NominalPhotoSolid

    criterion = (
        lower_source_delivery_draft(source_wire(), SOURCE).contract.requirements[0].verification
    )
    raw = criterion.model_dump(mode="json")
    raw.pop("source_text_policy")
    old = NominalPhotoSolid.model_validate(raw)
    assert old.model_dump(mode="json") == raw
    assert digest(old.model_dump(mode="json")) == digest(raw)


def test_forged_policy_cannot_use_a_valid_legacy_raw_record():
    prospective = preflight_source_delivery_draft(source_wire(), SOURCE)
    raw = deepcopy(FIXTURE["raw_attempts"][0])
    for row, lowered in zip(
        raw["requirements"], prospective.draft.contract.requirements, strict=True
    ):
        row["text"] = lowered.text
    legacy = preflight_source_delivery_draft(
        PhotoSourceDeliveryContractDraft.model_validate(raw), SOURCE
    )
    assert legacy.compilation["status"] == "ready", legacy.compilation
    forged = legacy.draft.model_copy(deep=True)
    forged.contract.requirements[0].verification.source_text_policy = "source_only_v2"
    with pytest.raises(ValueError):
        validate_photo_source_replay(
            document(legacy), "job", SOURCE, forged.contract, forged.attribution
        )
