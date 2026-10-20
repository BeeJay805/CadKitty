"""Prospective source-only delivery rows do not rewrite historical authority."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_intent_compilation import valid_draft
from tests.unit.workspace.test_verification_admission import box_contract

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.source_contract_draft import (
    SourceContractDraft,
    lower_source_contract_draft,
    source_proposal_record,
)
from forge_mind.workspace.source_contract_draft import (
    compile_source_contract_draft as _compile_source_contract_draft,
)
from forge_mind.workspace.source_contract_draft import (
    source_scope_diagnostics as _source_scope_diagnostics,
)


def compile_source_contract_draft(draft, source, **kwargs):
    return _compile_source_contract_draft(
        draft,
        source,
        source_only_requirement_ids=[row.id for row in draft.contract.requirements],
        **kwargs,
    )


def source_scope_diagnostics(draft, source):
    return _source_scope_diagnostics(
        draft, source, source_only_requirement_ids=[row.id for row in draft.contract.requirements]
    )


def source_wire(draft):
    """Explicitly author the new fixture shape; production accepts no old-wire fallback."""
    raw = draft.model_dump(mode="json")
    for row in raw["contract"]["requirements"]:
        if row.get("verification", {}).get("kind") == "digital_delivery":
            del row["text"]
    return raw


def deliveries():
    source = "Give me editable native CAD, STEP, STL and a prepared print package."
    rows = [
        {
            "id": identifier,
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
        for identifier, quote, deliverable in (
            ("native", "editable native CAD", "editable_native"),
            ("step", "STEP", "project_step"),
            ("stl", "STL", "project_stl"),
            ("printing", "a prepared print package", "prepared_print_package"),
        )
    ]
    return ContractDraft.model_validate(
        {"contract": {"requirements": rows}, "attribution": {}}
    ), source


def test_digital_lowering_copies_only_exact_source_text_and_retains_original_wire():
    draft, source = valid_draft()
    raw = source_wire(draft)
    before = deepcopy(raw)
    proposal = SourceContractDraft.model_validate(raw)
    lowered = lower_source_contract_draft(proposal)
    assert compile_source_contract_draft(lowered, source)["status"] == "ready"
    assert lowered.model_dump(mode="json") == draft.model_dump(mode="json")
    assert raw == before
    record = source_proposal_record(proposal, lowered)
    assert record == {
        "protocol": "source_contract_draft_v1",
        "raw_response": proposal.model_dump(mode="json"),
        "raw_response_hash": digest(proposal.model_dump(mode="json")),
        "source_only_requirement_ids": ["native", "printing"],
        "lowered_draft_hash": digest(lowered.model_dump(mode="json")),
    }
    assert all("text" not in row for row in record["raw_response"]["contract"]["requirements"][1:])
    assert record["raw_response"]["contract"]["requirements"][0] == draft.contract.requirements[
        0
    ].model_dump(mode="json")


def test_original_source003_generated_paraphrase_remains_invalid_on_legacy_path():
    draft, source = deliveries()
    draft.contract.requirements[
        -1
    ].text = (
        "Deliver a prepared print package covering all made parts and their occurrence quantities."
    )
    original = draft.model_dump(mode="json")
    assert compile_intent_draft(draft, source)["status"] == "invalid_interpretation"
    proposal = SourceContractDraft.model_validate(source_wire(draft))
    lowered = lower_source_contract_draft(proposal)
    assert lowered.contract.requirements[-1].text == "a prepared print package"
    assert compile_source_contract_draft(lowered, source)["status"] == "ready"
    assert draft.model_dump(mode="json") == original
    with pytest.raises(ValueError, match="SourceContractDraft"):
        lower_source_contract_draft(draft)


def test_wire_requires_explicit_full_text_branch_and_non_digital_full_fields():
    draft, _ = valid_draft()
    raw = source_wire(draft)
    raw["contract"]["requirements"][-1]["text"] = "authored extra scope"
    parsed = SourceContractDraft.model_validate(raw)
    assert (
        lower_source_contract_draft(parsed).contract.requirements[-1].text == "authored extra scope"
    )
    assert source_proposal_record(parsed)["source_only_requirement_ids"] == ["native"]
    raw = source_wire(draft)
    del raw["contract"]["requirements"][0]["text"]
    with pytest.raises(ValidationError):
        SourceContractDraft.model_validate(raw)


def test_mixed_geometry_and_file_sentence_can_explicitly_use_unchanged_full_text():
    source = "Make one 80 mm wide plate and give me STL."
    draft = ContractDraft.model_validate(
        {
            "contract": {
                "requirements": [
                    {
                        "id": "width",
                        "text": "80 mm wide",
                        "source_quote": "80 mm wide",
                        "metric": "extent_x",
                        "value": 80,
                    },
                    {
                        "id": "stl",
                        "text": "STL",
                        "source_quote": "STL",
                        "verification": {
                            "kind": "digital_delivery",
                            "scope": "digital_artifacts",
                            "deliverable": "project_stl",
                            "coverage": "all_made",
                            "source_quote": "STL",
                        },
                    },
                ]
            },
            "attribution": {},
        }
    )
    raw = draft.model_dump(mode="json")
    proposal = SourceContractDraft.model_validate(raw)
    assert source_proposal_record(proposal)["source_only_requirement_ids"] == []
    lowered = lower_source_contract_draft(proposal)
    assert lowered.model_dump(mode="json") == raw
    assert (
        _compile_source_contract_draft(lowered, source, source_only_requirement_ids=[])["status"]
        == "ready"
    )
    assert not any(
        row.verification is None and row.metric == "unmeasured"
        for row in lowered.contract.requirements
    )
    # The same quote cannot silently move out of source-only guarding once selected.
    assert (
        _compile_source_contract_draft(lowered, source, source_only_requirement_ids=["stl"])[
            "status"
        ]
        == "invalid_interpretation"
    )


def test_raw_source_proposal_can_be_retained_before_invalid_canonical_lowering():
    draft, _ = deliveries()
    raw = source_wire(draft)
    raw["contract"]["requirements"][0].update(metric="extent_x", value=80)
    proposal = SourceContractDraft.model_validate(raw)
    receipt = source_proposal_record(proposal)
    assert receipt["raw_response"]["contract"]["requirements"][0]["value"] == 80
    assert "lowered_draft_hash" not in receipt
    with pytest.raises(ValidationError, match="cannot mix scalar"):
        lower_source_contract_draft(proposal)


def test_entire_file_list_requires_each_critical_peer_and_retains_canonical_targets():
    draft, source = deliveries()
    assert compile_source_contract_draft(draft, source)["status"] == "ready"
    before = draft.model_dump(mode="json")
    missing = draft.model_copy(deep=True)
    missing.contract.requirements = [
        row for row in missing.contract.requirements if row.id != "stl"
    ]
    assert compile_intent_draft(missing, source)["status"] == "ready"
    assert source_scope_diagnostics(missing, source)
    assert draft.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "suffix",
    ["for my Ender 3", "including G-code", "that withstands heat", "waterproof", "for GizmoQ97"],
)
@pytest.mark.parametrize("separator", [" ", "\n", "; "])
def test_unknown_source_suffix_never_becomes_neutral_list_glue(suffix, separator):
    draft, source = deliveries()
    source = source[:-1] + separator + suffix + "."
    assert compile_source_contract_draft(draft, source)["status"] == "invalid_interpretation"


def test_physical_adjective_inside_quote_is_not_an_artifact_only_clause():
    draft, source = deliveries()
    row = draft.contract.requirements[-1]
    quote = "a waterproof prepared print package"
    row.text = row.source_quote = row.verification.source_quote = quote
    source = source.replace("a prepared print package", quote)
    assert source_scope_diagnostics(draft, source)


def test_only_retained_critical_unknown_can_cover_actual_extra_source_duty():
    draft, source = deliveries()
    source = source[:-1] + " for my Ender 3."
    unknown = Requirement(
        id="machine", text="Retain the machine-specific duty", source_quote="for my Ender 3"
    )
    draft.contract.requirements.append(unknown)
    assert compile_source_contract_draft(draft, source)["status"] == "ready"
    assert unknown.critical and unknown.metric == "unmeasured" and unknown.verification is None
    for fields in (
        {"critical": False},
        {"source": "assumption"},
        {"metric": "solid_count", "value": 1.0},
    ):
        changed = draft.model_copy(deep=True)
        changed.contract.requirements[-1] = Requirement.model_validate(
            {**unknown.model_dump(), **fields}
        )
        assert source_scope_diagnostics(changed, source)


def test_scalar_peer_cannot_mask_machine_scope_even_with_a_literal_count():
    draft, source = deliveries()
    source = source[:-1] + " for my Ender 3, and one solid."
    draft.contract.requirements.append(
        Requirement(
            id="solid",
            text="One solid",
            source_quote="for my Ender 3, and one solid",
            metric="solid_count",
            value=1,
        )
    )
    assert source_scope_diagnostics(draft, source)


def test_keyword_admitted_stl_peer_cannot_launder_machine_scope():
    draft, source = deliveries()
    source = source.replace("STL", "STL for my Ender 3")
    stl = next(row for row in draft.contract.requirements if row.id == "stl")
    stl.text = stl.source_quote = stl.verification.source_quote = "STL for my Ender 3"
    assert compile_intent_draft(draft, source)["status"] == "ready"
    errors = source_scope_diagnostics(draft, source)
    assert {"printing", "stl"} <= {row["requirement_id"] for row in errors}


def test_duplicate_quote_occurrences_refuse_ambiguous_source_binding():
    draft, source = deliveries()
    assert source_scope_diagnostics(draft, source + " Give me STL.")


def test_source_coverage_rechecks_corrected_rows_without_changing_retention():
    draft, source = deliveries()
    source = source[:-1] + " for my Ender 3."
    draft.contract.requirements.append(
        Requirement(id="machine", text="Machine duty", source_quote="for my Ender 3")
    )
    assert compile_source_contract_draft(draft, source)["status"] == "ready"
    corrected = draft.model_copy(deep=True)
    corrected.contract.requirements[-1] = Requirement(
        id="machine",
        text="Machine duty",
        source_quote="for my Ender 3",
        metric="solid_count",
        value=1,
    )
    result = compile_source_contract_draft(corrected, source, previous=draft)
    assert result["status"] == "invalid_interpretation"
    assert any(row["code"] == "incomplete_digital_source_scope" for row in result["diagnostics"])


def test_canonical_history_and_hashes_remain_unchanged_after_new_path_inspection():
    draft, source = valid_draft()
    before = draft.model_dump(mode="json")
    old_hash = draft.contract.fingerprint()
    source_scope_diagnostics(draft, source)
    lower_source_contract_draft(SourceContractDraft.model_validate(source_wire(draft)))
    assert before == draft.model_dump(mode="json")
    assert old_hash == draft.contract.fingerprint()


def test_actual_codex_schema_roundtrip_keeps_both_wire_branches_all_families_and_map():
    draft, _ = valid_draft()
    raw = source_wire(draft)
    # Explicitly retain a full-text digital peer alongside the source-only peer.
    raw["contract"]["requirements"][1] = draft.contract.requirements[1].model_dump(mode="json")
    box = box_contract().requirements[0].model_dump(mode="json")
    raw["contract"]["requirements"].extend(
        [
            box,
            {
                "id": "count",
                "text": "one solid",
                "source_quote": "one solid",
                "metric": "solid_count",
                "value": 1,
            },
            {"id": "unknown", "text": "Retain unknown duty", "source_quote": "unknown duty"},
            {
                "id": "inventory",
                "text": "one part named plate",
                "source_quote": "one part named plate",
                "verification": {
                    "kind": "nominal_subject_inventory",
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
                    "source_quote": "one part named plate",
                },
            },
            {
                "id": "scope",
                "text": "nominal CAD and digital files only",
                "source_quote": "nominal CAD and digital files only",
                "verification": {
                    "kind": "nominal_delivery_scope",
                    "scope": "software_policy",
                    "policy": "nominal_cad_and_digital_only",
                    "source_quote": "nominal CAD and digital files only",
                },
            },
        ]
    )
    raw["attribution"]["requirement_subjects"] = {
        "pattern": ["plate"],
        "body": ["plate"],
        "inventory": ["plate"],
    }
    raw["attribution"]["issues"] = [
        {
            "id": "unknown_input",
            "question": "What is the missing source value?",
            "reason": "Preserve uncertainty",
            "requirement_ids": ["unknown"],
        }
    ]
    proposal = SourceContractDraft.model_validate(raw)
    raw = proposal.model_dump(mode="json")
    schema = SourceContractDraft.model_json_schema()
    strict = _strict_schema(schema)
    assert (
        strict["$defs"]["IntentAttribution"]["properties"]["requirement_subjects"]["type"]
        == "array"
    )
    assert set(strict["$defs"]["SourceDigitalRequirement"]["properties"]) == set(
        Requirement.model_fields
    ) - {"text"}
    assert "text" in strict["$defs"]["Requirement"]["required"]
    wire = deepcopy(raw)
    wire["attribution"]["requirement_subjects"] = [
        {"key": key, "value": value}
        for key, value in raw["attribution"]["requirement_subjects"].items()
    ]
    # The actual strict transport returns null for optional fields; canonical
    # serializers deliberately omit these two legacy absence fields again.
    for row in wire["contract"]["requirements"]:
        if "text" in row and "verification" not in row:
            row["verification"] = None
    wire["attribution"]["issues"][0]["input_dependency"] = None
    restored = _restore_maps(wire, schema, schema)
    decoded = SourceContractDraft.model_validate(restored)
    assert decoded.model_dump(mode="json") == raw
    assert source_proposal_record(decoded)["source_only_requirement_ids"] == ["printing"]
    canonical = lower_source_contract_draft(decoded).model_dump(mode="json")
    expected = deepcopy(raw)
    for row in expected["contract"]["requirements"]:
        if "text" not in row:
            row["text"] = row["source_quote"]
    assert canonical == ContractDraft.model_validate(expected).model_dump(mode="json")
