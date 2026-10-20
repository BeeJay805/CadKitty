"""Compact wire equivalence, reference failures and actual provider conversion."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_intent_attribution import SOURCE, ambiguous_payload, contract
from tests.unit.workspace.test_intent_compilation import valid_draft, wrong_inventory_body_drafts

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.compact_source_contract_draft import (
    CompactSourceContractDraft,
    compact_source_proposal_record,
    lower_compact_source_draft,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft, IntentFact
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.source_contract_draft import (
    SourceContractDraft,
    compile_source_contract_draft,
    lower_source_contract_draft,
)

FAMILIES = (
    "nominal_through_hole_pattern",
    "nominal_perforated_box",
    "digital_delivery",
    "nominal_subject_inventory",
    "nominal_delivery_scope",
)
SCALARS = (
    "metric",
    "comparator",
    "value",
    "tolerance",
    "part_id",
    "axis",
    "center",
    "position_tolerance",
)


def fixture_wire(draft, *, compact_ids=(), source_only_ids=()):
    """Test-author encoding; full rows deliberately preserve invalid canonical quotes."""
    wire = CompactSourceContractDraft().model_dump(mode="json")
    quotes = {}

    def reference(text):
        if text not in quotes:
            quotes[text] = f"q_{len(quotes)}"
            wire["quotes"].append({"id": quotes[text], "text": text})
        return quotes[text]

    for row in draft.contract.requirements:
        raw = row.model_dump(mode="json")
        subjects = draft.attribution.requirement_subjects.get(row.id, [])
        wire["requirement_order"].append(row.id)
        if row.id not in compact_ids and row.id not in source_only_ids:
            wire["canonical_requirements"].append({"requirement": raw, "subject_ids": subjects})
            continue
        verification_id = None
        if row.verification is not None:
            criterion = row.verification.model_dump(mode="json")
            kind = criterion["kind"]
            fixed = {
                name
                for name, field in type(row.verification).model_json_schema()["properties"].items()
                if "const" in field
            }
            verification_id = f"v_{len(wire['requirements'])}"
            wire[kind].append(
                {
                    "id": verification_id,
                    "quote_ref": reference(criterion["source_quote"]),
                    **{
                        key: value
                        for key, value in criterion.items()
                        if key not in fixed | {"source_quote"}
                    },
                }
            )
        wire["requirements"].append(
            {
                "id": row.id,
                "text": None if row.id in source_only_ids else row.text,
                "source": row.source,
                "critical": row.critical,
                "quote_ref": reference(row.source_quote),
                "subject_ids": subjects,
                "verification_id": verification_id,
                "scalar": {key: raw[key] for key in SCALARS} if row.verification is None else None,
            }
        )
    for row in draft.attribution.subjects:
        raw = row.model_dump(mode="json")
        quote = raw.pop("source_quote")
        wire["subjects"].append({**raw, "quote_ref": reference(quote)})
    for row in draft.attribution.facts:
        raw = row.model_dump(mode="json")
        value = {key: raw.pop(key) for key in ("id", "property", "value")}
        quote = raw.pop("source_quote")
        header = {**raw, "quote_ref": reference(quote)}
        group = next(
            (
                item
                for item in wire["fact_groups"]
                if {key: value for key, value in item.items() if key != "values"} == header
            ),
            None,
        )
        if group is None:
            group = {**header, "values": []}
            wire["fact_groups"].append(group)
        group["values"].append(value)
        wire["fact_order"].append(row.id)
    wire["issues"] = [row.model_dump(mode="json") for row in draft.attribution.issues]
    wire["coverage_notes"] = list(draft.contract.coverage_notes)
    return wire


def parse(wire):
    return CompactSourceContractDraft.model_validate(wire)


def compile_wire(proposal, source, *, previous=None):
    record = compact_source_proposal_record(proposal, source)
    lowered = lower_compact_source_draft(proposal, source)
    return lowered, compile_source_contract_draft(
        lowered,
        source,
        source_only_requirement_ids=record["source_only_requirement_ids"],
        previous=previous,
    )


def golden_motor():
    payload = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("compact-source-motor003.json")
        .read_text(encoding="utf-8-sig")
    )
    return lower_source_contract_draft(
        SourceContractDraft.model_validate(payload["raw_draft"])
    ), payload["source"]


def test_compact_initial_draft_is_equivalent_and_retains_separate_raw_provenance():
    draft, source = valid_draft()
    wire = fixture_wire(
        draft,
        compact_ids=[row.id for row in draft.contract.requirements],
        source_only_ids=["native", "printing"],
    )
    before = deepcopy(wire)
    proposal = parse(wire)
    record = compact_source_proposal_record(proposal, source)
    assert "lowered_draft_hash" not in record
    lowered, result = compile_wire(proposal, source)
    assert lowered.model_dump(mode="json") == draft.model_dump(mode="json")
    assert result["status"] == "ready"
    retained = compact_source_proposal_record(proposal, source, lowered)
    assert retained == {
        "protocol": "compact_source_contract_draft_v1",
        "raw_response": proposal.model_dump(mode="json"),
        "raw_response_hash": digest(proposal.model_dump(mode="json")),
        "source_hash": digest(source),
        "source_only_requirement_ids": ["native", "printing"],
        "lowered_draft_hash": digest(lowered.model_dump(mode="json")),
    }
    assert wire == before
    changed = lowered.model_copy(deep=True)
    changed.contract.requirements[0].text += " changed"
    with pytest.raises(ValueError, match="does not lower"):
        compact_source_proposal_record(proposal, source, changed)
    with pytest.raises(ValueError, match="CompactSourceContractDraft"):
        lower_compact_source_draft(draft, source)


def test_retained_motor003_full_escape_preserves_all_22_facts_and_invalid_initial_family():
    draft, source = golden_motor()
    before = draft.model_dump(mode="json")
    assert len(draft.attribution.facts) == 22
    assert len(draft.contract.requirements) == 9
    wire = fixture_wire(draft)
    assert max(len(group["values"]) for group in wire["fact_groups"]) >= 8
    lowered = lower_compact_source_draft(parse(wire), source)
    assert lowered.model_dump(mode="json") == before
    assert compile_intent_draft(lowered, source) == compile_intent_draft(draft, source)
    assert compile_intent_draft(lowered, source)["status"] == "invalid_interpretation"
    assert [row.id for row in lowered.contract.requirements] == [
        row.id for row in draft.contract.requirements
    ]


def test_retained_22_facts_and_family_collections_share_only_declared_inapplicable_defaults():
    draft, source = golden_motor()
    expected = draft.model_copy(deep=True)
    compact_ids = []
    for row in expected.contract.requirements:
        if row.id == "r_geometry":
            continue  # Preserve the original mismatched parent/criterion quotes through full escape.
        compact_ids.append(row.id)
        if row.verification is not None:
            # This scalar selector is inactive for typed criteria. Actual criterion allowances stay .05.
            row.position_tolerance = Requirement.model_fields["position_tolerance"].default
    wire = fixture_wire(
        expected,
        compact_ids=compact_ids,
        source_only_ids=["r_native", "r_step", "r_stl", "r_print_package"],
    )
    before = deepcopy(wire)
    proposal = parse(wire)
    lowered = lower_compact_source_draft(proposal, source)
    assert lowered.model_dump(mode="json") == expected.model_dump(mode="json")
    assert len(lowered.attribution.facts) == 22
    assert len(lowered.contract.requirements) == 9
    holes = lowered.contract.requirements[2].verification.holes
    assert all(hole.diameter_tolerance_mm == 0.05 for hole in holes)
    assert all(hole.position_tolerance_mm == 0.05 for hole in holes)
    assert wire == before
    assert compact_source_proposal_record(proposal, source)["source_only_requirement_ids"] == [
        "r_native",
        "r_step",
        "r_stl",
        "r_print_package",
    ]


def test_full_escape_preserves_bad_embedded_quotes_for_unchanged_canonical_correction():
    draft, _, source = wrong_inventory_body_drafts()
    draft.contract.requirements[0].source_quote = "invalid parent quote for canonical diagnostics"
    wire = fixture_wire(draft)
    lowered = lower_compact_source_draft(parse(wire), source)
    assert lowered.model_dump(mode="json") == draft.model_dump(mode="json")
    assert compile_intent_draft(lowered, source) == compile_intent_draft(draft, source)


def test_nondefault_scalar_selection_and_assumption_flags_remain_explicit():
    source = "Use a 5 mm bore. Design assumption follows."
    requirement = Requirement(
        id="bore",
        text="Use a 5 mm bore",
        source_quote="5 mm bore",
        source="assumption",
        metric="hole_diameter",
        comparator="ge",
        value=5,
        tolerance=0.2,
        part_id="plate",
        axis="x",
        center=(10, -20),
        position_tolerance=0.3,
        critical=False,
    )
    draft = ContractDraft.model_validate(
        {"contract": {"requirements": [requirement]}, "attribution": {}}
    )
    proposal = parse(fixture_wire(draft, compact_ids=["bore"]))
    assert lower_compact_source_draft(proposal, source) == draft
    assert proposal.model_dump()["requirements"][0]["scalar"] == {
        name: getattr(requirement, name) for name in SCALARS
    }


def test_multiple_roles_genuine_issue_and_interleaved_fact_order_survive_grouping():
    attrs = ambiguous_payload()
    attrs["subjects"].append(
        {"id": "room", "label": "Room", "role": "environment", "source_quote": "room"}
    )
    attrs["facts"].append({**attrs["facts"][0], "id": "second_width"})
    attrs["issues"][0]["fact_ids"].append("second_width")
    draft = ContractDraft.model_validate({"contract": contract(), "attribution": attrs})
    source = SOURCE + " It goes in the room."
    wire = fixture_wire(draft)
    assert len(wire["fact_groups"]) == 2
    assert [row["id"] for row in wire["fact_groups"][0]["values"]] == [
        "sensor_width",
        "second_width",
    ]
    lowered = lower_compact_source_draft(parse(wire), source)
    assert lowered == draft
    assert [row.id for row in lowered.attribution.facts] == [
        "sensor_width",
        "screw_count",
        "second_width",
    ]
    assert compile_intent_draft(lowered, source)["status"] == "needs_customer_input"


def test_shared_criterion_does_not_merge_duplicate_duties_or_weaken_retention():
    draft, source = valid_draft()
    duplicate = draft.contract.requirements[0].model_copy(
        update={"id": "other_geometry"}, deep=True
    )
    draft.contract.requirements.append(duplicate)
    wire = fixture_wire(draft, compact_ids=[row.id for row in draft.contract.requirements])
    wire["requirements"][-1]["verification_id"] = wire["requirements"][0]["verification_id"]
    wire["nominal_through_hole_pattern"].pop()
    lowered = lower_compact_source_draft(parse(wire), source)
    assert lowered == draft
    dropped = lowered.model_copy(deep=True)
    dropped.contract.requirements.pop()
    diagnostics = compile_intent_draft(dropped, source, previous=lowered)["diagnostics"]
    assert any(row["code"] == "dropped_requirement" for row in diagnostics)


@pytest.mark.parametrize(
    "case",
    [
        "missing_quote",
        "altered_quote",
        "duplicate_quote",
        "missing_criterion",
        "duplicate_criterion",
        "unused_criterion",
        "parent_quote_mismatch",
        "duplicate_requirement",
        "missing_requirement_order",
        "duplicate_requirement_order",
        "duplicate_fact",
        "missing_fact_order",
        "duplicate_fact_order",
        "unknown_subject",
        "unknown_fact_requirement",
        "unknown_issue_fact",
        "both_scalar_and_criterion",
        "nondigital_null_text",
    ],
)
def test_invalid_wire_is_retained_before_reference_or_lowering_rejection(case):
    draft, source = valid_draft()
    if case in {
        "duplicate_fact",
        "missing_fact_order",
        "duplicate_fact_order",
        "unknown_fact_requirement",
    }:
        source += " Reference width is 5 mm."
        draft.attribution.facts.append(
            IntentFact(
                id="reference_width",
                subject_id=draft.attribution.subjects[0].id,
                property="reference width",
                value=5.0,
                unit="mm",
                source_quote="Reference width is 5 mm.",
                requirement_ids=[draft.contract.requirements[0].id],
            )
        )
    wire = fixture_wire(draft, compact_ids=[row.id for row in draft.contract.requirements])
    if case == "missing_quote":
        wire["requirements"][0]["quote_ref"] = "unknown"
    elif case == "altered_quote":
        wire["quotes"][0]["text"] += " invented source"
    elif case == "duplicate_quote":
        wire["quotes"].append(deepcopy(wire["quotes"][0]))
    elif case == "missing_criterion":
        wire["requirements"][0]["verification_id"] = "unknown"
    elif case == "duplicate_criterion":
        wire["digital_delivery"][0]["id"] = wire["nominal_through_hole_pattern"][0]["id"]
    elif case == "unused_criterion":
        wire["digital_delivery"].append({**wire["digital_delivery"][0], "id": "unused"})
    elif case == "parent_quote_mismatch":
        wire["requirements"][0]["quote_ref"] = wire["requirements"][1]["quote_ref"]
    elif case == "duplicate_requirement":
        wire["canonical_requirements"].append(
            {
                "requirement": draft.contract.requirements[0].model_dump(mode="json"),
                "subject_ids": [],
            }
        )
    elif case == "missing_requirement_order":
        wire["requirement_order"].pop()
    elif case == "duplicate_requirement_order":
        wire["requirement_order"].append(wire["requirement_order"][0])
    elif case == "duplicate_fact":
        wire["fact_groups"][0]["values"].append(deepcopy(wire["fact_groups"][0]["values"][0]))
    elif case == "missing_fact_order":
        wire["fact_order"].pop()
    elif case == "duplicate_fact_order":
        wire["fact_order"].append(wire["fact_order"][0])
    elif case == "unknown_subject":
        wire["requirements"][0]["subject_ids"] = ["unknown"]
    elif case == "unknown_fact_requirement":
        wire["fact_groups"][0]["requirement_ids"] = ["unknown"]
    elif case == "unknown_issue_fact":
        wire["issues"].append(
            {"id": "question", "question": "Which?", "reason": "Unknown", "fact_ids": ["unknown"]}
        )
    elif case == "both_scalar_and_criterion":
        wire["requirements"][0]["scalar"] = {
            key: Requirement.model_fields[key].default for key in SCALARS
        }
    elif case == "nondigital_null_text":
        wire["requirements"][0]["text"] = None
    proposal = parse(wire)
    record = compact_source_proposal_record(proposal, source)
    assert record["raw_response"] == proposal.model_dump(mode="json")
    assert "lowered_draft_hash" not in record
    with pytest.raises(ValueError):
        lower_compact_source_draft(proposal, source)


def test_missing_semantic_values_and_injected_fixed_literals_are_not_silently_filled():
    draft, source = valid_draft()
    wire = fixture_wire(draft, compact_ids=[row.id for row in draft.contract.requirements])
    for missing in ("axis", "subject_id", "holes"):
        altered = deepcopy(wire)
        del altered["nominal_through_hole_pattern"][0][missing]
        with pytest.raises(ValidationError):
            parse(altered)
    for name, value in (("kind", "digital_delivery"), ("scope", "digital_artifacts")):
        altered = deepcopy(wire)
        altered["nominal_through_hole_pattern"][0][name] = value
        with pytest.raises(ValidationError):
            parse(altered)
    altered = deepcopy(wire)
    altered["nominal_through_hole_pattern"][0]["holes"] = []
    with pytest.raises(ValidationError):
        parse(altered)
    altered = deepcopy(wire)
    altered["nominal_through_hole_pattern"][0]["holes"] *= 2
    proposal = parse(altered)
    compact_source_proposal_record(proposal, source)
    with pytest.raises(ValidationError, match="distinct"):
        lower_compact_source_draft(proposal, source)


def test_unicode_quotes_stay_literal_and_repeated_text_keeps_existing_source_scope_refusal():
    draft, source = valid_draft()
    source += " Unicode \u221230 mm, ±0.05 mm and café are literal."
    wire = fixture_wire(draft, compact_ids=[row.id for row in draft.contract.requirements])
    wire["quotes"].append({"id": "unicode", "text": "Unicode \u221230 mm, ±0.05 mm and café"})
    proposal = parse(wire)
    assert lower_compact_source_draft(proposal, source) == draft
    with pytest.raises(ValueError, match="occur exactly"):
        lower_compact_source_draft(proposal, source.replace("\u2212", "-"))
    wire = fixture_wire(draft, source_only_ids=["native", "printing"])
    proposal = parse(wire)
    _, result = compile_wire(proposal, source + " editable native CAD")
    assert result["status"] == "invalid_interpretation"
    assert compact_source_proposal_record(proposal, source)["source_only_requirement_ids"] == [
        "native",
        "printing",
    ]


def test_source_only_digital_does_not_cover_an_omitted_physical_conjunct():
    source = "Give me STL and withstand 10 kg."
    draft = ContractDraft.model_validate(
        {
            "contract": {
                "requirements": [
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
                    }
                ]
            },
            "attribution": {},
        }
    )
    proposal = parse(fixture_wire(draft, source_only_ids=["stl"]))
    _, result = compile_wire(proposal, source)
    assert result["status"] == "invalid_interpretation"
    draft.contract.requirements.append(
        Requirement(id="load", text="withstand 10 kg", source_quote="withstand 10 kg")
    )
    proposal = parse(fixture_wire(draft, source_only_ids=["stl"]))
    lowered, _ = compile_wire(proposal, source)
    assert lowered.contract.requirements[-1].critical
    assert lowered.contract.requirements[-1].verification is None
    assert lowered.contract.requirements[-1].metric == "unmeasured"


@pytest.mark.parametrize("case", ["motor", "ambiguity", "scalar", "through_pattern"])
def test_actual_provider_strict_schema_and_restoration_preserve_wire_and_canonical_values(case):
    if case == "motor":
        draft, source = golden_motor()
        compact_ids = []
        for row in draft.contract.requirements:
            if row.id != "r_geometry":
                compact_ids.append(row.id)
                if row.verification is not None:
                    row.position_tolerance = 0.1
    elif case == "ambiguity":
        draft = ContractDraft.model_validate(
            {"contract": contract(), "attribution": ambiguous_payload()}
        )
        source, compact_ids = SOURCE, []
    elif case == "scalar":
        source, compact_ids = "A 5 mm bore", ["bore"]
        draft = ContractDraft.model_validate(
            {
                "contract": {
                    "requirements": [
                        {
                            "id": "bore",
                            "text": source,
                            "source_quote": source,
                            "metric": "hole_diameter",
                            "value": 5,
                            "axis": "y",
                            "center": [2, -4],
                        }
                    ]
                },
                "attribution": {},
            }
        )
    else:
        draft, source = valid_draft()
        compact_ids = [row.id for row in draft.contract.requirements]
    proposal = parse(fixture_wire(draft, compact_ids=compact_ids))
    original_schema = CompactSourceContractDraft.model_json_schema()
    strict = _strict_schema(original_schema)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "default" not in node
            assert "prefixItems" not in node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(strict)
    for family in FAMILIES:
        ref = strict["properties"][family]["items"]["$ref"].split("/")[-1]
        family_schema = strict["$defs"][ref]
        assert {"id", "quote_ref"} <= family_schema["properties"].keys()
        assert not {"kind", "source_quote", "version", "scope"} & family_schema["properties"].keys()
    raw = proposal.model_dump(mode="json")
    restored = _restore_maps(raw, original_schema, original_schema)
    parsed = CompactSourceContractDraft.model_validate(restored)
    assert parsed.model_dump(mode="json") == raw
    assert lower_compact_source_draft(parsed, source).model_dump(mode="json") == draft.model_dump(
        mode="json"
    )
