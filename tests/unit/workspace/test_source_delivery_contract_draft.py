"""Source delivery profile: actual schemas, authority and canonical equivalence."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture
from tests.unit.workspace.test_compact_source_contract_draft import (
    SCALARS,
    golden_motor,
)
from tests.unit.workspace.test_compact_source_contract_draft import (
    fixture_wire as grouped_compact_wire,
)
from tests.unit.workspace.test_intent_attribution import SOURCE, ambiguous_payload, contract
from tests.unit.workspace.test_intent_compilation import valid_draft
from tests.unit.workspace.test_source_contract_draft import deliveries

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.compact_source_contract_draft import (
    CompactSourceContractDraft,
    compact_source_proposal_record,
    lower_compact_source_draft,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import ContractDraft
from forge_mind.workspace.intent_compilation import compile_intent_draft
from forge_mind.workspace.models import Requirement
from forge_mind.workspace.source_contract_draft import compile_source_contract_draft
from forge_mind.workspace.source_delivery_contract_draft import (
    SourceDeliveryContractDraft,
    lower_source_delivery_draft,
    source_delivery_proposal_record,
)


def parse(wire):
    return SourceDeliveryContractDraft.model_validate(wire)


def expected_direct(draft, selected):
    """Declare prospective differences; do not rewrite an old record or claim exact replay."""
    expected = draft.model_copy(deep=True)
    for row in expected.contract.requirements:
        if row.id in selected:
            row.text = row.source_quote
            for name in SCALARS:
                setattr(row, name, Requirement.model_fields[name].default)
    return expected


def compile_wire(proposal, source, *, previous=None, guarded_ids=None):
    record = source_delivery_proposal_record(proposal, source)
    lowered = lower_source_delivery_draft(proposal, source)
    return lowered, compile_source_contract_draft(
        lowered,
        source,
        source_only_requirement_ids=(
            record["source_only_requirement_ids"] if guarded_ids is None else guarded_ids
        ),
        previous=previous,
    )


def ordinary_from_escape(wire, identifier, *, criterion_id="criterion"):
    """Explicit test authoring of an ordinary nondigital/scalar row."""
    entry = next(
        row for row in wire["canonical_requirements"] if row["requirement"]["id"] == identifier
    )
    wire["canonical_requirements"].remove(entry)
    row = entry["requirement"]
    reference = f"q_ordinary_{identifier}"
    wire["quotes"].append({"id": reference, "text": row["source_quote"]})
    criterion = row.get("verification")
    if criterion is not None:
        canonical = Requirement.model_validate(row)
        constants = {
            name
            for name, field in type(canonical.verification)
            .model_json_schema()["properties"]
            .items()
            if "const" in field
        }
        wire[criterion["kind"]].append(
            {
                "id": criterion_id,
                "quote_ref": reference,
                **{
                    name: value
                    for name, value in criterion.items()
                    if name not in constants | {"source_quote"}
                },
            }
        )
    wire["requirements"].append(
        {
            "id": identifier,
            "text": row["text"],
            "source": row["source"],
            "critical": row["critical"],
            "quote_ref": reference,
            "subject_ids": entry["subject_ids"],
            "verification_id": criterion_id if criterion is not None else None,
            "scalar": None if criterion is not None else {name: row[name] for name in SCALARS},
        }
    )


def test_prospective_trial003_counterpart_has_no_generated_preparation_prose():
    legacy, source = golden_motor()
    # The frozen motor request is unchanged across trials. Explicitly author the
    # valid body criterion; retain both historical body duties and all 22 facts.
    legacy.contract.requirements[1].verification = deepcopy(
        legacy.contract.requirements[2].verification
    )
    legacy.contract.requirements[-1].text = "Deliver a prepared print package for the plate."
    original = legacy.model_dump(mode="json")
    assert compile_intent_draft(legacy, source)["status"] == "invalid_interpretation"
    with pytest.raises(ValueError, match="explicitly"):
        source_delivery_wire_fixture(legacy)
    ids = [
        row.id
        for row in legacy.contract.requirements
        if row.verification is not None and row.verification.kind == "digital_delivery"
    ]
    proposal = parse(source_delivery_wire_fixture(legacy, digital_requirement_ids=ids))
    retained = source_delivery_proposal_record(proposal, source)
    assert "lowered_draft_hash" not in retained
    assert retained["source_only_requirement_ids"] == ids
    assert all(
        set(row) == {"id", "source", "critical", "quote_ref", "subject_ids", "deliverable"}
        for row in retained["raw_response"]["digital_requirements"]
    )
    lowered, result = compile_wire(proposal, source)
    assert result["status"] == "ready"
    assert lowered == expected_direct(legacy, ids)
    assert lowered.contract.requirements[-1].text == "a prepared print package"
    assert legacy.model_dump(mode="json") == original
    assert compile_intent_draft(legacy, source)["status"] == "invalid_interpretation"
    receipt = source_delivery_proposal_record(proposal, source, lowered)
    assert receipt == {
        "protocol": "source_delivery_contract_draft_v1",
        "raw_response": proposal.model_dump(mode="json"),
        "raw_response_hash": digest(proposal.model_dump(mode="json")),
        "source_hash": digest(source),
        "source_only_requirement_ids": ids,
        "lowered_draft_hash": digest(lowered.model_dump(mode="json")),
    }
    changed = lowered.model_copy(deep=True)
    changed.contract.requirements[-1].text += " changed"
    with pytest.raises(ValueError, match="does not lower"):
        source_delivery_proposal_record(proposal, source, changed)


def test_all_five_deliverables_lower_to_the_same_source_backed_canonical_family():
    draft, source = deliveries()
    source = source[:-1] + ", and bound instructions."
    draft.contract.requirements.append(
        Requirement.model_validate(
            {
                "id": "instructions",
                "text": "Deliver bound instructions",
                "source_quote": "bound instructions",
                "verification": {
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "coverage": "all_made",
                    "deliverable": "bound_instruction_plan",
                    "source_quote": "bound instructions",
                },
            }
        )
    )
    ids = [row.id for row in draft.contract.requirements]
    lowered, result = compile_wire(
        parse(source_delivery_wire_fixture(draft, digital_requirement_ids=ids)), source
    )
    assert result["status"] == "ready"
    assert lowered == expected_direct(draft, ids)
    assert {row.verification.deliverable for row in lowered.contract.requirements} == {
        "editable_native",
        "project_step",
        "project_stl",
        "prepared_print_package",
        "bound_instruction_plan",
    }


@pytest.mark.parametrize("field", ["critical", "criterion_version"])
def test_record_rejects_boolean_numeric_aliases_in_unvalidated_supplied_canonical_models(field):
    draft, source = deliveries()
    ids = [row.id for row in draft.contract.requirements]
    proposal = parse(source_delivery_wire_fixture(draft, digital_requirement_ids=ids))
    lowered = lower_source_delivery_draft(proposal, source)
    altered = lowered.model_copy(deep=True)
    if field == "critical":
        altered.contract.requirements[0].critical = 1
    else:
        altered.contract.requirements[0].verification.version = True
    # Assignment validation is intentionally absent on canonical records. A supplied
    # model can therefore contain aliases that Python dict equality treats as equal.
    assert altered.model_dump(mode="json") == lowered.model_dump(mode="json")
    with pytest.raises(ValueError, match="does not lower"):
        source_delivery_proposal_record(proposal, source, altered)


def test_original_22_fact_v1_history_and_new_profile_remain_separate():
    legacy, source = golden_motor()
    before = legacy.model_dump(mode="json")
    old_wire = grouped_compact_wire(legacy)
    old_proposal = CompactSourceContractDraft.model_validate(old_wire)
    assert lower_compact_source_draft(old_proposal, source).model_dump(mode="json") == before
    assert (
        compact_source_proposal_record(old_proposal, source)["protocol"]
        == "compact_source_contract_draft_v1"
    )
    ids = [
        row.id
        for row in legacy.contract.requirements
        if row.verification is not None and row.verification.kind == "digital_delivery"
    ]
    wire = source_delivery_wire_fixture(legacy, digital_requirement_ids=ids)
    # Use actual shared headers to exercise grouped expansion, not only single-entry groups.
    grouped = grouped_compact_wire(legacy)
    wire["quotes"] = grouped["quotes"]
    wire["subjects"] = grouped["subjects"]
    wire["fact_groups"] = grouped["fact_groups"]
    # Rebind direct rows explicitly after replacing this test-authored quote pool.
    quote_ids = {row["text"]: row["id"] for row in wire["quotes"]}
    for row in wire["digital_requirements"]:
        quote = next(
            item.source_quote for item in legacy.contract.requirements if item.id == row["id"]
        )
        if quote not in quote_ids:
            reference = f"q_direct_{row['id']}"
            wire["quotes"].append({"id": reference, "text": quote})
            quote_ids[quote] = reference
        row["quote_ref"] = quote_ids[quote]
    proposal = parse(wire)
    lowered = lower_source_delivery_draft(proposal, source)
    assert lowered == expected_direct(legacy, ids)
    assert len(lowered.attribution.facts) == 22
    assert len(lowered.contract.requirements) == 9
    assert [row.id for row in lowered.attribution.facts] == wire["fact_order"]
    assert max(len(group["values"]) for group in wire["fact_groups"]) >= 8
    assert legacy.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "attack",
    [
        "digital_pool",
        "digital_full_escape",
        "digital_text",
        "digital_scalar",
        "digital_ref",
        "null_ordinary_text",
    ],
)
def test_new_schema_cannot_express_full_text_digital_or_selector_bypasses(attack):
    draft, _ = deliveries()
    wire = source_delivery_wire_fixture(
        draft, digital_requirement_ids=[row.id for row in draft.contract.requirements]
    )
    if attack == "digital_pool":
        wire["digital_delivery"] = []
    elif attack == "digital_full_escape":
        row = wire["digital_requirements"].pop()
        wire["canonical_requirements"].append(
            {
                "requirement": draft.contract.requirements[-1].model_dump(mode="json"),
                "subject_ids": row["subject_ids"],
            }
        )
    elif attack == "null_ordinary_text":
        row = wire["digital_requirements"].pop()
        wire["requirements"].append(
            {
                "id": row["id"],
                "source": row["source"],
                "critical": row["critical"],
                "subject_ids": row["subject_ids"],
                "quote_ref": row["quote_ref"],
                "text": None,
                "verification_id": None,
                "scalar": None,
            }
        )
    else:
        name, value = {
            "digital_text": ("text", "Generated words"),
            "digital_scalar": ("scalar", {}),
            "digital_ref": ("verification_id", "v1"),
        }[attack]
        wire["digital_requirements"][0][name] = value
    with pytest.raises(ValidationError):
        parse(wire)


@pytest.mark.parametrize(
    "attack", ["quote", "source", "duplicate", "order", "ordinary_ref", "family_duplicate"]
)
def test_raw_records_precede_reference_and_global_order_refusals(attack):
    draft, source = valid_draft()
    wire = source_delivery_wire_fixture(draft, digital_requirement_ids=["native", "printing"])
    ordinary_from_escape(wire, draft.contract.requirements[0].id)
    if attack == "quote":
        wire["digital_requirements"][0]["quote_ref"] = "unknown"
    elif attack == "source":
        wire["quotes"][0]["text"] += " invented source"
    elif attack == "duplicate":
        wire["digital_requirements"][0]["id"] = wire["requirements"][0]["id"]
    elif attack == "order":
        wire["requirement_order"].pop()
    elif attack == "ordinary_ref":
        # A future internal ID must not turn an unknown ordinary reference into digital proof.
        wire["requirements"][0]["verification_id"] = "source_delivery_0"
    else:
        wire["nominal_delivery_scope"].append(
            {"id": "criterion", "quote_ref": wire["requirements"][0]["quote_ref"]}
        )
    proposal = parse(wire)
    raw = source_delivery_proposal_record(proposal, source)
    assert raw["raw_response"] == proposal.model_dump(mode="json")
    assert "lowered_draft_hash" not in raw
    with pytest.raises(ValueError):
        lower_source_delivery_draft(proposal, source)


def test_internal_digital_references_do_not_collide_with_authored_nondigital_identifiers():
    draft, source = valid_draft()
    wire = source_delivery_wire_fixture(draft, digital_requirement_ids=["native", "printing"])
    ordinary_from_escape(wire, draft.contract.requirements[0].id, criterion_id="source_delivery_0")
    proposal = parse(wire)
    before = proposal.model_dump(mode="json")
    lowered, result = compile_wire(proposal, source)
    assert result["status"] == "ready"
    assert lowered == expected_direct(draft, ["native", "printing"])
    assert proposal.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "suffix",
    [
        "for my printer",
        "with specified material",
        "including G-code",
        "waterproof",
        "and withstand 10 kg",
        "with a 5 mm bore",
    ],
)
def test_file_fragments_cannot_cover_additional_source_duties(suffix):
    draft, source = deliveries()
    source = source[:-1] + " " + suffix + "."
    proposal = parse(
        source_delivery_wire_fixture(
            draft, digital_requirement_ids=[row.id for row in draft.contract.requirements]
        )
    )
    _, result = compile_wire(proposal, source)
    assert result["status"] == "invalid_interpretation"
    assert any(row["code"] == "incomplete_digital_source_scope" for row in result["diagnostics"])


@pytest.mark.parametrize(
    "critical,source_flag", [(False, "request"), (True, "assumption"), (True, "explicit")]
)
def test_ineligible_digital_source_flags_still_fail_the_unchanged_scope_guard(
    critical, source_flag
):
    draft, source = deliveries()
    wire = source_delivery_wire_fixture(
        draft, digital_requirement_ids=[row.id for row in draft.contract.requirements]
    )
    wire["digital_requirements"][0].update(critical=critical, source=source_flag)
    proposal = parse(wire)
    assert (
        len(source_delivery_proposal_record(proposal, source)["source_only_requirement_ids"]) == 4
    )
    _, result = compile_wire(proposal, source)
    assert result["status"] == "invalid_interpretation"


def test_genuine_ambiguity_nondefault_scalars_and_mixed_duties_retain_full_fields():
    attrs = ambiguous_payload()
    frozen = contract()
    frozen.requirements.append(
        Requirement(
            id="selector",
            text="45 mm width",
            source_quote="45 mm",
            metric="hole_diameter",
            comparator="ge",
            value=45,
            tolerance=0.3,
            part_id="carrier",
            axis="x",
            center=(1, -2),
            position_tolerance=0.2,
            source="assumption",
            critical=False,
        )
    )
    draft = ContractDraft.model_validate({"contract": frozen, "attribution": attrs})
    wire = source_delivery_wire_fixture(draft)
    ordinary_from_escape(wire, "selector")
    lowered = lower_source_delivery_draft(parse(wire), SOURCE)
    assert lowered == draft
    assert lowered.attribution.issues == draft.attribution.issues


def test_source_only_membership_survives_canonical_correction_and_repeated_quote_refusal():
    draft, source = deliveries()
    ids = [row.id for row in draft.contract.requirements]
    proposal = parse(source_delivery_wire_fixture(draft, digital_requirement_ids=ids))
    original, result = compile_wire(proposal, source)
    assert result["status"] == "ready"
    guarded = tuple(
        source_delivery_proposal_record(proposal, source)["source_only_requirement_ids"]
    )
    changed = original.model_copy(deep=True)
    changed.contract.requirements[-1].text += " for the plate"
    result = compile_source_contract_draft(
        changed, source, previous=original, source_only_requirement_ids=guarded
    )
    assert result["status"] == "invalid_interpretation"
    assert any(row["code"] == "changed_source_duty" for row in result["diagnostics"])
    dropped = original.model_copy(deep=True)
    dropped.contract.requirements.pop()
    result = compile_source_contract_draft(
        dropped, source, previous=original, source_only_requirement_ids=guarded
    )
    assert any(row["code"] == "dropped_requirement" for row in result["diagnostics"])
    _, repeated = compile_wire(proposal, source + " STEP")
    assert repeated["status"] == "invalid_interpretation"
    assert source_delivery_proposal_record(proposal, source)["source_only_requirement_ids"] == list(
        guarded
    )


def test_actual_strict_schema_and_map_restoration_exclude_digital_escape_and_preserve_values():
    draft, source = valid_draft()
    wire = source_delivery_wire_fixture(draft, digital_requirement_ids=["native", "printing"])
    ordinary_from_escape(wire, draft.contract.requirements[0].id)
    source += " Unicode \u221230 mm and café are literal."
    wire["quotes"].append({"id": "unicode", "text": "Unicode \u221230 mm and café"})
    proposal = parse(wire)
    schema = SourceDeliveryContractDraft.model_json_schema()
    strict = _strict_schema(schema)
    assert "digital_delivery" not in strict["properties"]
    assert "DigitalDelivery" not in strict["$defs"]
    digital_ref = strict["properties"]["digital_requirements"]["items"]["$ref"].split("/")[-1]
    assert set(strict["$defs"][digital_ref]["properties"]) == {
        "id",
        "source",
        "critical",
        "quote_ref",
        "subject_ids",
        "deliverable",
    }
    ordinary_ref = strict["properties"]["requirements"]["items"]["$ref"].split("/")[-1]
    assert strict["$defs"][ordinary_ref]["properties"]["text"]["type"] == "string"

    def inspect(node):
        if isinstance(node, dict):
            assert "default" not in node and "prefixItems" not in node
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)

    inspect(strict)
    raw = proposal.model_dump(mode="json")
    restored = _restore_maps(raw, schema, schema)
    parsed = SourceDeliveryContractDraft.model_validate(restored)
    assert parsed.model_dump(mode="json") == raw
    assert lower_source_delivery_draft(parsed, source) == expected_direct(
        draft, ["native", "printing"]
    )
    with pytest.raises(ValueError, match="occur exactly"):
        lower_source_delivery_draft(parsed, source.replace("\u2212", "-"))
