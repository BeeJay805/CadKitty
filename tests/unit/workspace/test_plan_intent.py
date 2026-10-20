"""Pure startup lowering controls; no model, geometry or worker runs."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.operation_plan import coverage, snapshot
from forge_mind.workspace.plan_intent import IntentPart, PlanIntent, lower_intent


def frozen_contract() -> Contract:
    return Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "width",
                    "text": "The main part is 20 mm wide",
                    "source_quote": "20 mm wide",
                    "metric": "extent_x",
                    "value": 20,
                    "part_id": "Main",
                },
                {
                    "id": "connection",
                    "text": "The two parts connect",
                    "source_quote": "two parts connect",
                },
                {
                    "id": "finish",
                    "text": "The completed design has the requested finish",
                    "source_quote": "requested finish",
                },
            ]
        }
    )


def intent_payload() -> dict:
    return {
        "objective": "Construct the requested assembly",
        "parts": [
            {
                "id": "Main",
                "role": "make",
                "purpose": "Support the connection",
                "requirement_ids": ["width", "connection"],
            },
            {
                "id": "standard_member",
                "role": "buy",
                "purpose": "Provide the mating member",
                "requirement_ids": ["connection"],
            },
            {"id": "Envelope", "role": "reference", "purpose": "Locate the installation space"},
        ],
        "first_part_id": "Main",
        "unresolved": ["Keep this exact question?", "  Keep surrounding spaces  "],
    }


def test_lowered_intent_equals_a_manually_authored_canonical_plan_and_preserves_inputs():
    payload = intent_payload()
    before_payload = deepcopy(payload)
    intent = PlanIntent.model_validate(payload)
    contract = frozen_contract()
    before_intent, before_contract = intent.model_dump(), contract.model_dump()
    expected = DesignPlan.model_validate(
        {
            "objective": "Construct the requested assembly",
            "parts": [
                "Main (make): Support the connection",
                "standard_member (buy): Provide the mating member",
                "Envelope (reference): Locate the installation space",
            ],
            "unresolved": payload["unresolved"],
            "operation_plan": {
                "part_ids": ["Main", "standard_member", "Envelope"],
                "steps": [
                    {
                        "id": "Main",
                        "part_id": "Main",
                        "expected_effect": "Main (make): Support the connection",
                        "requirement_ids": ["width", "connection"],
                    },
                    {
                        "id": "standard_member",
                        "part_id": "standard_member",
                        "expected_effect": "standard_member (buy): Provide the mating member",
                        "requirement_ids": ["connection"],
                    },
                    {
                        "id": "Envelope",
                        "part_id": "Envelope",
                        "expected_effect": "Envelope (reference): Locate the installation space",
                    },
                ],
                "checkpoint": "Construct the first meaningful make part: Main",
                "unresolved": payload["unresolved"],
            },
            "subsystems": [
                {
                    "id": "startup_part_0",
                    "name": "Support the connection",
                    "part_ids": ["Main"],
                    "requirement_ids": ["width", "connection"],
                },
                {
                    "id": "startup_part_1",
                    "name": "Provide the mating member",
                    "part_ids": ["standard_member"],
                    "requirement_ids": ["connection"],
                },
                {
                    "id": "startup_part_2",
                    "name": "Locate the installation space",
                    "part_ids": ["Envelope"],
                },
                {
                    "id": "request_scope",
                    "name": "Request scope",
                    "part_ids": ["Main", "standard_member", "Envelope"],
                    "requirement_ids": ["width", "connection", "finish"],
                },
            ],
        }
    )
    assert lower_intent(intent, contract) == expected
    assert lower_intent(intent, contract).model_dump() == expected.model_dump()
    assert payload == before_payload
    assert intent.model_dump() == before_intent and contract.model_dump() == before_contract


def test_omitted_requirements_stay_in_request_scope_without_invented_proof():
    payload = intent_payload()
    for part in payload["parts"]:
        part.pop("requirement_ids", None)
    plan = lower_intent(PlanIntent.model_validate(payload), frozen_contract())
    scope = plan.subsystems[-1]
    assert scope.id == "request_scope"
    assert scope.requirement_ids == ["width", "connection", "finish"]
    assert scope.part_ids == ["Main", "standard_member", "Envelope"]
    assert all(not subsystem.checks for subsystem in plan.subsystems)
    assert (
        not plan.requirement_dispositions and not plan.functional_interfaces and not plan.inspection
    )
    assert plan.operation_plan is not None
    assert not plan.operation_plan.frames and not plan.operation_plan.dimensions
    assert all(
        not step.feature_ids and not step.referenced_feature_ids
        for step in plan.operation_plan.steps
    )
    assert all(not step.requirement_ids for step in plan.operation_plan.steps)


def test_existing_feature_graph_does_not_complete_empty_startup_ownership():
    intent = PlanIntent.model_validate(
        {
            "objective": "Construct one part",
            "parts": [{"id": "part", "role": "make", "purpose": "Support the assembly"}],
            "first_part_id": "part",
        }
    )
    plan = lower_intent(intent, Contract())
    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "body", "name": "Body", "kind": "box", "params": {"x": 20, "y": 10, "z": 4}}
            ],
            "outputs": ["body"],
        }
    )
    report = coverage(snapshot(plan, 0, Contract()), document)
    assert report["graph_complete"] is False
    assert all(row["status"] == "pending" and not row["graph_present"] for row in report["steps"])
    assert all(not row["effect_verified"] for row in report["steps"])


@pytest.mark.parametrize("part_id", ["9bad", "part-name", "x" * 65])
def test_invalid_part_id_syntax_is_rejected(part_id):
    with pytest.raises(ValidationError, match="String should match pattern"):
        IntentPart(id=part_id, role="make", purpose="Construct the part")


@pytest.mark.parametrize("requirement_id", ["UPPER", "not a name", "x" * 65])
def test_invalid_requirement_id_syntax_is_rejected(requirement_id):
    with pytest.raises(ValidationError, match="String should match pattern"):
        IntentPart(
            id="part", role="make", purpose="Construct the part", requirement_ids=[requirement_id]
        )


def test_duplicate_part_ids_are_rejected():
    payload = intent_payload()
    payload["parts"].append(deepcopy(payload["parts"][0]))
    with pytest.raises(ValidationError, match="part IDs must be unique"):
        PlanIntent.model_validate(payload)


def test_duplicate_requirement_references_within_one_part_are_rejected():
    payload = intent_payload()
    payload["parts"][0]["requirement_ids"].append("width")
    with pytest.raises(ValidationError, match="requirement references must be unique"):
        PlanIntent.model_validate(payload)


def test_unknown_frozen_requirement_reference_is_rejected_by_canonical_validation():
    payload = intent_payload()
    payload["parts"][0]["requirement_ids"].append("unknown")
    with pytest.raises(ValueError, match="unknown frozen requirement"):
        lower_intent(PlanIntent.model_validate(payload), frozen_contract())


def test_part_scoped_requirement_cannot_be_attached_to_another_part():
    payload = intent_payload()
    payload["parts"][1]["requirement_ids"].append("width")
    with pytest.raises(ValueError, match="requirement belongs to a different part"):
        lower_intent(PlanIntent.model_validate(payload), frozen_contract())


@pytest.mark.parametrize("first_part_id", ["unknown", "standard_member", "Envelope"])
def test_first_part_must_exist_and_have_make_role(first_part_id):
    payload = intent_payload()
    payload["first_part_id"] = first_part_id
    with pytest.raises(ValidationError, match="first_part_id must reference"):
        PlanIntent.model_validate(payload)


@pytest.mark.parametrize("extra", ["checks", "status", "geometry", "hardware_count"])
def test_startup_intent_cannot_assert_extra_proof_fields(extra):
    payload = intent_payload()
    payload[extra] = []
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlanIntent.model_validate(payload)


def test_empty_inventory_and_empty_purpose_are_rejected():
    with pytest.raises(ValidationError, match="at least 1 item"):
        PlanIntent(objective="Construct", parts=[], first_part_id="part")
    with pytest.raises(ValidationError, match="at least 1 character"):
        IntentPart(id="part", role="make", purpose="")


def test_long_purpose_is_retained_but_subsystem_name_uses_canonical_limit():
    purpose = "Purpose " + "a" * 1992
    intent = PlanIntent(
        objective="Construct",
        parts=[IntentPart(id="part", role="make", purpose=purpose)],
        first_part_id="part",
    )
    plan = lower_intent(intent, Contract())
    assert plan.parts == [f"part (make): {purpose}"]
    assert plan.subsystems[0].name == purpose[:160]
    assert plan.operation_plan.steps[0].expected_effect == plan.parts[0]


def test_startup_inventory_has_no_part_count_cap(monkeypatch):
    for name in ("COMPLETION_SUBSYSTEM_LIMIT", "COMPLETION_CHECK_LIMIT", "COMPLETION_PLAN_BYTES"):
        monkeypatch.delenv("FORGE_WORKSPACE_" + name, raising=False)
    parts = [
        IntentPart(id=f"Part_{index}", role="make", purpose=f"Construct part {index}")
        for index in range(137)
    ]
    intent = PlanIntent(
        objective="Construct all declared parts", parts=parts, first_part_id="Part_136"
    )
    plan = lower_intent(intent, Contract())
    assert plan.operation_plan is not None
    assert len(plan.operation_plan.part_ids) == len(plan.operation_plan.steps) == 137
    assert len(plan.subsystems) == 138
    assert plan.subsystems[-2].id == "startup_part_136"
    assert plan.subsystems[-1].part_ids == [part.id for part in parts]
    assert "Part_136" in plan.operation_plan.checkpoint
