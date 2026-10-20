"""Discovered semantics and supplemental diagnostics preserve strict plan admission."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from forge_mind.workspace.completion_bindings import PREPARED_CHECK_TOOLS
from forge_mind.workspace.completion_models import FunctionalInterface, InterfaceState
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.operation_plan import plan_semantic_conflicts, snapshot, validate_plan


def _plan() -> dict[str, Any]:
    return {
        "objective": "Plan shared pivot parts",
        "operation_plan": {
            "part_ids": ["yoke", "sleeve"],
            "frames": [{"id": "pivot", "description": "Shared pivot", "part_id": None}],
            "steps": [
                {
                    "id": "build_yoke",
                    "part_id": "yoke",
                    "frame_id": "pivot",
                    "expected_effect": "Build the yoke",
                },
                {
                    "id": "build_sleeve",
                    "part_id": "sleeve",
                    "frame_id": "pivot",
                    "expected_effect": "Build the sleeve",
                },
            ],
        },
        "functional_interfaces": [
            {
                "id": "pivot_fit",
                "requirement_ids": ["fit"],
                "participant_paths": ["yoke", "sleeve"],
                "installed_occurrence_paths": ["sleeve"],
                "required_features": [{"occurrence_path": "yoke", "feature_id": "pivot_bore"}],
                "port_ids": ["yoke.pivot"],
                "states": [{"id": "neutral", "kind": "motion", "driver_values": {"tilt": 0.0}}],
                "duties": [
                    {
                        "id": "fit",
                        "kind": "fit",
                        "proof_scope": "nominal_fit",
                        "state_ids": ["neutral"],
                    }
                ],
            }
        ],
    }


def _contract() -> Contract:
    return Contract(
        requirements=[Requirement(id="fit", text="Fit pivot", source_quote="Fit pivot")]
    )


def test_plan_schema_exposes_enforced_cross_field_semantics() -> None:
    definitions = DesignPlan.model_json_schema()["$defs"]
    interface = definitions["FunctionalInterface"]["properties"]
    assert "No participant is inferred" in interface["participant_paths"]["description"]
    assert "participant_paths" in interface["installed_occurrence_paths"]["description"]
    assert "participant_paths" in interface["required_features"]["description"]
    assert "final dot" in interface["port_ids"]["description"]
    assert (
        "participant_paths"
        in definitions["InterfaceFeature"]["properties"]["occurrence_path"]["description"]
    )
    state = definitions["InterfaceState"]["properties"]
    assert "zero" in state["kind"]["description"]
    assert "nonempty" in state["driver_values"]["description"]
    assert (
        "Null (or omitted)" in definitions["OperationFrame"]["properties"]["part_id"]["description"]
    )
    assert "shared" in definitions["OperationStep"]["properties"]["frame_id"]["description"]
    assert (
        "without part scope"
        in definitions["DimensionFact"]["properties"]["frame_id"]["description"]
    )


def test_plan_schema_exposes_existing_completion_binding_routes() -> None:
    definitions = DesignPlan.model_json_schema()["$defs"]
    duty = definitions["InterfaceDuty"]["properties"]
    checks = duty["check_ids"]["description"]
    for (kind, scope), tools in PREPARED_CHECK_TOOLS.items():
        assert f"{kind}/{scope}: prepared {', '.join(sorted(tools))}" in checks
    assert "not duty.check_ids" in checks
    assert "current verified motion evidence, not check_ids" in checks
    assert "union must equal all interface participants" in checks
    assert "Unsupported duty/scope pairs remain unknown" in checks
    assert "do not waive missing construction" in checks
    states = duty["state_ids"]["description"]
    assert "allow empty state_ids" in states
    assert "requires nonempty" in states
    assert "entire driver_values dictionary" in states
    assert "not arbitrary requested poses" in states
    interface = definitions["FunctionalInterface"]["properties"]
    assert "at least two participants" in interface["mate_ids"]["description"]
    assert "no outside endpoint" in interface["mate_ids"]["description"]
    assert "at least one interface participant" in interface["participant_paths"]["description"]
    assert "bindings for every participant" in interface["participant_paths"]["description"]
    assert (
        "exact current revision ID"
        in definitions["PreparedCheck"]["properties"]["reference_revision_id"]["description"]
    )
    assert set(duty) == {
        "id",
        "kind",
        "proof_scope",
        "check_ids",
        "state_ids",
        "missing_inputs",
        "unsupported_checks",
        "physical_validation",
    }


def test_schema_descriptions_preserve_legacy_serialization_and_snapshot_hash() -> None:
    plan = DesignPlan(objective="Existing legacy plan")
    expected = {
        "objective": "Existing legacy plan",
        "parts": [],
        "interfaces": [],
        "assembly_sequence": [],
        "verification": [],
        "unresolved": [],
        "subsystems": [],
        "requirement_dispositions": [],
        "inspection": [],
    }
    assert plan.model_dump(mode="json") == expected
    expected_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert snapshot(plan, 1, _contract())["plan_hash"] == expected_hash
    state = InterfaceState(id="rest")
    assert state.model_dump(mode="json") == {"id": "rest", "kind": "reference", "driver_values": {}}
    interface = FunctionalInterface.model_validate(_plan()["functional_interfaces"][0])
    assert interface.model_dump(mode="json")["states"] == [
        {"id": "neutral", "kind": "motion", "driver_values": {"tilt": 0.0}}
    ]
    raw = _plan()["functional_interfaces"][0]
    assert interface.model_dump(mode="json") == {
        **raw,
        "mate_ids": [],
        "duties": [
            {
                **raw["duties"][0],
                "check_ids": [],
                "missing_inputs": [],
                "unsupported_checks": [],
                "physical_validation": [],
            }
        ],
    }


@pytest.mark.parametrize(
    ("kind", "drivers", "accepted"),
    [
        ("reference", {}, True),
        ("motion", {"tilt": 0.0}, True),
        ("reference", {"tilt": 0.0}, False),
        ("motion", {}, False),
    ],
)
def test_state_rules_include_zero_and_remain_strict(
    kind: str,
    drivers: dict[str, float],
    accepted: bool,
) -> None:
    raw = {"id": "neutral", "kind": kind, "driver_values": drivers}
    if accepted:
        assert InterfaceState.model_validate(raw).model_dump() == raw
    else:
        with pytest.raises(ValidationError, match="Motion states require"):
            InterfaceState.model_validate(raw)


def test_valid_shared_frame_and_zero_motion_have_no_conflicts() -> None:
    raw = _plan()
    plan = DesignPlan.model_validate(raw)
    validate_plan(plan, _contract())
    assert plan_semantic_conflicts(raw) == []
    assert plan_semantic_conflicts(plan) == []


def test_reports_all_frame_and_interface_conflicts_despite_invalid_child_state() -> None:
    raw = _plan()
    raw["operation_plan"]["frames"][0]["part_id"] = "yoke"
    raw["operation_plan"]["steps"].append(
        {
            "id": "inspect_sleeve",
            "part_id": "sleeve",
            "frame_id": "absent",
            "expected_effect": "Inspect the sleeve",
        }
    )
    interface = raw["functional_interfaces"][0]
    interface["participant_paths"] = ["yoke"]
    interface["required_features"].append({"occurrence_path": "nut", "feature_id": "body"})
    interface["port_ids"].append("nut.top.axis")
    interface["states"][0]["kind"] = "reference"
    interface["duties"][0]["state_ids"].append("missing_state")
    retained = copy.deepcopy(raw)
    conflicts = plan_semantic_conflicts(raw)
    assert raw == retained
    by_code = {row["code"]: row for row in conflicts}
    assert len(conflicts) == 7
    assert set(by_code) == {
        "frame_owner_mismatch",
        "unknown_frame",
        "installed_occurrence_not_participant",
        "feature_owner_not_participant",
        "port_owner_not_participant",
        "state_driver_mismatch",
        "unknown_duty_state",
    }
    assert by_code["frame_owner_mismatch"] == {
        "field_path": "/operation_plan/steps/1/frame_id",
        "record_id": "build_sleeve",
        "code": "frame_owner_mismatch",
        "conflicting_ids": ["pivot", "sleeve", "yoke"],
    }
    assert by_code["port_owner_not_participant"]["conflicting_ids"] == ["nut.top", "nut.top.axis"]
    assert by_code["state_driver_mismatch"]["record_id"] == "neutral"
    assert by_code["state_driver_mismatch"]["conflicting_ids"] == ["tilt"]
    with pytest.raises(ValidationError):
        DesignPlan.model_validate(raw)


def test_owned_frame_conflicts_do_not_relax_admission() -> None:
    raw = _plan()
    raw["operation_plan"]["frames"][0]["part_id"] = "yoke"
    plan = DesignPlan.model_validate(raw)
    assert [row["record_id"] for row in plan_semantic_conflicts(plan)] == ["build_sleeve"]
    with pytest.raises(ValueError, match="different part"):
        validate_plan(plan, _contract())
    raw["operation_plan"]["dimensions"] = [
        {
            "id": "unscoped",
            "value": 10,
            "source": "assumption",
            "source_quote": "Assume 10 mm",
            "frame_id": "pivot",
        }
    ]
    assert {row["record_id"] for row in plan_semantic_conflicts(raw)} == {
        "build_sleeve",
        "unscoped",
    }


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("installed_occurrence_paths", ["nut"], "installed_occurrence_not_participant"),
        (
            "required_features",
            [{"occurrence_path": "nut", "feature_id": "body"}],
            "feature_owner_not_participant",
        ),
        ("port_ids", ["nut.top.axis"], "port_owner_not_participant"),
    ],
)
def test_membership_conflicts_remain_strict_without_child_errors(
    field: str,
    replacement: Any,
    code: str,
) -> None:
    raw = _plan()
    raw["functional_interfaces"][0][field] = replacement
    assert [row["code"] for row in plan_semantic_conflicts(raw)] == [code]
    with pytest.raises(ValidationError, match="participants"):
        DesignPlan.model_validate(raw)


def test_ambiguous_or_malformed_references_do_not_invent_absence_or_ownership() -> None:
    raw = _plan()
    raw["operation_plan"]["frames"].append(
        {"id": "pivot", "description": "Conflicting declaration", "part_id": "yoke"}
    )
    interface = raw["functional_interfaces"][0]
    interface["participant_paths"] = "yoke"
    interface["states"] = [{"id": ["invalid"]}]
    interface["required_features"] = [{"occurrence_path": {}, "feature_id": "body"}]
    conflicts = plan_semantic_conflicts(raw)
    assert [row["code"] for row in conflicts] == ["duplicate_reference"]
    assert plan_semantic_conflicts(None) == []
    with pytest.raises(ValidationError):
        DesignPlan.model_validate(raw)


def test_duplicate_interface_references_are_reported_without_coercion() -> None:
    raw = _plan()
    interface = raw["functional_interfaces"][0]
    interface["installed_occurrence_paths"].append("sleeve")
    interface["required_features"].append(copy.deepcopy(interface["required_features"][0]))
    interface["states"].append(copy.deepcopy(interface["states"][0]))
    interface["duties"][0]["state_ids"].append("neutral")
    conflicts = plan_semantic_conflicts(raw)
    assert len(conflicts) == 4
    assert all(row["code"] == "duplicate_reference" for row in conflicts)
    with pytest.raises(ValidationError, match="references must be unique"):
        DesignPlan.model_validate(raw)
