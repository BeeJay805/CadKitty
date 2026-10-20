"""Native transport stays compact while existing production validation remains authoritative."""

import json

import pytest
from pydantic import ValidationError

from forge_mind.workspace.construction_reference import OPERATIONS
from forge_mind.workspace.models import Action, DesignPlan
from forge_mind.workspace.native_authoring import NativeAction, native_bootstrap


def test_compact_transport_preserves_every_action_and_uses_existing_plan_route():
    schema = NativeAction.model_json_schema()
    original = Action.model_json_schema()
    assert schema["properties"]["kind"] == original["properties"]["kind"]
    assert len(json.dumps(schema)) < len(json.dumps(original)) * 0.15
    plan = {"objective": "An editable block", "operation_plan": {"part_ids": ["block"]}}
    native = NativeAction(kind="plan", explanation="Save intent", document_json=json.dumps(plan))
    action = Action.model_validate(native.model_dump())
    assert action.plan is None
    assert DesignPlan.model_validate_json(action.document_json).objective == plan["objective"]
    with pytest.raises(ValidationError):
        NativeAction(kind="plan", explanation="Save intent", plan=plan)


def test_source_page_validation_and_extra_field_rejection_are_retained():
    with pytest.raises(ValidationError, match="source_page"):
        NativeAction(kind="source_page", explanation="Read source", query="not-an-owned-id")
    with pytest.raises(ValidationError):
        NativeAction(kind="finish", explanation="Saved draft", invented_field=True)


def test_optional_construction_patch_is_discoverable_and_absence_keeps_old_transport():
    plain = NativeAction(kind="finish", explanation="Saved draft")
    assert "plan_patch_json" not in plain.model_dump()
    patched = NativeAction(
        kind="document",
        explanation="Bind the explicit construction plan",
        plan_patch_json='{"objective":"Authored intent"}',
    )
    assert patched.model_dump()["plan_patch_json"] == patched.plan_patch_json
    discovery = native_bootstrap()["discovery"]
    assert "plan_patch_json" in discovery["construction_transaction"]
    assert "request_hash" in discovery["check_amendment"]


def test_bootstrap_contains_closed_core_schema_references_and_full_discovery():
    bootstrap = native_bootstrap()
    plan = bootstrap["plan_schema"]
    document = bootstrap["document_schema"]
    assert "operation_plan" in plan["properties"]
    assert "subsystems" not in plan["properties"]
    assert "PreparedCheck" not in plan.get("$defs", {})
    assert {"OperationPlan", "OperationStep", "DimensionFact", "OperationFrame"} <= set(
        plan["$defs"]
    )
    assert "assembly" in document["properties"]
    assert "AssemblySpec" in document["$defs"]
    assert set(bootstrap["feature_index"]) == set(OPERATIONS) - {"python"}
    assert set(bootstrap["discovery"]["all_actions"]) == set(
        Action.model_json_schema()["properties"]["kind"]["enum"]
    )
    for schema in (plan, document):
        pending = [schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if "$ref" in item:
                    assert item["$ref"].removeprefix("#/$defs/") in schema["$defs"]
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)


def test_bootstrap_is_fresh_and_does_not_change_shared_operation_reference():
    first = native_bootstrap()
    first["feature_index"]["box"]["required"].append("invented")
    second = native_bootstrap()
    assert second["feature_index"]["box"]["required"] == ["x", "y", "z"]
    assert OPERATIONS["box"]["required"] == ["x", "y", "z"]


def test_bootstrap_supplies_existing_native_operation_syntax_before_first_build():
    index = native_bootstrap()["feature_index"]
    assert "python" not in index
    for kind, record in OPERATIONS.items():
        if kind == "python":
            continue
        assert index[kind] == {"schema_query": f"feature:{kind}", **record}
    assert index["transform"]["optional"] == {
        "x": 0,
        "y": 0,
        "z": 0,
        "rx": 0,
        "ry": 0,
        "rz": 0,
    }
    assert index["cut"]["inputs"] == "target first, one or more tool solids next"
    assert index["cylinder"]["meaning"] == (
        "Positive radius and height; centered at origin, axis Z."
    )


def test_bootstrap_nested_optional_defaults_are_independent_between_calls():
    first = native_bootstrap()
    first["feature_index"]["transform"]["optional"]["x"] = 999
    second = native_bootstrap()
    assert second["feature_index"]["transform"]["optional"]["x"] == 0
    assert OPERATIONS["transform"]["optional"]["x"] == 0


def test_native_pattern_guidance_provides_a_supported_connected_part_route():
    guidance = native_bootstrap()["feature_index"]["pattern"]["meaning"]
    assert "explicit transformed Features and union" in guidance
    assert "Fuse copies in Python" not in guidance
