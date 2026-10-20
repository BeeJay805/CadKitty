"""Pure wire-schema controls for constrained map keys and nested model references."""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, RootModel, ValidationError

from forge_mind.llm.codex_client import _restore_maps, _strict_schema
from forge_mind.workspace.completion_models import InterfaceState
from forge_mind.workspace.models import Action

MapKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


def assert_closed(node: Any) -> None:
    if isinstance(node, dict):
        assert "patternProperties" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            assert_closed(value)
    elif isinstance(node, list):
        for value in node:
            assert_closed(value)


def action_payload(driver_values=None):
    return {
        "kind": "plan",
        "explanation": "Retain the declared motion inspection state",
        "plan": {
            "objective": "Inspect the declared state",
            "functional_interfaces": [
                {
                    "id": "connection",
                    "requirement_ids": ["movement"],
                    "participant_paths": ["member"],
                    "states": [
                        {
                            "id": "pose",
                            "kind": "motion",
                            "driver_values": driver_values
                            if driver_values is not None
                            else [{"key": "angle", "value": 45.0}],
                        }
                    ],
                    "duties": [
                        {
                            "id": "travel",
                            "kind": "travel",
                            "proof_scope": "sampled_clearance",
                            "state_ids": ["pose"],
                        }
                    ],
                }
            ],
        },
    }


def test_single_pattern_map_has_closed_pair_wire_objects_without_mutating_original():
    original = InterfaceState.model_json_schema()
    before = deepcopy(original)
    wire = _strict_schema(original)
    assert_closed(wire)
    values = wire["properties"]["driver_values"]
    assert values["type"] == "array"
    assert values["items"]["properties"]["value"]["type"] == "number"
    pattern = next(iter(original["properties"]["driver_values"]["patternProperties"]))
    assert values["items"]["properties"]["key"] == {"type": "string", "pattern": pattern}
    assert original == before


def test_full_action_optional_plan_ref_roundtrips_nested_motion_driver_map():
    original = Action.model_json_schema()
    assert_closed(_strict_schema(original))
    payload = action_payload()
    before = deepcopy(payload)
    action = Action.model_validate(_restore_maps(payload, original, original))
    assert action.plan is not None
    state = action.plan.functional_interfaces[0].states[0]
    assert state.driver_values == {"angle": 45.0}
    assert state.kind == "motion"
    assert payload == before
    already_restored = action.model_dump(mode="json")
    assert Action.model_validate(_restore_maps(already_restored, original, original)) == action


def test_duplicate_pattern_map_keys_are_rejected_before_model_validation():
    original = Action.model_json_schema()
    payload = action_payload([{"key": "angle", "value": 45.0}, {"key": "angle", "value": 60.0}])
    with pytest.raises(ValueError, match="duplicate map keys"):
        _restore_maps(payload, original, original)


@pytest.mark.parametrize("key", ["9angle", "angle with spaces", "angle/slash", ""])
def test_original_regex_key_validation_still_rejects_restored_invalid_names(key):
    original = Action.model_json_schema()
    restored = _restore_maps(action_payload([{"key": key, "value": 45.0}]), original, original)
    with pytest.raises(ValidationError, match="String should match pattern"):
        Action.model_validate(restored)


class DriverMap(RootModel[dict[MapKey, float]]):
    pass


class PlainMap(RootModel[dict[str, float]]):
    pass


class NestedMaps(BaseModel):
    groups: dict[MapKey, DriverMap]


class OptionalNestedMaps(BaseModel):
    design: NestedMaps | None = None


def test_nested_pattern_maps_restore_values_through_local_and_optional_refs():
    original = OptionalNestedMaps.model_json_schema()
    assert_closed(_strict_schema(original))
    payload = {
        "design": {
            "groups": [
                {"key": "first", "value": [{"key": "angle", "value": 12.5}]},
                {"key": "second", "value": [{"key": "offset", "value": 3.0}]},
            ]
        }
    }
    model = OptionalNestedMaps.model_validate(_restore_maps(payload, original, original))
    assert model.model_dump() == {
        "design": {"groups": {"first": {"angle": 12.5}, "second": {"offset": 3.0}}}
    }
    assert (
        OptionalNestedMaps.model_validate(
            _restore_maps({"design": None}, original, original)
        ).design
        is None
    )


def test_nested_ref_pattern_map_preserves_inner_key_validation():
    original = OptionalNestedMaps.model_json_schema()
    payload = {
        "design": {"groups": [{"key": "first", "value": [{"key": "INVALID", "value": 1.0}]}]}
    }
    with pytest.raises(ValidationError, match="String should match pattern"):
        OptionalNestedMaps.model_validate(_restore_maps(payload, original, original))


@pytest.mark.parametrize(
    "entry",
    [
        {"key": "angle", "value": 2.0, "unexpected": True},
        {"key": 7, "value": 2.0},
        ["angle", 2.0],
        None,
    ],
)
def test_malformed_map_pairs_reject_nonobjects_extra_fields_and_nonstring_keys(entry):
    original = InterfaceState.model_json_schema()
    with pytest.raises(ValueError, match="string key and value"):
        _restore_maps(
            {"id": "pose", "kind": "motion", "driver_values": [entry]}, original, original
        )


@pytest.mark.parametrize("output_model", [DriverMap, PlainMap])
@pytest.mark.parametrize(
    ("entry", "missing"),
    [({}, "key"), ({"value": 2.0}, "key"), ({"key": "angle"}, "value")],
)
def test_missing_map_pair_members_preserve_the_existing_key_error_contract(
    output_model, entry, missing
):
    original = output_model.model_json_schema()
    with pytest.raises(KeyError) as failure:
        _restore_maps([entry], original, original)
    assert failure.value.args == (missing,)


def test_missing_nested_ref_map_value_preserves_the_existing_key_error_contract():
    original = OptionalNestedMaps.model_json_schema()
    payload = {"design": {"groups": [{"key": "first", "value": [{"key": "angle"}]}]}}
    with pytest.raises(KeyError) as failure:
        _restore_maps(payload, original, original)
    assert failure.value.args == ("value",)


@pytest.mark.parametrize(
    "map_schema",
    [
        {"patternProperties": {"^a": {"type": "number"}, "^b": {"type": "string"}}},
        {
            "patternProperties": {"^a": {"type": "number"}},
            "additionalProperties": {"type": "string"},
        },
    ],
)
def test_heterogeneous_pattern_maps_are_not_silently_retyped(map_schema):
    with pytest.raises(ValueError, match="structured map output"):
        _strict_schema({"type": "object", **map_schema})
