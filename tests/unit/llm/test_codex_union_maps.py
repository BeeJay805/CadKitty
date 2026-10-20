"""Union map restoration preserves variant meaning before local model validation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, create_model

from forge_mind.llm.codex_client import _restore_maps, _strict_schema


class TextChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text"]
    text: str


class MapChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["map"]
    values: dict[str, Annotated[float, Field(gt=0)]]


class AnyChoice(BaseModel):
    choice: TextChoice | MapChoice


class TaggedChoice(BaseModel):
    choice: Annotated[TextChoice | MapChoice, Field(discriminator="kind")]


def restored(model, value):
    schema = model.model_json_schema()
    return _restore_maps(value, schema, schema)


@pytest.mark.parametrize("model", [AnyChoice, TaggedChoice])
def test_later_object_union_variant_restores_map_without_mutating_input(model):
    wire = {"choice": {"kind": "map", "values": [{"key": "width", "value": 10.0}]}}
    before = deepcopy(wire)
    original = model.model_json_schema()
    assert _strict_schema(original)["$defs"]["MapChoice"]["properties"]["values"]["type"] == "array"

    result = model.model_validate(restored(model, wire))

    assert result.choice.values == {"width": 10.0}
    assert wire == before
    assert model.model_validate(restored(model, result.model_dump())) == result


class TextFields(BaseModel):
    text: str


class MapFields(BaseModel):
    values: dict[str, float]


def test_required_object_fields_resolve_union_without_a_tag():
    model = RootModel[TextFields | MapFields]
    result = model.model_validate(restored(model, {"values": [{"key": "width", "value": 10}]}))
    assert isinstance(result.root, MapFields)
    assert result.root.values == {"width": 10}


def test_fixed_wire_fields_exclude_an_unrelated_optional_object_branch():
    class OptionalText(BaseModel):
        text: str | None = None

    model = RootModel[OptionalText | MapFields]
    value = {"values": [{"key": "width", "value": 10}]}
    result = model.model_validate(restored(model, value))
    assert isinstance(result.root, MapFields)
    assert result.root.values == {"width": 10}


def test_empty_fixed_wire_object_cannot_match_a_mapped_variant():
    class Empty(BaseModel):
        pass

    model = RootModel[Empty | MapFields]
    value = {"values": [{"key": "width", "value": 10}]}
    result = model.model_validate(restored(model, value))
    assert isinstance(result.root, MapFields)
    assert result.root.values == {"width": 10}

    # Empty properties do not make a homogeneous map into a fixed object.
    map_model = RootModel[dict[str, int] | None]
    schema = map_model.model_json_schema()
    schema["anyOf"][0]["properties"] = {}
    mapped = _restore_maps([{"key": "width", "value": 10}], schema, schema)
    assert map_model.model_validate(mapped).root == {"width": 10}


class RecursiveChoice(BaseModel):
    choice: TextChoice | MapChoice
    children: list[RecursiveChoice] = Field(default_factory=list)


def test_recursive_refs_restore_each_concrete_value():
    value = {
        "choice": {"kind": "text", "text": "root"},
        "children": [{"choice": {"kind": "map", "values": [{"key": "width", "value": 2}]}}],
    }
    result = RecursiveChoice.model_validate(restored(RecursiveChoice, value))
    assert result.children[0].choice.values == {"width": 2}


@pytest.mark.parametrize("model", [AnyChoice, TaggedChoice])
def test_selected_map_branch_still_rejects_duplicate_keys(model):
    value = {"choice": {"kind": "map", "values": [{"key": "x", "value": 1}] * 2}}
    with pytest.raises(ValueError, match="duplicate map keys"):
        restored(model, value)


@pytest.mark.parametrize("model", [AnyChoice, TaggedChoice])
def test_original_pydantic_constraints_still_validate_restored_map(model):
    value = {"choice": {"kind": "map", "values": [{"key": "x", "value": -1}]}}
    result = restored(model, value)
    assert result["choice"]["values"] == {"x": -1}
    with pytest.raises(ValidationError, match="greater than 0"):
        model.model_validate(result)


class Pair(BaseModel):
    key: str
    value: int


@pytest.mark.parametrize("reverse", [False, True])
def test_list_and_map_wire_ambiguity_is_rejected_independent_of_order(reverse):
    variants = [list[Pair], dict[str, int]]
    if reverse:
        variants.reverse()
    model = RootModel[variants[0] | variants[1]]
    with pytest.raises(ValueError, match="ambiguous"):
        restored(model, [{"key": "x", "value": 1}])


def test_identical_restored_outcomes_are_safe_for_local_union_validation():
    model = RootModel[dict[str, int] | dict[str, float]]
    value = restored(model, [{"key": "x", "value": 1}])
    assert value == {"x": 1}
    assert model.model_validate(value).root == {"x": 1}


@pytest.mark.parametrize("value", [True, 1])
def test_boolean_and_numeric_tags_do_not_share_a_branch(value):
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {"kind": {"const": True}, "values": {"type": "array", "items": {}}},
            },
            {
                "type": "object",
                "properties": {
                    "kind": {"const": 1},
                    "values": {"type": "object", "additionalProperties": {"type": "number"}},
                },
            },
        ]
    }
    wire = {"kind": value, "values": [{"key": "x", "value": 2}]}
    result = _restore_maps(wire, schema, schema)
    assert result["values"] == (wire["values"] if value is True else {"x": 2})


def test_discriminator_mapping_does_not_fall_back_to_another_branch():
    schema = TaggedChoice.model_json_schema()
    schema["properties"]["choice"]["discriminator"]["mapping"]["map"] = "#/$defs/TextChoice"
    wire = {"choice": {"kind": "map", "values": [{"key": "x", "value": 1}]}}
    with pytest.raises(ValueError, match=r"discriminator|union"):
        _restore_maps(wire, schema, schema)


def test_unknown_discriminator_is_rejected_without_guessing_a_map_branch():
    wire = {"choice": {"kind": "unknown", "values": [{"key": "x", "value": 1}]}}
    with pytest.raises(ValueError, match=r"discriminator|union"):
        restored(TaggedChoice, wire)


@pytest.mark.parametrize("empty", [False, True])
def test_ambiguous_empty_or_nonempty_lists_do_not_choose_a_map(empty):
    model = RootModel[list[Pair] | dict[str, int]]
    value = [] if empty else [{"key": "x", "value": 1}]
    with pytest.raises(ValueError, match="ambiguous"):
        restored(model, value)


@pytest.mark.parametrize("value", [True, 1])
@pytest.mark.parametrize("tag_keyword", ["type", "enum"])
def test_boolean_number_types_and_enums_select_only_the_compatible_structure(value, tag_keyword):
    boolean = "boolean" if tag_keyword == "type" else [True]
    integer = "integer" if tag_keyword == "type" else [1]
    schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tag": {tag_keyword: boolean},
                    "data": {"type": "array", "items": {}},
                },
            },
            {
                "type": "object",
                "properties": {
                    "tag": {tag_keyword: integer},
                    "data": {"type": "object", "additionalProperties": {"type": "integer"}},
                },
            },
        ]
    }
    wire = {"tag": value, "data": [{"key": "x", "value": 2}]}
    result = _restore_maps(wire, schema, schema)
    assert result["data"] == (wire["data"] if value is True else {"x": 2})


def test_untagged_oneof_uses_the_same_structural_resolution():
    schema = AnyChoice.model_json_schema()
    choice = schema["properties"]["choice"]
    choice["oneOf"] = choice.pop("anyOf")
    value = {"choice": {"kind": "map", "values": [{"key": "x", "value": 1}]}}
    result = AnyChoice.model_validate(_restore_maps(value, schema, schema))
    assert result.choice.values == {"x": 1}


def test_invalid_reference_cycle_still_fails_closed():
    schema = {
        "anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}],
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
    }
    with pytest.raises(ValueError, match="cyclic schema reference"):
        _restore_maps({}, schema, schema)


class Plain(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["plain"]
    text: str


class Measured(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["measured"]
    dimensions_mm: dict[str, float]


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: Annotated[Plain | Measured, Field(discriminator="kind")]


def test_actual_envelope_lowers_tagged_union_to_supported_wire_schema():
    original = Envelope.model_json_schema()
    before = deepcopy(original)
    wire = _strict_schema(original)

    assert "oneOf" not in json.dumps(wire)
    assert "discriminator" not in json.dumps(wire)
    assert wire["properties"]["result"]["anyOf"] == original["properties"]["result"]["oneOf"]
    assert wire["$defs"]["Measured"]["properties"]["dimensions_mm"]["type"] == "array"
    assert original == before
    value = {"result": {"kind": "measured", "dimensions_mm": [{"key": "width", "value": 50}]}}
    result = Envelope.model_validate(_restore_maps(value, original, original))
    assert result.result.dimensions_mm == {"width": 50}


def test_nested_nullable_list_and_map_tagged_unions_lower_with_original_refs():
    class Batch(BaseModel):
        results: list[Envelope | None]
        groups: dict[str, Envelope]

    original = Batch.model_json_schema()
    before = deepcopy(original)
    wire = _strict_schema(original)
    assert "oneOf" not in json.dumps(wire)
    assert "discriminator" not in json.dumps(wire)
    measured = {"result": {"kind": "measured", "dimensions_mm": [{"key": "width", "value": 50}]}}
    value = {"results": [None, measured], "groups": [{"key": "part", "value": measured}]}
    result = Batch.model_validate(_restore_maps(value, original, original))
    assert result.groups["part"].result.dimensions_mm == {"width": 50}
    assert original == before


def test_defaulted_and_multivalue_literal_tags_still_lower_safely():
    class DefaultPlain(BaseModel):
        kind: Literal["plain", "text"] = "plain"
        text: str

    class DefaultMeasured(BaseModel):
        kind: Literal["measured"] = "measured"
        dimensions_mm: dict[str, float]

    class DefaultEnvelope(BaseModel):
        result: Annotated[DefaultPlain | DefaultMeasured, Field(discriminator="kind")]

    wire = _strict_schema(DefaultEnvelope.model_json_schema())
    assert "anyOf" in wire["properties"]["result"]
    assert all("kind" in branch["required"] for branch in wire["$defs"].values())


def test_schema_keywords_used_as_property_and_definition_names_are_preserved():
    keyword_model = create_model("oneOf", oneOf=(str, ...), discriminator=(dict[str, int], ...))
    wrapper = create_model("KeywordEnvelope", content=(keyword_model, ...))
    original = wrapper.model_json_schema()
    before = deepcopy(original)
    wire = _strict_schema(original)
    assert "oneOf" in wire["$defs"]
    fields = wire["$defs"]["oneOf"]["properties"]
    assert fields["oneOf"]["type"] == "string"
    assert fields["discriminator"]["type"] == "array"
    assert original == before


@pytest.mark.parametrize(
    "schema",
    [
        {"oneOf": [{"type": "integer"}, {"type": "number"}]},
        {"oneOf": [{"type": "string"}, {"type": "number"}]},
        {"discriminator": {"propertyName": "kind"}, "anyOf": [{"type": "string"}]},
        {
            "oneOf": [
                {"type": "object", "properties": {"kind": {"const": "same"}}},
                {"type": "object", "properties": {"kind": {"enum": ["same", "other"]}}},
            ],
            "discriminator": {"propertyName": "kind"},
        },
        {
            "oneOf": [
                {"type": "object", "properties": {"kind": {"const": 1}}},
                {"type": "object", "properties": {"kind": {"const": 1.0}}},
            ]
        },
    ],
)
def test_unproven_exclusive_or_unsupported_discriminator_schemas_are_refused(schema):
    with pytest.raises(ValueError, match=r"exclusive|discriminator"):
        _strict_schema(schema)


def test_inconsistent_discriminator_mapping_is_refused_before_provider_call():
    schema = Envelope.model_json_schema()
    schema["properties"]["result"]["discriminator"]["mapping"]["measured"] = "#/$defs/Plain"
    with pytest.raises(ValueError, match="discriminator"):
        _strict_schema(schema)


def test_disjoint_boolean_and_numeric_tags_can_lower_without_conflation():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": True}}},
            {"type": "object", "properties": {"kind": {"const": 1}}},
        ]
    }
    wire = _strict_schema(schema)
    assert "anyOf" in wire
    assert [branch["properties"]["kind"]["const"] for branch in wire["anyOf"]] == [True, 1]
