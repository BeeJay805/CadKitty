"""Safe diagnostic metadata and correction arguments exclude private response values."""

import json

import pytest

from forge_mind.llm.output_contract import OutputContractError, response_correction

SECRET = "PRIVATE_VALIDATOR_AND_MAP_KEY_CANARY"
SCHEMA = {
    "type": "object",
    "properties": {
        "records": {"type": "object", "additionalProperties": {"$ref": "#/$defs/Entry"}}
    },
    "$defs": {"Entry": {"type": "object", "properties": {"count": {"type": "integer"}}}},
}


def test_feedback_keeps_only_approved_codes_and_schema_paths():
    error = OutputContractError(
        stage="pydantic_validation",
        schema=SCHEMA,
        retryable=True,
        violations=[
            {
                "type": "int_parsing",
                "loc": ("records", SECRET, "count"),
                "msg": SECRET,
                "input": SECRET,
                "ctx": {"error": ValueError(SECRET)},
            },
            {"type": SECRET, "loc": (SECRET,), "msg": SECRET},
        ],
    )
    feedback = error.feedback
    assert feedback["errors"] == [
        {"code": "int_parsing", "path": ["records", "<key>", "count"]},
        {"code": "invalid_value", "path": ["<unknown>"]},
    ]
    assert feedback["error_count"] == 2
    assert len(feedback["schema_sha256"]) == 64
    assert SECRET not in json.dumps(feedback)
    assert SECRET not in str(error)
    assert "response_sha256" not in feedback


def test_feedback_is_bounded_copied_and_properties_are_read_only():
    error = OutputContractError(
        stage="pydantic_validation",
        schema=SCHEMA,
        retryable=True,
        violations=[{"type": "int_type", "loc": ("records", SECRET, "count")} for _ in range(1000)],
    )
    feedback = error.feedback
    assert len(json.dumps(feedback)) <= 2048
    assert feedback["error_count"] == 1000
    assert len(feedback["errors"]) <= 12
    feedback["errors"][0]["path"].append(SECRET)
    assert SECRET not in json.dumps(error.feedback)
    with pytest.raises(AttributeError):
        error.stage = SECRET
    with pytest.raises(AttributeError):
        error.retryable = False


@pytest.mark.parametrize(
    "stage", ["map_restore", "utf8_decode", "missing_response", "output_overflow", SECRET]
)
def test_noncorrectable_stages_cannot_be_enabled_by_the_caller(stage):
    error = OutputContractError(stage=stage, schema={}, retryable=True)
    assert error.retryable is False
    assert SECRET not in json.dumps(error.feedback)
    with pytest.raises(ValueError, match="eligible"):
        response_correction({"system": "original"}, error)


def test_correction_changes_only_system_and_preserves_original_argument_objects():
    images = [b"image-one", b"image-two"]
    arguments = {
        "system": "Original system",
        "user": SECRET,
        "model": "gpt-6-astra",
        "max_tokens": 3000,
        "images_png": images,
        "image_labels": ["full", "detail"],
        "output_model": object(),
        "cache_system": True,
    }
    error = OutputContractError(
        stage="json_decode", schema=SCHEMA, retryable=True, violations=[{"type": "json_invalid"}]
    )
    corrected = response_correction(arguments, error)
    assert corrected is not arguments
    assert arguments["system"] == "Original system"
    assert corrected["system"].startswith(arguments["system"])
    assert SECRET not in corrected["system"]
    assert corrected.keys() == arguments.keys()
    for key in arguments.keys() - {"system"}:
        assert corrected[key] is arguments[key]


def test_union_variant_labels_and_recursive_refs_do_not_retain_values():
    schema = {
        "type": "object",
        "properties": {"choice": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}},
        "$defs": {
            "Node": {
                "title": SECRET,
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    "count": {"type": "integer"},
                },
            }
        },
    }
    error = OutputContractError(
        stage="pydantic_validation",
        schema=schema,
        violations=[{"type": "int_type", "loc": ("choice", SECRET, "children", 0, "count")}],
    )
    assert error.feedback["errors"][0]["path"] == [
        "choice",
        "<variant>",
        "children",
        "<item>",
        "count",
    ]
    assert SECRET not in json.dumps(error.feedback)
