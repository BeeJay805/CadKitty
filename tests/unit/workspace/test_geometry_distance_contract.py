"""Selection and result boundaries for exact saved-geometry questions."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.geometry_distance_contract import (
    DistanceContext,
    DistanceResult,
    DistanceSelector,
    distance_tool_result,
)


def context() -> dict:
    return {
        "project_id": "1" * 32,
        "revision_id": "2" * 32,
        "a": {
            "requested": {"body_index": 0},
            "body_index": 0,
            "body_signature": "3" * 64,
        },
        "b": {
            "requested": {"occurrence_path": "assembly/insert"},
            "body_index": 1,
            "body_signature": "4" * 64,
        },
        "source_hashes": {
            "part.step": "5" * 64,
            "topology.json": "6" * 64,
            "document.json": "7" * 64,
        },
        "saved_engine_fingerprint": "8" * 64,
        "query_engine_fingerprint": "9" * 64,
    }


def result() -> dict:
    return {
        **context(),
        "minimum_distance_mm": 5.0,
        "witness_a_mm": [0.0, 0.0, 0.0],
        "witness_b_mm": [3.0, 0.0, 4.0],
        "solution_count": 2,
        "native_completed": True,
        "native_inner_solution": False,
    }


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"body_index": 0, "occurrence_path": "part"},
        {"body_index": True},
        {"body_index": "0"},
        {"body_index": 0.0},
        {"body_index": -1},
        {"occurrence_path": ""},
        {"body_index": 0, "face_signature": "near the hole"},
        {"body_index": 0, "source": "part.step"},
    ],
)
def test_ambiguous_or_coerced_selection_is_refused(selection):
    with pytest.raises(ValidationError):
        DistanceSelector.model_validate(selection)


def test_zero_body_index_and_historical_engine_are_supported():
    measured = DistanceResult.model_validate(result())
    assert measured.a.body_index == 0
    assert measured.saved_engine_fingerprint != measured.query_engine_fingerprint
    assert measured.minimum_distance_mm == 5
    assert measured.solution_count == 2


def test_distinct_same_body_faces_are_supported():
    value = context()
    for key, signature in (("a", "a" * 64), ("b", "b" * 64)):
        value[key] = {
            "requested": {"body_index": 0, "face_signature": signature},
            "body_index": 0,
            "body_signature": "3" * 64,
            "face_signature": signature,
        }
    assert DistanceContext.model_validate(value).a.body_index == 0
    value["b"] = deepcopy(value["a"])
    with pytest.raises(ValidationError, match="distinct"):
        DistanceContext.model_validate(value)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("native_completed", False),
        ("native_completed", 1),
        ("native_inner_solution", "false"),
        ("minimum_distance_mm", float("nan")),
        ("minimum_distance_mm", -1.0),
        ("minimum_distance_mm", True),
        ("witness_b_mm", [0.0, 0.0, 4.0]),
        ("witness_b_mm", [3.0, float("inf"), 4.0]),
        ("solution_count", 0),
        ("solution_count", True),
        ("frame", "display_y_up"),
        ("scope", "This part fits correctly"),
    ],
)
def test_failed_or_inconsistent_measurement_is_refused(field, invalid):
    value = result()
    value[field] = invalid
    with pytest.raises(ValidationError):
        DistanceResult.model_validate(value)


def test_zero_distance_keeps_inner_flag_without_acceptance():
    value = result()
    value.update(
        minimum_distance_mm=0.0,
        witness_a_mm=[1.0, 2.0, 3.0],
        witness_b_mm=[1.0, 2.0, 3.0],
        native_inner_solution=True,
    )
    measured = DistanceResult.model_validate(value).model_dump()
    assert measured["native_inner_solution"] is True
    assert "status" not in measured and "pass" not in measured


def test_source_and_selection_binding_cannot_be_substituted():
    value = result()
    del value["source_hashes"]["topology.json"]
    with pytest.raises(ValidationError):
        DistanceResult.model_validate(value)
    value = result()
    value["a"]["body_index"] = 5
    with pytest.raises(ValidationError, match="requested index"):
        DistanceResult.model_validate(value)


def test_transport_rejects_a_result_from_another_request():
    value = result()
    a = DistanceSelector(body_index=0)
    b = DistanceSelector(occurrence_path="assembly/insert")
    observed = distance_tool_result(value, "1" * 32, "2" * 32, a, b)
    assert observed["minimum_distance_mm"] == 5
    with pytest.raises(ValueError, match="requested revision"):
        distance_tool_result(value, "1" * 32, "f" * 32, a, b)
    with pytest.raises(ValueError, match="requested revision"):
        distance_tool_result(value, "1" * 32, "2" * 32, b, a)
