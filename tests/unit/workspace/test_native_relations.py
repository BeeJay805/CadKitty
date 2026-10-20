"""Named arithmetic uses explicit domains and keeps historical binary records."""

import math

import pytest

from forge_mind.workspace.construction_reference import construction_reference
from forge_mind.workspace.document import DesignDocument, VariableRelation


@pytest.mark.parametrize(
    ("operation", "a", "b", "expected"),
    [
        ("add", 7, 3, 10),
        ("subtract", 7, 3, 4),
        ("multiply", 7, 3, 21),
        ("divide", 7, 2, 3.5),
        ("max", -7, -3, -3),
        ("min", -7, -3, -7),
        ("sin_deg", 30, 0, 0.5),
        ("sin_deg", 450, 0, 1),
        ("cos_deg", 60, 0, 0.5),
        ("cos_deg", -180, 0, -1),
        ("tan_deg", 45, 0, 1),
        ("tan_deg", -225, 0, -1),
        ("atan2_deg", 1, -1, 135),
        ("atan2_deg", -1, -1, -135),
        ("sqrt", 9, 0, 3),
        ("sqrt", 0, 0, 0),
    ],
)
def test_named_arithmetic_has_explicit_numerical_meaning(operation, a, b, expected):
    doc = DesignDocument(relations=[{"name": "answer", "operation": operation, "a": a, "b": b}])
    assert doc.values()["answer"] == pytest.approx(expected)


def test_ordered_relations_derive_length_and_angle_from_named_components():
    doc = DesignDocument(
        variables={"x": 3, "y": 4},
        relations=[
            {"name": "xx", "operation": "multiply", "a": "x", "b": "x"},
            {"name": "yy", "operation": "multiply", "a": "y", "b": "y"},
            {"name": "square_length", "operation": "add", "a": "xx", "b": "yy"},
            {"name": "length", "operation": "sqrt", "a": "square_length"},
            {"name": "direction", "operation": "atan2_deg", "a": "y", "b": "x"},
        ],
    )
    assert doc.values()["length"] == 5
    assert doc.values()["direction"] == pytest.approx(53.13010235415598)
    doc.variables["x"] = 0
    assert doc.values()["length"] == 4
    assert doc.values()["direction"] == 90


@pytest.mark.parametrize("angle", [-450, -270, -90, 90, 270, 450, 90 + 1e-11])
def test_tangent_refuses_singular_angles(angle):
    with pytest.raises(ValueError, match="tan_deg is undefined"):
        DesignDocument(relations=[{"name": "slope", "operation": "tan_deg", "a": angle}])


@pytest.mark.parametrize(
    ("relation", "message"),
    [
        ({"operation": "sqrt", "a": -1}, "nonnegative"),
        ({"operation": "atan2_deg", "a": 0, "b": 0}, "nonzero vector"),
        ({"operation": "divide", "a": 1, "b": 0}, "division by zero"),
        ({"operation": "multiply", "a": 1e308, "b": 1e308}, "must be finite"),
        ({"operation": "add", "a": "later", "b": 1}, "ordered"),
    ],
)
def test_invalid_domains_and_dependencies_refuse_before_cad(relation, message):
    with pytest.raises(ValueError, match=message):
        DesignDocument(relations=[{"name": "answer", **relation}])


@pytest.mark.parametrize("operation", ["sin_deg", "cos_deg", "tan_deg", "sqrt"])
@pytest.mark.parametrize("ignored_operand", [1, -1, "unused"])
def test_unary_operations_never_ignore_a_supplied_operand(operation, ignored_operand):
    with pytest.raises(ValueError, match="Unary variable relations"):
        VariableRelation(name="answer", operation=operation, a=4, b=ignored_operand)


@pytest.mark.parametrize(
    "operation", ["add", "subtract", "multiply", "divide", "max", "min", "atan2_deg"]
)
def test_binary_operations_still_require_b(operation):
    with pytest.raises(ValueError, match="explicit b operand"):
        VariableRelation(name="answer", operation=operation, a=4)


def test_nonfinite_mutated_named_operand_cannot_become_a_derived_value():
    doc = DesignDocument(
        variables={"x": 1},
        relations=[{"name": "answer", "operation": "min", "a": "x", "b": 2}],
    )
    doc.variables["x"] = math.nan
    with pytest.raises(ValueError, match="operands must be finite"):
        doc.values()


def test_legacy_binary_serialization_has_no_new_fields_or_defaults():
    legacy = {"name": "total", "operation": "add", "a": "x", "b": 2.0}
    assert VariableRelation.model_validate(legacy).model_dump(mode="json") == legacy
    doc = DesignDocument(variables={"x": 3}, relations=[legacy])
    assert doc.model_dump(mode="json")["relations"] == [legacy]
    assert doc.values()["total"] == 5


def test_relation_discovery_defines_degrees_domains_and_editable_example():
    reference = construction_reference("relations")
    assert set(reference["unary"]) == {"sin_deg", "cos_deg", "tan_deg", "sqrt"}
    assert "a=y and b=x" in reference["angles"]
    assert "literal zero" in reference["rules"]
    doc = DesignDocument.model_validate(construction_reference("countersunk_plate")["document"])
    assert doc.values()["sink_depth"] == pytest.approx(3)
    doc.variables["included_angle"] = 82
    assert doc.values()["sink_depth"] == pytest.approx(3 / math.tan(math.radians(41)))
