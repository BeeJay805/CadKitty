"""Literal count regression; no native CAD/model execution or invented source facts."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

from forge_mind.workspace.intent_attribution import (
    ContractDraft,
    IntentAttribution,
    validate_attribution,
)
from forge_mind.workspace.intent_compilation import _retention_diagnostics
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.source_numbers import (
    COUNT_WORDS,
    POSITIVE_COUNT_WORDS,
    count_word_values,
    source_word_values,
)
from forge_mind.workspace.verification_admission import _inventory_source_item, _source_targets
from forge_mind.workspace.verification_models import NominalInventoryItem, NominalThroughHolePattern


def attribution(quote, *, value=2, unit="holes"):
    return IntentAttribution.model_validate(
        {
            "subjects": [
                {"id": "plate", "label": "Plate", "role": "designed", "source_quote": "plate"}
            ],
            "facts": [
                {
                    "id": "count",
                    "subject_id": "plate",
                    "property": "visible count",
                    "value": float(value),
                    "unit": unit,
                    "source_quote": quote,
                    "requirement_ids": ["duty"],
                }
            ],
            "requirement_subjects": {"duty": ["plate"]},
        }
    )


def source_contract(quote, metric="unmeasured", value=None):
    return Contract(
        requirements=[
            Requirement(
                id="duty", text="Retained duty", source_quote=quote, metric=metric, value=value
            )
        ]
    )


def pure_ground_contract():
    """Execute the unchanged real function without measure's native module imports.

    The full measure module imports build123d transitively. This bounded test
    compiles its exact ground_contract AST with its real pure dependencies. The
    full module/pipeline remains covered by root's qualified backend gate.
    """
    # tests/unit/workspace -> repository is three parent steps above tests.
    source = Path(__file__).resolve().parents[3] / "src/forge_mind/workspace/measure.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ground_contract"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "math": math,
        "re": re,
        "Contract": Contract,
        "Any": object,
        "count_word_values": count_word_values,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["ground_contract"]


def test_original_both_visible_holes_fact_retains_exact_quote_and_physical_duty():
    quote = "Both visible holes must remain open"
    source = "Copy this plate. " + quote + "."
    contract = source_contract(quote)
    attrs = attribution(quote)
    before = (attrs.model_dump(), contract.model_dump())
    validate_attribution(attrs, contract, source)
    assert (attrs.model_dump(), contract.model_dump()) == before
    assert contract.requirements[0].metric == "unmeasured"
    assert contract.requirements[0].critical


@pytest.mark.parametrize(
    "quote,unit,value",
    [
        ("Both visible holes", "holes", 3),
        ("Both mm dimensions", "mm", 2),
        ("Both centimetres", "centimetres", 2),
        ("Both Hz signals", "Hz", 2),
        ("Both unknown_unit", "unknown_unit", 2),
        ("Bothy holes", "holes", 2),
    ],
)
def test_pair_word_cannot_invent_other_counts_or_physical_dimensions(quote, unit, value):
    with pytest.raises(ValueError, match="must occur literally"):
        validate_attribution(
            attribution(quote, value=value, unit=unit), source_contract(quote), "plate " + quote
        )


def test_source_quote_and_unit_provenance_still_required():
    quote = "Both holes"
    with pytest.raises(ValueError, match="quote must occur exactly"):
        validate_attribution(attribution(quote), source_contract(quote), "plate TWO holes")
    with pytest.raises(ValueError, match="unit must occur literally"):
        validate_attribution(
            attribution(quote, unit="screws"), source_contract(quote), "plate " + quote
        )


@pytest.mark.parametrize(
    "word,value",
    [("zero", 0), ("no", 0), ("one", 1), ("single", 1), ("two", 2), ("four", 4), ("twelve", 12)],
)
def test_legacy_word_values_and_positive_grammar_scope_are_retained(word, value):
    assert source_word_values(word + " holes", unit="holes") == [float(value)]
    assert source_word_values(word + " mm", unit="mm") == [float(value)]
    assert (word in POSITIVE_COUNT_WORDS) == (value > 0)


def test_valid_pair_fact_value_is_protected_during_unrelated_source_correction():
    quote = "Both visible holes"
    source = "Copy this plate. " + quote
    previous = ContractDraft(contract=source_contract(quote), attribution=attribution(quote))
    assert _retention_diagnostics(previous, previous.model_copy(deep=True), source) == []
    changed = previous.model_copy(deep=True)
    changed.attribution.facts[0].value = 3.0
    diagnostics = _retention_diagnostics(previous, changed, source)
    assert any(
        row["field"] == "attribution.facts.count" and row["code"] == "changed_source_attribution"
        for row in diagnostics
    )


def test_shared_vocabulary_is_immutable_and_does_not_match_subwords():
    assert COUNT_WORDS["both"] == POSITIVE_COUNT_WORDS["both"] == 2
    assert count_word_values("Both holes, TWO bores, nobody, bothy") == [2.0, 2.0]
    with pytest.raises(TypeError):
        COUNT_WORDS["both"] = 3


@pytest.mark.parametrize("metric", ["solid_count", "hole_count"])
def test_actual_scalar_grounder_accepts_both_only_for_count_metrics(metric):
    quote = "Both holes" if metric == "hole_count" else "Both solids"
    grounded = pure_ground_contract()(source_contract(quote, metric, 2.0), quote)
    assert grounded.requirements[0].metric == metric
    wrong = pure_ground_contract()(source_contract(quote, metric, 3.0), quote)
    assert wrong.requirements[0].metric == "unmeasured"


def test_actual_scalar_grounder_does_not_derive_two_mm_from_both():
    quote = "Both mm dimensions"
    assert (
        pure_ground_contract()(source_contract(quote, "extent_x", 2.0), quote)
        .requirements[0]
        .metric
        == "unmeasured"
    )
    for word in ("no", "zero"):
        text = word + " holes"
        assert (
            pure_ground_contract()(source_contract(text, "hole_count", 0.0), text)
            .requirements[0]
            .metric
            == "hole_count"
        )


@pytest.mark.parametrize("metric,noun", [("hole_count", "holes"), ("solid_count", "solids")])
@pytest.mark.parametrize("word", ["Both", "Two"])
@pytest.mark.parametrize("measurement,wrong_count", [("5 cm", 20.0), ("5 inches", 127.0)])
def test_count_target_cannot_acquire_a_length_conversion(
    metric, noun, word, measurement, wrong_count
):
    quote = f"{word} {noun} are {measurement} apart."
    ground = pure_ground_contract()
    assert ground(source_contract(quote, metric, 2.0), quote).requirements[0].metric == metric
    assert (
        ground(source_contract(quote, metric, wrong_count), quote).requirements[0].metric
        == "unmeasured"
    )


@pytest.mark.parametrize(
    "metric,quote,value",
    [("extent_x", "Width 5 cm", 50.0), ("hole_diameter", "Diameter 5 inches", 127.0)],
)
def test_length_metric_unit_conversions_remain_supported(metric, quote, value):
    assert (
        pure_ground_contract()(source_contract(quote, metric, value), quote).requirements[0].metric
        == metric
    )


def test_typed_inventory_accepts_definite_pair_without_broadening_zero_scope():
    item = NominalInventoryItem(
        subject_id="plate", subject_role="designed", required_name="plate_pair", quantity=2
    )
    assert _inventory_source_item(item, "Make both nominal CAD plates named plate_pair.")
    for word in ("no", "zero", "three", "bothy"):
        with pytest.raises(ValueError, match="quantity"):
            _inventory_source_item(item, f"Make {word} nominal CAD plates named plate_pair.")


def test_typed_hole_count_uses_shared_pair_word_and_still_rejects_wrong_count():
    quote = "Both uniform cylindrical through-holes parallel to local Z: diameter 5 mm at local XY (-5, 0) and (5, 0) mm."
    criterion = NominalThroughHolePattern(
        kind="nominal_through_hole_pattern",
        scope="nominal_cad",
        subject_id="plate",
        subject_role="designed",
        frame="definition_local",
        axis="z",
        count_policy="exact",
        source_quote=quote,
        holes=[
            {
                "center_mm": [x, 0],
                "diameter_mm": 5.0,
                "diameter_tolerance_mm": 0.0,
                "position_tolerance_mm": 0.0,
            }
            for x in (-5.0, 5.0)
        ],
    )
    requirement = Requirement(
        id="pattern", text="Nominal CAD pattern", source_quote=quote, verification=criterion
    )
    _source_targets(requirement, "Nominal CAD only. " + quote)
    for word in ("three", "no", "zero", "bothy"):
        changed = requirement.model_copy(update={"source_quote": quote.replace("Both", word)})
        with pytest.raises(ValueError, match="count"):
            _source_targets(changed, "Nominal CAD only. " + changed.source_quote)
