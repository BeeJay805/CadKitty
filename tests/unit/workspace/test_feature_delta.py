"""Scoped feature deltas preserve source and reject unapplied graph changes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.workspace.document import DesignDocument, Feature
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.feature_delta import FeatureChange, FeatureDelta, apply_feature_delta


def assembly() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "name": "Retained assembly",
            "variables": {"width": 30, "bore": 4},
            "relations": [{"name": "radius", "operation": "divide", "a": "bore", "b": 2}],
            "delivery_guidance": {"installation": "Install the pin and retain it."},
            "motion_rig": {
                "drivers": [
                    {"id": "travel", "name": "Travel", "unit": "mm", "minimum": 0, "maximum": 10}
                ],
                "groups": [
                    {
                        "id": "slide",
                        "name": "Slide",
                        "driver": "travel",
                        "kind": "slide",
                        "occurrences": ["pair/pin"],
                    }
                ],
            },
            "assembly": {
                "parts": [
                    {
                        "id": "plate",
                        "name": "Plate",
                        "source": "Retain imported machining reference",
                        "features": [
                            {
                                "id": "imported",
                                "name": "Imported reference",
                                "kind": "import",
                                "asset_id": "d" * 32,
                            },
                            {
                                "id": "body",
                                "name": "Plate blank",
                                "kind": "box",
                                "params": {"y": 12, "z": 5},
                                "bindings": {"x": "width"},
                            },
                        ],
                        "output": "body",
                        "ports": [{"name": "axis", "kind": "bore", "diameter": "bore"}],
                        "production": {
                            "role": "make",
                            "process": "cnc",
                            "material": {"name": "Aluminum", "grade": "Declared grade"},
                        },
                    },
                    {
                        "id": "pin",
                        "name": "Bought pin",
                        "features": [
                            {
                                "id": "body",
                                "name": "Pin",
                                "kind": "cylinder",
                                "params": {"height": 10},
                                "bindings": {"radius": "radius"},
                            }
                        ],
                        "output": "body",
                        "ports": [{"name": "axis", "kind": "shaft", "diameter": "bore"}],
                        "production": {
                            "role": "buy",
                            "procurement": {"supplier": "Declared supplier", "part_number": "PIN"},
                        },
                    },
                ],
                "subassemblies": [
                    {
                        "id": "pair",
                        "name": "Pin connection",
                        "children": [
                            {"id": "plate", "definition": "plate"},
                            {"id": "pin", "definition": "pin"},
                        ],
                    }
                ],
                "roots": [
                    {"id": "pair", "definition": "pair"},
                    {"id": "spare", "definition": "plate", "position": [40, 0, 0]},
                ],
                "mates": [{"id": "pin_fit", "a": "pair/plate.axis", "b": "pair/pin.axis"}],
            },
        }
    )


def flat(*, implicit: bool = False) -> DesignDocument:
    return DesignDocument(
        name="Plate",
        features=[Feature(id="body", name="Blank", kind="box", params={"x": 30, "y": 12, "z": 5})],
        outputs=[] if implicit else ["body"],
    )


def basis(document: DesignDocument) -> EditBasis:
    return EditBasis(
        revision_id="a" * 32,
        document_hash=document.fingerprint(),
        plan_version=1,
        plan_hash="b" * 64,
        contract_hash="c" * 64,
        input_version=0,
        source_fingerprint="d" * 64,
    )


def hole_change(part_id: str | None, *, output: str | None = "drilled") -> FeatureChange:
    return FeatureChange(
        part_id=part_id,
        append=[
            Feature(
                id="hole", name="Hole cutter", kind="cylinder", params={"radius": 2, "height": 5}
            ),
            Feature(id="drilled", name="Drilled plate", kind="cut", inputs=["body", "hole"]),
        ],
        output=output,
    )


def test_append_with_explicit_output_preserves_every_unrelated_record():
    document = assembly()
    before = document.model_dump()
    delta = FeatureDelta(basis=basis(document), changes=[hole_change("plate")])
    result = apply_feature_delta(document, delta)
    assert document.model_dump() == before
    actual = result.model_dump()
    expected = document.model_dump()
    expected["assembly"]["parts"][0]["features"].extend(
        feature.model_dump() for feature in delta.changes[0].append
    )
    expected["assembly"]["parts"][0]["output"] = "drilled"
    assert actual == expected
    assert result.assembly.parts[0].features[0].asset_id == "d" * 32
    assert result.assembly.parts[1].production.procurement.part_number == "PIN"


def test_update_keeps_position_output_and_all_other_features():
    document = assembly()
    replacement = (
        document.assembly.parts[0].features[1].model_copy(update={"params": {"y": 14, "z": 5}})
    )
    result = apply_feature_delta(
        document,
        FeatureDelta(
            basis=basis(document), changes=[FeatureChange(part_id="plate", update=[replacement])]
        ),
    )
    assert [feature.id for feature in result.assembly.parts[0].features] == ["imported", "body"]
    assert result.assembly.parts[0].output == "body"
    assert result.assembly.parts[0].features[1].params == {"y": 14, "z": 5}
    assert document.assembly.parts[0].features[1].params == {"y": 12, "z": 5}
    result.assembly.parts[0].ports[0].diameter = 99
    assert document.assembly.parts[0].ports[0].diameter == "bore"


def test_flat_append_without_output_does_not_rewire_finished_geometry():
    document = flat()
    result = apply_feature_delta(
        document,
        FeatureDelta(basis=basis(document), changes=[hole_change(None, output=None)]),
    )
    assert result.outputs == ["body"]
    assert result.features[-1].id == "drilled"


def test_implicit_flat_output_requires_explicit_selection_for_append():
    document = flat(implicit=True)
    before = document.model_dump_json()
    with pytest.raises(ValueError, match="implicit outputs"):
        apply_feature_delta(
            document,
            FeatureDelta(basis=basis(document), changes=[hole_change(None, output=None)]),
        )
    assert document.model_dump_json() == before
    result = apply_feature_delta(
        document, FeatureDelta(basis=basis(document), changes=[hole_change(None)])
    )
    assert result.outputs == ["drilled"]


@pytest.mark.parametrize(
    "mode", ["append_existing", "update_missing", "missing_input", "forward_input"]
)
def test_invalid_feature_edit_preserves_original_exactly(mode):
    document = flat()
    body = document.features[0].model_copy(deep=True)
    if mode == "append_existing":
        change = FeatureChange(append=[body])
    elif mode == "update_missing":
        change = FeatureChange(update=[body.model_copy(update={"id": "missing"})])
    elif mode == "missing_input":
        change = FeatureChange(
            append=[Feature(id="cut", name="Invalid cut", kind="cut", inputs=["body", "missing"])]
        )
    else:
        change = FeatureChange(
            append=[
                Feature(id="cut", name="Forward cut", kind="cut", inputs=["body", "hole"]),
                Feature(id="hole", name="Hole", kind="cylinder", params={"radius": 2, "height": 5}),
            ]
        )
    before = document.model_dump_json()
    with pytest.raises(ValueError):
        apply_feature_delta(document, FeatureDelta(basis=basis(document), changes=[change]))
    assert document.model_dump_json() == before


def test_later_part_failure_cannot_apply_an_earlier_valid_change():
    document = assembly()
    before = document.model_dump_json()
    with pytest.raises(ValueError, match="do not exist"):
        apply_feature_delta(
            document,
            FeatureDelta(
                basis=basis(document),
                changes=[
                    hole_change("plate"),
                    FeatureChange(
                        part_id="pin", update=[Feature(id="unknown", name="Unknown", kind="box")]
                    ),
                ],
            ),
        )
    assert document.model_dump_json() == before


@pytest.mark.parametrize("part_id", [None, "unknown"])
def test_assembly_requires_existing_explicit_scope(part_id):
    document = assembly()
    with pytest.raises(ValueError, match="existing part scope"):
        apply_feature_delta(
            document, FeatureDelta(basis=basis(document), changes=[hole_change(part_id)])
        )


def test_flat_rejects_named_scope():
    document = flat()
    with pytest.raises(ValueError, match="null part scope"):
        apply_feature_delta(
            document, FeatureDelta(basis=basis(document), changes=[hole_change("main")])
        )


@pytest.mark.parametrize("target", ["unknown", "hole"])
def test_unknown_or_suppressed_output_is_rejected(target):
    document = flat()
    change = hole_change(None, output=target)
    if target == "hole":
        change.append[0].suppressed = True
    before = document.model_dump_json()
    with pytest.raises(ValueError):
        apply_feature_delta(document, FeatureDelta(basis=basis(document), changes=[change]))
    assert document.model_dump_json() == before


def test_suppressed_dependency_cannot_hide_behind_unsuppressed_output():
    document = flat()
    change = hole_change(None)
    change.append[0].suppressed = True
    with pytest.raises(ValueError, match="suppressed: hole"):
        apply_feature_delta(document, FeatureDelta(basis=basis(document), changes=[change]))


@pytest.mark.parametrize(
    "problem", ["empty", "duplicate_id", "append_update_overlap", "duplicate_scope"]
)
def test_ambiguous_requests_are_rejected_before_reduction(problem):
    document = flat()
    body = document.features[0]
    with pytest.raises(ValidationError):
        if problem == "empty":
            FeatureDelta(basis=basis(document), changes=[])
        elif problem == "duplicate_id":
            FeatureChange(append=[body, body])
        elif problem == "append_update_overlap":
            FeatureChange(append=[body], update=[body])
        else:
            FeatureDelta(basis=basis(document), changes=[hole_change(None), hole_change(None)])


def test_stale_hash_and_unbound_source_are_rejected():
    document = flat()
    current = basis(document)
    for stale in (
        current.model_copy(update={"document_hash": "0" * 64}),
        current.model_copy(update={"revision_id": None, "document_hash": None}),
    ):
        before = document.model_dump_json()
        with pytest.raises(ValueError, match="document basis changed"):
            apply_feature_delta(document, FeatureDelta(basis=stale, changes=[hole_change(None)]))
        assert document.model_dump_json() == before


def test_request_cannot_write_ports_or_arbitrary_document_fields():
    with pytest.raises(ValidationError):
        FeatureChange.model_validate({"output": "body", "ports": []})
    with pytest.raises(ValidationError):
        FeatureChange.model_validate({"output": "body", "remove": ["body"]})
