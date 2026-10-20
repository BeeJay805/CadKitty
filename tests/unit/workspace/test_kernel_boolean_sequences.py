"""Boolean intermediates preserve authored material operations and operand order."""

import itertools
import math

import pytest

from forge_mind.workspace import kernel
from forge_mind.workspace.document import DesignDocument


def feature(name, kind, *, inputs=(), **params):
    return {"id": name, "name": name, "kind": kind, "inputs": list(inputs), "params": params}


def document(features):
    return DesignDocument.model_validate({"features": features, "outputs": [features[-1]["id"]]})


def bridge_features():
    return [
        feature("end", "box", x=2, y=2, z=2),
        feature("left", "transform", inputs=["end"], x=-4),
        feature("right", "transform", inputs=["end"], x=4),
        feature("bridge", "box", x=8, y=2, z=2),
    ]


@pytest.mark.parametrize("operands", list(itertools.permutations(["left", "right", "bridge"])))
@pytest.mark.slow
def test_native_bridge_union_all_authored_orders_fuse_the_same_material(operands):
    design = document([*bridge_features(), feature("joined", "union", inputs=operands)])
    shape, history = kernel.compile_document(design, {})
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(40.0, abs=1e-7)
    assert history[-1]["volume_mm3"] == pytest.approx(40.0, abs=1e-7)
    assert design.features[-1].inputs == list(operands)


@pytest.mark.slow
def test_native_final_disjoint_union_retains_existing_multiple_solid_semantics():
    shape, history = kernel.compile_document(
        document([*bridge_features(), feature("separate", "union", inputs=["left", "right"])]),
        {},
    )
    assert shape.is_valid and len(shape.solids()) == 2
    assert shape.volume == pytest.approx(16.0, abs=1e-7)
    assert history[-1]["solid_count"] == 2


def split_features():
    return [
        feature("bar", "box", x=10, y=2, z=2),
        feature("slot", "box", x=2, y=4, z=4),
        feature("islands", "cut", inputs=["bar", "slot"]),
    ]


@pytest.mark.slow
@pytest.mark.parametrize("continuation", ["none", "cut", "union", "intersect"])
def test_native_multi_island_cut_remains_a_boolean_operand(continuation):
    features = split_features()
    volume, count = 32.0, 2
    if continuation == "cut":
        features += [
            feature("second_slot", "transform", inputs=["slot"], x=-3),
            feature("trimmed", "cut", inputs=["islands", "second_slot"]),
        ]
        volume, count = 24.0, 3
    elif continuation == "union":
        features += [feature("restored", "union", inputs=["islands", "slot"])]
        # The cutter is taller/wider than the bar: adding it retains that authored material.
        volume, count = 64.0, 1
    elif continuation == "intersect":
        features += [
            feature("clip", "box", x=10, y=1, z=2),
            feature("clipped", "intersect", inputs=["islands", "clip"]),
        ]
        volume, count = 16.0, 2
    shape, history = kernel.compile_document(document(features), {})
    assert shape.is_valid and len(shape.solids()) == count
    assert shape.volume == pytest.approx(volume, abs=1e-7)
    assert history[-1]["solid_count"] == count


@pytest.mark.slow
def test_native_sequential_cut_normalizes_split_before_the_next_cutter():
    features = [
        *split_features()[:2],
        feature("second_slot", "transform", inputs=["slot"], x=-3),
        feature("trimmed", "cut", inputs=["bar", "slot", "second_slot"]),
    ]
    shape, _ = kernel.compile_document(document(features), {})
    assert len(shape.solids()) == 3 and shape.volume == pytest.approx(24.0, abs=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("operation", ["intersect", "cut"])
def test_native_empty_boolean_remains_an_explicit_geometry_failure(operation):
    features = bridge_features()
    if operation == "cut":
        features.append(feature("identical_left", "transform", inputs=["left"]))
    inputs = ["left", "right"] if operation == "intersect" else ["left", "identical_left"]
    with pytest.raises(ValueError, match="valid positive-volume geometry"):
        kernel.compile_document(
            document([*features, feature("empty", operation, inputs=inputs)]), {}
        )


@pytest.mark.slow
def test_native_miniature_rod_fuses_separated_ends_before_drilling():
    features = [
        feature("end", "cylinder", radius=2, height=2),
        feature("other_end", "transform", inputs=["end"], y=8),
        feature("shank", "box", x=2, y=8, z=2),
        feature("shank_at_centers", "transform", inputs=["shank"], y=4),
        feature("stock", "union", inputs=["end", "other_end", "shank_at_centers"]),
        feature("bore", "cylinder", radius=0.5, height=4),
        feature("other_bore", "transform", inputs=["bore"], y=8),
        feature("rod", "cut", inputs=["stock", "bore", "other_bore"]),
    ]
    shape, history = kernel.compile_document(document(features), {})
    # Two radius-2 ends plus the shank, minus both exact half-strip overlaps and bores.
    expected = 32 + 37 * math.pi / 3 - 4 * math.sqrt(3)
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(expected, abs=1e-7)
    assert next(row for row in history if row["id"] == "stock")["solid_count"] == 1


def test_pure_normalization_preserves_result_order_and_existing_shapes(monkeypatch):
    class Results(list):
        pass

    class Compound:
        def __init__(self, *, children):
            self.children = list(children)

    monkeypatch.setattr(kernel.b, "ShapeList", Results)
    monkeypatch.setattr(kernel.b, "Compound", Compound)
    first, second = object(), object()
    normalized = kernel._normalize_boolean_result(Results([first, second]))
    assert normalized.children == [first, second]
    assert kernel._normalize_boolean_result(first) is first


@pytest.mark.parametrize("empty", [None, []])
def test_pure_empty_boolean_cannot_become_a_candidate(monkeypatch, empty):
    class Results(list):
        pass

    monkeypatch.setattr(kernel.b, "ShapeList", Results)
    with pytest.raises(ValueError, match="valid positive-volume geometry"):
        kernel._normalize_boolean_result(None if empty is None else Results())
