"""Detailed products retain bounded exact geometry and explicit selection degradation."""

import time

import build123d as b
import pytest

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.topology import topology_index


def document(features, output):
    return DesignDocument.model_validate({"features": features, "outputs": [output]})


def feature(id, kind, params, inputs=None):
    return {"id": id, "name": id, "kind": kind, "params": params, "inputs": inputs or []}


def test_dense_square_vent_pattern_and_body_picking():
    doc = document(
        [
            feature("panel", "box", {"x": 44, "y": 44, "z": 1}),
            feature("pore", "box", {"x": 1, "y": 1, "z": 3}),
            feature("row", "pattern", {"count": 20, "x": 2}, ["pore"]),
            feature("field", "pattern", {"count": 20, "y": 2}, ["row"]),
            feature("centered", "transform", {"x": -19, "y": -19}, ["field"]),
            feature("vent", "cut", {}, ["panel", "centered"]),
        ],
        "vent",
    )
    started = time.monotonic()
    shape, history = compile_document(doc, {})
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(44 * 44 - 400)
    assert len(shape.faces()) > 1200
    resource = next(r["resources"] for r in history if r["id"] == "field")
    assert resource["faces"] == 2400
    assert resource["pattern_preallocation_estimate"]["instances"] == 400
    index = topology_index(shape)
    assert index["selection_scope"] == "body"
    assert index["detail_policy"]["mode"] == "body_index"
    assert {r["kind"] for r in index["entities"]} == {"body"}
    assert len(index["meshes"]) == 1
    assert index["pick_tessellation"]["triangles"] <= 300000
    assert index["pick_tessellation"]["relative_linear_deflection"] == 0.2
    assert index["topology_counts"]["faces"] == len(shape.faces())
    print(
        {
            "seconds": round(time.monotonic() - started, 3),
            "faces": len(shape.faces()),
            "edges": len(shape.edges()),
        }
    )


def test_exponential_pattern_refused_before_compound(monkeypatch):
    import forge_mind.workspace.kernel as kernel

    monkeypatch.setattr(kernel, "PATTERN_INSTANCE_LIMIT", 512)

    doc = document(
        [
            feature("pore", "box", {"x": 1, "y": 1, "z": 1}),
            feature("row", "pattern", {"count": 128, "x": 2}, ["pore"]),
            feature("huge", "pattern", {"count": 128, "y": 2}, ["row"]),
        ],
        "huge",
    )
    real = kernel.b.Compound
    allocations = []

    def tracked(*args, **kwargs):
        allocations.append(len(kwargs.get("children", [])))
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel.b, "Compound", tracked)
    with pytest.raises(ValueError, match="expanded pattern instances requires 16384, limit is 512"):
        compile_document(doc, {})
    assert allocations and max(allocations) == 128


def test_ordinary_part_preserves_face_edge_picking():
    index = topology_index(b.Box(4, 5, 6))
    assert index["selection_scope"] == "face_edge_body"
    assert len(index["meshes"]) == 6
    assert len([r for r in index["entities"] if r["kind"] == "edge"]) == 12
    assert all(len(r["polyline"]) == 33 for r in index["entities"] if r["kind"] == "edge")


def test_configured_pattern_count_policy_is_enforced(monkeypatch):
    import forge_mind.workspace.kernel as kernel

    monkeypatch.setattr(kernel, "PATTERN_COUNT_LIMIT", 128)
    doc = document(
        [
            feature("a", "box", {"x": 1, "y": 1, "z": 1}),
            feature("p", "pattern", {"count": 129}, ["a"]),
        ],
        "p",
    )
    with pytest.raises(ValueError, match="requires 129, limit is 128"):
        compile_document(doc, {})


def test_pattern_above_previous_count_ceiling_without_operator_policy(monkeypatch):
    import forge_mind.workspace.kernel as kernel

    monkeypatch.setattr(kernel, "PATTERN_COUNT_LIMIT", None)
    shape, _ = compile_document(
        document(
            [
                feature("a", "box", {"x": 1, "y": 1, "z": 1}),
                feature("p", "pattern", {"count": 129, "x": 2}, ["a"]),
            ],
            "p",
        ),
        {},
    )
    assert len(shape.solids()) == 129
