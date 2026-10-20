"""Pure selected-neighborhood requirements. No CAD or worker execution."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.workspace import geometry_neighbors_contract as subject


def entity(kind, digest, identifier):
    return {
        "id": identifier,
        "kind": kind,
        "signature": digest * 64,
        "body_index": 0,
        "body_signature": "c" * 64,
        "center": [0.0, 0.0, 0.0],
        "min": [-1.0, -1.0, 0.0],
        "max": [1.0, 1.0, 0.0],
        "measure": 4.0,
        "unit": "mm^2" if kind == "face" else "mm",
        **(
            {"surface": "GeomType.PLANE", "normal": [0.0, 0.0, 1.0]}
            if kind == "face"
            else {"curve": "GeomType.LINE"}
        ),
    }


def result():
    return {
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "selector": {
            "requested": {"body_index": 0, "kind": "face", "signature": "f" * 64},
            "body_index": 0,
            "body_signature": "c" * 64,
        },
        "source_hashes": dict.fromkeys(subject.SOURCE_NAMES, "1" * 64),
        "saved_engine_fingerprint": "2" * 64,
        "query_engine_fingerprint": "3" * 64,
        "focal": entity("face", "f", "face_focal"),
        "relation": "face_boundary_edges",
        "neighbors": [
            {"entity": entity("edge", "d", "edge_first"), "incidence_count": 1},
            {"entity": entity("edge", "e", "edge_seam"), "incidence_count": 2},
        ],
        "offset": 0,
        "limit": 40,
        "total_neighbors": 2,
        "total_incidences": 3,
        "next_offset": None,
        "native_completed": True,
        "coverage": "complete_selected_neighborhood",
        "scope": subject.SCOPE,
    }


def test_native_free_contract_imports():
    tree = ast.parse(Path(subject.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(
        module and module.startswith(("OCP", "build123d", "forge_mind.workspace.kernel"))
        for module in modules
    )


@pytest.mark.parametrize(
    "change",
    [
        {"body_index": None},
        {"occurrence_path": "part"},
        {"body_index": True},
        {"body_index": -1},
        {"body_index": "0"},
        {"kind": "body"},
        {"signature": "A" * 64},
        {"signature": "short"},
        {"other": True},
    ],
)
def test_strict_selection_refuses_wrong_identity(change):
    value = {"body_index": 0, "kind": "face", "signature": "a" * 64, **change}
    with pytest.raises(ValidationError):
        subject.NeighborSelector.model_validate(value)


def test_occurrence_selector_and_resolved_body_mismatch():
    selected = subject.NeighborSelector(
        occurrence_path="nested/left", kind="edge", signature="a" * 64
    )
    assert (
        subject.ResolvedNeighborSelector(
            requested=selected, body_index=4, body_signature="b" * 64
        ).body_index
        == 4
    )
    indexed = subject.NeighborSelector(body_index=2, kind="edge", signature="a" * 64)
    with pytest.raises(ValidationError, match="requested index"):
        subject.ResolvedNeighborSelector(requested=indexed, body_index=3, body_signature="b" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("offset", -1),
        ("offset", True),
        ("offset", "0"),
        ("limit", 0),
        ("limit", 101),
        ("limit", 1.0),
    ],
)
def test_paging_arguments_are_bounded_and_strict(field, value):
    with pytest.raises(ValidationError):
        subject.NeighborQuery.model_validate(
            {"selector": result()["selector"]["requested"], field: value}
        )


def test_seam_count_and_transport_preservation_without_input_mutation():
    value = result()
    before = json.dumps(value, sort_keys=True)
    selected = subject.NeighborSelector.model_validate(value["selector"]["requested"])
    output = subject.neighbors_tool_result(value, "a" * 32, "b" * 32, selected, 0, 40)
    assert output["total_neighbors"] == 2 and output["total_incidences"] == 3
    assert [row["incidence_count"] for row in output["neighbors"]] == [1, 2]
    assert output["focal"]["signature"] == "f" * 64
    assert output["frame"] == subject.FRAME and output["operation"] == "topology_neighbors"
    assert json.dumps(value, sort_keys=True) == before


def test_verified_zero_and_out_of_range_pages_are_not_missing_coverage():
    empty = result()
    empty.update(neighbors=[], total_neighbors=0, total_incidences=0)
    assert subject.NeighborResult.model_validate(empty).next_offset is None
    beyond = result()
    beyond.update(offset=20, neighbors=[])
    parsed = subject.NeighborResult.model_validate(beyond)
    assert parsed.total_incidences == 3 and parsed.total_neighbors == 2 and not parsed.neighbors
    first = result()
    first.update(limit=1, neighbors=first["neighbors"][:1], next_offset=1)
    assert subject.NeighborResult.model_validate(first).next_offset == 1
    last = result()
    last.update(offset=1, limit=1, neighbors=last["neighbors"][1:])
    assert subject.NeighborResult.model_validate(last).neighbors[0].incidence_count == 2


@pytest.mark.parametrize("offset", [0, 1, 99])
def test_zero_unique_neighbors_requires_zero_incidences_at_every_page(offset):
    value = result()
    value.update(neighbors=[], total_neighbors=0, total_incidences=1, offset=offset)
    with pytest.raises(ValidationError, match="empty native neighborhood"):
        subject.NeighborResult.model_validate(value)


@pytest.mark.parametrize(
    "tamper",
    [
        "focal_kind",
        "focal_signature",
        "wrong_body",
        "wrong_neighbor_kind",
        "duplicate_id",
        "duplicate_signature",
        "bad_unit",
        "nan",
        "inverted_bounds",
        "bad_scope",
        "bad_coverage",
        "not_completed",
        "bool_completed",
        "wrong_relation",
        "wrong_total",
        "wrong_incidence",
        "bool_incidence",
        "bad_next",
        "extra_source",
        "wrong_page",
        "mesh",
    ],
)
def test_partial_or_mismatched_response_refused(tamper):
    value = result()
    neighbor = value["neighbors"][0]
    if tamper == "focal_kind":
        value["focal"]["kind"] = "edge"
    elif tamper == "focal_signature":
        value["focal"]["signature"] = "a" * 64
    elif tamper == "wrong_body":
        neighbor["entity"]["body_index"] = 1
    elif tamper == "wrong_neighbor_kind":
        neighbor["entity"] = entity("face", "d", "face_wrong")
    elif tamper == "duplicate_id":
        value["neighbors"][1]["entity"]["id"] = neighbor["entity"]["id"]
    elif tamper == "duplicate_signature":
        value["neighbors"][1]["entity"]["signature"] = neighbor["entity"]["signature"]
    elif tamper == "bad_unit":
        neighbor["entity"]["unit"] = "mm^2"
    elif tamper == "nan":
        neighbor["entity"]["center"][0] = float("nan")
    elif tamper == "inverted_bounds":
        neighbor["entity"]["min"][0] = 2.0
    elif tamper == "bad_scope":
        value["scope"] = "Smooth and safe"
    elif tamper == "bad_coverage":
        value["coverage"] = "incomplete"
    elif tamper == "not_completed":
        value["native_completed"] = False
    elif tamper == "bool_completed":
        value["native_completed"] = 1
    elif tamper == "wrong_relation":
        value["relation"] = "edge_incident_faces"
    elif tamper == "wrong_total":
        value["total_neighbors"] = 3
    elif tamper == "wrong_incidence":
        value["total_incidences"] = 2
    elif tamper == "bool_incidence":
        neighbor["incidence_count"] = True
    elif tamper == "bad_next":
        value["next_offset"] = 2
    elif tamper == "extra_source":
        value["source_hashes"]["other.step"] = "a" * 64
    elif tamper == "wrong_page":
        value["offset"] = 1
    else:
        neighbor["entity"]["mesh"] = {"triangles": []}
    with pytest.raises(ValidationError):
        subject.NeighborResult.model_validate(value)


def test_tool_result_binds_requested_ids_selection_page_and_bytes(monkeypatch):
    value = result()
    selector = subject.NeighborSelector.model_validate(value["selector"]["requested"])
    for project, revision, offset, limit in [
        ("c" * 32, "b" * 32, 0, 40),
        ("a" * 32, "c" * 32, 0, 40),
        ("a" * 32, "b" * 32, 1, 40),
        ("a" * 32, "b" * 32, 0, 20),
    ]:
        with pytest.raises(ValueError, match="requested"):
            subject.neighbors_tool_result(value, project, revision, selector, offset, limit)
    other = selector.model_copy(update={"signature": "e" * 64})
    with pytest.raises(ValueError, match="requested"):
        subject.neighbors_tool_result(value, "a" * 32, "b" * 32, other, 0, 40)
    monkeypatch.setattr(subject, "MAX_RESPONSE_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        subject.neighbors_tool_result(deepcopy(value), "a" * 32, "b" * 32, selector, 0, 40)
