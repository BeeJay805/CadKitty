"""Native saved-STEP neighbor requirements; root runs in the serialized CAD slot."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import build123d as cad
import pytest

from forge_mind.workspace import geometry_neighbors_native as subject
from forge_mind.workspace.assembly import occurrence_label
from forge_mind.workspace.geometry_neighbors_contract import (
    NeighborSelector,
    ResolvedNeighborSelector,
)
from forge_mind.workspace.topology import signature

pytestmark = pytest.mark.slow


def saved(tmp_path, shape):
    path = tmp_path / "actual.step"
    cad.export_step(shape, path)
    loaded = cad.import_step(path)
    bodies = list(loaded.solids())
    entities = []
    for index, body in enumerate(bodies):
        digest = signature(body)
        for kind, items in (("body", [body]), ("face", body.faces()), ("edge", body.edges())):
            for ordinal, item in enumerate(items):
                entities.append(
                    {
                        "id": f"{kind}_{index}_{ordinal}",
                        "kind": kind,
                        "signature": signature(item),
                        "body_index": index,
                        "body_signature": digest,
                    }
                )
    return (
        path,
        {"frame": "cad_z_up_mm", "entities": entities, "topology_counts": {"bodies": len(bodies)}},
        bodies,
    )


def select(topology, body_index, kind, digest, occurrence=None):
    body = next(
        row
        for row in topology["entities"]
        if row["kind"] == "body" and row["body_index"] == body_index
    )
    request = NeighborSelector(
        kind=kind,
        signature=digest,
        **({"body_index": body_index} if occurrence is None else {"occurrence_path": occurrence}),
    )
    return ResolvedNeighborSelector(
        requested=request, body_index=body_index, body_signature=body["signature"]
    )


def face_at(body, z):
    faces = [
        face
        for face in body.faces()
        if abs(face.bounding_box().min.Z - z) < 1e-7 and abs(face.bounding_box().max.Z - z) < 1e-7
    ]
    assert len(faces) == 1
    return faces[0]


def test_box_face_four_boundaries_and_edge_two_faces_with_native_facts(tmp_path):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_box(4, 5, 6))
    before = path.read_bytes(), json.dumps(topology, sort_keys=True)
    focal = face_at(bodies[0], 6)
    request = select(topology, 0, "face", signature(focal))
    result = subject.inspect_selected_neighbors(path, topology, request, 0, 40)
    assert result["relation"] == "face_boundary_edges"
    assert result["total_neighbors"] == result["total_incidences"] == 4
    assert sorted(row["entity"]["measure"] for row in result["neighbors"]) == pytest.approx(
        [4, 4, 5, 5]
    )
    assert all(
        row["incidence_count"] == 1 and row["entity"]["unit"] == "mm" for row in result["neighbors"]
    )
    assert result["focal"]["measure"] == pytest.approx(20)
    assert result["focal"]["min"] == pytest.approx([0, 0, 6])
    assert result["focal"]["max"] == pytest.approx([4, 5, 6])
    assert result["focal"]["normal"] == pytest.approx([0, 0, 1])
    edge = result["neighbors"][0]["entity"]
    reverse = subject.inspect_selected_neighbors(
        path, topology, select(topology, 0, "edge", edge["signature"]), 0, 40
    )
    assert reverse["total_neighbors"] == reverse["total_incidences"] == 2
    assert {row["entity"]["surface"] for row in reverse["neighbors"]} == {"GeomType.PLANE"}
    assert request.requested.signature in {
        row["entity"]["signature"] for row in reverse["neighbors"]
    }
    assert (path.read_bytes(), json.dumps(topology, sort_keys=True)) == before


def test_cylinder_seam_repeats_and_ring_has_two_distinct_faces(tmp_path):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_cylinder(2, 5))
    lateral = [face for face in bodies[0].faces() if "CYLINDER" in str(face.geom_type)]
    assert len(lateral) == 1
    selected = select(topology, 0, "face", signature(lateral[0]))
    result = subject.inspect_selected_neighbors(path, topology, selected, 0, 40)
    assert result["total_neighbors"] == 3 and result["total_incidences"] == 4
    assert sorted(row["incidence_count"] for row in result["neighbors"]) == [1, 1, 2]
    seam = next(row for row in result["neighbors"] if row["incidence_count"] == 2)
    assert "LINE" in seam["entity"]["curve"] and seam["entity"]["measure"] == pytest.approx(5)
    reverse = subject.inspect_selected_neighbors(
        path, topology, select(topology, 0, "edge", seam["entity"]["signature"]), 0, 40
    )
    assert reverse["total_neighbors"] == 1 and reverse["total_incidences"] == 2
    assert reverse["neighbors"][0]["entity"]["signature"] == selected.requested.signature
    ring = next(row for row in result["neighbors"] if "CIRCLE" in row["entity"]["curve"])
    round_result = subject.inspect_selected_neighbors(
        path, topology, select(topology, 0, "edge", ring["entity"]["signature"]), 0, 40
    )
    assert round_result["total_neighbors"] == round_result["total_incidences"] == 2
    assert {row["entity"]["surface"] for row in round_result["neighbors"]} == {
        "GeomType.PLANE",
        "GeomType.CYLINDER",
    }


def test_inner_wire_is_included_not_only_outer_perimeter(tmp_path):
    pipe = cad.Solid.make_cylinder(6, 8).cut(cad.Solid.make_cylinder(2, 8))
    path, topology, bodies = saved(tmp_path, pipe)
    top = face_at(bodies[0], 8)
    result = subject.inspect_selected_neighbors(
        path, topology, select(topology, 0, "face", signature(top)), 0, 40
    )
    assert result["total_neighbors"] == result["total_incidences"] == 2
    assert sorted(row["entity"]["measure"] for row in result["neighbors"]) == pytest.approx(
        [4 * math.pi, 12 * math.pi]
    )
    assert result["focal"]["measure"] == pytest.approx(32 * math.pi)
    assert all(row["entity"]["curve"] == "GeomType.CIRCLE" for row in result["neighbors"])


def test_paging_is_saved_order_and_full_ring_counts_survive_all_pages(tmp_path):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_cylinder(2, 5))
    face = next(face for face in bodies[0].faces() if "CYLINDER" in str(face.geom_type))
    selector = select(topology, 0, "face", signature(face))
    whole = subject.inspect_selected_neighbors(path, topology, selector, 0, 40)
    topology["entities"] = list(reversed(topology["entities"]))
    reversed_order = subject.inspect_selected_neighbors(path, topology, selector, 0, 40)
    assert [row["entity"]["id"] for row in reversed_order["neighbors"]] == list(
        reversed([row["entity"]["id"] for row in whole["neighbors"]])
    )
    pages = [
        subject.inspect_selected_neighbors(path, topology, selector, offset, 1)
        for offset in range(3)
    ]
    assert [page["neighbors"][0] for page in pages] == reversed_order["neighbors"]
    assert all(page["total_neighbors"] == 3 and page["total_incidences"] == 4 for page in pages)
    assert [page["next_offset"] for page in pages] == [1, 2, None]
    beyond = subject.inspect_selected_neighbors(path, topology, selector, 99, 1)
    assert beyond["neighbors"] == [] and beyond["next_offset"] is None
    assert beyond["total_neighbors"] == 3 and beyond["total_incidences"] == 4


def test_repeated_named_instances_keep_world_positions_and_body_identity(tmp_path):
    definition = cad.Solid.make_cylinder(2, 5)
    first = deepcopy(definition).moved(cad.Location((10, 0, 0)))
    second = deepcopy(definition).moved(cad.Location((30, 0, 0)))
    first.label, second.label = occurrence_label("nested/a"), occurrence_label("nested/b")
    path, topology, bodies = saved(tmp_path, cad.Compound(children=[first, second]))
    observed_edges = []
    for index, body in enumerate(bodies):
        x = 10 if body.center().X < 20 else 30
        occurrence = "nested/a" if x == 10 else "nested/b"
        focal = face_at(body, 5)
        result = subject.inspect_selected_neighbors(
            path, topology, select(topology, index, "face", signature(focal), occurrence), 0, 40
        )
        assert result["focal"]["center"] == pytest.approx([x, 0, 5])
        assert (
            result["total_neighbors"] == 1
            and result["neighbors"][0]["entity"]["body_index"] == index
        )
        edge = result["neighbors"][0]["entity"]
        assert edge["min"] == pytest.approx([x - 2, -2, 5])
        assert edge["max"] == pytest.approx([x + 2, 2, 5])
        # The native edge center is its parameter midpoint, on this circular rim.
        assert math.dist(edge["center"], [x, 0, 5]) == pytest.approx(2)
        observed_edges.append((x, edge["center"]))
        wrong = "nested/b" if occurrence == "nested/a" else "nested/a"
        with pytest.raises(ValueError, match="occurrence label"):
            subject.inspect_selected_neighbors(
                path, topology, select(topology, index, "face", signature(focal), wrong), 0, 40
            )
    observed_edges.sort()
    assert [
        b - a for a, b in zip(observed_edges[0][1], observed_edges[1][1], strict=True)
    ] == pytest.approx([20, 0, 0])


@pytest.mark.parametrize(
    "tamper",
    [
        "body_only",
        "missing_focal",
        "duplicate_focal",
        "missing_neighbor",
        "duplicate_neighbor",
        "duplicate_neighbor_id",
        "wrong_body",
        "bad_frame",
    ],
)
def test_missing_and_ambiguous_inventory_never_becomes_empty_page(tmp_path, tamper):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_cylinder(2, 5))
    focal = next(face for face in bodies[0].faces() if "CYLINDER" in str(face.geom_type))
    digest = signature(focal)
    selector = select(topology, 0, "face", digest)
    focal_row = next(
        row for row in topology["entities"] if row["kind"] == "face" and row["signature"] == digest
    )
    edges = [row for row in topology["entities"] if row["kind"] == "edge"]
    if tamper == "body_only":
        topology["entities"] = [row for row in topology["entities"] if row["kind"] == "body"]
    elif tamper == "missing_focal":
        topology["entities"].remove(focal_row)
    elif tamper == "duplicate_focal":
        topology["entities"].append(dict(focal_row))
    elif tamper == "missing_neighbor":
        topology["entities"].remove(edges[-1])
    elif tamper == "duplicate_neighbor":
        topology["entities"].append(dict(edges[-1]))
    elif tamper == "duplicate_neighbor_id":
        edges[-1]["id"] = edges[0]["id"]
    elif tamper == "wrong_body":
        topology["entities"][0]["signature"] = "a" * 64
    else:
        topology["frame"] = "local_guess"
    with pytest.raises(ValueError):
        subject.inspect_selected_neighbors(path, topology, selector, 99, 1)


def test_missing_incident_face_is_refused_even_off_page(tmp_path):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_cylinder(2, 5))
    ring = next(edge for edge in bodies[0].edges() if "CIRCLE" in str(edge.geom_type))
    selector = select(topology, 0, "edge", signature(ring))
    result = subject.inspect_selected_neighbors(path, topology, selector, 0, 40)
    missing = result["neighbors"][-1]["entity"]["signature"]
    topology["entities"] = [
        row
        for row in topology["entities"]
        if not (row["kind"] == "face" and row["signature"] == missing)
    ]
    with pytest.raises(ValueError, match="missing or ambiguous"):
        subject.inspect_selected_neighbors(path, topology, selector, 99, 1)


@pytest.mark.parametrize(
    "returned",
    [
        None,
        SimpleNamespace(wrapped=None),
        SimpleNamespace(wrapped=SimpleNamespace(IsNull=lambda: True)),
    ],
)
def test_invalid_native_import_refused(tmp_path, monkeypatch, returned):
    path, topology, bodies = saved(tmp_path, cad.Solid.make_box(4, 5, 6))
    selector = select(topology, 0, "face", signature(face_at(bodies[0], 6)))
    monkeypatch.setattr(cad, "import_step", lambda path: returned)
    with pytest.raises(ValueError, match="invalid or empty"):
        subject.inspect_selected_neighbors(path, topology, selector, 0, 40)


def test_native_validity_and_explorer_errors_do_not_become_empty(tmp_path, monkeypatch):
    from OCP import BRepCheck

    path, topology, bodies = saved(tmp_path, cad.Solid.make_box(4, 5, 6))
    selector = select(topology, 0, "face", signature(face_at(bodies[0], 6)))

    def failed_explorer(face):
        raise RuntimeError("Native traversal failed")

    monkeypatch.setattr(subject, "_face_edge_uses", failed_explorer)
    with pytest.raises(RuntimeError, match="traversal failed"):
        subject.inspect_selected_neighbors(path, topology, selector, 0, 40)
    monkeypatch.setattr(
        BRepCheck, "BRepCheck_Analyzer", lambda raw: SimpleNamespace(IsValid=lambda: False)
    )
    with pytest.raises(ValueError, match="invalid or empty"):
        subject.inspect_selected_neighbors(path, topology, selector, 0, 40)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_nonfinite_native_measurements_refused(value):
    with pytest.raises(ValueError, match="invalid/nonfinite"):
        subject._number(value)
