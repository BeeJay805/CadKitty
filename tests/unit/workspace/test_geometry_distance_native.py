"""Native saved-STEP distance requirements; root executes in the serialized CAD slot."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import build123d as cad
import pytest

from forge_mind.workspace import geometry_distance_native as subject
from forge_mind.workspace.assembly import occurrence_label
from forge_mind.workspace.geometry_distance_contract import (
    DistanceSelector,
    ResolvedDistanceSelector,
)
from forge_mind.workspace.topology import signature

pytestmark = pytest.mark.slow


def saved(tmp_path, shape):
    path = tmp_path / "actual.step"
    cad.export_step(shape, path)
    loaded = cad.import_step(path)
    solids = list(loaded.solids())
    entities = []
    for index, body in enumerate(solids):
        body_sig = signature(body)
        entities.append(
            {
                "id": f"body{index}",
                "kind": "body",
                "body_index": index,
                "signature": body_sig,
                "body_signature": body_sig,
            }
        )
        entities.extend(
            {
                "id": f"face{index}_{n}",
                "kind": "face",
                "body_index": index,
                "signature": signature(face),
                "body_signature": body_sig,
            }
            for n, face in enumerate(body.faces())
        )
    topology = {
        "frame": "cad_z_up_mm",
        "entities": entities,
        "topology_counts": {"bodies": len(solids)},
    }
    return path, topology, solids


def selection(topology, index, face=None, occurrence=None):
    body = next(
        row for row in topology["entities"] if row["kind"] == "body" and row["body_index"] == index
    )
    requested = (
        DistanceSelector(body_index=index, face_signature=face)
        if occurrence is None
        else DistanceSelector(occurrence_path=occurrence, face_signature=face)
    )
    return ResolvedDistanceSelector(
        requested=requested, body_index=index, body_signature=body["signature"], face_signature=face
    )


def face_at(body, axis, coordinate):
    faces = [
        face
        for face in body.faces()
        if abs(tuple(face.bounding_box().min)[axis] - coordinate) < 1e-6
        and abs(tuple(face.bounding_box().max)[axis] - coordinate) < 1e-6
    ]
    assert len(faces) == 1
    return signature(faces[0])


def test_trimmed_faces_distance5_not_infinite_plane4_and_bytes_unchanged(tmp_path):
    a = cad.Pos(0, 0, -1) * cad.Box(1, 1, 1, align=(cad.Align.MIN,) * 3)
    b = cad.Pos(4, 0, 4) * cad.Box(1, 1, 1, align=(cad.Align.MIN,) * 3)
    path, topology, solids = saved(tmp_path, cad.Compound(children=[a, b]))
    before = path.read_bytes(), json.dumps(topology, sort_keys=True)
    ia = next(i for i, s in enumerate(solids) if s.bounding_box().max.Z < 1)
    ib = 1 - ia
    result = subject.measure_selected_distance(
        path,
        topology,
        selection(topology, ia, face_at(solids[ia], 2, 0)),
        selection(topology, ib, face_at(solids[ib], 2, 4)),
    )
    assert result["minimum_distance_mm"] == pytest.approx(5, abs=1e-7)
    assert abs(result["witness_a_mm"][2] - result["witness_b_mm"][2]) == pytest.approx(4)
    assert math.dist(result["witness_a_mm"], result["witness_b_mm"]) == pytest.approx(5)
    assert result["native_completed"] is True and result["solution_count"] >= 1
    assert (path.read_bytes(), json.dumps(topology, sort_keys=True)) == before


def test_same_body_opposite_trimmed_faces(tmp_path):
    path, topology, solids = saved(tmp_path, cad.Box(4, 5, 6, align=(cad.Align.MIN,) * 3))
    result = subject.measure_selected_distance(
        path,
        topology,
        selection(topology, 0, face_at(solids[0], 2, 0)),
        selection(topology, 0, face_at(solids[0], 2, 6)),
    )
    assert result["minimum_distance_mm"] == pytest.approx(6)


def test_curved_faces_preserve_repeated_instance_world_poses(tmp_path):
    cylinder = cad.Solid.make_cylinder(2, 5)
    # Production assembly makes independent native definitions before assigning labels.
    first = deepcopy(cylinder).moved(cad.Location((10, 0, 0)))
    second = deepcopy(cylinder).moved(cad.Location((30, 0, 0)))
    first.label = occurrence_label("repeat/a")
    second.label = occurrence_label("repeat/b")
    path, topology, solids = saved(tmp_path, cad.Compound(children=[first, second]))
    signatures = []
    for body in solids:
        lateral = [face for face in body.faces() if "CYLINDER" in str(face.geom_type)]
        assert len(lateral) == 1
        signatures.append(signature(lateral[0]))
    result = subject.measure_selected_distance(
        path,
        topology,
        selection(
            topology, 0, signatures[0], "repeat/a" if solids[0].center().X < 20 else "repeat/b"
        ),
        selection(
            topology, 1, signatures[1], "repeat/a" if solids[1].center().X < 20 else "repeat/b"
        ),
    )
    assert result["minimum_distance_mm"] == pytest.approx(16)
    points = sorted(result[key][0] for key in ("witness_a_mm", "witness_b_mm"))
    assert points == pytest.approx([12, 28])


@pytest.mark.parametrize("offset,scale", [(10, 1), (8, 1), (4, 0.2)])
def test_touch_overlap_containment_zero_without_collision_classification(tmp_path, offset, scale):
    a = cad.Box(10, 10, 10, align=(cad.Align.MIN,) * 3)
    b = cad.Pos(offset, 0 if scale == 1 else 4, 0 if scale == 1 else 4) * cad.Box(
        10 * scale, 10 * scale, 10 * scale, align=(cad.Align.MIN,) * 3
    )
    path, topology, _ = saved(tmp_path, cad.Compound(children=[a, b]))
    result = subject.measure_selected_distance(
        path, topology, selection(topology, 0), selection(topology, 1)
    )
    assert result["minimum_distance_mm"] == pytest.approx(0)
    assert math.dist(result["witness_a_mm"], result["witness_b_mm"]) <= 1e-6
    assert set(result) == {
        "minimum_distance_mm",
        "witness_a_mm",
        "witness_b_mm",
        "solution_count",
        "native_completed",
        "native_inner_solution",
    }
    assert type(result["native_inner_solution"]) is bool


def test_body_only_inventory_is_supported_but_missing_face_is_refused(tmp_path):
    path, topology, solids = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)])
    )
    face = signature(solids[0].faces()[0])
    a, b = selection(topology, 0), selection(topology, 1)
    topology["entities"] = [row for row in topology["entities"] if row["kind"] == "body"]
    assert subject.measure_selected_distance(path, topology, a, b)[
        "minimum_distance_mm"
    ] == pytest.approx(3)
    with pytest.raises(ValueError, match="face inventory"):
        subject.measure_selected_distance(path, topology, selection(topology, 0, face), b)


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate_body",
        "wrong_body_signature",
        "duplicate_face",
        "missing_face",
        "bad_native_face",
        "bad_frame",
    ],
)
def test_saved_selection_tamper_and_ambiguity_refused(tmp_path, tamper):
    path, topology, solids = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)])
    )
    face = signature(solids[0].faces()[0])
    a = selection(topology, 0, face)
    b = selection(topology, 1)
    if tamper == "duplicate_body":
        topology["entities"].append(dict(topology["entities"][0]))
    elif tamper == "wrong_body_signature":
        topology["entities"][0]["signature"] = "a" * 64
    elif tamper == "duplicate_face":
        topology["entities"].append(
            dict(next(row for row in topology["entities"] if row.get("signature") == face))
        )
    elif tamper == "missing_face":
        topology["entities"] = [row for row in topology["entities"] if row.get("signature") != face]
    elif tamper == "bad_native_face":
        for row in topology["entities"]:
            if row.get("signature") == face:
                row["signature"] = "b" * 64
        a = selection(topology, 0, "b" * 64)
    else:
        topology["frame"] = "other"
    with pytest.raises(ValueError):
        subject.measure_selected_distance(path, topology, a, b)


def test_identical_entities_and_coincident_ambiguous_bodies_refused(tmp_path):
    box = cad.Box(2, 2, 2)
    path, topology, _ = saved(tmp_path, cad.Compound(children=[box, cad.Pos(5, 0, 0) * box]))
    a = selection(topology, 0)
    with pytest.raises(ValueError, match="distinct"):
        subject.measure_selected_distance(path, topology, a, a)
    path, topology, _ = saved(tmp_path, cad.Compound(children=[box, box.moved(cad.Location())]))
    with pytest.raises(ValueError, match="ambiguous"):
        subject.measure_selected_distance(
            path, topology, selection(topology, 0), selection(topology, 1)
        )


@pytest.mark.parametrize(
    "returned", [None, SimpleNamespace(wrapped=SimpleNamespace(IsNull=lambda: True))]
)
def test_invalid_or_empty_native_import_refused(tmp_path, monkeypatch, returned):
    path, topology, _ = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)])
    )
    a, b = selection(topology, 0), selection(topology, 1)
    monkeypatch.setattr(cad, "import_step", lambda path: returned)
    with pytest.raises(ValueError, match="invalid or empty"):
        subject.measure_selected_distance(path, topology, a, b)


class DistanceOperation:
    def __init__(self, **changes):
        self.values = {
            "done": True,
            "count": 1,
            "distance": 3.0,
            "a": (0.0, 0.0, 0.0),
            "b": (3.0, 0.0, 0.0),
            "inner": False,
            **changes,
        }
        self.performed = False

    def LoadS1(self, value):  # noqa: N802 - mirrors the native OCCT API
        pass

    def LoadS2(self, value):  # noqa: N802 - mirrors the native OCCT API
        pass

    def Perform(self):  # noqa: N802 - mirrors the native OCCT API
        self.performed = True

    def IsDone(self):  # noqa: N802 - mirrors the native OCCT API
        return self.performed and self.values["done"]

    def NbSolution(self):  # noqa: N802 - mirrors the native OCCT API
        return self.values["count"]

    def Value(self):  # noqa: N802 - mirrors the native OCCT API
        return self.values["distance"]

    def PointOnShape1(self, index):  # noqa: N802 - mirrors the native OCCT API
        return self.point(self.values["a"])

    def PointOnShape2(self, index):  # noqa: N802 - mirrors the native OCCT API
        return self.point(self.values["b"])

    def InnerSolution(self):  # noqa: N802 - mirrors the native OCCT API
        return self.values["inner"]

    @staticmethod
    def point(values):
        return SimpleNamespace(X=lambda: values[0], Y=lambda: values[1], Z=lambda: values[2])


@pytest.mark.parametrize(
    "changes",
    [
        {"done": False},
        {"count": 0},
        {"count": True},
        {"distance": float("nan")},
        {"distance": -1.0},
        {"distance": True},
        {"a": (float("inf"), 0, 0)},
        {"b": (4, 0, 0)},
        {"inner": 1},
    ],
)
def test_failed_nonfinite_or_inconsistent_native_measurement_refused(
    tmp_path, monkeypatch, changes
):
    path, topology, _ = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)])
    )
    operation = DistanceOperation(**changes)
    monkeypatch.setattr(subject, "_distance_operation", lambda: operation)
    with pytest.raises(ValueError):
        subject.measure_selected_distance(
            path, topology, selection(topology, 0), selection(topology, 1)
        )
    assert operation.performed


def test_wrong_or_missing_occurrence_label_cannot_select_another_body(tmp_path):
    first, second = cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)
    first.label = occurrence_label("first")
    second.label = occurrence_label("second")
    path, topology, solids = saved(tmp_path, cad.Compound(children=[first, second]))
    index = next(i for i, body in enumerate(solids) if body.center().X < 2)
    for wrong_path in ("missing", "second"):
        with pytest.raises(ValueError, match="occurrence label"):
            subject.measure_selected_distance(
                path,
                topology,
                selection(topology, index, occurrence=wrong_path),
                selection(topology, 1 - index),
            )


def test_native_invalid_topology_refused_before_distance(tmp_path, monkeypatch):
    from OCP import BRepCheck

    path, topology, _ = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(5, 0, 0) * cad.Box(2, 2, 2)])
    )
    a, b = selection(topology, 0), selection(topology, 1)
    monkeypatch.setattr(
        BRepCheck, "BRepCheck_Analyzer", lambda wrapped: SimpleNamespace(IsValid=lambda: False)
    )
    monkeypatch.setattr(
        subject, "_distance_operation", lambda: pytest.fail("Distance ran before native validation")
    )
    with pytest.raises(ValueError, match="invalid or empty"):
        subject.measure_selected_distance(path, topology, a, b)
