"""Independent saved-shape controls for complete finite cylinder translation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from types import SimpleNamespace

import build123d as cad
import pytest
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS_Shape

from forge_mind.workspace import geometry_access_native as subject
from forge_mind.workspace import geometry_distance_native as distance_native
from forge_mind.workspace.assembly import occurrence_label
from forge_mind.workspace.geometry_access_contract import (
    AccessQuery,
    ResolvedAccessBody,
    evaluate_query,
)
from forge_mind.workspace.topology import signature

pytestmark = pytest.mark.slow


def evaluated(mode="static", travel=12, **cylinder):
    query = AccessQuery.model_validate(
        {
            "cylinder": {
                "center_mm": [0, 0, 0],
                "axis": [0, 0, 1],
                "radial_reference": [1, 0, 0],
                "radius_mm": 2,
                "length_mm": 10,
                **cylinder,
            },
            "motion": {"mode": mode, **({} if mode == "static" else {"distance_mm": travel})},
            "obstacles": [{"body_index": 0}],
        }
    )
    return evaluate_query(query)


def box(lower, upper):
    return cad.Pos(*lower) * cad.Box(
        *(b - a for a, b in zip(lower, upper, strict=True)), align=(cad.Align.MIN,) * 3
    )


def saved(tmp_path, shape):
    path = tmp_path / "saved.step"
    cad.export_step(shape, path)
    solids = list(cad.import_step(path).solids())
    topology = {
        "frame": "cad_z_up_mm",
        "topology_counts": {"bodies": len(solids)},
        "entities": [
            {
                "kind": "body",
                "body_index": i,
                "id": f"body{i}",
                "signature": signature(s),
                "body_signature": signature(s),
            }
            for i, s in enumerate(solids)
        ],
    }
    return path, topology, solids


def selector(topology, index=0, path=None):
    row = next(row for row in topology["entities"] if row["body_index"] == index)
    return ResolvedAccessBody(
        requested={"body_index": index} if path is None else {"occurrence_path": path},
        body_index=index,
        body_signature=row["signature"],
    )


def run(tmp_path, obstacle, mode="static", travel=12, **cylinder):
    path, topology, solids = saved(tmp_path, obstacle)
    c, m = evaluated(mode, travel, **cylinder)
    before = hashlib.sha256(path.read_bytes()).hexdigest(), json.dumps(topology, sort_keys=True)
    result = subject.inspect_selected_cylindrical_access(
        path, topology, c, m, [selector(topology)], len(solids)
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        json.dumps(topology, sort_keys=True),
    ) == before
    return result


@pytest.mark.parametrize(
    "mode,travel,volume,lower,upper",
    [
        ("static", 0, 40 * math.pi, [-2, -2, -5], [2, 2, 5]),
        ("axial", 12, 88 * math.pi, [-2, -2, -5], [2, 2, 17]),
        ("axial", -12, 88 * math.pi, [-2, -2, -17], [2, 2, 5]),
        ("transverse", 12, 40 * math.pi + 480, [-2, -2, -5], [14, 2, 5]),
        ("transverse", -12, 40 * math.pi + 480, [-14, -2, -5], [2, 2, 5]),
    ],
)
def test_exact_union_volume_bounds_and_endpoint_containment(mode, travel, volume, lower, upper):
    c, m = evaluated(mode, travel)
    sweep = subject._sweep(c, m)
    facts = subject._facts(sweep)
    assert BRepCheck_Analyzer(sweep.wrapped).IsValid()
    assert len(sweep.solids()) == 1
    assert facts["volume_mm3"] == pytest.approx(volume, abs=1e-6)
    assert facts["bbox_min_mm"] == pytest.approx(lower, abs=1e-6)
    assert facts["bbox_max_mm"] == pytest.approx(upper, abs=1e-6)
    for center in (m.start_center_mm, m.end_center_mm):
        cylinder = cad.Pos(*center) * cad.Cylinder(2, 10)
        components = subject._common_result(subject._common_operation(sweep, cylinder))
        assert sum(row["volume_mm3"] for row in components) == pytest.approx(40 * math.pi, abs=1e-6)


@pytest.mark.parametrize(
    "lower,upper,volume,distance,relation",
    [
        ((2, -0.5, -1), (4, 0.5, 1), 0, 0, "contact_or_unresolved"),
        ((2.25, -0.5, -1), (4.25, 0.5, 1), 0, 0.25, "separated"),
        ((-0.5, -0.5, -1), (0.5, 0.5, 1), 2, 0, "positive_volume_interference"),
        ((-3, -3, -6), (3, 3, 6), 40 * math.pi, 0, "positive_volume_interference"),
    ],
)
def test_touch_gap_overlap_and_containment_are_distinct(
    tmp_path, lower, upper, volume, distance, relation
):
    result = run(tmp_path, box(lower, upper))
    row = result["observations"][0]
    assert row["overlap_volume_mm3"] == pytest.approx(volume, abs=1e-6)
    assert row["minimum_distance_mm"] == pytest.approx(distance, abs=1e-6)
    assert row["relation"] == relation
    assert sum(item["volume_mm3"] for item in row["overlap_components"]) == pytest.approx(
        volume, abs=1e-6
    )
    assert math.dist(row["witness_sweep_mm"], row["witness_obstacle_mm"]) == pytest.approx(
        distance, abs=1e-6
    )


@pytest.mark.parametrize(
    "mode,lower,upper,volume",
    [
        ("axial", (-0.5, -0.5, 5.5), (0.5, 0.5, 6.5), 1),
        ("transverse", (5.5, -0.5, -1), (6.5, 0.5, 1), 2),
    ],
)
def test_midpath_blocker_is_clear_of_both_endpoints_but_full_path_detects_it(
    tmp_path, mode, lower, upper, volume
):
    obstacle = box(lower, upper)
    _c, m = evaluated(mode)
    for center in (m.start_center_mm, m.end_center_mm):
        endpoint = cad.Pos(*center) * cad.Cylinder(2, 10)
        assert subject._common_result(subject._common_operation(endpoint, obstacle)) == []
    row = run(tmp_path, obstacle, mode)["observations"][0]
    assert row["overlap_volume_mm3"] == pytest.approx(volume, abs=1e-6)
    assert row["relation"] == "positive_volume_interference"


def test_full_cable_span_detects_blocker_beyond_short_supported_gauge(tmp_path):
    obstacle = box((19.5, 9.5, -0.5), (20.5, 10.5, 0.5))
    _full_c, full_m = evaluated(
        "transverse", 20, axis=[1, 0, 0], radial_reference=[0, 1, 0], radius_mm=3, length_mm=52
    )
    short_c, short_m = evaluated(
        "transverse", 20, axis=[1, 0, 0], radial_reference=[0, 1, 0], radius_mm=3, length_mm=20
    )
    assert (
        subject._common_result(
            subject._common_operation(subject._sweep(short_c, short_m), obstacle)
        )
        == []
    )
    for center in (full_m.start_center_mm, full_m.end_center_mm):
        endpoint = cad.Plane(origin=center, x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * cad.Cylinder(3, 52)
        assert subject._common_result(subject._common_operation(endpoint, obstacle)) == []
    row = run(
        tmp_path,
        obstacle,
        "transverse",
        20,
        axis=[1, 0, 0],
        radial_reference=[0, 1, 0],
        radius_mm=3,
        length_mm=52,
    )["observations"][0]
    assert row["overlap_volume_mm3"] == pytest.approx(1, abs=1e-6)


def test_arbitrary_world_frame_preserves_complete_geometry(tmp_path):
    result = run(
        tmp_path,
        box((9, -7.5, 13.5), (11, -6.5, 14.5)),
        "transverse",
        12,
        center_mm=[10, -7, 20],
        axis=[1, 0, 0],
        radial_reference=[0, 0, -1],
    )
    assert result["sweep"]["volume_mm3"] == pytest.approx(40 * math.pi + 480, abs=1e-6)
    assert result["sweep"]["bbox_min_mm"] == pytest.approx([5, -9, 6], abs=1e-6)
    assert result["sweep"]["bbox_max_mm"] == pytest.approx([15, -5, 22], abs=1e-6)
    assert result["observations"][0]["overlap_volume_mm3"] == pytest.approx(2, abs=1e-6)


def test_repeated_saved_instances_named_selection_and_explicit_omission(tmp_path):
    definition = cad.Solid.make_box(2, 2, 2).moved(cad.Location((-1, -1, -1)))
    a = deepcopy(definition).moved(cad.Location((10, 0, 0)))
    b = deepcopy(definition).moved(cad.Location((20, 0, 0)))
    a.label, b.label = occurrence_label("repeat/a"), occurrence_label("repeat/b")
    path, topology, solids = saved(tmp_path, cad.Compound(children=[a, b]))
    first = next(i for i, body in enumerate(solids) if body.center().X < 15)
    second = 1 - first
    c, m = evaluated(center_mm=[10, 0, 0])
    result = subject.inspect_selected_cylindrical_access(
        path,
        topology,
        c,
        m,
        [selector(topology, second, "repeat/b"), selector(topology, first, "repeat/a")],
        2,
    )
    assert [row["overlap_volume_mm3"] for row in result["observations"]] == pytest.approx([0, 8])
    assert result["observations"][0]["minimum_distance_mm"] == pytest.approx(7)
    omitted = subject.inspect_selected_cylindrical_access(
        path, topology, c, m, [selector(topology, second, "repeat/b")], 2
    )
    assert len(omitted["observations"]) == 1
    with pytest.raises(ValueError, match="duplicate body"):
        subject.inspect_selected_cylindrical_access(
            path,
            topology,
            c,
            m,
            [selector(topology, first), selector(topology, first, "repeat/a")],
            2,
        )
    with pytest.raises(ValueError, match="occurrence label"):
        subject.inspect_selected_cylindrical_access(
            path, topology, c, m, [selector(topology, first, "repeat/b")], 2
        )


@pytest.mark.parametrize("kind", ["missing", "duplicate", "signature", "count", "frame"])
def test_incomplete_or_tampered_entire_inventory_refuses(tmp_path, kind):
    path, topology, solids = saved(
        tmp_path, cad.Compound(children=[cad.Box(2, 2, 2), cad.Pos(20, 0, 0) * cad.Box(2, 2, 2)])
    )
    selection = selector(topology)
    if kind == "missing":
        topology["entities"].pop()
    elif kind == "duplicate":
        topology["entities"].append(dict(topology["entities"][0]))
    elif kind == "signature":
        topology["entities"][1]["signature"] = "b" * 64
    elif kind == "count":
        topology["topology_counts"]["bodies"] = 1
    else:
        topology["frame"] = "local"
    with pytest.raises(ValueError):
        subject.inspect_selected_cylindrical_access(
            path, topology, *evaluated(), [selection], len(solids)
        )


def test_ambiguous_coincident_body_correspondence_refuses(tmp_path):
    shape = cad.Solid.make_box(2, 2, 2)
    path, topology, solids = saved(
        tmp_path, cad.Compound(children=[deepcopy(shape), deepcopy(shape)])
    )
    with pytest.raises(ValueError, match="ambiguous"):
        subject.inspect_selected_cylindrical_access(
            path, topology, *evaluated(), [selector(topology)], len(solids)
        )


class CommonOperation:
    def __init__(self, done=True, raw=None):
        self.done, self.raw, self.built = done, raw, False

    def Build(self):  # noqa: N802 - installed OCCT API
        self.built = True

    def IsDone(self):  # noqa: N802 - installed OCCT API
        return self.built and self.done

    def Shape(self):  # noqa: N802 - installed OCCT API
        assert self.built and self.done, "Shape read before confirmed native completion"
        return self.raw


def test_failed_common_or_none_cannot_become_empty_success():
    for operation in (CommonOperation(done=False), CommonOperation(raw=None)):
        with pytest.raises(ValueError):
            subject._common_result(operation)
        assert operation.built
    assert subject._common_result(CommonOperation(raw=TopoDS_Shape())) == []


def test_invalid_nonnull_common_refuses(monkeypatch):
    from OCP import BRepCheck

    monkeypatch.setattr(
        BRepCheck, "BRepCheck_Analyzer", lambda shape: SimpleNamespace(IsValid=lambda: False)
    )
    with pytest.raises(ValueError, match="result is invalid"):
        subject._common_result(CommonOperation(raw=cad.Box(1, 1, 1).wrapped))


def test_all_positive_components_retained_and_cap_excess_refuses():
    solids = [cad.Pos(i * 3, 0, 0) * cad.Box(1, 1, 1) for i in range(33)]
    first = subject._common_result(CommonOperation(raw=cad.Compound(children=solids[:3]).wrapped))
    assert len(first) == 3
    assert sum(row["volume_mm3"] for row in first) == pytest.approx(3)
    with pytest.raises(ValueError, match="component limit"):
        subject._common_result(CommonOperation(raw=cad.Compound(children=solids).wrapped))


@pytest.mark.parametrize("returned", [None, SimpleNamespace(wrapped=None)])
def test_invalid_saved_native_import_refuses(tmp_path, monkeypatch, returned):
    path, topology, _ = saved(tmp_path, cad.Box(1, 1, 1))
    selection = selector(topology)
    monkeypatch.setattr(cad, "import_step", lambda path: returned)
    with pytest.raises(ValueError, match="invalid or empty"):
        subject.inspect_selected_cylindrical_access(path, topology, *evaluated(), [selection], 1)


@pytest.mark.parametrize("done,value", [(False, 1), (True, float("nan"))])
def test_failed_distance_cannot_return_clearance(tmp_path, monkeypatch, done, value):
    operation = SimpleNamespace(
        LoadS1=lambda s: None,
        LoadS2=lambda s: None,
        Perform=lambda: None,
        IsDone=lambda: done,
        NbSolution=lambda: 1,
        Value=lambda: value,
    )
    monkeypatch.setattr(distance_native, "_distance_operation", lambda: operation)
    with pytest.raises(ValueError):
        run(tmp_path, box((10, 10, 10), (11, 11, 11)))


def test_oblique_or_static_translated_native_motion_refuses():
    c, m = evaluated("transverse")
    m = m.model_copy(update={"translation_mm": (12, 0, 1), "end_center_mm": (12, 0, 1)})
    with pytest.raises(ValueError, match="local motion"):
        subject._sweep(c, m)
    c, m = evaluated()
    m = m.model_copy(update={"translation_mm": (1, 0, 0), "end_center_mm": (1, 0, 0)})
    with pytest.raises(ValueError, match="cannot translate"):
        subject._sweep(c, m)
