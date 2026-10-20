"""Requirements for native-free saved occurrence focus selection and framing."""

from __future__ import annotations

import copy
import json
import math

import pytest

from forge_mind.workspace.occurrence_focus import (
    FOCUS_SCOPE,
    OccurrenceFocus,
    focus_view_metadata,
    resolve_focus,
    validate_focus_frame,
)

BODY = "a" * 64
FACE = "c" * 64
EDGE = "e" * 64


def inventory():
    return {
        "frame": "cad_z_up_mm",
        "selection_scope": "body",
        "detail_policy": {"mode": "full_index", "face_edge_picking_available": False},
        "entities": [
            {
                "id": "body_a_0",
                "kind": "body",
                "body_index": 0,
                "signature": BODY,
                "body_signature": BODY,
                "min": [0, -10, 0],
                "max": [100, 10, 20],
            },
            {
                "id": "face_c_0",
                "kind": "face",
                "body_index": 0,
                "signature": FACE,
                "body_signature": BODY,
                "min": [10, -6, 5],
                "max": [14, -2, 5],
            },
            {
                "id": "edge_e_0",
                "kind": "edge",
                "body_index": 0,
                "signature": EDGE,
                "body_signature": BODY,
                "min": [10, -6, 5],
                "max": [14, -6, 5],
            },
        ],
        # Body meshing does not remove normal assembly face/edge fact inventory.
        "meshes": [{"id": "body_a_0", "body_index": 0}],
    }


def selected(kind="face", **kwargs):
    return OccurrenceFocus(kind=kind, signature=FACE if kind == "face" else EDGE, **kwargs)


def test_saved_planar_face_and_edge_with_whole_body_mesh_and_no_mutation():
    topology = inventory()
    before = json.dumps(topology, sort_keys=True)
    face = resolve_focus(topology, 0, selected())
    edge = resolve_focus(topology, 0, selected("edge", padding=1.5))
    assert face["entity_id"] == "face_c_0"
    assert face["center_mm"] == [12.0, -4.0, 5.0]
    assert face["radius_mm"] == pytest.approx(math.sqrt(32))
    assert edge["center_mm"] == [12.0, -6.0, 5.0]
    assert edge["radius_mm"] == 3.0
    assert face["scope"] == FOCUS_SCOPE
    assert face["request"] == {
        "kind": "face",
        "signature": FACE,
        "view": "isometric",
        "padding": 2.0,
    }
    assert json.dumps(topology, sort_keys=True) == before
    # Returned arrays do not alias the saved topology.
    face["body_bounds"]["min"][0] = 999
    assert topology["entities"][0]["min"][0] == 0


def test_repeated_translated_selection_uses_world_position_and_correct_body():
    topology = inventory()
    translated = copy.deepcopy(topology["entities"])
    shift = [200, 40, -30]
    for row in translated:
        row["body_index"] = 1
        row["body_signature"] = "b" * 64
        row["signature"] = {"body": "b", "face": "d", "edge": "f"}[row["kind"]] * 64
        row["id"] += "_translated"
        for key in ("min", "max"):
            row[key] = [a + b for a, b in zip(row[key], shift, strict=True)]
    topology["entities"].extend(translated)
    first = resolve_focus(topology, 0, selected())
    second = resolve_focus(topology, 1, OccurrenceFocus(kind="face", signature="d" * 64))
    assert second["body_signature"] == "b" * 64
    assert second["entity_id"] == "face_c_0_translated"
    assert second["center_mm"] == [212.0, 36.0, -25.0]
    assert second["radius_mm"] == first["radius_mm"]
    assert focus_view_metadata(second)["context_center_mm"] == [250.0, 40.0, -20.0]
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, OccurrenceFocus(kind="face", signature="d" * 64))


@pytest.mark.parametrize(
    "change",
    [
        "missing_inventory",
        "bad_inventory_row",
        "missing_body",
        "duplicate_body",
        "duplicate_face",
        "duplicate_id",
        "duplicate_body_id",
        "stale_signature",
        "body_hash_mismatch",
        "entity_body_hash_mismatch",
        "malformed_body_hash",
        "malformed_id",
        "boolean_body_index",
        "wrong_frame",
        "missing_face",
    ],
)
def test_saved_identity_refusals(change):
    topology = inventory()
    rows = topology["entities"]
    if change == "missing_inventory":
        topology.pop("entities")
    elif change == "bad_inventory_row":
        rows.append(None)
    elif change == "missing_body":
        rows.pop(0)
    elif change == "duplicate_body":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "duplicate_face":
        duplicate = copy.deepcopy(rows[1])
        duplicate["id"] = "another_face"
        rows.append(duplicate)
    elif change == "duplicate_id":
        rows[2]["id"] = rows[1]["id"]
    elif change == "duplicate_body_id":
        rows[2]["id"] = rows[0]["id"]
    elif change == "stale_signature":
        rows[1]["signature"] = "d" * 64
    elif change == "body_hash_mismatch":
        rows[0]["body_signature"] = "b" * 64
    elif change == "entity_body_hash_mismatch":
        rows[1]["body_signature"] = "b" * 64
    elif change == "malformed_body_hash":
        rows[0]["signature"] = rows[0]["body_signature"] = "not-a-hash"
    elif change == "malformed_id":
        rows[1]["id"] = ""
    elif change == "boolean_body_index":
        rows[0]["body_index"] = False
    elif change == "wrong_frame":
        topology["frame"] = "screen_pixels"
    else:
        topology["entities"] = rows[:1]
        topology["detail_policy"]["mode"] = "body_index"
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, selected())


@pytest.mark.parametrize("index", [True, -1, "0", 0.0, 1])
def test_invalid_or_unavailable_body_selection(index):
    with pytest.raises(ValueError):
        resolve_focus(inventory(), index, selected())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "5", None])
def test_invalid_saved_coordinates(value):
    topology = inventory()
    topology["entities"][1]["min"][0] = value
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, selected())


@pytest.mark.parametrize(
    "change", ["unordered", "outside", "zero_diagonal", "body_zero", "wrong_length"]
)
def test_invalid_saved_bounds(change):
    topology = inventory()
    entity, body = topology["entities"][1], topology["entities"][0]
    if change == "unordered":
        entity["min"][0] = 15
    elif change == "outside":
        entity["min"][1] = -10.001
    elif change == "zero_diagonal":
        entity["max"] = entity["min"].copy()
    elif change == "body_zero":
        body["max"] = body["min"].copy()
    else:
        entity["max"].append(7)
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, selected())


def test_roundoff_bounds_allowance_does_not_change_saved_entity_bounds():
    topology = inventory()
    topology["entities"][1]["max"][0] = 100.00001
    assert resolve_focus(topology, 0, selected())["entity_bounds"]["max"][0] == 100.00001
    topology["entities"][1]["max"][0] = 100.01
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, selected())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "body"},
        {"signature": "A" * 64},
        {"signature": "a" * 63},
        {"signature": b"a" * 64},
        {"signature": "a" * 64 + "\n"},
        {"view": "perspective"},
        {"padding": "2.0"},
        {"padding": True},
        {"padding": 1.0},
        {"padding": 8.1},
        {"padding": float("nan")},
        {"padding": float("inf")},
        {"extra": 1},
    ],
)
def test_public_selector_rejects_invalid_fields(kwargs):
    value = {"kind": "face", "signature": FACE, **kwargs}
    with pytest.raises(ValueError):
        OccurrenceFocus.model_validate(value)


def test_tuple_list_normalization_roundtrips_without_mutation():
    frame = resolve_focus(inventory(), 0, selected())
    original = copy.deepcopy(frame)
    frame["center_mm"] = tuple(frame["center_mm"])
    for key in ("entity_bounds", "body_bounds"):
        frame[key] = {name: tuple(vector) for name, vector in frame[key].items()}
    assert validate_focus_frame(frame) == original
    assert isinstance(frame["center_mm"], tuple)
    assert validate_focus_frame(json.loads(json.dumps(frame))) == original


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing",
        "center",
        "radius",
        "scope",
        "padding",
        "body",
        "request_missing",
        "request_extra",
    ],
)
def test_resolved_frame_tampering_is_rejected(change):
    frame = resolve_focus(inventory(), 0, selected())
    if change == "extra":
        frame["extra"] = 1
    elif change == "missing":
        frame.pop("entity_id")
    elif change == "center":
        frame["center_mm"][0] += 1
    elif change == "radius":
        frame["radius_mm"] += 1
    elif change == "scope":
        frame["scope"] = "exact measured fit"
    elif change == "padding":
        frame["request"]["padding"] = 3.0
    elif change == "body":
        frame["body_index"] = False
    elif change == "request_missing":
        frame["request"].pop("view")
    else:
        frame["request"]["extra"] = 1
    with pytest.raises(ValueError):
        validate_focus_frame(frame)


@pytest.mark.parametrize("view", ["isometric", "front", "right", "rear", "top", "bottom"])
def test_camera_axes_are_right_handed_orthonormal_and_frames_keep_world_units(view):
    frame = resolve_focus(inventory(), 0, selected(view=view))
    metadata = focus_view_metadata(frame)
    assert set(metadata) == {
        "view",
        "basis",
        "context_center_mm",
        "context_radius_mm",
        "detail_center_mm",
        "detail_radius_mm",
        "all_selected_body_triangles_depth_tested",
        "region_marker",
    }
    assert metadata["context_center_mm"] == [50.0, 0.0, 10.0]
    assert metadata["context_radius_mm"] == pytest.approx(math.sqrt(10800) / 2 * 1.1)
    assert metadata["detail_center_mm"] == [12.0, -4.0, 5.0]
    assert metadata["detail_radius_mm"] == pytest.approx(math.sqrt(32))
    assert metadata["all_selected_body_triangles_depth_tested"] is True
    assert metadata["region_marker"] == "projected saved entity bounds; not exact entity outline"
    axes = list(zip(*metadata["basis"], strict=True))
    for i, axis in enumerate(axes):
        assert sum(x * x for x in axis) == pytest.approx(1.0)
        for other in axes[i + 1 :]:
            assert sum(a * b for a, b in zip(axis, other, strict=True)) == pytest.approx(
                0.0, abs=1e-15
            )
    right, up, toward = axes
    cross = [
        right[1] * up[2] - right[2] * up[1],
        right[2] * up[0] - right[0] * up[2],
        right[0] * up[1] - right[1] * up[0],
    ]
    assert cross == pytest.approx(toward)
    if view == "front":
        assert right == pytest.approx([1.0, 0.0, 0.0], abs=1e-15)
        assert toward[1] == pytest.approx(-math.cos(math.radians(2)))
    assert json.loads(json.dumps(metadata)) == metadata


def test_tiny_nonzero_edge_uses_documented_display_radius_floor():
    topology = inventory()
    edge = topology["entities"][2]
    edge["min"], edge["max"] = [1, 0, 1], [1.00000001, 0, 1]
    assert resolve_focus(topology, 0, selected("edge"))["radius_mm"] == 1e-6


def test_unrepresentable_derived_camera_extent_refuses():
    topology = inventory()
    topology["entities"][0]["min"] = [-1e308, 0, 0]
    topology["entities"][0]["max"] = [1e308, 20, 20]
    with pytest.raises(ValueError):
        resolve_focus(topology, 0, selected())
