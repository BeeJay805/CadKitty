"""Proposed focused regressions for root-integrated geometry_results helper."""

import math

import build123d as b
import numpy as np
import pytest

from forge_mind.workspace.geometry_results import boolean_overlap_components, boolean_volume
from forge_mind.workspace.motion_check import review_poses


def test_disconnected_overlap_retains_every_component_in_world_coordinates():
    target = b.Compound(
        children=[b.Pos(0, 0, 0) * b.Box(4, 4, 4), b.Pos(20, 0, 0) * b.Box(2, 2, 2)]
    )
    tool = b.Pos(10, 0, 0) * b.Box(40, 10, 10)
    common = target & tool
    components = boolean_overlap_components(common)
    assert len(components) == 2
    assert sorted(c["volume_mm3"] for c in components) == pytest.approx([8, 64])
    assert math.fsum(c["volume_mm3"] for c in components) == pytest.approx(boolean_volume(common))
    ordered = sorted(components, key=lambda c: c["bbox_min_mm"][0])
    assert ordered[0]["bbox_min_mm"] == pytest.approx([-2, -2, -2])
    assert ordered[0]["bbox_max_mm"] == pytest.approx([2, 2, 2])
    assert ordered[1]["bbox_min_mm"] == pytest.approx([19, -1, -1])
    assert ordered[1]["bbox_max_mm"] == pytest.approx([21, 1, 1])
    assert all(c["frame"] == "cad_z_up_mm" for c in components)


def test_nested_list_tuple_and_none_preserve_components():
    first, second = b.Box(2, 2, 2), b.Pos(20, 0, 0) * b.Box(1, 1, 1)
    components = boolean_overlap_components([first, (None, second)])
    assert len(components) == 2
    assert [c["volume_mm3"] for c in components] == pytest.approx([8, 1])
    assert boolean_overlap_components(None) == []
    assert boolean_overlap_components([]) == []


def test_small_positive_component_is_not_filtered_by_pair_collision_threshold():
    small = b.Box(0.01, 0.01, 0.01)
    components = boolean_overlap_components(small)
    assert len(components) == 1
    assert components[0]["volume_mm3"] == pytest.approx(1e-6)


def test_motion_witness_reports_components_after_world_transform():
    shape = b.Box(4, 4, 4)
    transform = np.eye(4)
    transform[0, 3] = 30
    result = review_poses(
        [shape, shape], ["a", "b"], ["a", "b"], [({"angle": 0}, [transform, transform])]
    )
    assert result["status"] == "fail"
    assert result["coverage"] == "discrete_only"
    assert result["collision_count"] == 1
    components = result["collisions"][0]["overlap_components"]
    assert len(components) == 1
    assert components[0]["volume_mm3"] == pytest.approx(64)
    assert components[0]["bbox_min_mm"] == pytest.approx([28, -2, -2])
    assert components[0]["bbox_max_mm"] == pytest.approx([32, 2, 2])


def test_localization_failure_keeps_known_collision_and_marks_incomplete(monkeypatch):
    from forge_mind.workspace import motion_check

    def broken(shape):
        raise ValueError("Invalid component bounds")

    monkeypatch.setattr(motion_check, "boolean_overlap_components", broken)
    result = review_poses(
        [b.Box(4, 4, 4), b.Box(4, 4, 4)], ["a", "b"], ["a", "b"], [({}, [np.eye(4), np.eye(4)])]
    )
    assert result["status"] == "fail"
    assert result["coverage"] == "incomplete"
    assert result["collision_count"] == 1
    assert result["collisions"][0]["overlap_mm3"] == pytest.approx(64)
    assert result["errors"] == ["a / b: ValueError"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_volume_readback_raises(value):
    class Broken:
        volume = value

    with pytest.raises(ValueError, match="Boolean volume"):
        boolean_overlap_components(Broken())


def test_missing_positive_volume_decomposition_raises():
    class Broken:
        volume = 1.0
        is_valid = True

        def solids(self):
            return []

    with pytest.raises(ValueError, match="component volumes"):
        boolean_overlap_components(Broken())


def test_reference_retains_all_collisions_beyond_dynamic_witness_limit():
    shapes = [b.Box(1, 1, 1) for _ in range(21)]
    result = review_poses(
        shapes,
        [str(i) for i in range(21)],
        [None] * 21,
        [({}, [np.eye(4)] * 21)],
        reference_values={},
    )
    assert result["reference_completed"] is True
    assert len(result["collisions"]) == 200
    assert len(result["reference_collisions"]) == 210


def test_later_incomplete_motion_does_not_erase_complete_reference(monkeypatch):
    from forge_mind.workspace import motion_check

    original = motion_check.boolean_overlap_components
    calls = 0

    def break_second(shape):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Late dynamic readback failure")
        return original(shape)

    monkeypatch.setattr(motion_check, "boolean_overlap_components", break_second)
    result = review_poses(
        [b.Box(1, 1, 1)] * 2,
        ["a", "b"],
        ["a", "b"],
        [({"angle": 0}, [np.eye(4)] * 2), ({"angle": 1}, [np.eye(4)] * 2)],
        reference_values={"angle": 0},
    )
    assert result["coverage"] == "incomplete"
    assert result["reference_completed"] is True
    assert len(result["reference_collisions"]) == 1


def test_reference_cannot_reuse_an_earlier_unrecorded_collision():
    with pytest.raises(ValueError, match="checked first"):
        review_poses(
            [b.Box(1, 1, 1)] * 2,
            ["a", "b"],
            [None, None],
            [({"angle": 1}, [np.eye(4)] * 2), ({"angle": 0}, [np.eye(4)] * 2)],
            reference_values={"angle": 0},
        )
