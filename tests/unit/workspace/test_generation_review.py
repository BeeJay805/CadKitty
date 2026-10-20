"""Native boolean controls and truthful incomplete geometry for development review."""

from __future__ import annotations

import build123d as b
import pytest
from scripts import workspace_generation_review as review


@pytest.mark.parametrize("offset, expected", [(4, 0), (2, 0), (1, 4)])
def test_real_disjoint_touching_and_overlapping_intersections(offset, expected):
    left = b.Box(2, 2, 2).solids()[0]
    right = (b.Pos(offset, 0, 0) * b.Box(2, 2, 2)).solids()[0]
    result = left & right
    assert review.boolean_solid_volume(result) == pytest.approx(expected)


def test_real_multiple_solid_intersection_and_split_difference():
    pair = b.Compound(children=[b.Pos(x, 0, 0) * b.Box(2, 2, 2) for x in (-3, 3)])
    intersection = b.Box(10, 2, 2).solids()[0] & pair
    assert isinstance(intersection, b.Compound)
    assert len(intersection.solids()) == 2
    assert review.boolean_solid_volume(intersection) == pytest.approx(16)
    difference = b.Box(10, 2, 2).solids()[0] - b.Box(2, 4, 4).solids()[0]
    assert isinstance(difference, b.ShapeList)
    assert len(difference) == 2
    assert review.boolean_solid_volume(difference) == pytest.approx(32)
    assert review.boolean_solid_volume(pair) == pytest.approx(16)


def test_real_complete_difference_and_disjoint_difference():
    small = b.Box(2, 2, 2).solids()[0]
    empty = small - b.Box(4, 4, 4).solids()[0]
    assert isinstance(empty, b.ShapeList) and not empty
    assert review.boolean_solid_volume(empty) == 0
    unchanged = small - b.Pos(4, 0, 0) * b.Box(2, 2, 2)
    assert review.boolean_solid_volume(unchanged) == pytest.approx(8)


@pytest.mark.parametrize("unknown", [object(), [], {}, 0, False, "empty"])
def test_unknown_boolean_result_never_scores_zero(unknown):
    with pytest.raises(TypeError, match="Unsupported boolean result type"):
        review.boolean_solid_volume(unknown)


def test_unknown_shapelist_member_never_scores_zero():
    with pytest.raises(TypeError, match="Unsupported boolean result type"):
        review.boolean_solid_volume(b.ShapeList([object()]))


def test_partial_enclosure_reports_missing_features_without_exception(tmp_path):
    """Correct envelopes cannot stand in for cavity, ports or blind pilot geometry."""
    partial = b.Compound(
        children=[
            b.Pos(0, 0, 18) * b.Box(108, 76, 36),
            b.Pos(0, 0, 38.4) * b.Box(108, 76, 4),
            b.Pos(0, 0, 15.8) * b.Box(84, 48, 1.6),
        ]
    )
    step = tmp_path / "partial-enclosure.step"
    b.export_step(partial, str(step))
    result = review.check("development_enclosure_assembly", step)
    checks = result["checks"]
    assert checks["three_displayed_solids"]
    assert checks["108x76x40_4_assembly_bounds"]
    for missing in (
        "no_positive_volume_interference",
        "floor_and_open_cavity_samples",
        "all_four_actual_lid_clearance_holes",
        "all_four_actual_base_blind_pilots",
        "actual_PCB_blind_pilots",
        "requested_cable_window_samples",
        "eight_requested_vent_window_samples",
        "stable_part_ids_and_declared_production_roles",
    ):
        assert checks[missing] is False, missing
    assert result["declared_export_checks_pass"] is False
    assert result["full_task_success"] is None
    assert result["physical_readiness"] == "unverified"


def test_partial_impeller_handles_empty_radial_difference_truthfully(tmp_path):
    """An enclosed cylinder has no radial overshoot but is not an eight-blade impeller."""
    partial = b.Pos(0, 0, 12) * b.Cylinder(45, 24)
    step = tmp_path / "partial-impeller.step"
    b.export_step(partial, str(step))
    result = review.check("development_twisted_impeller", step)
    assert result["checks"]["no_material_outside_tip_radius45"]
    assert not result["checks"]["eight_blades_all_three_sections"]
    assert not result["checks"]["actual8mm_full_span_bore"]
    assert result["declared_export_checks_pass"] is False
