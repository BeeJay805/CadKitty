"""Pure coverage-bound and replay controls; no native CAD construction."""

from __future__ import annotations

import copy
from itertools import pairwise

import pytest
from tests.unit.workspace.test_photo_profile import image_bytes, profile_request

from forge_mind.workspace.photo_profile import analyze_profile, profile_digest
from forge_mind.workspace.photo_profile_native import _observed_loops, compare_profile_curves


def square(x=0):
    return {"role": "outer", "points_mm": [[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]}


def square_saved(x=0):
    points = square(x)["points_mm"]
    return {
        "role": "outer",
        "curves": [{"kind": "line", "a": a, "b": b} for a, b in pairwise(points)],
    }


def test_complete_boundary_bound_contains_exact_matching_and_shifted_truth():
    same = compare_profile_curves([square()], [square_saved()])
    assert same["symmetric_residual_interval_mm"][0] == 0
    assert same["symmetric_residual_interval_mm"][1] <= 0.010001
    shifted = compare_profile_curves([square()], [square_saved(0.25)])
    low, high = shifted["symmetric_residual_interval_mm"]
    assert low <= 0.25 <= high
    assert low > 0.24
    assert "pass" not in shifted.values()
    assert all(item["complete_boundary_length_mm"] == 4 for item in same["directed_residuals"])


def test_extra_hole_retained_and_detected_without_outer_loop_masking():
    outer = square_saved()
    hole = {
        "role": "inner",
        "curves": [{"kind": "circle", "center_mm": [0.5, 0.5], "radius_mm": 0.1}],
    }
    result = compare_profile_curves([square()], [outer, hole])
    assert result["inner_loop_count_difference"] == 1
    assert result["symmetric_residual_interval_mm"][0] > 0.39


def test_source_metric_projection_is_replayed_not_trusted_by_label():
    image = image_bytes(holes=True)
    observation = analyze_profile(image, profile_request(image))
    assert len(_observed_loops(observation)) == 3
    for mutation in ("matrix", "metric", "source", "request"):
        changed = copy.deepcopy(observation)
        if mutation == "matrix":
            changed["calibration"]["pixel_to_plane_mm"][0][2] += 1
        elif mutation == "metric":
            changed["components"][0]["loops"][0]["points_mm"][1][0] += 1
        elif mutation == "source":
            changed["vision_image_sha256"] = "f" * 64
        else:
            changed["request"]["threshold"] = 100
        # Even a new self-computed digest cannot bypass replay/binding checks.
        changed.pop("observation_sha256")
        changed["observation_sha256"] = profile_digest(changed)
        with pytest.raises(ValueError):
            _observed_loops(changed)


def test_no_metric_observation_cannot_be_compared_as_metric():
    image = image_bytes()
    result = analyze_profile(image, profile_request(image, calibration_anchors=[], check_points=[]))
    with pytest.raises(ValueError, match="four calibration"):
        _observed_loops(result)


def test_clipped_outline_cannot_gain_coverage_by_rehashing():
    image = image_bytes(clipped=True)
    observation = analyze_profile(image, profile_request(image))
    with pytest.raises(ValueError, match="complete selected"):
        _observed_loops(observation)


def test_huge_boundary_refuses_instead_of_silently_undersampling():
    observed = [{"points_mm": [[0, 0], [100000, 0], [100000, 1], [0, 0]]}]
    with pytest.raises(ValueError, match="sample budget"):
        compare_profile_curves(observed, [square_saved()])
