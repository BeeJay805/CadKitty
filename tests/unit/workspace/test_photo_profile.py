"""Independent pixel fixtures exercise measured coordinates, not physical claims."""

from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.photo_profile import _calibration, _project, analyze_profile
from forge_mind.workspace.photo_profile_models import PhotoProfileRequest


def image_bytes(*, holes=False, extra=False, clipped=False, light=False):
    image = Image.new("RGB", (120, 100), "black" if light else "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0 if clipped else 20, 20, 90, 75), fill="white" if light else "black")
    if holes:
        draw.ellipse((35, 35, 49, 49), fill="black" if light else "white")
        draw.rectangle((65, 50, 75, 60), fill="black" if light else "white")
    if extra:
        draw.rectangle((100, 82, 110, 92), fill="white" if light else "black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def profile_request(image, **changes):
    def point(x, y):
        return {"pixel": [x, y], "plane_mm": [x / 2, -y / 2]}

    return {
        "vision_image_sha256": hashlib.sha256(image).hexdigest(),
        "calibration_anchors": [point(5, 5), point(115, 5), point(115, 95), point(5, 95)],
        "check_points": [point(60, 10)],
        "foreground": "dark",
        "threshold": 127,
        "fit_tolerance_mm": 0.3,
        "assumptions": {
            "planar": True,
            "distortion": "negligible_declared",
            "source_origin": "independent_test_measurements",
        },
        "thickness_mm": 4.0,
        "through_cuts": True,
        **changes,
    }


def test_metric_profile_preserves_every_hole_and_builds_editable_separate_cuts():
    image = image_bytes(holes=True)
    result = analyze_profile(image, profile_request(image))
    assert result["status"] == "measured"
    assert result["complete_visible_component_count"] == 1
    component = result["components"][0]
    assert component["inner_loop_count"] == 2
    assert len(component["loops"]) == 3
    assert component["loops"][0]["role"] == "outer"
    outer = np.array(component["loops"][0]["points_mm"])
    assert np.ptp(outer[:, 0]) == pytest.approx(35.5, abs=1e-9)
    assert np.ptp(outer[:, 1]) == pytest.approx(28.0, abs=1e-9)
    document = DesignDocument.model_validate(result["proposal_document"])
    assert document.variables["thickness_mm"] == 4
    assert [f.kind for f in document.features].count("cut") == 2
    assert [f.kind for f in document.features].count("sketch") == 3
    assert document.outputs == ["loop_2_cut"]
    assert "requirements_satisfied" not in result
    assert "pass" not in result.values()


def test_no_calibration_retains_useful_complete_pixel_geometry_without_metric_invention():
    image = image_bytes(holes=True)
    result = analyze_profile(image, profile_request(image, calibration_anchors=[], check_points=[]))
    assert result["status"] == "advisory"
    assert result["proposal_document"] is None
    assert result["components"][0]["inner_loop_count"] == 2
    assert all(
        loop["points_px"] and "points_mm" not in loop for loop in result["components"][0]["loops"]
    )


@pytest.mark.parametrize("vertical_sign", [-1, 1])
def test_all_proposed_polygon_cutters_use_positive_xy_winding_without_rewriting_observations(
    vertical_sign,
):
    image = image_bytes(holes=True)
    request = profile_request(image, allow_circle_fits=False)
    for anchor in [*request["calibration_anchors"], *request["check_points"]]:
        anchor["plane_mm"][1] *= vertical_sign
    result = analyze_profile(image, request)
    document = DesignDocument.model_validate(result["proposal_document"])
    areas = []
    for feature in document.features:
        if feature.kind != "sketch":
            continue
        points = {point.id: (point.x, point.y) for point in feature.sketch.points}
        area = (
            sum(
                points[entity.points[0]][0] * points[entity.points[1]][1]
                - points[entity.points[1]][0] * points[entity.points[0]][1]
                for entity in feature.sketch.entities
            )
            / 2
        )
        areas.append(area)
    assert len(areas) == 3
    assert all(area > 0 for area in areas), areas
    # The inner source contours keep their observed orientation for comparison.
    source_areas = [loop["signed_area_mm2"] for loop in result["components"][0]["loops"]]
    assert all(source_areas[0] * area < 0 for area in source_areas[1:])


def test_no_check_points_are_explicitly_unverified_but_allow_estimated_proposal():
    image = image_bytes()
    result = analyze_profile(image, profile_request(image, check_points=[]))
    assert result["calibration"]["independent_check_status"] == "not_run"
    assert result["status"] == "advisory"
    assert result["proposal_document"] is not None
    assert any("unverified" in warning for warning in result["warnings"])


def test_pixel_outline_is_a_reduced_display_view_not_a_replacement_measurement():
    image = image_bytes(holes=True)
    result = analyze_profile(image, profile_request(image, calibration_anchors=[], check_points=[]))
    loops = result["components"][0]["loops"]
    assert len(loops) == 3
    for loop in loops:
        display = loop["pixel_outline"]
        assert display["points_px"][0] == display["points_px"][-1]
        assert display["display_tolerance_px"] == 0.5
        assert len(display["points_px"]) <= len(loop["points_px"])
        assert "points_mm" not in loop
    assert len(loops[0]["pixel_outline"]["points_px"]) < len(loops[0]["points_px"])
    assert len(loops[0]["points_px"]) > 200


@pytest.mark.parametrize(
    "field,value", [("thickness_mm", None), ("through_cuts", None), ("through_cuts", False)]
)
def test_hidden_depth_or_blind_semantics_never_invented(field, value):
    image = image_bytes(holes=True)
    result = analyze_profile(image, profile_request(image, **{field: value}))
    assert result["components"][0]["inner_loop_count"] == 2
    assert result["proposal_document"] is None


def test_no_holes_can_explicitly_declare_no_through_cuts():
    image = image_bytes()
    result = analyze_profile(image, profile_request(image, through_cuts=False))
    assert result["proposal_document"] is not None


def test_supplied_wrong_independent_check_blocks_metric_proposal():
    image = image_bytes()
    request = profile_request(image)
    request["check_points"][0]["plane_mm"][0] += 5
    result = analyze_profile(image, request)
    assert result["calibration"] is None
    assert result["proposal_document"] is None
    assert any("check residual" in text for text in result["warnings"])


@pytest.mark.parametrize(
    "assumptions",
    [
        {"planar": False, "distortion": "negligible_declared", "source_origin": "unknown plane"},
        {"planar": True, "distortion": "uncorrected", "source_origin": "unknown lens"},
    ],
)
def test_unsupported_acquisition_keeps_pixels_but_no_metric_proposal(assumptions):
    image = image_bytes()
    result = analyze_profile(image, profile_request(image, assumptions=assumptions))
    assert result["components"]
    assert result["calibration"] is None
    assert result["proposal_document"] is None


def test_multiple_components_require_explicit_seed_and_inventory_remains_complete():
    image = image_bytes(extra=True)
    result = analyze_profile(image, profile_request(image))
    assert result["complete_visible_component_count"] == 2
    assert result["selected_component"] is None
    assert result["proposal_document"] is None
    selected = analyze_profile(image, profile_request(image, seed_pixel=[30, 30]))
    assert len(selected["components"]) == 2
    assert selected["selected_component"] == 1
    assert selected["proposal_document"] is not None


@pytest.mark.parametrize("seed", [[0, 0], [150, 150]])
def test_seed_cannot_select_background_or_outside_image(seed):
    image = image_bytes()
    with pytest.raises(ValueError, match="seed"):
        analyze_profile(image, profile_request(image, seed_pixel=seed))


def test_clipped_component_is_retained_without_fabricating_closed_object():
    image = image_bytes(clipped=True)
    result = analyze_profile(image, profile_request(image))
    assert result["components"][0]["touches_region_boundary"]
    assert result["proposal_document"] is None


def test_explicit_region_uses_full_normalized_coordinates_and_rejects_cropped_part():
    image = image_bytes(holes=True)
    complete = analyze_profile(image, profile_request(image, region=[10, 10, 100, 85]))
    assert complete["components"][0]["bounds_px"] == [20, 20, 90, 75]
    assert complete["proposal_document"] is not None
    cropped = analyze_profile(image, profile_request(image, region=[30, 10, 100, 85]))
    assert cropped["proposal_document"] is None
    assert cropped["components"][0]["touches_region_boundary"]


def test_explicit_light_foreground_recovers_same_geometry():
    dark, light = image_bytes(holes=True), image_bytes(holes=True, light=True)
    a = analyze_profile(dark, profile_request(dark))
    b = analyze_profile(light, profile_request(light, foreground="light"))
    assert a["components"] == b["components"]


def test_tampered_image_refused_before_measurement():
    image = image_bytes()
    with pytest.raises(ValueError, match="hash"):
        analyze_profile(image, profile_request(image, vision_image_sha256="0" * 64))


def test_check_point_cannot_reuse_fitting_anchor():
    image = image_bytes()
    request = profile_request(image)
    request["check_points"] = request["calibration_anchors"][:1]
    with pytest.raises(ValueError, match="separate"):
        analyze_profile(image, request)


def test_collinear_calibration_returns_pixels_without_metric_target():
    image = image_bytes()
    anchors = [{"pixel": [10 + i * 10, 10], "plane_mm": [i * 5, 0]} for i in range(4)]
    result = analyze_profile(image, profile_request(image, calibration_anchors=anchors))
    assert result["proposal_document"] is None
    assert result["calibration"] is None


def test_actual_perspective_transform_recovers_independent_metric_points():
    # Independent rational pinhole plane projection; no production projector builds input.
    def photograph(x, y):
        denominator = 1 + 0.01 * x + 0.003 * y
        return [
            4 * (3 * x + 0.4 * y + 15) / denominator,
            4 * (0.3 * x + 2.8 * y + 15) / denominator,
        ]

    plane = [(0, 0), (50, 0), (50, 40), (0, 40)]
    image = Image.new("RGB", (640, 560), "white")
    ImageDraw.Draw(image).polygon(
        [tuple(photograph(x, y)) for x, y in [(10, 10), (35, 10), (35, 30), (10, 30)]], fill="black"
    )
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    anchors = [{"pixel": photograph(x, y), "plane_mm": [x, y]} for x, y in plane]
    checks = [{"pixel": photograph(20, 5), "plane_mm": [20, 5]}]
    request = PhotoProfileRequest.model_validate(
        profile_request(raw, calibration_anchors=anchors, check_points=checks)
    )
    calibration = _calibration(request, 640, 560)
    truth = np.array([[7, 17], [31, 26], [42, 8]])
    recovered = _project([photograph(x, y) for x, y in truth], calibration["pixel_to_plane_mm"])
    assert np.max(np.abs(recovered - truth)) < 1e-10
    result = analyze_profile(raw, request)
    measured = np.array(result["components"][0]["loops"][0]["points_mm"])
    assert np.ptp(measured[:, 0]) == pytest.approx(25, abs=0.9)
    assert np.ptp(measured[:, 1]) == pytest.approx(20, abs=0.9)
    assert result["proposal_document"] is not None


def test_optional_circle_fit_outputs_editable_circle_and_explicit_hole():
    image = Image.new("RGB", (120, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 15, 95, 90), fill="black")
    draw.ellipse((42, 35, 62, 55), fill="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()
    result = analyze_profile(raw, profile_request(raw, allow_circle_fits=True))
    assert result["components"][0]["inner_loop_count"] == 1
    assert result["components"][0]["loops"][0]["fit"]["kind"] == "circle"
    circle = result["components"][0]["loops"][0]["fit"]
    assert circle["radius_mm"] == pytest.approx(19, abs=0.3)
    document = DesignDocument.model_validate(result["proposal_document"])
    assert document.features[0].sketch.entities[0].kind == "circle"
    assert math.isfinite(circle["contour_chord_guard_mm"])
