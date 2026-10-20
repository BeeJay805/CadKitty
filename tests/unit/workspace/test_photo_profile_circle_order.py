"""Prepared pure circle-traversal controls. Do not run before root release."""

from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from forge_mind.workspace.photo_profile import _fit_loop, analyze_profile
from forge_mind.workspace.photo_profile_models import PhotoProfileRequest


def request(tolerance=0.2):
    return PhotoProfileRequest(
        vision_image_sha256="0" * 64,
        foreground="dark",
        threshold=127,
        fit_tolerance_mm=tolerance,
        assumptions={
            "planar": True,
            "distortion": "negligible_declared",
            "source_origin": "independent analytic regression",
        },
        allow_circle_fits=True,
    )


def closed(vertices):
    return np.asarray([*vertices, vertices[0]], dtype=float)


def polar(radius, angle_degrees):
    angle = math.radians(angle_degrees)
    return [radius * math.cos(angle), radius * math.sin(angle)]


def fitted_kind(points, tolerance=0.2):
    try:
        return _fit_loop(points, request(tolerance))["kind"]
    except ValueError:
        return "refused"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seam", [0, 137])
def test_thin_c_ring_cannot_become_a_filled_circle(reverse, seam):
    angles = [5 + 0.5 * i for i in range(701)]
    vertices = [polar(10.05, angle) for angle in angles]
    vertices += [polar(9.95, angle) for angle in reversed(angles)]
    if reverse:
        vertices.reverse()
    vertices = vertices[seam:] + vertices[:seam]
    points = closed(vertices)
    before = points.copy()
    assert fitted_kind(points) != "circle"
    assert np.array_equal(points, before)


@pytest.mark.parametrize("reverse", [False, True])
def test_genuine_near_circle_accepts_both_windings(reverse):
    vertices = [polar(10 + 0.01 * math.sin(math.radians(3 * i)), i * 0.5) for i in range(720)]
    if reverse:
        vertices.reverse()
    points = closed(vertices)
    before = points.copy()
    assert fitted_kind(points, 0.1) == "circle"
    assert np.array_equal(points, before)


def test_zero_winding_nonmonotone_traversal_refuses_circle():
    arc = [polar(10, angle) for angle in range(356)]
    assert fitted_kind(closed(arc + list(reversed(arc)))) != "circle"


def test_double_boundary_traversal_refuses_circle():
    points = closed([polar(10, i * 0.5) for i in range(1440)])
    assert fitted_kind(points) != "circle"


def test_shuffled_global_angular_coverage_is_not_ordered_coverage():
    vertices = [polar(10, i * 2) for i in range(180)]
    shuffled = vertices[::2] + vertices[1::2]
    assert fitted_kind(closed(shuffled)) != "circle"


def test_actual_neighbor_long_chord_cannot_use_sorted_neighbor_bound():
    vertices = [polar(10, i * 10) for i in range(36)]
    vertices[4], vertices[14] = vertices[14], vertices[4]
    assert fitted_kind(closed(vertices)) != "circle"


def test_valid_ordered_gap_has_its_actual_chord_bound():
    points = closed([polar(10, i) for i in range(336)])
    result = _fit_loop(points, request(0.3))
    assert result["kind"] == "circle"
    expected = 10 * (1 - math.cos(math.radians(25) / 2))
    assert result["contour_chord_guard_mm"] >= expected - 1e-10
    assert result["contour_chord_guard_mm"] <= 0.3


def test_real_raster_circle_stairs_remain_supported():
    image = Image.new("RGB", (240, 240), "white")
    ImageDraw.Draw(image).ellipse((25, 25, 215, 215), fill="black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    raw = stream.getvalue()

    def anchor(x, y):
        return {"pixel": [x, y], "plane_mm": [x / 10, y / 10]}

    body = request(0.1).model_dump(mode="json")
    body.update(
        vision_image_sha256=hashlib.sha256(raw).hexdigest(),
        calibration_anchors=[anchor(5, 5), anchor(235, 5), anchor(235, 235), anchor(5, 235)],
        check_points=[anchor(120, 10)],
        thickness_mm=2.0,
        through_cuts=True,
    )
    result = analyze_profile(raw, body)
    assert result["components"][0]["loops"][0]["fit"]["kind"] == "circle"
    assert result["proposal_document"] is not None
