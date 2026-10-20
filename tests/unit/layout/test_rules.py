"""Shared layout rules (ADR-0006 Stage 2): pure helpers used by BOTH interpreters.

These lock the behaviour the two interpreters now depend on. The last two tests
are parity checks: the helper must reproduce the exact corner geometry each
interpreter used inline before the extraction, so the refactor is geometry-
preserving (the NEMA 17 31 mm bolt square and the Pi 4 58 x 49 standoff rect).
"""

from __future__ import annotations

import math

from forge_mind.layout.rules import centered_rect_corners, edge_land, is_floor_down
from forge_mind.types import Vec3

# --- is_floor_down -----------------------------------------------------------


def test_is_floor_down_true_for_pure_plus_z() -> None:
    assert is_floor_down(Vec3(x=0.0, y=0.0, z=1.0))


def test_is_floor_down_false_for_minus_z() -> None:
    assert not is_floor_down(Vec3(x=0.0, y=0.0, z=-1.0))


def test_is_floor_down_false_for_horizontal() -> None:
    assert not is_floor_down(Vec3(x=0.0, y=1.0, z=0.0))
    assert not is_floor_down(Vec3(x=1.0, y=0.0, z=0.0))


def test_is_floor_down_true_when_z_dominates_a_tilt() -> None:
    # z positive and >= |x|,|y| in magnitude -> still floor-down.
    assert is_floor_down(Vec3(x=0.2, y=-0.3, z=0.9))


def test_is_floor_down_boundary_z_equals_x_magnitude() -> None:
    # abs(z) >= abs(x) is inclusive: exactly equal still counts as +Z dominant.
    assert is_floor_down(Vec3(x=0.5, y=0.0, z=0.5))


def test_is_floor_down_false_when_x_exceeds_z() -> None:
    assert not is_floor_down(Vec3(x=0.6, y=0.0, z=0.5))


# --- edge_land ---------------------------------------------------------------


def test_edge_land_basic_value() -> None:
    # boundary at 30, feature centre 15.5, radius 1.6 -> land 12.9.
    assert math.isclose(edge_land(30.0, 15.5, 1.6), 12.9)


def test_edge_land_zero_at_exact_boundary() -> None:
    assert edge_land(17.1, 15.5, 1.6) == 0.0


def test_edge_land_negative_when_feature_crosses_boundary() -> None:
    # A hole whose edge pokes past the boundary yields a negative land.
    assert edge_land(15.0, 15.5, 1.6) < 0.0


def test_edge_land_centred_feature_is_half_extent_minus_radius() -> None:
    # center_offset 0 (a central bore): land = half_extent - radius.
    assert math.isclose(edge_land(30.0, 0.0, 11.0), 19.0)


# --- centered_rect_corners ---------------------------------------------------


def test_centered_rect_corners_order_and_tags() -> None:
    corners = centered_rect_corners(10.0, 20.0)
    assert [tag for tag, _, _ in corners] == ["nw", "ne", "sw", "se"]


def test_centered_rect_corners_signs() -> None:
    by_tag = {tag: (x, y) for tag, x, y in centered_rect_corners(10.0, 20.0)}
    assert by_tag["nw"] == (-5.0, 10.0)
    assert by_tag["ne"] == (5.0, 10.0)
    assert by_tag["sw"] == (-5.0, -10.0)
    assert by_tag["se"] == (5.0, -10.0)


def test_centered_rect_corners_square_reproduces_nema17_bolt_square() -> None:
    # Parity with interpreter.py: a 31mm square -> bolts at +/-15.5 in both axes.
    corners = centered_rect_corners(31.0, 31.0)
    for _tag, x, y in corners:
        assert abs(abs(x) - 15.5) < 1e-9
        assert abs(abs(y) - 15.5) < 1e-9


def test_centered_rect_corners_reproduces_pi4_standoff_rectangle() -> None:
    # Parity with pi4_interpreter.py: the 58 x 49 hole rectangle.
    by_tag = {tag: (x, y) for tag, x, y in centered_rect_corners(58.0, 49.0)}
    assert by_tag["ne"] == (29.0, 24.5)
    assert by_tag["sw"] == (-29.0, -24.5)
