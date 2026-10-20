"""Deterministic placement patterns (pure geometry, no LLM, no compile)."""

from __future__ import annotations

import math

from forge_mind.layout.placement import LayoutPattern, derive_positions


def _sep_ok(positions: list[tuple[float, float]], min_center_dist: float) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if math.hypot(dx, dy) < min_center_dist - 1e-9:
                return False
    return True


def test_bolt_circle_places_n_on_one_pitch_circle() -> None:
    # The round-plate analog of the corner pattern: N features equidistant from the centre.
    pos = derive_positions(LayoutPattern.BOLT_CIRCLE, 6, 1.7, min_wall=4.0, slack=6.0)
    assert len(pos) == 6
    radii = {round(math.hypot(x, y), 6) for x, y in pos}
    assert len(radii) == 1 and radii.pop() > 0.0  # all on ONE pitch circle
    assert _sep_ok(pos, 2 * 1.7 + 4.0)  # adjacent holes keep >= 2r + min_wall


def test_bolt_circle_clears_a_central_feature() -> None:
    central_r, r, m = 11.0, 1.7, 4.0
    pos = derive_positions(
        LayoutPattern.BOLT_CIRCLE, 4, r, min_wall=m, central_radius=central_r, slack=6.0
    )
    for x, y in pos:
        assert math.hypot(x, y) >= central_r + m + r - 1e-9  # pitch radius clears the centre


def test_single_feature_is_centered() -> None:
    assert derive_positions(LayoutPattern.CENTERED, 1, 2.0, min_wall=4.0) == [(0.0, 0.0)]
    # any pattern collapses to centered for a lone feature on an empty plate
    assert derive_positions(LayoutPattern.CORNER, 1, 2.0, min_wall=4.0) == [(0.0, 0.0)]


def test_end_offset_pushes_single_to_plus_x() -> None:
    pos = derive_positions(LayoutPattern.END_OFFSET, 1, 2.0, min_wall=4.0)
    assert len(pos) == 1 and pos[0][0] > 0.0 and pos[0][1] == 0.0


def test_corner_four_is_a_2x2_square() -> None:
    pos = derive_positions(LayoutPattern.CORNER, 4, 1.7, min_wall=4.0, slack=6.0)
    assert len(pos) == 4
    xs = sorted({round(x, 6) for x, _ in pos})
    ys = sorted({round(y, 6) for _, y in pos})
    assert len(xs) == 2 and len(ys) == 2  # exactly two distinct x and y -> a rectangle
    assert xs[0] == -xs[1] and ys[0] == -ys[1]  # centered
    assert _sep_ok(pos, 2 * 1.7 + 4.0)  # holes keep >= 2r+min_wall apart


def test_corner_two_is_a_pair() -> None:
    pos = derive_positions(LayoutPattern.CORNER, 2, 1.7, min_wall=4.0)
    assert len(pos) == 2
    assert _sep_ok(pos, 2 * 1.7 + 4.0)


def test_row_is_collinear() -> None:
    pos = derive_positions(LayoutPattern.ROW, 4, 1.7, min_wall=4.0, slack=6.0)
    assert len({round(y, 6) for _, y in pos}) == 1  # all same y -> a line
    assert _sep_ok(pos, 2 * 1.7 + 4.0)


def test_grid_nine_is_3x3() -> None:
    pos = derive_positions(LayoutPattern.GRID, 9, 1.5, min_wall=3.0, slack=4.0)
    assert len(pos) == 9
    assert len({round(x, 6) for x, _ in pos}) == 3
    assert len({round(y, 6) for _, y in pos}) == 3
    assert _sep_ok(pos, 2 * 1.5 + 3.0)


def test_corner_clears_a_central_feature() -> None:
    # bearing (central r=11) + 4 holes: every hole centre must clear the bearing disk + margin.
    central_r, r, m = 11.0, 1.7, 4.0
    pos = derive_positions(
        LayoutPattern.CORNER, 4, r, min_wall=m, central_radius=central_r, slack=6.0
    )
    for x, y in pos:
        assert math.hypot(x, y) >= central_r + m + r - 1e-9


def test_single_peripheral_with_central_is_offset_not_centered() -> None:
    # a lone hole alongside a bearing must NOT sit at the origin (it would collide).
    pos = derive_positions(LayoutPattern.CORNER, 1, 1.7, min_wall=4.0, central_radius=11.0)
    assert pos != [(0.0, 0.0)]
    assert math.hypot(*pos[0]) >= 11.0 + 4.0 + 1.7 - 1e-9


def test_perimeter_spaces_many_features() -> None:
    pos = derive_positions(LayoutPattern.PERIMETER, 8, 1.5, min_wall=3.0, slack=4.0)
    assert len(pos) == 8
    assert _sep_ok(pos, 2 * 1.5 + 3.0)


def test_empty() -> None:
    assert derive_positions(LayoutPattern.CORNER, 0, 1.0, min_wall=4.0) == []
