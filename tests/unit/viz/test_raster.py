"""Analytic visibility controls, independent of triangle painter ordering."""

import numpy as np
import pytest

from forge_mind.viz.raster import MAX_RASTER_PIXELS, rasterize_orthographic

COLORS = np.array([[0.1, 0.7, 0.2], [0.9, 0.1, 0.1]])


def coordinates(size=256, radius=1.1):
    axis = (np.arange(size) + 0.5) * (2 * radius / size) - radius
    return np.meshgrid(axis, axis)


def overlap():
    return np.array(
        [
            [[-1, -1, -2], [1, -1, -2], [0, 1, 10]],
            [[-0.4, -0.95, 0], [0.4, -0.95, 0], [0, -0.75, 0]],
        ],
        dtype=float,
    )


def test_sloped_triangle_mean_cannot_hide_locally_nearer_nonintersecting_triangle():
    image = rasterize_orthographic(overlap(), COLORS, 1.1, 256)
    x, y = coordinates()
    # The red plane is Z=0. The green plane is Z=6Y+4, below red
    # throughout this red triangle. Their mean depths give the wrong order.
    inner = (y > -0.95 + 1e-4) & (y < -0.75 - 1e-4) & (abs(x) < 0.4 * (-0.75 - y) / 0.2 - 1e-4)
    assert inner.sum() == 1104
    assert np.all(image[inner] == np.rint(COLORS[1] * 255))


def test_crossing_planes_change_visibility_within_one_triangle():
    triangles = np.array(
        [[[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[-1, -1, -2], [1, -1, 2], [0, 1, 0]]], dtype=float
    )
    image = rasterize_orthographic(triangles, COLORS, 1.1, 256)
    x, y = coordinates()
    interior = (y > -1 + 1e-4) & (y < 1 - 1e-4) & (abs(x) < (1 - y) / 2 - 1e-4) & (abs(x) > 1e-4)
    assert interior.sum() > 26000
    # Independent plane equations: Z=0 and Z=2X.
    expected = np.rint(COLORS[(x[interior] > 0).astype(int)] * 255)
    assert np.array_equal(image[interior], expected)


def test_cavity_opening_reveals_rear_plate_only_inside_hole():
    triangles = []
    for x0, x1, y0, y1 in [
        (-1, -0.3, -1, 1),
        (0.3, 1, -1, 1),
        (-0.3, 0.3, -1, -0.3),
        (-0.3, 0.3, 0.3, 1),
    ]:
        triangles.extend(
            [[[x0, y0, 2], [x1, y0, 2], [x1, y1, 2]], [[x0, y0, 2], [x1, y1, 2], [x0, y1, 2]]]
        )
    triangles.extend([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]])
    colors = np.vstack([np.tile(COLORS[0], (8, 1)), np.tile(COLORS[1], (2, 1))])
    image = rasterize_orthographic(np.array(triangles), colors, 1.1, 256)
    x, y = coordinates()
    panel = (
        (abs(x) < 0.99) & (abs(y) < 0.99) & (abs(abs(x) - 0.3) > 1e-4) & (abs(abs(y) - 0.3) > 1e-4)
    )
    opening = (abs(x) < 0.3) & (abs(y) < 0.3)
    assert panel.sum() > 52000
    assert np.array_equal(image[panel], np.rint(COLORS[opening[panel].astype(int)] * 255))


def test_face_order_and_winding_do_not_change_non_coplanar_visibility():
    triangles = overlap()
    expected = rasterize_orthographic(triangles, COLORS, 1.1, 256)
    assert np.array_equal(rasterize_orthographic(triangles[::-1], COLORS[::-1], 1.1, 256), expected)
    assert np.array_equal(rasterize_orthographic(triangles[:, ::-1], COLORS, 1.1, 256), expected)


def test_every_visible_triangle_survives_multiple_batches_without_mesh_sampling():
    triangles = []
    for y in range(65):
        for x in range(65):
            x, y_center = x - 32, y - 32
            triangles.append(
                [
                    [x - 0.35, y_center - 0.3, 0],
                    [x + 0.35, y_center - 0.3, 0],
                    [x, y_center + 0.35, 0],
                ]
            )
    ids = np.arange(1, len(triangles) + 1)
    palette = np.column_stack([ids % 256, ids // 256, np.full(len(ids), 128)]).astype(np.uint8)
    image = rasterize_orthographic(np.array(triangles), palette / 255, 50)

    def packed(colors):
        return (
            (colors[..., 0].astype(np.uint32) << 16)
            | (colors[..., 1].astype(np.uint32) << 8)
            | colors[..., 2]
        )

    assert len(triangles) == 4225
    assert set(packed(palette)) <= set(np.unique(packed(image)))


def test_shared_edges_have_no_white_cracks_and_edge_on_faces_cover_no_pixels():
    triangles = np.array(
        [
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
            [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            [[0, -1, -1], [0, 1, -1], [0, 1, 1]],
        ],
        dtype=float,
    )
    colors = np.vstack([COLORS[0], COLORS[0], COLORS[1]])
    image = rasterize_orthographic(triangles, colors, 1.1, 256)
    x, y = coordinates()
    inside = (abs(x) < 1) & (abs(y) < 1)
    assert np.all(image[inside] == np.rint(COLORS[0] * 255))
    assert np.all(image[~inside] == 255)


@pytest.mark.parametrize(
    "field,value",
    [
        ("triangles", np.empty((0, 3, 3))),
        ("triangles", np.zeros((1, 3, 2))),
        ("triangles", np.full((2, 3, 3), np.nan)),
        ("colors", np.zeros((1, 3))),
        ("colors", np.full((2, 3), np.inf)),
        ("colors", np.full((2, 3), -0.1)),
        ("colors", np.full((2, 3), 1.1)),
        ("radius", 0),
        ("radius", -1),
        ("radius", np.inf),
        ("radius", np.nan),
        ("size", 0),
        ("size", MAX_RASTER_PIXELS + 1),
        ("size", 1.5),
        ("size", True),
    ],
)
def test_invalid_input_and_unbounded_raster_resolution_fail_before_allocating_panel(
    monkeypatch, field, value
):
    inputs = {"triangles": overlap(), "colors": COLORS, "radius": 1.1, "size": 256}
    inputs[field] = value

    def allocation_forbidden(*args, **kwargs):
        raise AssertionError("Invalid input allocated a raster buffer")

    monkeypatch.setattr(np, "full", allocation_forbidden)
    with pytest.raises(ValueError):
        rasterize_orthographic(**inputs)
