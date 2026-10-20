"""Real focus pixels retain opaque context and saved world placement."""

from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh
from PIL import Image

from forge_mind.viz import focus
from forge_mind.workspace.occurrence_focus import FOCUS_SCOPE, focus_view_metadata


def frame_for(mesh, lower, upper, view="front"):
    bounds = {
        "min": np.asarray(lower, dtype=float).tolist(),
        "max": np.asarray(upper, dtype=float).tolist(),
    }
    return {
        "request": {"kind": "face", "signature": "a" * 64, "view": view, "padding": 2.0},
        "body_index": 0,
        "body_signature": "b" * 64,
        "entity_id": "small_face",
        "entity_bounds": bounds,
        "body_bounds": {"min": mesh.bounds[0].tolist(), "max": mesh.bounds[1].tolist()},
        "center_mm": ((np.asarray(lower) + upper) / 2).tolist(),
        "radius_mm": float(np.linalg.norm(np.asarray(upper) - lower)),
        "scope": FOCUS_SCOPE,
    }


def tiny_feature():
    post = trimesh.creation.box([2, 2, 2])
    post.apply_translation([35, 0, 2])
    mesh = trimesh.util.concatenate([trimesh.creation.box([100, 60, 2]), post])
    return mesh, frame_for(mesh, [34, -1, 1], [36, -1, 3])


def capture_rasters(monkeypatch):
    rows = []
    original = focus.rasterize_orthographic

    def collect(triangles, colors, radius, size):
        pixels = original(triangles, colors, radius, size=size)
        rows.append((triangles.copy(), colors.copy(), radius, pixels.copy()))
        return pixels

    monkeypatch.setattr(focus, "rasterize_orthographic", collect)
    return rows


def region_pixels(pixels, frame, center, radius, basis):
    lower, upper = frame["entity_bounds"]["min"], frame["entity_bounds"]["max"]
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    projected = (corners - center) @ basis
    coordinates = (projected[:, :2] / radius + 1) * len(pixels) / 2
    start = np.maximum(np.floor(coordinates.min(axis=0)).astype(int), 0)
    stop = np.minimum(np.ceil(coordinates.max(axis=0)).astype(int), len(pixels))
    patch = pixels[start[1] : stop[1], start[0] : stop[0]]
    return np.count_nonzero(np.any(patch < 240, axis=2))


def test_tiny_feature_gains_real_pixel_area_and_keeps_same_camera(tmp_path, monkeypatch):
    mesh, frame = tiny_feature()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    rows = capture_rasters(monkeypatch)
    output = tmp_path / "focus.png"
    metadata = focus.render_occurrence_focus(mesh, output, frame)
    assert metadata == focus_view_metadata(frame)
    assert len(rows) == 2
    assert all(len(row[0]) == len(mesh.faces) for row in rows)
    basis = np.asarray(metadata["basis"])
    centers = [metadata["context_center_mm"], metadata["detail_center_mm"]]
    for row, center in zip(rows, centers, strict=True):
        assert row[0] == pytest.approx((mesh.triangles - center) @ basis)
    offset = (np.asarray(centers[0]) - centers[1]) @ basis
    assert rows[1][0] - rows[0][0] == pytest.approx(np.broadcast_to(offset, rows[0][0].shape))
    context = region_pixels(rows[0][3], frame, centers[0], rows[0][2], basis)
    detail = region_pixels(rows[1][3], frame, centers[1], rows[1][2], basis)
    assert context > 0
    assert detail > context * 50
    assert mesh.vertices == pytest.approx(vertices)
    assert np.array_equal(mesh.faces, faces)
    with Image.open(output) as image:
        assert image.size == (1680, 1008)
    assert 0 < output.stat().st_size <= focus.MAX_FOCUS_IMAGE_BYTES


def test_same_body_occluder_outside_focus_box_stays_opaque(tmp_path, monkeypatch):
    target = trimesh.creation.box([2, 2, 1])
    target.apply_translation([0, 0, -3])
    cover = trimesh.creation.box([10, 10, 1])
    # One selected display body contains both the hidden target and outer material.
    mesh = trimesh.util.concatenate([cover, target])
    frame = frame_for(mesh, [-1, -1, -3.5], [1, 1, -2.5], view="top")
    rows = capture_rasters(monkeypatch)
    focus.render_occurrence_focus(mesh, tmp_path / "opaque.png", frame)
    assert all(len(row[0]) == len(mesh.faces) for row in rows)
    pixels = rows[1][3]
    assert np.mean(np.any(pixels < 240, axis=2)) > 0.99
    # A target-only crop would show white around it: this is the meaningful negative control.
    metadata = focus_view_metadata(frame)
    expected = focus.rasterize_orthographic(
        (target.triangles - metadata["detail_center_mm"]) @ np.asarray(metadata["basis"]),
        focus._shaded_facecolors(target.face_normals),
        metadata["detail_radius_mm"],
        size=focus.FOCUS_PANEL_PIXELS,
    )
    assert np.mean(np.all(expected == 255, axis=2)) > 0.3


def test_translated_selected_body_keeps_identical_view_pixels(tmp_path, monkeypatch):
    mesh, frame = tiny_feature()
    rows = capture_rasters(monkeypatch)
    original = focus.render_occurrence_focus(mesh, tmp_path / "original.png", frame)
    offset = np.asarray([120, -65, 42])
    shifted = mesh.copy()
    shifted.apply_translation(offset)
    moved = frame_for(shifted, np.asarray([34, -1, 1]) + offset, np.asarray([36, -1, 3]) + offset)
    updated = focus.render_occurrence_focus(shifted, tmp_path / "translated.png", moved)
    assert updated["detail_center_mm"] == pytest.approx(
        np.asarray(original["detail_center_mm"]) + offset
    )
    assert updated["context_center_mm"] == pytest.approx(
        np.asarray(original["context_center_mm"]) + offset
    )
    assert np.array_equal(rows[0][3], rows[2][3])
    assert np.array_equal(rows[1][3], rows[3][3])


def test_locator_marks_bounds_and_detail_window_with_literal_readable_text(tmp_path, monkeypatch):
    mesh, frame = tiny_feature()
    figures = []
    original = plt.close

    def capture(figure):
        if hasattr(figure, "axes"):
            figures.append(figure)
        original(figure)

    monkeypatch.setattr(focus.plt, "close", capture)
    metadata = focus.render_occurrence_focus(mesh, tmp_path / "labels.png", frame)
    fig = figures[-1]
    assert len(fig.axes[0].patches) == 2
    assert len(fig.axes[1].patches) == 1
    window = fig.axes[0].patches[1]
    offset = (np.asarray(metadata["detail_center_mm"]) - metadata["context_center_mm"]) @ metadata[
        "basis"
    ]
    assert window.get_xy() == pytest.approx(offset[:2] - metadata["detail_radius_mm"])
    assert window.get_width() == 2 * metadata["detail_radius_mm"]
    assert any("not an exact outline" in text.get_text() for text in fig.texts)
    for text in fig.texts:
        assert text.get_fontsize() >= 12
        bounds = text.get_window_extent(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.width
        assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height


@pytest.mark.parametrize("problem", ["outside", "nonfinite", "changed_radius"])
def test_bad_mesh_or_frame_cannot_produce_image(tmp_path, problem):
    mesh, frame = tiny_feature()
    if problem == "outside":
        mesh.apply_translation([1000, 0, 0])
    elif problem == "nonfinite":
        mesh.vertices[0, 0] = np.nan
    else:
        frame = deepcopy(frame)
        frame["radius_mm"] += 1
    with pytest.raises(ValueError):
        focus.render_occurrence_focus(mesh, tmp_path / "bad.png", frame)
    assert not (tmp_path / "bad.png").exists()


def test_failure_closes_figure_and_output_budget_is_enforced(tmp_path, monkeypatch):
    mesh, frame = tiny_feature()
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("raster failed")

    with monkeypatch.context() as context:
        context.setattr(focus, "rasterize_orthographic", fail)
        with pytest.raises(OSError, match="raster failed"):
            focus.render_occurrence_focus(mesh, tmp_path / "failed.png", frame)
    assert plt.get_fignums() == before
    monkeypatch.setattr(focus, "MAX_FOCUS_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="output bound"):
        focus.render_occurrence_focus(mesh, tmp_path / "oversize.png", frame)
    assert plt.get_fignums() == before
