import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh
from PIL import Image

from forge_mind.viz import render


def test_camera_basis_preserves_handedness_and_front_right_top_bottom():
    for _, elevation, azimuth in render._VIEWS:
        basis = render._orthographic_basis(elevation, azimuth)
        assert basis.T @ basis == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(basis) == pytest.approx(1)
    # Looking from -Y: screen right is +X and screen up is +Z.
    assert np.array([1, 0, 0]) @ render._orthographic_basis(0, -90) == pytest.approx([1, 0, 0])
    assert np.array([0, 0, 1]) @ render._orthographic_basis(0, -90) == pytest.approx([0, 1, 0])
    # Looking from +X: screen right is +Y. Top and bottom must reverse depth.
    assert np.array([0, 1, 0]) @ render._orthographic_basis(0, 0) == pytest.approx([1, 0, 0])
    assert (np.array([0, 0, 1]) @ render._orthographic_basis(90, 0))[2] == pytest.approx(1)
    assert (np.array([0, 0, 1]) @ render._orthographic_basis(-90, 0))[2] == pytest.approx(-1)


def test_six_views_keep_all_triangles_colors_and_common_uncropped_scale(tmp_path, monkeypatch):
    mesh = trimesh.util.concatenate(
        [trimesh.creation.box([20, 10, 4]), trimesh.creation.icosphere(subdivisions=1, radius=3)]
    )
    mesh.apply_translation([100, -30, 70])
    source = tmp_path / "asymmetric.stl"
    mesh.export(source)
    captured = []
    original = render.rasterize_orthographic

    def collect(vertices, colors, radius):
        captured.append((vertices.copy(), colors.copy(), radius))
        return original(vertices, colors, radius)

    monkeypatch.setattr(render, "rasterize_orthographic", collect)
    before = plt.get_fignums()
    output = render.render_workspace_preview(source, tmp_path / "preview.png")
    assert plt.get_fignums() == before
    assert Image.open(output).size == (1800, 1200)
    assert len(captured) == 6
    loaded = trimesh.load(source, force="mesh")
    triangles = loaded.triangles - loaded.bounds.mean(axis=0)
    colors = render._shaded_facecolors(loaded.face_normals)
    radius = np.linalg.norm(loaded.extents) / 2 * 1.1
    for (vertices, facecolors, frame_radius), (_, elevation, azimuth) in zip(
        captured, render._VIEWS, strict=True
    ):
        projected = triangles @ render._orthographic_basis(elevation, azimuth)
        assert len(vertices) == len(mesh.faces)
        assert vertices == pytest.approx(projected)
        assert facecolors == pytest.approx(colors)
        assert frame_radius == pytest.approx(radius)
        assert np.max(np.abs(vertices[:, :, :2])) < radius


def test_render_failure_closes_figure(tmp_path, monkeypatch):
    source = tmp_path / "box.stl"
    trimesh.creation.box().export(source)
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(render, "rasterize_orthographic", fail)
    with pytest.raises(OSError, match="write failed"):
        render.render_workspace_preview(source, tmp_path / "preview.png")
    assert plt.get_fignums() == before
