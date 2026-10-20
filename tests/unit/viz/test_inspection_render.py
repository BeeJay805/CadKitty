import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh
from PIL import Image

from forge_mind.viz import render


def test_inspection_keeps_complete_view_frame_and_every_selected_triangle(tmp_path, monkeypatch):
    mesh = trimesh.creation.box([4, 6, 8])
    mesh.apply_translation([20, -10, 5])
    source = tmp_path / "selected.stl"
    mesh.export(source)
    bounds = np.array([[-30, -40, -15], [40, 20, 25]])
    captured = []
    original = render.rasterize_orthographic

    def collect(vertices, colors, radius):
        captured.append(vertices.copy())
        assert radius == pytest.approx(np.linalg.norm(bounds[1] - bounds[0]) / 2 * 1.1)
        return original(vertices, colors, radius)

    monkeypatch.setattr(render, "rasterize_orthographic", collect)
    output = render.render_workspace_preview(
        source,
        tmp_path / "inspection.png",
        frame_bounds=bounds,
        inspection_label="Reference-excluded inspection · 2 bodies hidden",
    )
    assert Image.open(output).size == (1800, 1200)
    loaded = trimesh.load(source, force="mesh")
    triangles = loaded.triangles - bounds.mean(axis=0)
    for vertices, (_, elevation, azimuth) in zip(captured, render._VIEWS, strict=True):
        projected = triangles @ render._orthographic_basis(elevation, azimuth)
        assert vertices == pytest.approx(projected)
    assert len(captured) == 6


@pytest.mark.parametrize(
    "bounds",
    [
        [[1, 1, 1], [-1, -1, -1]],
        [[-1, -1, -1], [float("nan"), 1, 1]],
        [[-1, -1, -1], [float("inf"), 1, 1]],
        [[-1, -1], [1, 1]],
        [[0, 0, 0], [0.1, 0.1, 0.1]],
    ],
)
def test_bad_or_cropping_frame_rejected_without_leaking_figures(tmp_path, bounds):
    source = tmp_path / "box.stl"
    trimesh.creation.box().export(source)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="frame"):
        render.render_workspace_preview(source, tmp_path / "out.png", frame_bounds=np.array(bounds))
    assert plt.get_fignums() == before
    assert not (tmp_path / "out.png").exists()
