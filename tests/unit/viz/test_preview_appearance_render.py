import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh
from PIL import Image

from forge_mind.viz import render
from forge_mind.viz.raster import rasterize_orthographic
from forge_mind.workspace.preview_appearance import preview_appearance


def test_colored_render_keeps_every_triangle_and_literal_bounded_legend(tmp_path, monkeypatch):
    front = trimesh.creation.box([4, 2, 3])
    rear = front.copy()
    rear.apply_translation([0, 5, 0])
    mesh = trimesh.util.concatenate([front, rear])
    palette = np.vstack([np.tile([1, 0, 0], (12, 1)), np.tile([0, 0, 1], (12, 1))])
    captured, text_objects = [], []
    raster = render.rasterize_orthographic
    save = plt.Figure.savefig

    def collect(vertices, colors, radius):
        captured.append((vertices.copy(), colors.copy()))
        return raster(vertices, colors, radius)

    def capture_texts(self, *args, **kwargs):
        text_objects.extend(t for legend in self.legends for t in legend.get_texts())
        return save(self, *args, **kwargs)

    monkeypatch.setattr(render, "rasterize_orthographic", collect)
    monkeypatch.setattr(plt.Figure, "savefig", capture_texts)
    output = render.render_workspace_preview(
        tmp_path / "unused.stl",
        tmp_path / "colored.png",
        display_mesh=mesh,
        face_colors=palette,
        legend=(
            ("#FF0000", "$\\invalid{x}$\nmake, qty 1"),
            ("#0000FF", "Device\nreference only, qty 1"),
        ),
        appearance_note="3 part labels omitted. " + "Display declarations only. " * 7,
    )
    assert Image.open(output).size == (1800, 1362)
    assert len(captured) == 6 and all(len(vertices) == 24 for vertices, _ in captured)
    expected = render._shaded_facecolors(mesh.face_normals, palette)
    assert all(colors == pytest.approx(expected) for _, colors in captured)
    assert all(not text.get_parse_math() for text in text_objects)
    assert text_objects[0].get_text() == "$\\invalid{x}$\nmake, qty 1"


def test_role_colors_are_opaque_and_depth_keeps_nearest_surface():
    face = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
    near = face + np.array([0, 0, 2])
    colors = np.array([[0, 0, 1], [1, 0, 0]])
    image = rasterize_orthographic(np.array([face, near]), colors, 2, size=32)
    assert image[16, 16].tolist() == [255, 0, 0]
    reversed_image = rasterize_orthographic(np.array([near, face]), colors[::-1], 2, size=32)
    assert np.array_equal(image, reversed_image)


@pytest.mark.parametrize("long_note", [False, True])
def test_maximum_producer_legend_and_note_fit_canvas_below_geometry(
    tmp_path, monkeypatch, long_note
):
    parts = [
        {"id": f"part{i}", "name": "W" * 200, "production": {"role": "reference"}}
        for i in range(20)
    ]
    rows = [
        {"path": f"root/{p['id']}/{i}", "definition": p["id"], "body_index": index}
        for index, (p, i) in enumerate((p, i) for p in parts for i in range(100))
    ]
    appearance = preview_appearance(
        {"assembly": {"parts": parts}}, {"occurrences": rows}, len(rows)
    )
    assert len(appearance.legend) == 8 and appearance.omitted_labels == 12
    assert all(
        "qty 100" in label and len(label.splitlines()[0]) == 24 for _, label in appearance.legend
    )
    note = (
        "W" * 240
        if long_note
        else f"{appearance.omitted_labels} part labels omitted. {appearance.diagnostic}"
    )
    save = plt.Figure.savefig
    measured = []

    def check_layout(self, *args, **kwargs):
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        legend_texts = [text for key in self.legends for text in key.get_texts()]
        note_texts = [text for text in self.texts if text is not self._suptitle]
        boxes = [text.get_window_extent(renderer) for text in legend_texts + note_texts]
        panel_bottom = min(ax.get_window_extent(renderer).y0 for ax in self.axes)
        for box in boxes:
            assert self.bbox.contains(box.x0, box.y0)
            assert self.bbox.contains(box.x1, box.y1)
            assert box.y1 < panel_bottom
        assert max(text.get_window_extent(renderer).y1 for text in note_texts) < min(
            text.get_window_extent(renderer).y0 for text in legend_texts
        )
        assert all(text.get_fontsize() == 12 for text in legend_texts)
        measured.append(True)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", check_layout)
    render.render_workspace_preview(
        tmp_path / "unused.stl",
        tmp_path / "legend.png",
        display_mesh=trimesh.creation.box(),
        face_colors=np.array([[0.4, 0.5, 0.6]]),
        legend=appearance.legend,
        appearance_note=note,
    )
    assert measured == [True]


@pytest.mark.parametrize(
    "palette", [np.zeros((2, 3)), np.full((12, 3), np.nan), np.ones((12, 4)), np.full((12, 3), 2)]
)
def test_invalid_triangle_palette_refuses_instead_of_hiding_faces(palette):
    with pytest.raises(ValueError, match="RGB"):
        render._shaded_facecolors(trimesh.creation.box().face_normals, palette)
