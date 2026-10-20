"""Pure geometry and byte-boundary controls; these tests do not render."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
import sys
import zlib

import numpy as np
import pytest

from forge_mind.workspace import product_photo_render as worker
from forge_mind.workspace.product_photo_contract import FRAME, SCOPE, VERSION


def box_points(size, shift=(0, 0, 0)):
    return np.array(list(itertools.product(*[(-x / 2, x / 2) for x in size]))) + shift


@pytest.mark.parametrize(
    "size,shift",
    [
        ((80, 50, 6), (0, 0, 0)),
        ((1000, 2, 1), (0, 0, 0)),
        ((2, 1, 1000), (0, 0, 0)),
        ((80, 50, 6), (100_000, -80_000, 1200)),
    ],
)
def test_camera_contains_every_actual_point_with_margin(size, shift):
    points = box_points(size, shift)
    before = points.copy()
    camera = worker.fitted_camera(points)
    position, focal, up = np.asarray(camera["position"])
    forward = focal - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    relative = points - position
    depth = relative @ forward
    assert np.all(depth > 0)
    half_angle = math.tan(math.radians(camera["view_angle_degrees"] / 2))
    projected = np.column_stack((relative @ right, relative @ actual_up)) / (
        depth[:, None] * half_angle
    )
    assert np.max(np.abs(projected)) <= 1 / 1.2 + 1e-10
    assert camera["projection"] == "perspective"
    assert np.array_equal(points, before)


def test_camera_translation_preserves_fit():
    points = box_points((80, 50, 6))
    shift = np.array([125_000, -210_000, 8000])
    a, b = worker.fitted_camera(points), worker.fitted_camera(points + shift)
    assert b["distance_mm"] == pytest.approx(a["distance_mm"], abs=1e-9)
    assert np.asarray(b["position"])[:2] == pytest.approx(np.asarray(a["position"])[:2] + shift)
    assert b["maximum_absolute_ndc"] == pytest.approx(a["maximum_absolute_ndc"])


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[1, 2, 3]],
        [[0, 0, float("nan")], [1, 1, 1]],
        [[-1e308, 0, 0], [1e308, 1, 1]],
        [[1, 2], [3, 4]],
    ],
)
def test_camera_refuses_invalid_and_overflowing_inputs(points):
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        worker.fitted_camera(points)


def test_triangles_keep_exact_coordinates_without_repair():
    source = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype="<f8")
    before = source.tobytes()
    result = worker.validated_triangles(source)
    assert result.tobytes() == before
    result[0, 0, 0] = 2
    assert source.tobytes() == before


@pytest.mark.parametrize(
    "triangles",
    [
        [],
        [[[0, 0, 0], [1, 0, 0], [2, 0, 0]]],
        [[[0, 0, 0], [float("inf"), 0, 0], [0, 1, 0]]],
        [[[0, 0, 0], [1e200, 0, 0], [0, 1e200, 0]]],
        [[[0, 0, 0], [1e-200, 0, 0], [0, 1e-200, 0]]],
    ],
)
def test_triangles_refuse_empty_degenerate_nonfinite_and_extreme_scales(triangles):
    with pytest.raises(ValueError):
        worker.validated_triangles(triangles)


def test_triangle_limit_is_not_repaired(monkeypatch):
    monkeypatch.setattr(worker, "MAX_TRIANGLES", 1)
    with pytest.raises(ValueError, match="excessive"):
        worker.validated_triangles([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]] * 2)


class Context:
    def __init__(self, supported):
        self.supported = supported

    def SupportsOpenGL(self):  # noqa: N802 - match the VTK API
        return self.supported

    def GetOpenGLSupportMessage(self):  # noqa: N802 - match the VTK API
        return "No supported graphics context"

    def ReportCapabilities(self):  # noqa: N802 - match the VTK API
        return "Vendor: fixture\nRenderer: fixture\nVersion: fixture\nExtensions: " + "x" * 5000


def test_context_requires_positive_support_and_omits_extensions():
    assert (
        worker.require_graphics_context(Context(1))
        == "Vendor: fixture\nRenderer: fixture\nVersion: fixture"
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        worker.require_graphics_context(Context(0))


def test_optional_renderer_unavailable_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(RuntimeError, match="optional PyVista/VTK renderer could not load"):
        worker._render(b"not parsed", "metal")


def png_bytes():
    def chunk(kind, content):
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    scanline = b"\0" + bytes([160, 170, 180]) * 1024
    return (
        worker.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 1024, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanline * 1024))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def staged(tmp_path):
    mesh = b"fixture binary STL".ljust(80, b"\0") + struct.pack(
        "<I12fH", 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0
    )
    (tmp_path / "part.stl").write_bytes(mesh)
    request = {
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "mesh_sha256": hashlib.sha256(mesh).hexdigest(),
        "preset": "metal",
        "frame": FRAME,
        "version": VERSION,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    rendering = {
        "versions": {"renderer": "mocked byte-boundary fixture"},
        "camera": worker.fitted_camera([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        "texture": {"scope": "mocked fixture"},
        "ssao": {"scope": "mocked fixture"},
        "lights": [{"intensity": 0.05}],
        "shadow_method": "mocked fixture",
        "geometry": {
            "triangles": 1,
            "triangle_sha256": "c" * 64,
            "triangle_positions_unchanged": True,
            "bounds_mm": [[0, 0, 0], [1, 1, 0]],
        },
    }
    return path, request, mesh, rendering


def test_worker_publishes_exact_source_bound_receipt_and_only_two_outputs(staged, monkeypatch):
    path, request, mesh, rendering = staged
    image = png_bytes()
    seen = []

    def fake_render(content, preset):
        seen.append((content, preset))
        return image, rendering

    monkeypatch.setattr(worker, "_render", fake_render)
    result = worker.render_product_photo(path)
    assert seen == [(mesh, "metal")]
    assert {k: result[k] for k in request} == request
    assert result["scope"] == SCOPE
    assert result["rendering"] == rendering
    assert result["image"] == {
        "file": "photo.png",
        "sha256": hashlib.sha256(image).hexdigest(),
        "width": 1024,
        "height": 1024,
    }
    assert (path.parent / "photo.png").read_bytes() == image
    assert json.loads((path.parent / "receipt.json").read_bytes()) == result
    assert (path.parent / "part.stl").read_bytes() == mesh
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "part.stl",
        "photo.png",
        "receipt.json",
        "request.json",
    ]


def test_worker_refuses_changed_source_before_optional_render(staged, monkeypatch):
    path, _, _, _ = staged
    (path.parent / "part.stl").write_bytes(b"changed source")
    monkeypatch.setattr(worker, "_render", lambda *_: pytest.fail("renderer must not run"))
    with pytest.raises(ValueError, match="source hash"):
        worker.render_product_photo(path)
    assert not (path.parent / "photo.png").exists()


@pytest.mark.parametrize("changed", ["mesh", "request"])
def test_worker_refuses_midrender_input_change(staged, monkeypatch, changed):
    path, _, _, rendering = staged

    def fake_render(*_):
        target = path if changed == "request" else path.parent / "part.stl"
        target.write_bytes(target.read_bytes() + b" ")
        return png_bytes(), rendering

    monkeypatch.setattr(worker, "_render", fake_render)
    with pytest.raises(ValueError, match="changed during rendering"):
        worker.render_product_photo(path)
    assert not (path.parent / "photo.png").exists()
    assert not (path.parent / "receipt.json").exists()


def test_worker_refuses_existing_output(staged, monkeypatch):
    path, _, _, _ = staged
    image = path.parent / "photo.png"
    image.write_bytes(b"preserved")
    monkeypatch.setattr(worker, "_render", lambda *_: pytest.fail("renderer must not run"))
    with pytest.raises(ValueError, match="already exists"):
        worker.render_product_photo(path)
    assert image.read_bytes() == b"preserved"


def test_worker_refuses_duplicate_request_key(staged):
    path, _, _, _ = staged
    path.write_text(path.read_text(encoding="utf-8")[:-1] + ', "version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        worker.render_product_photo(path)


def test_worker_refuses_oversized_image_before_publication(staged, monkeypatch):
    path, _, _, rendering = staged
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 10)
    monkeypatch.setattr(worker, "_render", lambda *_: (png_bytes(), rendering))
    with pytest.raises(ValueError, match="oversized"):
        worker.render_product_photo(path)
    assert not (path.parent / "photo.png").exists()


def test_binary_triangle_limit_precedes_optional_render(staged, monkeypatch):
    path, _, _, _ = staged
    monkeypatch.setattr(worker, "MAX_TRIANGLES", 0)
    monkeypatch.setattr(worker, "_render", lambda *_: pytest.fail("renderer must not run"))
    with pytest.raises(ValueError, match="triangle count"):
        worker.render_product_photo(path)
