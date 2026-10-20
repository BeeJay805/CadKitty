"""Selected mesh rendering retains identity and never displays omitted bodies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import build123d as b
import numpy as np
import pytest
from PIL import Image

from forge_mind.viz import render
from forge_mind.workspace import occurrence_render as worker
from forge_mind.workspace.topology import topology_index


@pytest.fixture
def topology():
    first = b.Pos(20, 30, 40) * b.Box(10, 6, 4)
    second = b.Pos(200, -80, 90) * b.Box(20, 10, 8)
    return json.loads(
        json.dumps(topology_index(b.Compound(children=[first, second]), assembly=True))
    )


def stage(tmp_path, topology, body_index=0):
    content = json.dumps(topology).encode()
    (tmp_path / "topology.json").write_bytes(content)
    request = {
        "revision_id": "a" * 32,
        "occurrence_path": "group/part",
        "body_index": body_index,
        "topology_sha256": hashlib.sha256(content).hexdigest(),
        "step_sha256": "b" * 64,
        "document_sha256": "c" * 64,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path, request


@pytest.mark.parametrize("body_index", [0, 1])
def test_selected_occurrence_isolated_in_all_six_views(tmp_path, topology, monkeypatch, body_index):
    path, request = stage(tmp_path, topology, body_index)
    captured = []
    original = render.rasterize_orthographic

    def capture(vertices, colors, radius):
        captured.append(vertices.copy())
        return original(vertices, colors, radius)

    monkeypatch.setattr(render, "rasterize_orthographic", capture)
    receipt = worker.render_occurrence(path)
    assert all(receipt[key] == value for key, value in request.items())
    assert receipt["selected_body_count"] == 1
    assert receipt["omitted_body_count"] == 1
    assert receipt["omitted_body_indices"] == [1 - body_index]
    assert receipt["frame"] == "cad_z_up_mm_world"
    assert len(captured) == 6
    selected_mesh = next(row for row in topology["meshes"] if row["body_index"] == body_index)
    assert all(len(vertices) == len(selected_mesh["triangles"]) for vertices in captured)
    points = np.asarray(selected_mesh["vertices"])
    assert receipt["frame_bounds"]["min"] == pytest.approx(points.min(axis=0))
    assert receipt["frame_bounds"]["max"] == pytest.approx(points.max(axis=0))
    assert receipt["display"]["source_purpose"] == topology["pick_tessellation"]["purpose"]
    output = tmp_path / "inspection.png"
    assert receipt["image"] == {
        "file": "inspection.png",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt
    with Image.open(output) as image:
        assert image.size == (1800, 1200)
        pixels = np.asarray(image.convert("RGB"))
        assert np.count_nonzero(np.any(pixels < 220, axis=2)) > 20_000
    assert len(output.read_bytes()) <= worker.MAX_IMAGE_BYTES
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "inspection.png",
        "receipt.json",
        "request.json",
        "topology.json",
    ]


def test_all_selected_face_meshes_are_retained(tmp_path):
    topology = json.loads(json.dumps(topology_index(b.Box(10, 6, 4))))
    assert topology["selection_scope"] == "face_edge_body"
    path, _ = stage(tmp_path, topology)
    mesh, count, discarded = worker._selected_mesh(topology, 0)
    assert count == 1
    assert discarded == 0
    assert len(mesh.faces) == sum(len(row["triangles"]) for row in topology["meshes"])
    receipt = worker.render_occurrence(path)
    assert receipt["omitted_body_count"] == 0


@pytest.mark.parametrize("shape_name", ["sphere", "bored_chamfered_cylinder"])
def test_native_curved_topology_renders_without_source_repair(tmp_path, shape_name):
    if shape_name == "sphere":
        shape = b.Sphere(8)
    else:
        ring = b.Cylinder(8, 20).cut(b.Cylinder(4, 22))
        shape = b.chamfer(ring.edges().filter_by(b.GeomType.CIRCLE), length=0.5)
    topology = json.loads(json.dumps(topology_index(shape)))
    path, _ = stage(tmp_path, topology)
    before = (tmp_path / "topology.json").read_bytes()
    receipt = worker.render_occurrence(path)
    assert (tmp_path / "topology.json").read_bytes() == before
    assert receipt["display"]["triangles"] > 100
    discarded = receipt["display"]["discarded_degenerate_triangles"]
    assert discarded > 0 if shape_name == "sphere" else discarded == 0
    assert receipt["display"]["triangles"] + discarded == sum(
        len(row["triangles"]) for row in topology["meshes"]
    )
    assert "no surface repair or coverage certification" in receipt["display"]["triangle_filter"]
    with Image.open(tmp_path / "inspection.png") as image:
        assert image.size == (1800, 1200)


def test_collapsed_display_triangle_count_and_empty_mesh_rejection(tmp_path, topology):
    data = deepcopy(topology)
    data["meshes"][0]["triangles"].append([0, 0, 1])
    path, _ = stage(tmp_path, data)
    receipt = worker.render_occurrence(path)
    assert receipt["display"]["discarded_degenerate_triangles"] == 1
    assert receipt["display"]["triangles"] == len(topology["meshes"][0]["triangles"])
    data["meshes"][0]["triangles"] = [[0, 0, 1]]
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    path, _ = stage(rejected, data)
    with pytest.raises(ValueError, match="no nondegenerate"):
        worker.render_occurrence(path)
    assert not (rejected / "inspection.png").exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing_body",
        "duplicate_body",
        "duplicate_mesh",
        "unknown_mesh",
        "bad_index",
        "nan_vertex",
        "infinite_vertex",
        "empty_triangles",
        "missing_selected_mesh",
        "bad_frame",
        "body_signature",
        "escaped_vertex",
        "boolean_index",
    ],
)
def test_invalid_mesh_or_metadata_never_produces_image(tmp_path, topology, change):
    data = deepcopy(topology)
    selected = data["meshes"][0]
    if change == "missing_body":
        data["entities"] = [
            r for r in data["entities"] if not (r["kind"] == "body" and r["body_index"] == 0)
        ]
    elif change == "duplicate_body":
        row = deepcopy(next(r for r in data["entities"] if r["kind"] == "body"))
        row["id"] += "_duplicate"
        data["entities"].append(row)
    elif change == "duplicate_mesh":
        data["meshes"].append(deepcopy(selected))
    elif change == "unknown_mesh":
        selected["id"] = "unknown"
    elif change == "bad_index":
        selected["triangles"][0][0] = len(selected["vertices"])
    elif change in {"nan_vertex", "infinite_vertex"}:
        selected["vertices"][0][0] = float("nan") if change == "nan_vertex" else float("inf")
    elif change == "empty_triangles":
        selected["triangles"] = []
    elif change == "missing_selected_mesh":
        data["meshes"] = [r for r in data["meshes"] if r["body_index"] != 0]
    elif change == "body_signature":
        next(r for r in data["entities"] if r["kind"] == "body")["body_signature"] = "0" * 64
    elif change == "escaped_vertex":
        selected["vertices"][0][0] = 1_000_000
    elif change == "boolean_index":
        selected["triangles"][0][0] = True
    else:
        data["frame"] = "camera_coordinates"
    path, _ = stage(tmp_path, data)
    with pytest.raises(ValueError):
        worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_staged_source_mismatch_and_output_overwrite_are_rejected(tmp_path, topology):
    path, _ = stage(tmp_path, topology)
    content = (tmp_path / "topology.json").read_bytes()
    (tmp_path / "topology.json").write_bytes(content + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.render_occurrence(path)
    (tmp_path / "topology.json").write_bytes(content)
    (tmp_path / "inspection.png").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="already exist"):
        worker.render_occurrence(path)
    assert (tmp_path / "inspection.png").read_bytes() == b"preserve"


def test_incomplete_selected_face_coverage_rejected(tmp_path):
    data = topology_index(b.Box(10, 6, 4))
    data["meshes"].pop()
    path, _ = stage(tmp_path, data)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        worker.render_occurrence(path)


@pytest.mark.parametrize("body_index", [True, -1, 100, "0"])
def test_invalid_requested_body_never_selects_another(tmp_path, topology, body_index):
    path, _ = stage(tmp_path, topology, body_index)
    with pytest.raises(ValueError):
        worker.render_occurrence(path)


def test_duplicate_json_and_bounded_inputs_rejected(tmp_path, topology, monkeypatch):
    path, _ = stage(tmp_path, topology)
    path.write_text('{"revision_id":"a","revision_id":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        worker.render_occurrence(path)
    path, _ = stage(tmp_path, topology)
    monkeypatch.setattr(worker, "MAX_TOPOLOGY_BYTES", 10)
    with pytest.raises(ValueError, match="bounded regular"):
        worker.render_occurrence(path)


def test_hardlinked_request_is_rejected(tmp_path, topology):
    path, _ = stage(tmp_path, topology)
    os.link(path, tmp_path / "linked-request.json")
    with pytest.raises(ValueError, match="bounded regular"):
        worker.render_occurrence(path)


def test_mesh_count_and_output_size_bounds(tmp_path, topology, monkeypatch):
    path, _ = stage(tmp_path, topology)
    with monkeypatch.context() as patched:
        patched.setattr(worker, "MAX_TRIANGLES", 1)
        with pytest.raises(ValueError, match="count bound"):
            worker.render_occurrence(path)
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="bounded regular"):
        worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_cli_creates_only_scoped_regular_outputs(tmp_path, topology):
    path, request = stage(tmp_path, topology)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(
        [sys.executable, "-B", "-m", "forge_mind.workspace.occurrence_render", str(path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["body_index"] == request["body_index"]
    assert receipt["occurrence_path"] == request["occurrence_path"]
    assert (
        receipt["image"]["sha256"]
        == hashlib.sha256((tmp_path / "inspection.png").read_bytes()).hexdigest()
    )
    assert all(p.is_file() and not p.is_symlink() for p in tmp_path.iterdir())
