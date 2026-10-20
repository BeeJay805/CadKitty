"""Focus worker re-resolves exact staged inventory without native CAD fixtures."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
import trimesh
from PIL import Image

from forge_mind.workspace import occurrence_render as worker
from forge_mind.workspace.occurrence_focus import (
    OccurrenceFocus,
    focus_view_metadata,
    resolve_focus,
)


@pytest.fixture
def topology():
    entities, meshes = [], []
    for index, offset in enumerate(([20, 30, 40], [120, -65, 42])):
        mesh = trimesh.creation.box([10, 6, 4])
        mesh.apply_translation(offset)
        lower, upper = mesh.bounds.tolist()
        body_signature = ("b" if index == 0 else "c") * 64
        entities.append(
            {
                "id": f"body_{index}",
                "kind": "body",
                "body_index": index,
                "signature": body_signature,
                "body_signature": body_signature,
                "min": lower,
                "max": upper,
            }
        )
        entities.append(
            {
                "id": f"face_{index}",
                "kind": "face",
                "body_index": index,
                "signature": "d" * 64,
                "body_signature": body_signature,
                "min": [upper[0], lower[1], lower[2]],
                "max": upper,
            }
        )
        entities.append(
            {
                "id": f"edge_{index}",
                "kind": "edge",
                "body_index": index,
                "signature": "e" * 64,
                "body_signature": body_signature,
                "min": [upper[0], lower[1], lower[2]],
                "max": [upper[0], upper[1], lower[2]],
            }
        )
        meshes.append(
            {
                "id": f"body_{index}",
                "body_index": index,
                "vertices": mesh.vertices.tolist(),
                "triangles": mesh.faces.tolist(),
            }
        )
    return {
        "frame": "cad_z_up_mm",
        "selection_scope": "body",
        "entities": entities,
        "meshes": meshes,
        "topology_counts": {"bodies": 2, "faces": 2, "edges": 2},
        "pick_tessellation": {
            "purpose": "Whole-body display with detailed entity facts",
            "relative_linear_deflection": 0.01,
            "angular_deflection_radians": 0.1,
        },
    }


def stage(directory, topology, body_index=0, focus=None):
    content = json.dumps(topology).encode()
    (directory / "topology.json").write_bytes(content)
    request = {
        "revision_id": "a" * 32,
        "occurrence_path": f"group/part{body_index}",
        "body_index": body_index,
        "topology_sha256": hashlib.sha256(content).hexdigest(),
        "step_sha256": "f" * 64,
        "document_sha256": "0" * 64,
    }
    if focus is not None:
        request["focus"] = focus
    path = directory / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path, request


@pytest.mark.parametrize("kind,signature,body_index", [("face", "d", 0), ("edge", "e", 1)])
def test_actual_focus_image_from_whole_body_mesh_and_detailed_facts(
    tmp_path, topology, kind, signature, body_index
):
    frame = resolve_focus(
        topology, body_index, OccurrenceFocus(kind=kind, signature=signature * 64, view="right")
    )
    path, request = stage(tmp_path, topology, body_index, frame)
    before = (tmp_path / "topology.json").read_bytes()
    receipt = worker.render_occurrence(path)
    assert receipt["focus"] == frame
    assert receipt["focus_view"] == focus_view_metadata(frame)
    assert receipt["focus_view"]["all_selected_body_triangles_depth_tested"] is True
    assert receipt["focus"]["entity_id"] == f"{kind}_{body_index}"
    assert receipt["omitted_body_indices"] == [1 - body_index]
    assert receipt["selected_body_count"] == 1
    assert receipt["occurrence_path"] == request["occurrence_path"]
    assert receipt["display"]["triangles"] == len(topology["meshes"][body_index]["triangles"])
    assert (tmp_path / "topology.json").read_bytes() == before
    with Image.open(tmp_path / "inspection.png") as image:
        assert image.size == (1680, 1008)
        assert np.count_nonzero(np.any(np.asarray(image.convert("RGB")) < 230, axis=2)) > 10000
    assert (
        receipt["image"]["sha256"]
        == hashlib.sha256((tmp_path / "inspection.png").read_bytes()).hexdigest()
    )
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8")) == receipt


def test_legacy_request_receipt_shape_and_six_view_image_remain_unchanged(tmp_path, topology):
    path, request = stage(tmp_path, topology)
    receipt = worker.render_occurrence(path)
    assert set(receipt) == set(request) | {
        "image",
        "frame",
        "selected_body_count",
        "omitted_body_count",
        "omitted_body_indices",
        "frame_bounds",
        "display",
        "source_binding",
    }
    assert "focus" not in receipt and "focus_view" not in receipt
    with Image.open(tmp_path / "inspection.png") as image:
        assert image.size == (1800, 1200)


@pytest.mark.parametrize(
    "change", ["entity_id", "body_signature", "body_index", "body_bounds", "null"]
)
def test_changed_resolved_frame_is_refused_before_rendering(tmp_path, topology, change):
    frame = resolve_focus(topology, 0, OccurrenceFocus(kind="face", signature="d" * 64))
    if change == "entity_id":
        frame["entity_id"] = "face_not_selected"
    elif change == "body_signature":
        frame["body_signature"] = "9" * 64
    elif change == "body_index":
        frame["body_index"] = 1
    elif change == "body_bounds":
        frame["body_bounds"]["min"][0] -= 1
    path, request = stage(tmp_path, topology, focus=frame)
    if change == "null":
        request["focus"] = None
        path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError):
        worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_body", "stale"])
def test_staged_focus_inventory_must_still_resolve(tmp_path, topology, change):
    frame = resolve_focus(topology, 0, OccurrenceFocus(kind="face", signature="d" * 64))
    data = deepcopy(topology)
    face = next(row for row in data["entities"] if row["id"] == "face_0")
    if change == "missing":
        data["entities"].remove(face)
    elif change == "duplicate":
        copied = dict(face, id="extra_face")
        data["entities"].append(copied)
    elif change == "wrong_body":
        face["body_index"] = 1
    else:
        face["signature"] = "8" * 64
    path, _ = stage(tmp_path, data, focus=frame)
    with pytest.raises(ValueError):
        worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()


def test_wrong_renderer_metadata_and_image_budget_leave_no_published_output(
    tmp_path, topology, monkeypatch
):
    from forge_mind.viz import focus as renderer

    frame = resolve_focus(topology, 0, OccurrenceFocus(kind="face", signature="d" * 64))
    path, _ = stage(tmp_path, topology, focus=frame)
    original = renderer.render_occurrence_focus

    def wrong(mesh, output, resolved):
        value = original(mesh, output, resolved)
        value["detail_radius_mm"] *= 2
        return value

    with monkeypatch.context() as context:
        context.setattr(renderer, "render_occurrence_focus", wrong)
        with pytest.raises(ValueError, match="different view metadata"):
            worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()
    assert not (tmp_path / "receipt.json").exists()
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="bounded regular"):
        worker.render_occurrence(path)
    assert not (tmp_path / "inspection.png").exists()
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob("occurrence-render-*"))
