"""Native STEP identity coverage for the display-only role selector."""

import json
from copy import deepcopy
from pathlib import Path

import build123d as b
import pytest
import trimesh

from forge_mind.validation_provenance import file_sha256
from forge_mind.viz import render
from forge_mind.workspace.assembly import assembly_evidence, named_solids, occurrence_label
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.preview_inspection import add_inspection_preview
from forge_mind.workspace.preview_selection import select_preview_bodies
from forge_mind.workspace.topology import same_exported_solid


def test_nested_repeated_moved_step_bodies_keep_identity_and_full_geometry(tmp_path):
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": identifier,
                        "name": identifier,
                        "production": {"role": role},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": size, "y": 5, "z": 4},
                            }
                        ],
                        "output": "body",
                    }
                    for identifier, role, size in [
                        ("fixture", "make", 10),
                        ("context", "reference", 12),
                        ("pin", "buy", 2),
                        ("unknown", "unspecified", 3),
                    ]
                ],
                "subassemblies": [
                    {
                        "id": "pair",
                        "name": "Nested pair",
                        "children": [
                            {"id": "fixture", "definition": "fixture"},
                            {"id": "context", "definition": "context", "position": [0, 10, 0]},
                        ],
                    }
                ],
                "roots": [
                    {"id": "left", "definition": "pair"},
                    {
                        "id": "right",
                        "definition": "pair",
                        "position": [30, 40, 10],
                        "rotation": [0, 0, 90],
                    },
                    {"id": "pin", "definition": "pin", "position": [60, 0, 0]},
                    {"id": "unknown", "definition": "unknown", "position": [70, 0, 0]},
                ],
            }
        }
    )
    source, history = compile_document(document, {}, tmp_path)
    step = tmp_path / "part.step"
    b.export_step(source, step)
    original_bytes = step.read_bytes()
    imported = b.import_step(step)
    record = next(item for item in history if item["id"] == "__assembly__")
    evidence = assembly_evidence(document, imported, record)
    bodies = list(imported.solids())
    named = named_solids(imported)
    selected = select_preview_bodies(document.model_dump(mode="json"), evidence, len(bodies))
    assert selected.status == "reference_excluded"
    assert len(selected.included_body_indices) == 4
    assert set(selected.omitted_occurrence_ids) == {"left/context", "right/context"}
    for row in evidence["occurrences"]:
        actual = bodies[row["body_index"]]
        assert same_exported_solid(actual, named[occurrence_label(row["path"])])
        assert (row["body_index"] in selected.included_body_indices) == (
            row["definition"] != "context"
        )
    volumes = [body.volume for body in bodies]
    # Exercise exactly the selected-solid compound used by the inspection exporter.
    filtered = b.Compound(children=[bodies[index] for index in selected.included_body_indices])
    assert b.export_stl(
        filtered, tmp_path / "inspection.stl", tolerance=0.15, angular_tolerance=0.35
    )
    assert len(imported.solids()) == 6
    assert [body.volume for body in imported.solids()] == pytest.approx(volumes)
    assert set(named_solids(imported)) == set(named)
    assert assembly_evidence(document, imported, record) == evidence
    assert step.read_bytes() == original_bytes


@pytest.fixture
def preview_case(tmp_path):
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "fixture",
                        "name": "Curved fixture",
                        "production": {"role": "make"},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "sphere",
                                "params": {"radius": 20},
                            }
                        ],
                        "output": "body",
                    },
                    {
                        "id": "reference",
                        "name": "Reference",
                        "production": {"role": "reference"},
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 2, "y": 2, "z": 2},
                            }
                        ],
                        "output": "body",
                    },
                ],
                "roots": [
                    {"id": "fixture", "definition": "fixture"},
                    {"id": "reference", "definition": "reference"},
                ],
            }
        }
    )
    source, history = compile_document(document, {}, tmp_path)
    b.export_step(source, tmp_path / "part.step")
    (tmp_path / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
    shape = b.import_step(tmp_path / "part.step")
    # Reproduce the production order: fine exports precede separate coarse preview.
    b.export_stl(shape, tmp_path / "part.stl", tolerance=0.001, angular_tolerance=0.1)
    fresh = b.import_step(tmp_path / "part.step")
    b.export_stl(fresh, tmp_path / "full-preview.stl", tolerance=0.15, angular_tolerance=0.35)
    bounds = trimesh.load(tmp_path / "full-preview.stl", force="mesh").bounds
    render.render_workspace_preview(tmp_path / "full-preview.stl", tmp_path / "render.png")
    record = next(row for row in history if row["id"] == "__assembly__")
    evidence = assembly_evidence(document, shape, record)
    result = {
        "revision_id": "a" * 32,
        "engine_fingerprint": "b" * 64,
        "status": "fail",
        "facts": {"assembly": evidence, "manufacturing_ready": False},
        "artifact_hashes": {
            name: file_sha256(tmp_path / name)
            for name in ("part.step", "part.stl", "document.json", "render.png")
        },
    }
    return tmp_path, shape, document.model_dump(mode="json"), result, bounds


def test_curved_companion_uses_coarse_framing_without_changing_authoritative_outputs(preview_case):
    directory, shape, _, result, _ = preview_case
    before = deepcopy(result)
    volumes = [solid.volume for solid in shape.solids()]
    named = set(named_solids(shape))
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is not None, receipt
    assert receipt["selection"]["omitted_occurrence_ids"] == ("reference",)
    assert result["status"] == before["status"]
    assert result["facts"]["manufacturing_ready"] is False
    assert result["facts"]["assembly"] == before["facts"]["assembly"]
    for name, digest in before["artifact_hashes"].items():
        assert file_sha256(directory / name) == digest
    assert [solid.volume for solid in shape.solids()] == pytest.approx(volumes)
    assert set(named_solids(shape)) == named
    assert not (directory / "preview-inspection.stl").exists()
    assert json.loads((directory / "render-inspection.json").read_text()) == json.loads(
        json.dumps(receipt)
    )


def test_locked_previous_image_never_reused_or_registered(preview_case, monkeypatch):
    directory, _, _, result, _ = preview_case
    image = directory / "render-inspection.png"
    image.write_bytes(b"stale optional image")
    result["artifact_hashes"].update({image.name: "old", "render-inspection.json": "old"})
    original = Path.unlink

    def locked(path, *args, **kwargs):
        if path == image:
            raise PermissionError("held reader")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is None
    assert receipt["cleanup_errors"]
    assert receipt["render_error"]
    assert image.name not in result["artifact_hashes"]
    assert image.read_bytes() == b"stale optional image"
    assert result["status"] == "fail"


def test_intermediate_cleanup_failure_remains_diagnostic(preview_case, monkeypatch):
    directory, _, _, result, _ = preview_case
    intermediate = directory / "preview-inspection.stl"
    original = Path.unlink

    def locked(path, *args, **kwargs):
        if path == intermediate and path.exists():
            raise PermissionError("held reader")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is not None, receipt
    assert any(intermediate.name in error for error in receipt["cleanup_errors"])
    assert result["status"] == "fail"


def test_metadata_write_failure_does_not_register_companion(preview_case, monkeypatch):
    directory, _, _, result, _ = preview_case
    metadata = directory / "render-inspection.json"
    result["artifact_hashes"].update({"render-inspection.png": "old", metadata.name: "old"})
    original = Path.write_text

    def locked(path, *args, **kwargs):
        if path == metadata:
            raise PermissionError("held reader")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", locked)
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is None
    assert receipt["metadata_error"]
    assert "render-inspection.png" not in result["artifact_hashes"]
    assert metadata.name not in result["artifact_hashes"]
    assert result["status"] == "fail"


def test_failed_repeat_removes_previous_optional_hash(preview_case, monkeypatch):
    _, _, _, result, _ = preview_case
    assert add_inspection_preview(*preview_case)["inspection"] is not None

    def failed(*args, **kwargs):
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(render, "render_workspace_preview", failed)
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is None
    assert "renderer failed" in receipt["render_error"]
    assert "render-inspection.png" not in result["artifact_hashes"]
    assert result["status"] == "fail"


def test_false_export_cannot_render_previous_intermediate(preview_case, monkeypatch):
    directory, _, _, result, _ = preview_case
    intermediate = directory / "preview-inspection.stl"
    intermediate.write_bytes((directory / "full-preview.stl").read_bytes())
    monkeypatch.setattr(b, "export_stl", lambda *args, **kwargs: False)
    receipt = add_inspection_preview(*preview_case)
    assert receipt["inspection"] is None
    assert receipt["render_error"]
    assert "render-inspection.png" not in result["artifact_hashes"]
