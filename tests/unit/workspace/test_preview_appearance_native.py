"""Prepared native checks; root serializes execution with other CAD work."""

import json
from copy import deepcopy

import build123d as b
import numpy as np
import pytest
import trimesh

from forge_mind.validation_provenance import file_sha256
from forge_mind.viz import render
from forge_mind.workspace.assembly import assembly_evidence
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.measure import measure_revision
from forge_mind.workspace.models import Contract
from forge_mind.workspace.preview_appearance import render_primary_preview
from forge_mind.workspace.preview_inspection import add_inspection_preview
from forge_mind.workspace.tessellation import export_verified_tessellation


@pytest.fixture
def native_case(tmp_path):
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
                                "params": {"x": 6, "y": 4, "z": 3},
                            }
                        ],
                        "output": "body",
                    }
                    for identifier, role in [("holder", "make"), ("device", "reference")]
                ],
                "subassemblies": [
                    {
                        "id": "pair",
                        "name": "Pair",
                        "children": [
                            {"id": "holder", "definition": "holder"},
                            {"id": "device", "definition": "device", "position": [0, 8, 0]},
                        ],
                    }
                ],
                "roots": [
                    {"id": "left", "definition": "pair"},
                    {
                        "id": "right",
                        "definition": "pair",
                        "position": [20, 0, 0],
                        "rotation": [0, 0, 90],
                    },
                ],
            }
        }
    )
    shape, history = compile_document(document, {}, tmp_path)
    b.export_step(shape, tmp_path / "part.step")
    (tmp_path / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
    (tmp_path / "program.py").write_text(
        "# saved declaration fixture; not executed", encoding="utf-8"
    )
    export_verified_tessellation(tmp_path)
    trimesh.Scene(trimesh.load(tmp_path / "part.stl", force="mesh")).export(tmp_path / "part.glb")
    b.export_stl(
        b.import_step(tmp_path / "part.step"),
        tmp_path / "old-preview.stl",
        tolerance=0.15,
        angular_tolerance=0.35,
    )
    render.render_workspace_preview(tmp_path / "old-preview.stl", tmp_path / "render.png")
    imported = b.import_step(tmp_path / "part.step")
    record = next(row for row in history if row["id"] == "__assembly__")
    evidence = assembly_evidence(document, imported, record)
    return tmp_path, imported, document.model_dump(mode="json"), evidence


def test_same_invocation_color_render_preserves_saved_geometry_measurements_and_companion_binding(
    native_case,
):
    directory, shape, document, evidence = native_case
    hashes = {
        name: file_sha256(directory / name)
        for name in ("part.step", "part.stl", "part.glb", "document.json")
    }
    before = measure_revision(
        directory, "a" * 32, Contract(), structured_assembly=True
    ).model_dump()
    bounds, appearance = render_primary_preview(directory, shape, document, evidence)
    assert appearance["body_colors"] and len(appearance["body_colors"]) == 4
    after = measure_revision(directory, "a" * 32, Contract(), structured_assembly=True).model_dump()
    before["artifact_hashes"].pop("render.png")
    after["artifact_hashes"].pop("render.png")
    assert after == before
    assert all(file_sha256(directory / name) == digest for name, digest in hashes.items())
    after["facts"]["assembly"] = evidence
    unchanged = deepcopy(after)
    receipt = add_inspection_preview(directory, shape, document, after, bounds)
    assert receipt["primary"]["sha256"] == file_sha256(directory / "render.png")
    assert receipt["inspection"] is not None
    assert after["geometry_valid"] == unchanged["geometry_valid"]
    assert after["findings"] == unchanged["findings"]
    assert json.loads((directory / "render-inspection.json").read_text()) == json.loads(
        json.dumps(receipt)
    )


def test_display_export_failure_is_visible_neutral_fallback_without_saved_geometry_changes(
    native_case, monkeypatch
):
    directory, shape, document, evidence = native_case
    before = {name: file_sha256(directory / name) for name in ("part.step", "part.stl", "part.glb")}
    export = b.export_stl
    calls = []

    def fail_first(shape, filename, **kwargs):
        calls.append(str(filename))
        if len(calls) == 1:
            raise RuntimeError("optional color tessellation fault")
        return export(shape, filename, **kwargs)

    monkeypatch.setattr(b, "export_stl", fail_first)
    bounds, appearance = render_primary_preview(directory, shape, document, evidence)
    assert appearance["body_colors"] == () and appearance["legend"] == ()
    assert "display tessellation failed (RuntimeError)" in appearance["diagnostic"]
    assert np.isfinite(bounds).all()
    assert all(file_sha256(directory / name) == digest for name, digest in before.items())
    assert (
        not list(directory.glob("preview-body-*.stl")) and not (directory / "preview.stl").exists()
    )


@pytest.mark.parametrize("failure", [TimeoutError, MemoryError])
def test_fatal_display_limits_propagate(native_case, monkeypatch, failure):
    directory, shape, document, evidence = native_case

    def fail(*args, **kwargs):
        raise failure("fatal resource limit")

    monkeypatch.setattr(b, "export_stl", fail)
    with pytest.raises(failure, match="fatal resource limit"):
        render_primary_preview(directory, shape, document, evidence)
