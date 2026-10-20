"""Native curved solids and retained source cross actual export/process boundaries."""

import build123d as b
import pytest

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def test_editable_smooth_loft_exports_actual_curved_solid(tmp_path):
    profiles = [
        {
            "id": f"section{i}",
            "name": f"Section {i}",
            "kind": "profile",
            "params": {"x": x, "ry": 90, "radius_x": radius, "radius_y": radius * 0.75},
            "bindings": {"radius_x": "radius"} if i == 2 else {},
        }
        for i, (x, radius) in enumerate([(-100, 2), (-60, 15), (0, 20), (60, 12), (100, 1)])
    ]
    doc = DesignDocument.model_validate(
        {
            "variables": {"radius": 20},
            "features": [
                *profiles,
                {
                    "id": "body",
                    "name": "Smooth body",
                    "kind": "loft",
                    "inputs": [p["id"] for p in profiles],
                },
            ],
            "outputs": ["body"],
        }
    )
    shape, history = compile_document(doc, {})
    b.export_step(shape, str(tmp_path / "curved.step"))
    exported = b.import_step(str(tmp_path / "curved.step"))
    assert exported.is_valid and len(exported.solids()) == 1
    assert any("BSPLINE" in str(f.geom_type) for f in exported.faces())
    assert exported.volume == pytest.approx(shape.volume, rel=1e-6)
    assert all(h["state"] == "built" for h in history)
    doc.variables["radius"] = 25
    changed, _ = compile_document(doc, {})
    assert changed.is_valid and changed.volume > shape.volume


def test_python_feature_imports_inputs_and_cannot_change_parent_evaluator_globals():
    original = b.Box
    doc = DesignDocument.model_validate(
        {
            "variables": {"length": 10},
            "features": [
                {"id": "base", "name": "Base", "kind": "box", "params": {"x": 10, "y": 2, "z": 3}},
                {
                    "id": "program",
                    "name": "Retained source",
                    "kind": "python",
                    "inputs": ["base"],
                    "bindings": {"length": "length"},
                    "code": "b.Box = lambda *args, **kwargs: inputs[0]\nresult = inputs[0] + b.Pos(params['length'],0,0) * b.Box(params['length'],2,3)",
                },
            ],
            "outputs": ["program"],
        }
    )
    shape, history = compile_document(doc, {})
    assert b.Box is original
    assert shape.is_valid and shape.volume == pytest.approx(120)
    assert len(history[-1]["source_sha256"]) == 64
    assert "unverified" in history[-1]["history_scope"]


def test_python_feature_failure_does_not_return_fabricated_geometry():
    doc = DesignDocument.model_validate(
        {
            "features": [
                {
                    "id": "failed",
                    "name": "Failed source",
                    "kind": "python",
                    "code": "raise RuntimeError('deliberate failure')",
                }
            ],
            "outputs": ["failed"],
        }
    )
    with pytest.raises(ValueError, match="deliberate failure"):
        compile_document(doc, {})
