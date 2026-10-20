"""Distinct part identities can share bytes without aliasing occurrence/export identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build123d as b
import pytest

from forge_mind.workspace.assembly import assembly_evidence
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def equal_definitions() -> DesignDocument:
    feature = {"id": "body", "name": "Body", "kind": "box", "params": {"x": 10, "y": 6, "z": 4}}
    return DesignDocument.model_validate(
        {
            "variables": {"offset": 25},
            "assembly": {
                "parts": [
                    {"id": identifier, "name": identifier, "features": [feature], "output": "body"}
                    for identifier in ("left_hand_nut", "right_hand_nut")
                ],
                "roots": [
                    {"id": "left", "definition": "left_hand_nut"},
                    {"id": "right", "definition": "right_hand_nut", "position": ["offset", 0, 0]},
                ],
            },
        }
    )


def test_equal_definitions_share_cache_but_retain_exports_and_placed_identities(
    tmp_path: Path,
) -> None:
    first, warm = tmp_path / "first", tmp_path / "warm"
    first.mkdir()
    warm.mkdir()
    document = equal_definitions()
    _, cold_history = compile_document(document, {}, first)
    manifest = json.loads((first / "parts.json").read_text())
    assert manifest["left_hand_nut"]["file"] == manifest["right_hand_nut"]["file"]
    assert len(list(first.glob("*.brep"))) == 1
    assert cold_history[-1]["built"] == 2
    document.variables["offset"] = 40
    shape, history = compile_document(document, {}, warm, first)
    assert history[-1]["reused"] == 2
    assert history[-1]["built"] == 0
    assert len(list(warm.glob("*.brep"))) == 1
    assert pytest.approx(50) == shape.bounding_box().size.X
    for identifier in ("left_hand_nut", "right_hand_nut"):
        exported = b.import_step(warm / f"definition_{identifier}.step")
        assert len(exported.solids()) == 1
        assert exported.volume == pytest.approx(240)
    b.export_step(shape, warm / "assembly.step")
    evidence = assembly_evidence(document, b.import_step(warm / "assembly.step"), history[-1])
    assert {(row["path"], row["definition"]) for row in evidence["occurrences"]} == {
        ("left", "left_hand_nut"),
        ("right", "right_hand_nut"),
    }
    assert evidence["occurrence_count"] == 2
    assert evidence["unique_parts"] == 2


def test_shared_destination_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, warm = tmp_path / "first", tmp_path / "warm"
    first.mkdir()
    warm.mkdir()
    document = equal_definitions()
    compile_document(document, {}, first)
    original_export = b.export_step

    def corrupt_after_first_definition(shape: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_export(shape, target, *args, **kwargs)
        if Path(target).name == "definition_left_hand_nut.step":
            next(warm.glob("*.brep")).write_bytes(b"tampered after first definition")
        return result

    monkeypatch.setattr(b, "export_step", corrupt_after_first_definition)
    with pytest.raises(ValueError, match="Shared compiled part cache changed"):
        compile_document(document, {}, warm, first)


def test_unowned_existing_warm_destination_is_not_reused(tmp_path: Path) -> None:
    first, warm = tmp_path / "first", tmp_path / "warm"
    first.mkdir()
    warm.mkdir()
    document = equal_definitions()
    compile_document(document, {}, first)
    target = warm / next(first.glob("*.brep")).name
    target.write_bytes(b"unrelated existing artifact")
    with pytest.raises(FileExistsError):
        compile_document(document, {}, warm, first)
    assert target.read_bytes() == b"unrelated existing artifact"
