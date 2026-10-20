"""blueprint_fields (the read side of the editable blueprint, ADR-0060): a stored LayoutSpec ->
the closed catalog of editable leaves the UI renders as controls. No geometry build, no LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge_mind.api.patch import blueprint_fields
from forge_mind.layout.patch import SpecPatchError

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


def _spec(name: str) -> dict[str, Any]:
    return json.loads((FIX / f"{name}.layoutspec.json").read_text())


def test_catalog_lists_editable_fields() -> None:
    fields = blueprint_fields(_spec("bench001_l_bracket"))
    by_path = {f["path"]: f for f in fields}
    # the known editable parameter exercised by the /patch tests
    assert "extrusion_2020.m5_spacing_mm" in by_path
    f = by_path["extrusion_2020.m5_spacing_mm"]
    assert f["kind"] == "number"
    assert f["current"] == 20.0
    # the union discriminator is identity, never an editable parameter
    assert "pattern" not in by_path
    # every entry carries the UI-render shape
    for entry in fields:
        assert set(entry) == {"path", "kind", "current", "description", "choices"}
        assert entry["kind"] in {"number", "boolean", "enum", "string"}


def test_invalid_spec_refuses() -> None:
    # Not a usable parametric model -> refuse, never a fabricated catalog.
    with pytest.raises(SpecPatchError):
        blueprint_fields({"pattern": "nope"})
