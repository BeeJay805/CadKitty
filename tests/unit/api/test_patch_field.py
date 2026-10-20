"""patch_part_structured (ADR-0060 slice B): a deterministic field+op+value edit -> re-derived,
strictly re-validated part, with NO LLM. The slider/dropdown apply path. Mirrors the patch_part
proofs (real geometry) but without a mapper call; refusals ship nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge_mind.api.patch import patch_part_structured

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


def _spec(fixture: str) -> dict[str, Any]:
    return json.loads((FIX / f"{fixture}.layoutspec.json").read_text())


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = node
    return out


@pytest.mark.slow
def test_structured_edit_ships_and_survives(tmp_path: Path) -> None:
    """Same bench-1 edit as the prose path, but applied STRUCTURALLY (no LLM): it lands, the rest of
    the spec is byte-stable, the rebuilt part PASSES, artifacts exist, and it costs nothing."""
    spec_data = _spec("bench001_l_bracket")
    res = patch_part_structured(
        spec_data, "extrusion_2020.m5_spacing_mm", "increase", 5.0, tmp_path
    )
    assert res.status == "ok" and res.ok is not None, (res.refused, res.error)
    assert res.ok.gen_path == "standard:patch"
    assert res.ok.cost_usd == 0.0  # deterministic: no mapper call, free
    assert res.ok.validation.passed
    assert res.ok.patch is not None
    assert res.ok.patch.field_path == "extrusion_2020.m5_spacing_mm"
    assert (res.ok.patch.old_value, res.ok.patch.new_value) == (20.0, 25.0)
    assert res.ok.spec is not None
    flat_new = _flatten(res.ok.spec)
    diffs = sorted(p for p, v in _flatten(spec_data).items() if flat_new.get(p) != v)
    assert diffs == ["extrusion_2020.m5_spacing_mm"]
    for name in ("part.stl", "part.glb", "part.layoutspec.json"):
        assert (tmp_path / name).is_file(), f"missing {name}"
    assert res.lane is not None and res.lane.outcome == "shipped" and res.lane.cost_usd == 0.0


def test_structured_unknown_field_refuses(tmp_path: Path) -> None:
    """A field that isn't in the closed catalog refuses (no geometry built, nothing shipped)."""
    res = patch_part_structured(_spec("bench001_l_bracket"), "no_such.field", "set", 1.0, tmp_path)
    assert res.status == "refused" and res.refused is not None
    assert "Can't apply that edit" in res.refused.reason
    assert not (tmp_path / "part.stl").exists()
    assert res.lane is not None and res.lane.outcome == "refused"


def test_structured_invalid_spec_errors(tmp_path: Path) -> None:
    res = patch_part_structured({"pattern": "nope"}, "x", "set", 1.0, tmp_path)
    assert res.status == "error" and res.error is not None and res.error.stage == "spec"
