"""patch_part end-to-end (ADR-0037): stored spec + one edit -> re-derived, strictly
re-validated part; refusals on infeasible / failing edits with nothing shipped.

Money-free (the mapper is a fake client); the geometry path (interpret -> compile ->
validate) is REAL, so these are the before/after bench-part proofs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.api.patch import patch_part
from forge_mind.layout.patch import SpecEdit
from forge_mind.llm.cost import LLMUsage

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


def _spec(fixture: str) -> dict[str, Any]:
    return json.loads((FIX / f"{fixture}.layoutspec.json").read_text())


class _FakeClient:
    def __init__(self, edit: SpecEdit) -> None:
        self.edit = edit

    def generate_structured(self, **kw: Any) -> tuple[SpecEdit, LLMUsage]:
        return self.edit, LLMUsage(model="fake", input_tokens=10, output_tokens=5, cost_usd=0.002)


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
def test_l_bracket_spacing_patch_ships_and_survives(tmp_path: Path) -> None:
    """Bench part 1 (before/after): 'move the extrusion holes 5mm further apart' on the
    NEMA17 L-bracket. The edit lands, the rest of the spec is byte-stable, the rebuilt
    part PASSES the strict gates, and all artifacts exist."""
    spec_data = _spec("bench001_l_bracket")
    fake = _FakeClient(
        SpecEdit(mappable=True, field_path="extrusion_2020.m5_spacing_mm", op="increase", value=5.0)
    )
    res = patch_part(
        spec_data,
        "move the extrusion holes 5mm further apart",
        tmp_path,
        client=fake,  # type: ignore[arg-type]
    )
    assert res.status == "ok" and res.ok is not None, (res.refused, res.error)
    assert res.ok.gen_path == "standard:patch"
    assert res.ok.validation.passed
    assert res.ok.patch is not None
    assert res.ok.patch.field_path == "extrusion_2020.m5_spacing_mm"
    assert (res.ok.patch.old_value, res.ok.patch.new_value) == (20.0, 25.0)
    # Edit-survival at the spec level: ONLY the targeted leaf changed.
    assert res.ok.spec is not None
    flat_old = _flatten(spec_data)
    flat_new = _flatten(res.ok.spec)
    diffs = sorted(p for p in flat_old if flat_old[p] != flat_new[p])
    assert diffs == ["extrusion_2020.m5_spacing_mm"]
    assert flat_new["extrusion_2020.m5_spacing_mm"] == 25.0
    # Artifacts for the viewer + the engineer, and the NEW parametric model for the next edit.
    for name in ("part.stl", "part.glb", "part.step", "part.layoutspec.json"):
        assert (tmp_path / name).is_file(), f"missing {name}"
    persisted = json.loads((tmp_path / "part.layoutspec.json").read_text())
    assert persisted == res.ok.spec
    assert res.lane is not None and res.lane.outcome == "shipped"


@pytest.mark.slow
def test_tray_standoff_patch_ships(tmp_path: Path) -> None:
    """Bench part 2 (before/after): 'raise the standoffs to 6mm' on the Pi-4 tray."""
    spec_data = _spec("bench002_pi4_case")
    fake = _FakeClient(
        SpecEdit(mappable=True, field_path="standoffs.height_mm", op="set", value=6.0)
    )
    res = patch_part(spec_data, "raise the standoffs to 6mm", tmp_path, client=fake)  # type: ignore[arg-type]
    assert res.status == "ok" and res.ok is not None, (res.refused, res.error)
    assert res.ok.validation.passed
    assert res.ok.spec is not None and res.ok.spec["standoffs"]["height_mm"] == 6.0
    diffs = sorted(
        p
        for p, v in _flatten(_spec("bench002_pi4_case")).items()
        if _flatten(res.ok.spec).get(p) != v
    )
    assert diffs == ["standoffs.height_mm"]


def test_infeasible_edit_refuses_with_reason_and_ships_nothing(tmp_path: Path) -> None:
    """An edit that cannot fit the user's hard envelope is an HONEST refusal carrying the
    LayoutError reason -- never a silently re-cut or wrong part (the moat)."""
    spec_data = _spec("bench002_pi4_case")
    # The user's stated hard max == the current envelope: any growth must refuse.
    spec_data["max_envelope_mm"] = dict(spec_data["envelope_mm"])
    fake = _FakeClient(
        SpecEdit(mappable=True, field_path="clearance_mm", op="increase", value=30.0)
    )
    res = patch_part(spec_data, "add 30mm of clearance around the board", tmp_path, client=fake)  # type: ignore[arg-type]
    assert res.status == "refused" and res.refused is not None, (res.status, res.error)
    assert "Can't apply that edit" in res.refused.reason
    assert not (tmp_path / "part.stl").exists()
    assert not (tmp_path / "part.glb").exists()
    assert res.lane is not None and res.lane.outcome == "refused"


def test_failed_validation_refuses_no_false_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-false-PASS is absolute: if the rebuilt part fails the strict gates, /patch
    refuses (naming the checks) instead of shipping."""
    import forge_mind.api.patch as patch_mod

    failed = SimpleNamespace(
        passed=False,
        overall=SimpleNamespace(value="fail"),
        checks=[
            SimpleNamespace(id="features.min_wall_thickness", status=SimpleNamespace(value="fail"))
        ],
    )
    monkeypatch.setattr(patch_mod, "validate", lambda *a, **k: failed)
    spec_data = _spec("bench001_l_bracket")
    fake = _FakeClient(
        SpecEdit(mappable=True, field_path="extrusion_2020.m5_spacing_mm", op="increase", value=5.0)
    )
    res = patch_part(spec_data, "move the holes apart", tmp_path, client=fake)  # type: ignore[arg-type]
    assert res.status == "refused" and res.refused is not None
    assert "features.min_wall_thickness" in res.refused.reason
    assert not (tmp_path / "part.stl").exists()
    assert res.lane is not None and res.lane.outcome == "validation_failed"


def test_unmappable_edit_refuses_with_mapper_reason(tmp_path: Path) -> None:
    fake = _FakeClient(
        SpecEdit(mappable=False, reason="Adding another hole is not a parameter edit yet.")
    )
    res = patch_part(_spec("bench001_l_bracket"), "add a third hole", tmp_path, client=fake)  # type: ignore[arg-type]
    assert res.status == "refused" and res.refused is not None
    assert res.refused.reason == "Adding another hole is not a parameter edit yet."
    assert res.lane is not None and res.lane.outcome == "refused"
    assert res.lane.cost_usd == pytest.approx(0.002)  # the mapper call is still accounted


def test_invalid_spec_payload_errors(tmp_path: Path) -> None:
    res = patch_part({"pattern": "nope"}, "make it wider", tmp_path)
    assert res.status == "error" and res.error is not None
    assert res.error.stage == "spec"
    assert res.lane is not None and res.lane.outcome == "failed_stage"
