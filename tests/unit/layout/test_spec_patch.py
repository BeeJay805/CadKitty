"""Spec-level parametric patch (ADR-0037): edit-survival at the spec level.

The contract under test: an edit changes EXACTLY ONE leaf of the stored LayoutSpec
(everything else byte-stable), out-of-catalog / wrong-kind / out-of-range edits
REFUSE (SpecPatchError, with a user-facing reason), and the geometry re-derives the
edited dimension through the same deterministic interpreter. All money-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.patch import (
    SpecEdit,
    SpecPatchError,
    apply_spec_edit,
    editable_fields,
    map_edit,
    render_catalog,
    validate_spec_payload,
)
from forge_mind.llm.cost import LLMUsage

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


def _load(fixture: str) -> Any:
    return validate_spec_payload(json.loads((FIX / f"{fixture}.layoutspec.json").read_text()))


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


# ---------------------------------------------------------------------------
# Catalog: closed, deterministic, identity excluded.
# ---------------------------------------------------------------------------


def test_catalog_covers_tray_parameters_and_excludes_identity() -> None:
    spec = _load("bench002_pi4_case")
    cat = {f.path: f for f in editable_fields(spec)}
    assert "pattern" not in cat  # the union discriminator is identity, never editable
    assert cat["clearance_mm"].kind == "number"
    assert cat["standoffs.height_mm"].current == 4.0
    assert cat["ventilation"].kind == "enum"
    assert cat["ventilation"].choices == ("none", "slots", "grid")
    # None-valued optional classified from its annotation, still addressable.
    assert cat["board_component_id"].kind == "string"


def test_catalog_addresses_list_items_by_index() -> None:
    from forge_mind.layout.composable_spec import (
        BaseArchetype,
        ComposablePlateSpec,
        Face,
        InterfacePlacement,
        InterfaceRef,
    )
    from forge_mind.types import Vec3

    spec = ComposablePlateSpec(
        name="plate",
        envelope_mm=Vec3(x=80.0, y=60.0, z=10.0),
        material="PLA",
        min_wall_mm=3.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_TOP, spacing_mm=20.0
            )
        ],
    )
    cat = {f.path: f for f in editable_fields(spec)}
    assert cat["interfaces.0.spacing_mm"].current == 20.0
    edited, old, new = apply_spec_edit(
        spec,
        SpecEdit(mappable=True, field_path="interfaces.0.spacing_mm", op="increase", value=5.0),
    )
    assert (old, new) == (20.0, 25.0)
    assert edited.interfaces[0].spacing_mm == 25.0


def test_render_catalog_lists_paths_and_choices() -> None:
    text = render_catalog(_load("bench002_pi4_case"))
    assert "standoffs.height_mm" in text
    assert "choices=['none', 'slots', 'grid']" in text


# ---------------------------------------------------------------------------
# THE edit-survival property: one edit -> exactly one changed leaf.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["bench001_l_bracket", "bench002_pi4_case"])
def test_every_edit_touches_only_its_leaf(fixture: str) -> None:
    spec = _load(fixture)
    flat_old = _flatten(spec.model_dump(mode="json"))
    covered = 0
    for f in editable_fields(spec):
        if f.current is None:
            continue
        if f.kind == "number":
            edit = SpecEdit(mappable=True, field_path=f.path, op="increase", value=1.0)
        elif f.kind == "enum":
            other = next((c for c in (f.choices or ()) if c != f.current), None)
            if other is None:
                continue
            edit = SpecEdit(mappable=True, field_path=f.path, op="set", value=other)
        else:
            continue
        new_spec, _, _ = apply_spec_edit(spec, edit)
        flat_new = _flatten(new_spec.model_dump(mode="json"))
        assert set(flat_new) == set(flat_old)
        diffs = sorted(p for p in flat_old if flat_old[p] != flat_new[p])
        assert diffs == [f.path], f"edit to {f.path} changed {diffs}"
        covered += 1
    assert covered >= 5  # the loop genuinely exercised the catalog


def test_increase_and_decrease_do_the_math() -> None:
    spec = _load("bench001_l_bracket")
    _up, old, new = apply_spec_edit(
        spec,
        SpecEdit(
            mappable=True, field_path="extrusion_2020.m5_spacing_mm", op="increase", value=5.0
        ),
    )
    assert (old, new) == (20.0, 25.0)
    _down, old2, new2 = apply_spec_edit(
        spec,
        SpecEdit(
            mappable=True, field_path="extrusion_2020.m5_spacing_mm", op="decrease", value=2.5
        ),
    )
    assert (old2, new2) == (20.0, 17.5)


def test_spacing_edit_re_derives_geometry() -> None:
    """The edited dimension lands in the DERIVED geometry (interpreter re-run), and the
    program does not change wholesale (same op count -- not a re-cut)."""
    spec = _load("bench001_l_bracket")
    new_spec, _, _ = apply_spec_edit(
        spec,
        SpecEdit(
            mappable=True, field_path="extrusion_2020.m5_spacing_mm", op="increase", value=5.0
        ),
    )
    prog_old = interpret_layout(spec)
    prog_new = interpret_layout(new_spec)
    assert len(prog_old.operations) == len(prog_new.operations)
    old_ops = {op.op_id: op.model_dump(mode="json") for op in prog_old.operations}
    new_ops = {op.op_id: op.model_dump(mode="json") for op in prog_new.operations}
    assert set(old_ops) == set(new_ops), "edit must not add/remove/rename operations"
    changed = sorted(i for i in old_ops if old_ops[i] != new_ops[i])
    # ONLY the M5 extrusion-bolt pair moves; nothing unrelated drifts (no re-cut).
    assert changed == ["ext_bolt_left", "ext_bolt_right"], changed
    lx, rx = (new_ops[i]["position"]["x"] for i in ("ext_bolt_left", "ext_bolt_right"))
    assert abs(rx - lx) == pytest.approx(25.0)  # 20mm spacing + the user's 5


# ---------------------------------------------------------------------------
# Refusal gates: out-of-catalog, wrong kind, out of range, unmappable.
# ---------------------------------------------------------------------------


def test_out_of_catalog_path_refuses() -> None:
    spec = _load("bench002_pi4_case")
    with pytest.raises(SpecPatchError, match="not an editable parameter"):
        apply_spec_edit(
            spec, SpecEdit(mappable=True, field_path="lid_thickness", op="set", value=3.0)
        )


def test_wrong_kind_refuses() -> None:
    spec = _load("bench002_pi4_case")
    with pytest.raises(SpecPatchError, match="increase/decrease applies to numbers"):
        apply_spec_edit(
            spec, SpecEdit(mappable=True, field_path="ventilation", op="increase", value=1.0)
        )
    with pytest.raises(SpecPatchError, match="needs a numeric value"):
        apply_spec_edit(
            spec, SpecEdit(mappable=True, field_path="clearance_mm", op="set", value="big")
        )
    with pytest.raises(SpecPatchError, match="must be one of"):
        apply_spec_edit(
            spec, SpecEdit(mappable=True, field_path="ventilation", op="set", value="holes")
        )


def test_out_of_range_value_refuses() -> None:
    spec = _load("bench002_pi4_case")
    with pytest.raises(SpecPatchError, match="out of range"):
        apply_spec_edit(
            spec, SpecEdit(mappable=True, field_path="min_wall_mm", op="decrease", value=10.0)
        )  # 2.0 - 10.0 = -8.0 violates gt=0 -> refuse, never a broken spec


def test_unmappable_edit_refuses_with_reason() -> None:
    spec = _load("bench002_pi4_case")
    with pytest.raises(SpecPatchError, match="state a quantity"):
        apply_spec_edit(
            spec, SpecEdit(mappable=False, reason="Please state a quantity for the change.")
        )


def test_invalid_spec_payload_refuses() -> None:
    with pytest.raises(SpecPatchError, match="parametric model"):
        validate_spec_payload({"pattern": "not_a_pattern", "name": "x"})


# ---------------------------------------------------------------------------
# map_edit: the call carries the catalog; the emission is returned verbatim.
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, edit: SpecEdit) -> None:
        self.edit = edit
        self.calls: list[dict[str, Any]] = []

    def generate_structured(self, **kw: Any) -> tuple[SpecEdit, LLMUsage]:
        self.calls.append(kw)
        return self.edit, LLMUsage(model="fake", input_tokens=10, output_tokens=5, cost_usd=0.001)


def test_map_edit_prompts_with_the_catalog() -> None:
    spec = _load("bench002_pi4_case")
    canned = SpecEdit(mappable=True, field_path="standoffs.height_mm", op="set", value=6.0)
    fake = _FakeClient(canned)
    edit, usage = map_edit(spec, "raise the standoffs to 6mm", client=fake)  # type: ignore[arg-type]
    assert edit is canned
    assert usage.cost_usd == 0.001
    (call,) = fake.calls
    assert "standoffs.height_mm" in call["user"]  # the closed menu IS the prompt
    assert "raise the standoffs to 6mm" in call["user"]
    assert call["output_model"] is SpecEdit
