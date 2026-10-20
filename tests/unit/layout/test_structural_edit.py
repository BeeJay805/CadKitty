"""apply_spec_structural_edit (ADR-0060 slice C): deterministic ADD / REMOVE of an interface on the
composable-plate ``interfaces`` list, re-validated through the LayoutSpec union. No geometry here --
this proves the spec-level contract: byte-stability of the untouched interfaces, the min/max-length
and vocabulary refusals, and that a 3rd interface routes through the generic derive path instead of
the legacy-delegate KeyError (the latent interpreter gap this slice fixes). The full rebuild + strict
gates are proven in tests/unit/api/test_patch_structure.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.patch import (
    SpecPatchError,
    apply_spec_structural_edit,
    validate_spec_payload,
)
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"

_CLEARANCE = {
    "ref": "clearance_hole",
    "face": "plate_bottom",
    "component_id": "M3",
    "fit": "normal",
}


def _base(n_interfaces: int = 2) -> ComposablePlateSpec:
    """The canonical adapter plate: NEMA17 face + a 2020 extrusion mount (default spacing ->
    the legacy fast-path delegate, the layout a real edit starts from)."""
    interfaces = [
        InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP),
        InterfacePlacement(
            ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, spacing_mm=20.0
        ),
    ]
    return ComposablePlateSpec(
        name="adapter",
        envelope_mm=Vec3(x=120.0, y=120.0, z=8.0),
        material="PLA",
        min_wall_mm=3.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=interfaces[:n_interfaces],
    )


def test_add_appends_and_keeps_others_byte_stable() -> None:
    spec = _base()
    new_spec, field_path, old_value, new_value = apply_spec_structural_edit(
        spec, "add", placement=_CLEARANCE
    )
    assert len(new_spec.interfaces) == 3
    assert field_path == "interfaces.2"
    assert old_value is None
    assert new_value is not None and "clearance_hole" in new_value
    # the two original interfaces are untouched (value-equal after the dump->revalidate round trip)
    assert list(new_spec.interfaces[:2]) == list(spec.interfaces)
    assert new_spec.interfaces[2].ref is InterfaceRef.CLEARANCE_HOLE


def test_remove_drops_the_index_and_keeps_the_rest() -> None:
    spec = _base()
    new_spec, field_path, old_value, new_value = apply_spec_structural_edit(spec, "remove", index=1)
    assert len(new_spec.interfaces) == 1
    assert new_spec.interfaces[0] == spec.interfaces[0]
    assert field_path == "interfaces.1"
    assert new_value is None
    assert old_value is not None and "extrusion_2020" in old_value


def test_remove_last_interface_refuses() -> None:
    """min_length=1 -- a part with no interfaces is not a valid plate, so removing the last refuses."""
    spec = _base(n_interfaces=1)
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "remove", index=0)


def test_add_beyond_max_length_refuses() -> None:
    """max_length=10 -- the 11th interface refuses (deterministic union constraint)."""
    crowded = ComposablePlateSpec(
        name="crowded",
        envelope_mm=Vec3(x=200.0, y=200.0, z=8.0),
        material="PLA",
        min_wall_mm=3.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.CLEARANCE_HOLE, face=Face.PLATE_TOP, component_id="M3"
            )
            for _ in range(10)
        ],
    )
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(crowded, "add", placement=_CLEARANCE)


def test_add_unknown_ref_refuses() -> None:
    spec = _base()
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "add", placement={"ref": "not_a_ref", "face": "plate_top"})


def test_add_missing_placement_refuses() -> None:
    spec = _base()
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "add", placement=None)


def test_remove_out_of_range_index_refuses() -> None:
    spec = _base()
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "remove", index=9)


def test_remove_without_index_refuses() -> None:
    spec = _base()
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "remove", index=None)


def test_structural_edit_on_non_composable_spec_refuses() -> None:
    """An L-bracket has no add/remove-able interfaces list -> a clear refusal, never a crash."""
    lbracket = validate_spec_payload(
        json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())
    )
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(lbracket, "add", placement=_CLEARANCE)


def test_three_interfaces_interpret_without_keyerror() -> None:
    """Regression for the legacy-delegate gap: adding a 3rd interface to the canonical nema+ext
    plate must route to the GENERIC derive path, not KeyError in the 2-placement fast-path."""
    spec = _base()
    new_spec, *_ = apply_spec_structural_edit(spec, "add", placement=_CLEARANCE)
    program = interpret_layout(new_spec)  # must NOT raise KeyError
    assert len(program.hardware_features) >= 1


def test_unknown_op_refuses() -> None:
    spec = _base()
    with pytest.raises(SpecPatchError):
        apply_spec_structural_edit(spec, "frobnicate", placement=_CLEARANCE)  # type: ignore[arg-type]
