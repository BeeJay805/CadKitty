"""patch_part_structural (ADR-0060 slice C): a deterministic ADD / REMOVE of an interface ->
re-derived, strictly re-validated part, with NO LLM. The structural twin of patch_part_structured:
the UI adds/removes a feature from the blueprint and the deterministic spine rebuilds + re-validates.
Real geometry; no-false-PASS refusals ship nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.api.patch import patch_part_structural
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.patch import validate_spec_payload
from forge_mind.types import Vec3

_CLEARANCE = {
    "ref": "clearance_hole",
    "face": "plate_bottom",
    "component_id": "M3",
    "fit": "normal",
}


def _base_data() -> dict[str, Any]:
    """The canonical adapter plate as a stored-spec dict (what rides the /generate response)."""
    spec = ComposablePlateSpec(
        name="adapter",
        envelope_mm=Vec3(x=120.0, y=120.0, z=8.0),
        material="PLA",
        min_wall_mm=3.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, spacing_mm=20.0
            ),
        ],
    )
    return spec.model_dump(mode="json")


@pytest.mark.slow
def test_add_interface_ships_and_survives(tmp_path: Path) -> None:
    """Add a clearance hole to the canonical plate: it re-derives through the generic path (NOT the
    legacy-delegate KeyError), PASSES the strict gates, ships free, and the two original interfaces
    are byte-stable."""
    spec_data = _base_data()
    res = patch_part_structural(spec_data, "add", tmp_path, placement=_CLEARANCE)
    assert res.status == "ok" and res.ok is not None, (res.refused, res.error)
    assert res.ok.gen_path == "standard:patch"
    assert res.ok.cost_usd == 0.0  # deterministic, no mapper call
    assert res.ok.validation.passed
    assert res.ok.patch is not None
    assert res.ok.patch.op == "add"
    assert res.ok.patch.field_path == "interfaces.2"
    assert res.ok.spec is not None
    assert len(res.ok.spec["interfaces"]) == 3
    base_norm = validate_spec_payload(spec_data).model_dump(mode="json")
    assert res.ok.spec["interfaces"][0] == base_norm["interfaces"][0]
    assert res.ok.spec["interfaces"][1] == base_norm["interfaces"][1]
    for name in ("part.stl", "part.glb", "part.layoutspec.json"):
        assert (tmp_path / name).is_file(), f"missing {name}"
    assert res.lane is not None and res.lane.outcome == "shipped" and res.lane.cost_usd == 0.0


@pytest.mark.slow
def test_remove_interface_ships_and_survives(tmp_path: Path) -> None:
    """Remove the extrusion mount: the remaining single-face plate re-derives, PASSES, and ships."""
    spec_data = _base_data()
    res = patch_part_structural(spec_data, "remove", tmp_path, index=1)
    assert res.status == "ok" and res.ok is not None, (res.refused, res.error)
    assert res.ok.validation.passed
    assert res.ok.patch is not None and res.ok.patch.op == "remove"
    assert res.ok.spec is not None and len(res.ok.spec["interfaces"]) == 1
    base_norm = validate_spec_payload(spec_data).model_dump(mode="json")
    assert res.ok.spec["interfaces"][0] == base_norm["interfaces"][0]
    assert (tmp_path / "part.stl").is_file()


def test_remove_last_interface_refuses(tmp_path: Path) -> None:
    """Removing the only interface refuses (min_length=1); nothing is built or shipped."""
    one = ComposablePlateSpec(
        name="adapter",
        envelope_mm=Vec3(x=120.0, y=120.0, z=8.0),
        material="PLA",
        min_wall_mm=3.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP)],
    ).model_dump(mode="json")
    res = patch_part_structural(one, "remove", tmp_path, index=0)
    assert res.status == "refused" and res.refused is not None
    assert "Can't apply that edit" in res.refused.reason
    assert not (tmp_path / "part.stl").exists()
    assert res.lane is not None and res.lane.outcome == "refused"


def test_add_unknown_ref_refuses(tmp_path: Path) -> None:
    """A ref outside the closed interface menu refuses (no geometry built)."""
    res = patch_part_structural(
        _base_data(), "add", tmp_path, placement={"ref": "not_a_ref", "face": "plate_top"}
    )
    assert res.status == "refused" and res.refused is not None
    assert not (tmp_path / "part.stl").exists()
    assert res.lane is not None and res.lane.outcome == "refused"


def test_structural_on_non_composable_refuses(tmp_path: Path) -> None:
    """A pi4 tray (walled cavity, no interfaces list) can't take add/remove -> a clear refusal."""
    tray = {"pattern": "pi4_case", "name": "tray"}
    res = patch_part_structural(tray, "add", tmp_path, placement=_CLEARANCE)
    # Either the payload is not a usable parametric model (error) or it has no interfaces (refused);
    # both ship nothing. The structural edit never crashes.
    assert res.status in {"refused", "error"}
    assert not (tmp_path / "part.stl").exists()


def test_invalid_spec_errors(tmp_path: Path) -> None:
    res = patch_part_structural({"pattern": "nope"}, "add", tmp_path, placement=_CLEARANCE)
    assert res.status == "error" and res.error is not None and res.error.stage == "spec"
