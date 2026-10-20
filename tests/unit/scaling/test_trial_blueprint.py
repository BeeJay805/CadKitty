"""ADR-0017: the engine's decontaminated trial Blueprint must NAME the component variant
so the LLM dispatches the right component_id (else a new family is marked menu-mismatch and
wrongly rejected in the paid run). Deterministic ($0) -- asserts the request TEXT, no LLM."""

from __future__ import annotations

from forge_mind.layout.composable_spec import BaseArchetype, Face, InterfaceRef
from forge_mind.scaling.engine import Perturbation
from forge_mind.scaling.enumerate import FamilyShape, Placement
from forge_mind.scaling.trial import build_blueprint
from forge_mind.types import Vec3

_PERT = Perturbation(4.0, Vec3(x=400.0, y=400.0, z=40.0), "nominal")


def _summary(shape: FamilyShape) -> str:
    return build_blueprint(shape, _PERT, 0).summary


def test_blueprint_names_new_hardware_variants() -> None:
    s = FamilyShape(
        BaseArchetype.FLAT_PLATE,
        (
            Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP, "nema23"),
            Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM, "ext_3030"),
        ),
    )
    summary = _summary(s)
    assert "NEMA 23" in summary, summary  # so the LLM picks component_id=nema23
    assert "3030" in summary, summary  # ...and ext_3030


def test_blueprint_keeps_legacy_default_wording() -> None:
    # Legacy defaults (component_id None) keep the generic noun -> the LLM omits component_id,
    # matching the None-keyed control shapes (no change to the validated controls).
    s = FamilyShape(
        BaseArchetype.FLAT_PLATE,
        (
            Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP, None),
            Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM, None),
        ),
    )
    summary = _summary(s)
    assert "NEMA 17" in summary and "2020" in summary, summary


def test_blueprint_names_fan_size() -> None:
    s = FamilyShape(
        BaseArchetype.FLAT_PLATE, (Placement(InterfaceRef.FAN_FACE, Face.PLATE_TOP, "fan_60mm"),)
    )
    assert "60 mm" in _summary(s)
