"""LayoutSpec — the discriminated union over supported part-family patterns, and
the interpreter dispatch (ADR-0004, ADR-0005, ADR-0008).

``LayoutSpec`` is a `pattern`-discriminated union of the concrete specs. The
LLM generator emits one member; ``interpret_layout`` routes it to the matching
deterministic interpreter. Adding a fourth pattern = add its spec + interpreter +
one dispatch branch (no other call site changes).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from forge_mind.dsl.program import PartProgram
from forge_mind.layout.adapter_interpreter import interpret_adapter_plate
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.arm_interpreter import ArmSpec, interpret_arm
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec, interpret_computed_body
from forge_mind.layout.interpreter import interpret as interpret_l_bracket
from forge_mind.layout.pi4_interpreter import interpret_electronics_tray
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.spec import LBracketMotorMountSpec, Pattern

# Discriminated on the `pattern` literal each concrete spec pins. The composable plate
# (ADR-0014) is the generator's emit going forward; the three legacy plate specs stay in
# the union so directly-constructed legacy specs + fixtures still interpret unchanged.
# ComputedBodySpec (ADR-0019) is the computed-geometry track (a standalone strut); ArmSpec
# is the first ARM (computed strut + NEMA-17 motor mount). Both additive.
LayoutSpec = Annotated[
    LBracketMotorMountSpec
    | ElectronicsTraySpec
    | AdapterPlateSpec
    | ComposablePlateSpec
    | ComputedBodySpec
    | ArmSpec,
    Field(discriminator="pattern"),
]


def interpret_layout(
    spec: LBracketMotorMountSpec
    | ElectronicsTraySpec
    | AdapterPlateSpec
    | ComposablePlateSpec
    | ComputedBodySpec
    | ArmSpec,
) -> PartProgram:
    """Route a LayoutSpec to its deterministic interpreter."""
    if spec.pattern is Pattern.COMPOSABLE_PLATE:
        assert isinstance(spec, ComposablePlateSpec)
        return interpret_composable_plate(spec)
    if spec.pattern is Pattern.COMPUTED_BODY:
        assert isinstance(spec, ComputedBodySpec)
        return interpret_computed_body(spec)
    if spec.pattern is Pattern.MOTOR_ARM:
        assert isinstance(spec, ArmSpec)
        return interpret_arm(spec)
    if spec.pattern is Pattern.L_BRACKET_MOTOR_MOUNT:
        assert isinstance(spec, LBracketMotorMountSpec)
        return interpret_l_bracket(spec)
    if spec.pattern is Pattern.ELECTRONICS_TRAY:
        assert isinstance(spec, ElectronicsTraySpec)
        return interpret_electronics_tray(spec)
    if spec.pattern is Pattern.ADAPTER_PLATE:
        assert isinstance(spec, AdapterPlateSpec)
        return interpret_adapter_plate(spec)
    raise ValueError(f"no interpreter for pattern {spec.pattern!r}")


__all__ = ["LayoutSpec", "interpret_layout"]
