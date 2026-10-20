"""Integration: the kit on a REAL interpreter-produced part (offline, no LLM).

Proves the wiring end of the contract -- the composable interpreter actually carries
``hardware_features`` onto the program, and ``build_kit`` turns those into the right buyable
lines + assembly steps. Complements the unit tests, which use hand-built feature lists.
"""

from __future__ import annotations

from forge_mind.kit import build_kit
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.types import Vec3


def _bearing_plate() -> ComposablePlateSpec:
    """A 608 bearing seat + 2 M3 clearance holes + 2 M3 heat-set bosses on a flat plate.

    Five interfaces covering all three buyable kinds (bearing, screw, insert) in one feasible part.
    """
    interfaces = [
        InterfacePlacement(
            ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
        ),
        *[
            InterfacePlacement(
                ref=InterfaceRef.CLEARANCE_HOLE,
                face=Face.PLATE_TOP,
                component_id="M3",
                fit="normal",
            )
            for _ in range(2)
        ],
        *[
            InterfacePlacement(
                ref=InterfaceRef.HEAT_SET_BOSS, face=Face.PLATE_TOP, component_id="M3"
            )
            for _ in range(2)
        ],
    ]
    return ComposablePlateSpec(
        name="kit-bearing-plate",
        envelope_mm=Vec3(x=200.0, y=200.0, z=20.0),
        material="PETG",
        min_wall_mm=4.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=interfaces,
    )


def test_kit_from_a_real_composable_part() -> None:
    program = interpret_composable_plate(_bearing_plate())
    # The interpreter must have carried the registry-backed features onto the program.
    assert program.hardware_features, "interpreter did not carry hardware_features"

    kit = build_kit(program, material="PETG")
    by_component = {line.component: line for line in kit.bom.lines}

    assert by_component["608 deep-groove ball bearing"].quantity == 1
    assert by_component["M3 socket-head cap screw"].quantity == 2
    assert by_component["M3 heat-set threaded insert"].quantity == 2
    assert kit.bom.unmapped == []  # everything mapped to the registry
    assert kit.bom.total_items == 5

    # ISO provenance for the bearing, PROVISIONAL flag surfaced for the insert.
    assert "ISO 15" in by_component["608 deep-groove ball bearing"].source
    assert by_component["M3 heat-set threaded insert"].provisional

    # Make-it reflects the part: material carried, inserts -> install note + step.
    assert kit.instructions.material == "PETG"
    assert kit.instructions.insert_install_note is not None
    joined = " | ".join(kit.instructions.steps)
    assert "Heat-set" in joined
    assert "Press" in joined  # the bearing press step
