"""Measure the buildable-kit outputs (BOM + make-it) on a few real parts -- OFFLINE, $0.

Builds several composable parts through the deterministic interpreter (no LLM), compiles +
validates each, then prints the registry-backed Bill of Materials and the advisory make-it
instructions. Renders ONE sample captioned with its recipe (the standing render convention).

Run:  uv run python scripts/measure_kit.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.compiler.compile import compile_program
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
from forge_mind.validator.run import validate
from forge_mind.viz.render import render_part_to_png

RENDER_DIR = Path.home() / "Desktop" / "forge-mind-adr0017-renders"
MATERIAL = "PETG"


def _plate(
    name: str, base: BaseArchetype, interfaces: list[InterfacePlacement]
) -> ComposablePlateSpec:
    return ComposablePlateSpec(
        name=name,
        envelope_mm=Vec3(x=200.0, y=200.0, z=20.0),
        material=MATERIAL,
        min_wall_mm=4.0,
        base=base,
        interfaces=interfaces,
    )


def _clr() -> InterfacePlacement:
    return InterfacePlacement(
        ref=InterfaceRef.CLEARANCE_HOLE, face=Face.PLATE_TOP, component_id="M3", fit="normal"
    )


def _heat() -> InterfacePlacement:
    return InterfacePlacement(
        ref=InterfaceRef.HEAT_SET_BOSS, face=Face.PLATE_TOP, component_id="M3"
    )


def _bearing() -> InterfacePlacement:
    return InterfacePlacement(
        ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
    )


def _magnet() -> InterfacePlacement:
    return InterfacePlacement(
        ref=InterfaceRef.MAGNET_POCKET, face=Face.PLATE_TOP, component_id="6x3"
    )


# (label/recipe, spec) -- the label doubles as the render caption (standing convention).
CASES: list[tuple[str, ComposablePlateSpec]] = [
    (
        "flat plate: a 608 bearing seat + 2 M3 clearance holes + 2 M3 heat-set bosses",
        _plate(
            "kit-bearing-plate",
            BaseArchetype.FLAT_PLATE,
            [_bearing(), _clr(), _clr(), _heat(), _heat()],
        ),
    ),
    (
        "open box: a 608 bearing seat + 4 M3 clearance holes on a walled floor",
        _plate(
            "kit-open-box", BaseArchetype.OPEN_BOX, [_bearing(), _clr(), _clr(), _clr(), _clr()]
        ),
    ),
    (
        "round plate: a 6x3 magnet pocket + 2 M3 clearance holes",
        _plate("kit-round-magnet", BaseArchetype.ROUND_PLATE, [_magnet(), _clr(), _clr()]),
    ),
]


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="forge-kit-"))
    first_render: Path | None = None

    for recipe, spec in CASES:
        print("=" * 88)
        print(f"PART: {spec.name}")
        print(f"recipe: {recipe}")
        program = interpret_composable_plate(spec)
        model = compile_program(program, work / spec.name)
        report = validate(model, program, material=MATERIAL)
        kit = build_kit(program, material=MATERIAL, report=report)

        print(f"validator: {report.overall.value.upper()}")
        print(f"\nBILL OF MATERIALS ({kit.bom.total_items} items):")
        for line in kit.bom.lines:
            print(f"  - {line.render()}   [{line.source}]")
        for flagged in kit.bom.unmapped:
            print(f"  ! unmapped: {flagged}")

        print("\nMAKE IT (advisory):")
        print(
            f"  material {kit.instructions.material}, layer {kit.instructions.layer_height_mm:g} mm"
        )
        print(f"  {kit.instructions.wall_note}")
        for i, step in enumerate(kit.instructions.steps, 1):
            print(f"  {i}. {step}")
        if kit.instructions.insert_install_note:
            print(f"  note: {kit.instructions.insert_install_note}")
        print()

        if first_render is None:
            png = RENDER_DIR / "kit_bearing_plate.png"
            render_part_to_png(model.stl_path, png, title=spec.name, caption=recipe)
            first_render = png

    print("=" * 88)
    if first_render is not None:
        print(f"rendered sample (captioned with its recipe): {first_render}")


if __name__ == "__main__":
    main()
