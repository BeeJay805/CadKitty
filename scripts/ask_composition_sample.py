"""Composition Step-4 askability (PAID, <=$3): are the new composed parts reachable by a
natural-language prompt through the real product path?

3a PRODUCT PATH: a hardware prompt -> intent -> blueprint -> ADR-0020 router -> a composable_plate
   spec that PLACES standards hardware -> interpret -> validate (the part comes out).
3b PER-PATH PROOF: generate_for_path(COMPOSABLE_PLATE) on a hardware-described blueprint, isolating
   the composable-path emission from upstream blueprint variance.

Reports routed (-> composable_plate, with >=1 hardware placement) + validated count + spend.

Run: uv run python scripts/ask_composition_sample.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.schema import Blueprint, BlueprintVec3, FeatureSpec, InterfaceSpec
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.composable_hardware import HARDWARE_REFS
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.router import GenerationPath, RoutedLayoutGenerator
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="ask_composition_"))
BUDGET_STOP_USD = 2.7
_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()

PROMPTS = [
    "A flat PLA mounting plate with two M5 bolt clearance holes and a heat-set insert boss for an M3 screw. 4mm walls.",
    "A flat plate with a press-fit seat for a 608 ball bearing and an M4 bolt clearance hole. PLA, 4mm min wall.",
    "A flat bracket plate with a captive M4 nut pocket and a 6 mm dowel locating hole, PLA.",
    "A flat plate with a counterbored M5 screw hole and a pocket for a 6x3 disc magnet. PLA, 4mm walls.",
]


def _hw_placements(spec: object) -> int:
    return (
        sum(1 for p in spec.interfaces if p.ref in HARDWARE_REFS)
        if isinstance(spec, ComposablePlateSpec)
        else 0
    )


def _validate_spec(spec: object, tag: str) -> bool:
    program = interpret_layout(spec)  # type: ignore[arg-type]
    model = compile_program(program, WORK / tag)
    return validate(model, program, material="PLA").passed


def _hw_blueprint() -> Blueprint:
    return Blueprint(
        name="hw_plate",
        summary="A flat PLA mounting plate that hosts standards hardware: a press-fit 608 ball-bearing "
        "seat, two M4 bolt clearance holes, and a heat-set insert boss for an M3 screw. A drilled/"
        "bossed flat plate -- not a motor mount, not an enclosure.",
        envelope_mm=BlueprintVec3(x=140.0, y=120.0, z=40.0),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=4.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="plate",
                kind="plate",
                purpose="host bearing + bolt + insert hardware",
                approx_size_mm=BlueprintVec3(x=120.0, y=100.0, z=4.0),
            )
        ],
        required_interfaces=[
            InterfaceSpec(
                name="bolt", interface_type=InterfaceType.M5_TNUT, on_feature="plate", facing="+Z"
            )
        ],
    )


def main() -> None:
    total = 0.0
    routed = validated = 0
    print("=== 3a PRODUCT PATH: NL prompt -> router -> composable_plate with hardware ===")
    for i, prompt in enumerate(PROMPTS):
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            break
        try:
            intent, u = _intent(prompt)
            total += u.cost_usd
            blueprint, u = _blueprint(intent)
            total += u.cost_usd
            spec, usages = _router(blueprint)
            total += sum(x.cost_usd for x in usages)
            fam = spec.pattern.value
            nhw = _hw_placements(spec)
            is_comp = fam == "composable_plate" and nhw >= 1
            ok = is_comp and _validate_spec(spec, f"p{i}")
            routed += int(is_comp)
            validated += int(ok)
            print(
                f"  t{i}: route={fam:<16} hardware_placements={nhw} validated={ok} "
                f"{'OK' if ok else '(see per-path proof)'}"
            )
        except Exception as e:
            print(f"  t{i}: ERROR {type(e).__name__}: {str(e)[:80]}")

    print(
        "\n=== 3b PER-PATH PROOF: generate_for_path(COMPOSABLE_PLATE) on a hardware blueprint ==="
    )
    if total <= BUDGET_STOP_USD:
        try:
            spec, usage = _router.generate_for_path(
                GenerationPath.COMPOSABLE_PLATE, _hw_blueprint()
            )
            total += usage.cost_usd
            nhw = _hw_placements(spec)
            ok = nhw >= 1 and _validate_spec(spec, "perpath")
            refs = (
                [p.ref.value for p in spec.interfaces]
                if isinstance(spec, ComposablePlateSpec)
                else []
            )
            print(f"  emitted refs={refs} hardware_placements={nhw} validated={ok}")
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {str(e)[:80]}")

    print(
        f"\n  routed-to-composable-with-hardware: {routed}/{len(PROMPTS)}   routed+validated: {validated}/{len(PROMPTS)}"
    )
    print(f"  TOTAL SPEND: ${total:.3f}  (cap ${BUDGET_STOP_USD + 0.3:.1f})")


if __name__ == "__main__":
    main()
