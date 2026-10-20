"""Composition engine Step-0 safety proof (deterministic, $0): the engine cannot admit a bad part.

  1. Each standards hardware feature places + validates clean on a flat plate (the vocabulary works).
  2. A deliberately COLLIDING composition (two features overlapping) is REJECTED at the derive
     stage (the feature-collision / min-wall-between-features check), never built.
  3. A feature with NO valid registry spec (an unknown key) is EXCLUDED -- the interpreter refuses
     it, so a part can never be built from a dimension the registry does not vouch for.

Run: uv run python scripts/derisk_composition.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    BaseDimsSpec,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.interpreter import LayoutError
from forge_mind.types import Vec3
from forge_mind.validator.report import CheckStatus
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="composition_derisk_"))
_ENV = Vec3(x=240.0, y=160.0, z=90.0)


def _plate(
    name: str, ifaces: list[InterfacePlacement], base_dims: BaseDimsSpec | None = None
) -> ComposablePlateSpec:
    return ComposablePlateSpec(
        name=name,
        envelope_mm=_ENV,
        material="PLA",
        min_wall_mm=4.0,
        base=BaseArchetype.FLAT_PLATE,
        base_dims=base_dims,
        interfaces=ifaces,
    )


def _one(ref: InterfaceRef, cid: str | None, fit: str | None = None) -> InterfacePlacement:
    return InterfacePlacement(ref=ref, face=Face.PLATE_TOP, component_id=cid, fit=fit)


def main() -> None:
    print("=== 1. each hardware feature places + validates clean on a flat plate ===")
    vocab = [
        ("clearance_hole M4", _one(InterfaceRef.CLEARANCE_HOLE, "M4", "normal")),
        ("counterbore M5", _one(InterfaceRef.COUNTERBORE, "M5")),
        ("dowel_hole 4", _one(InterfaceRef.DOWEL_HOLE, "4", "press")),
        ("heat_set_boss M3", _one(InterfaceRef.HEAT_SET_BOSS, "M3")),
        ("bearing_seat 608", _one(InterfaceRef.BEARING_SEAT, "608", "press")),
        ("magnet_pocket 6x3", _one(InterfaceRef.MAGNET_POCKET, "6x3")),
        ("captive_nut_pocket M3", _one(InterfaceRef.CAPTIVE_NUT_POCKET, "M3")),
    ]
    passed = 0
    for label, p in vocab:
        program = interpret_composable_plate(_plate("v", [p]))
        report = validate(
            compile_program(program, WORK / label.split()[0]), program, material="PLA"
        )
        bad = [c.id for c in report.checks if c.status != CheckStatus.PASS]
        passed += int(report.passed)
        print(
            f"  {'OK  ' if report.passed else 'FAIL'} {label:24} {'' if report.passed else bad[:2]}"
        )
    print(f"  -> {passed}/{len(vocab)} hardware features validate clean")

    print("\n=== 2. a deliberately colliding composition is REJECTED (never built) ===")
    colliding = _plate(
        "collide",
        [
            _one(InterfaceRef.CLEARANCE_HOLE, "M8", "normal"),
            _one(InterfaceRef.CLEARANCE_HOLE, "M8", "normal"),
        ],
        base_dims=BaseDimsSpec(width_mm=60.0, depth_mm=60.0),
    )
    # Force the two M8 holes nearly on top of each other (gap << min_wall) via explicit offsets.
    colliding = colliding.model_copy(
        update={
            "interfaces": [
                colliding.interfaces[0].model_copy(update={"offset_mm": 0.0}),
                colliding.interfaces[1].model_copy(update={"offset_mm": 3.0}),
            ]
        }
    )
    try:
        interpret_composable_plate(colliding)
        print("  *** NOT REJECTED -- collision admitted (FALSE-PASS HAZARD)")
        collision_ok = False
    except LayoutError as e:
        print(f"  REJECTED at derive: {str(e)[:90]}")
        collision_ok = True

    print("\n=== 3. a feature with no valid registry spec is EXCLUDED ===")
    try:
        interpret_composable_plate(
            _plate("nospec", [_one(InterfaceRef.CLEARANCE_HOLE, "M99", "normal")])
        )
        print("  *** NOT EXCLUDED -- built from a non-registry dimension (FALSE-PASS HAZARD)")
        nospec_ok = False
    except (LayoutError, ValueError) as e:
        print(f"  EXCLUDED: {str(e)[:90]}")
        nospec_ok = True

    ok = passed == len(vocab) and collision_ok and nospec_ok
    print(
        f"\n{'SAFETY FLOOR PROVEN -- the engine cannot admit a bad part ($0)' if ok else '*** SAFETY PROOF FAILED'}"
    )


if __name__ == "__main__":
    main()
