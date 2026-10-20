"""Part B end-to-end validation (PAID, small): a raw lattice-arm prompt -> a validated,
lighter, lattice-bodied arm.

  1. GRAMMAR  — every router schema FITS (path:motor_arm now carries body + density).
  2. ARM      — a "lightweight lattice arm holding a NEMA 17" prompt routes -> motor_arm, the
                generator picks body=lattice, the part validates + the motor face is certified.
                Mass is reported AND lighter than the SAME beam built SOLID at equal envelope.

The solid baseline is a solid version of the lattice arm's beam (equal envelope) -- the honest
lattice-vs-solid comparison (a lattice body lightens a stiff full-section beam; it is heavier
than the minimal faceted strut, which is a different, smaller geometry).

Run: uv run python scripts/validate_lattice_arm.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.arm_sizing import arm_pad_width, derive_nema17_arm_envelope
from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.components.registry import NEMA_17
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.interfaces import place_nema17_face
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.router import RoutedLayoutGenerator, check_grammar
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="lattice_arm_validate_"))
BUDGET_STOP_USD = 1.0
_TIP_GAP_MM = 15.0
_CERT = (
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
)

_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()


def _solid_beam_arm_mass(reach: str, min_wall: float) -> float:
    """Mass of a SOLID version of the lattice arm's beam at the same (derived) envelope."""
    env = derive_nema17_arm_envelope(reach, min_wall)
    pad = arm_pad_width(NEMA_17, min_wall)
    body_len = env.x - _TIP_GAP_MM - pad / 2.0
    pad_thick = max(min_wall, 5.0)
    face_cx = body_len / 2.0 + _TIP_GAP_MM
    pad_cz = -env.z / 2.0 + pad_thick / 2.0
    ops: list[Operation] = [
        Box(
            op_id="beam",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=0.0),
            width=body_len,
            depth=pad,
            height=env.z,
        ),
        Box(
            op_id="mount_pad",
            mode=OpMode.ADD,
            position=Vec3(x=face_cx, y=0.0, z=pad_cz),
            width=pad,
            depth=pad,
            height=pad_thick,
        ),
        *place_nema17_face(center=Vec3(x=face_cx, y=0.0, z=pad_cz), thickness=pad_thick),
    ]
    decl = InterfaceDeclaration(
        name="motor_face",
        interface_type=InterfaceType.NEMA17_FACE,
        location=Vec3(x=face_cx, y=0.0, z=pad_cz + pad_thick / 2.0),
        normal=Vec3(x=0.0, y=0.0, z=1.0),
    )
    program = PartProgram(
        metadata=Metadata(
            name="solid_beam_arm", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=ops,
        interfaces=[decl],
        constraints=Constraints(min_wall_thickness=min_wall, max_overhang_deg=45.0, max_bbox=env),
    )
    model = compile_program(program, WORK / "solid_beam")
    report = validate(program=program, model=model, material="PLA")
    return report.metrics.mass_g if report.metrics else float("nan")


def _run_full(prompt: str, tag: str) -> tuple[str, str, bool, bool, float | None, float, str]:
    cost = 0.0
    intent, u = _intent(prompt)
    cost += u.cost_usd
    blueprint, u = _blueprint(intent)
    cost += u.cost_usd
    spec, usages = _router(blueprint)
    cost += sum(x.cost_usd for x in usages)
    body = getattr(spec, "body", "-")
    reach = getattr(spec, "reach", "standard")
    program = interpret_layout(spec)
    model = compile_program(program, WORK / tag)
    report = validate(program=program, model=model, material=blueprint.material)
    certified = all(
        any(c.id == cid and c.status.value == "pass" for c in report.checks) for cid in _CERT
    )
    mass = report.metrics.mass_g if report.metrics else None
    return spec.pattern.value, body, report.passed, certified, mass, cost, reach


def main() -> None:
    total = 0.0
    print("=== 1. GRAMMAR (real API) ===")
    for label, fits, detail in check_grammar():
        print(f"  {'FIT ' if fits else 'OVER'} {label:<24} {detail}")

    print(
        "\n=== 2. LATTICE ARM from a raw prompt -> motor_arm / body=lattice, validated + lighter ==="
    )
    prompts = [
        "A lightweight drone arm with an internal lattice that holds a NEMA 17 motor at its far end. As light as possible. PLA.",
        "A long lattice-bodied robot arm carrying a NEMA 17 stepper at the tip, weight-optimized, PLA.",
    ]
    for t, prompt in enumerate(prompts):
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            break
        try:
            route, body, passed, certified, mass, c, reach = _run_full(prompt, f"larm-{t}")
            total += c
            solid = _solid_beam_arm_mass(reach, 4.0)
            lighter = mass is not None and mass < solid
            saved = f"{100 * (1 - mass / solid):.0f}%" if lighter and mass else "-"
            ok = (
                "OK"
                if (route == "motor_arm" and body == "lattice" and passed and certified and lighter)
                else "(check)"
            )
            print(
                f"  t{t}: route={route:<10} body={body:<8} reach={reach:<8} passed={passed} "
                f"certified={certified} mass={mass and round(mass, 1)}g vs solid-beam {solid:.1f}g saved={saved} {ok}"
            )
        except Exception as e:
            print(f"  t{t}: ERROR {type(e).__name__}: {e}")

    print(f"\nTOTAL SPEND: ${total:.3f}  (budget stop ${BUDGET_STOP_USD})")


if __name__ == "__main__":
    main()
