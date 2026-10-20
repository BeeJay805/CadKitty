"""FIRST ARM Step-3 validation (PAID, <=$5; cost-tracked with a hard stop).

3a GRAMMAR  — the arm path's schema FITS the real API (with the rest).
3b ARM      — a "drone arm that holds a NEMA 17 motor" prompt routes -> motor_arm, the strut +
              mount assemble into one valid body, the motor interface is certified, mass reported.
3c REGRESSION — a couple of existing-bench router trials still route + validate (no regression).
Plus a definitive per-path proof (generate_for_path(MOTOR_ARM, ...)) isolating the arm
emission from upstream blueprint variance.

Run: uv run python scripts/validate_arm.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.schema import Blueprint, BlueprintVec3, FeatureSpec, InterfaceSpec
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.router import GenerationPath, RoutedLayoutGenerator, check_grammar
from forge_mind.validator.run import validate

REPO = Path.cwd()
WORK = Path(tempfile.mkdtemp(prefix="arm_validate_"))
BUDGET_STOP_USD = 4.5
_MOTOR_FACE_CHECKS = (
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
)

_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()


def _certified(report) -> bool:  # type: ignore[no-untyped-def]
    ck = {c.id: c.status.value for c in report.checks}
    return all(ck.get(c) == "pass" for c in _MOTOR_FACE_CHECKS)


def _run_full(prompt: str, tag: str) -> tuple[str, bool, bool, float | None, float]:
    cost = 0.0
    intent, u = _intent(prompt)
    cost += u.cost_usd
    blueprint, u = _blueprint(intent)
    cost += u.cost_usd
    spec, usages = _router(blueprint)
    cost += sum(x.cost_usd for x in usages)
    program = interpret_layout(spec)
    model = compile_program(program, WORK / tag)
    report = validate(model, program, material=blueprint.material)
    mass = report.metrics.mass_g if report.metrics else None
    return spec.pattern.value, report.passed, _certified(report), mass, cost


def _arm_blueprint() -> Blueprint:
    return Blueprint(
        name="drone_arm",
        summary="A drone arm: a structural strut that holds a NEMA 17 motor "
        "at its tip. A load-bearing member carrying a motor mount.",
        envelope_mm=BlueprintVec3(x=180, y=60, z=30),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=4.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="arm",
                kind="strut",
                purpose="span and carry the motor at the tip",
                approx_size_mm=BlueprintVec3(x=180, y=14, z=14),
            )
        ],
        required_interfaces=[
            InterfaceSpec(
                name="motor_face",
                interface_type=InterfaceType.NEMA17_FACE,
                on_feature="arm",
                facing="+Z",
            )
        ],
    )


def main() -> None:
    total = 0.0
    print("=== 3a GRAMMAR (real API) ===")
    for label, fits, detail in check_grammar():
        print(f"  {'FIT ' if fits else 'OVER'} {label:<24} {detail}")

    print("\n=== 3b ARM from a prompt -> motor_arm, assembled + certified + mass ===")
    arm_prompt = "A drone arm that holds a NEMA 17 motor at its tip: a lightweight structural strut spanning ~180mm with the motor mount at the end. PLA, 4mm min wall."
    for t in range(2):
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            return
        try:
            route, passed, certified, mass, c = _run_full(arm_prompt, f"arm-{t}")
            total += c
            ok = "OK" if (route == "motor_arm" and passed and certified) else "(see per-path proof)"
            print(
                f"  t{t}: route={route:<16} passed={passed} motor_certified={certified} mass={mass} {ok}"
            )
        except Exception as e:
            print(f"  t{t}: ERROR {type(e).__name__}: {e}")

    print("\n=== 3c REGRESSION (existing benches via router) ===")
    for bench, expected in [
        ("bench-001-nema17-mount-v1", "composable_plate"),
        ("bench-002-pi4-case-v1", "pi4_case"),
    ]:
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            return
        prompt = (REPO / "benchmarks" / bench / "prompt.txt").read_text().strip()
        try:
            route, passed, _cert, _mass, c = _run_full(prompt, bench)
            total += c
            ok = "OK" if (route == expected and passed) else "*** REGRESSION ***"
            print(f"  {bench:<32} route={route:<16} passed={passed} {ok}")
        except Exception as e:
            print(f"  {bench:<32} ERROR {type(e).__name__}: {e}")

    print("\n=== per-path proof: generate_for_path(MOTOR_ARM) -> validated arm + mass ===")
    if total <= BUDGET_STOP_USD:
        try:
            bp = _arm_blueprint()
            spec, usage = _router.generate_for_path(GenerationPath.MOTOR_ARM, bp)
            total += usage.cost_usd
            program = interpret_layout(spec)
            model = compile_program(program, WORK / "arm-direct")
            report = validate(model, program, material=bp.material)
            mass = report.metrics.mass_g if report.metrics else None
            print(
                f"  spec={spec.pattern.value} passed={report.passed} motor_certified={_certified(report)} mass={mass}"
            )
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")

    print(f"\nTOTAL SPEND: ${total:.3f}  (budget stop ${BUDGET_STOP_USD})")


if __name__ == "__main__":
    main()
