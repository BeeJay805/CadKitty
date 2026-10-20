"""ADR-0020 Step-6 router validation (PAID, <=$5 authorized; cost-tracked with a hard stop).

6a GRAMMAR  — classifier + every per-path schema FIT the real API (the headline the old
              single wrapper failed).
6b REGRESSION — a few live trials route the EXISTING benches (001/003/006 -> composable_plate,
              002 -> electronics_tray) through the ROUTER and validate (must not regress).
6c UNBLOCK  — a structural-strut prompt routes to computed_body end-to-end; plus a definitive
              per-path proof that computed_body generates + validates with a reported mass.

The router flow is run EXPLICITLY (no direct-DSL fallback) so a router regression is visible
rather than masked by the fallback.

Run: uv run python scripts/validate_router.py
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
WORK = Path(tempfile.mkdtemp(prefix="router_validate_"))
BUDGET_STOP_USD = 4.5

_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()


def _run_full(prompt: str, tag: str) -> tuple[str, bool, float | None, float]:
    """Full router flow: prompt -> intent -> blueprint -> router -> interpret -> validate."""
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
    return spec.pattern.value, report.passed, mass, cost


def _strut_blueprint() -> Blueprint:
    return Blueprint(
        name="hex_strut",
        summary="A standalone hexagonal structural strut: a load-bearing beam "
        "member that spans a distance and carries force. A structural member, NOT a mounting "
        "plate and NOT an enclosure.",
        envelope_mm=BlueprintVec3(x=120, y=16, z=16),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=3.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="member",
                kind="strut",
                purpose="carry axial/bending load",
                approx_size_mm=BlueprintVec3(x=120, y=14, z=14),
            )
        ],
        required_interfaces=[
            InterfaceSpec(
                name="frame_mount",
                interface_type=InterfaceType.M5_TNUT,
                on_feature="member",
                facing="-X",
            )
        ],
    )


def main() -> None:
    total = 0.0
    print("=== 6a GRAMMAR (real API) ===")
    for label, fits, detail in check_grammar():
        print(f"  {'FIT ' if fits else 'OVER'} {label:<24} {detail}")

    print("\n=== 6b REGRESSION via router (existing patterns must NOT regress) ===")
    benches = [
        ("bench-001-nema17-mount-v1", "composable_plate"),
        ("bench-003-nema17-2020-adapter-plate-v1", "composable_plate"),
        ("bench-006-fan-mount-v1", "composable_plate"),
        ("bench-002-pi4-case-v1", "pi4_case"),
    ]
    for bench, expected in benches:
        prompt = (REPO / "benchmarks" / bench / "prompt.txt").read_text().strip()
        for t in range(2):
            if total > BUDGET_STOP_USD:
                print(f"  BUDGET STOP at ${total:.2f}")
                return
            try:
                route, passed, _mass, c = _run_full(prompt, f"{bench}-{t}")
                total += c
                ok = "OK" if (route == expected and passed) else "*** REGRESSION ***"
                print(f"  {bench:<40} t{t}: route={route:<16} passed={passed} {ok}")
            except Exception as e:
                print(f"  {bench:<40} t{t}: ERROR {type(e).__name__}: {e}")

    print("\n=== 6c UNBLOCK: structural strut -> computed_body ===")
    strut_prompt = (
        "A lightweight hexagonal structural strut: a load-bearing beam member about 120 mm long "
        "that spans between two points and carries axial and bending load. It is a structural "
        "member, not a mounting plate and not an enclosure. PLA, 3 mm minimum wall."
    )
    for t in range(2):
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            return
        try:
            route, passed, mass, c = _run_full(strut_prompt, f"strut-{t}")
            total += c
            tag = (
                "OK"
                if (route == "computed_body" and passed)
                else "(routed elsewhere; see definitive proof)"
            )
            print(f"  full-pipeline t{t}: route={route:<16} passed={passed} mass={mass} {tag}")
        except Exception as e:
            print(f"  full-pipeline t{t}: ERROR {type(e).__name__}: {e}")

    # Definitive per-path proof (isolates the computed_body EMISSION from upstream variance).
    if total <= BUDGET_STOP_USD:
        try:
            bp = _strut_blueprint()
            spec, usage = _router.generate_for_path(GenerationPath.COMPUTED_BODY, bp)
            total += usage.cost_usd
            program = interpret_layout(spec)
            model = compile_program(program, WORK / "strut-direct")
            report = validate(model, program, material=bp.material)
            mass = report.metrics.mass_g if report.metrics else None
            print(
                f"  per-path computed_body: spec={spec.pattern.value} passed={report.passed} mass={mass}"
            )
        except Exception as e:
            print(f"  per-path computed_body: ERROR {type(e).__name__}: {e}")

    print(f"\nTOTAL SPEND: ${total:.3f}  (budget stop ${BUDGET_STOP_USD})")


if __name__ == "__main__":
    main()
