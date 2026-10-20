"""LATTICE Step-3 validation (PAID, <=$3; cost-tracked with a hard stop).

3a GRAMMAR    — every router schema FITS the real API, incl path:computed_body now carrying
                the 2-member FacetedStrutOp|LatticeOp union.
3b LATTICE    — a "lightweight lattice structural member" prompt routes -> computed_body, the
                generator picks op.kind=lattice, the part validates, and mass is REPORTED AND
                LIGHTER than a solid of the same envelope (the lattice payoff).
3c REGRESSION — a couple of existing-bench router trials still route + validate (no regression).
Plus a definitive per-path proof (generate_for_path(COMPUTED_BODY, ...)) isolating the lattice
emission from upstream blueprint variance.

Run: uv run python scripts/validate_lattice.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.schema import Blueprint, BlueprintVec3, FeatureSpec, InterfaceSpec
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.computed_ops import LatticeOp
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.router import GenerationPath, RoutedLayoutGenerator, check_grammar
from forge_mind.types import Vec3
from forge_mind.validator.metrics import density_for
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="lattice_validate_"))
BUDGET_STOP_USD = 2.7

_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()


def _solid_mass_g(env: Vec3, material: str) -> float:
    return (env.x * env.y * env.z / 1000.0) * density_for(material)


def _is_lattice(spec: object) -> bool:
    return isinstance(spec, ComputedBodySpec) and isinstance(spec.op, LatticeOp)


def _validate_spec(
    spec: object, tag: str, material: str
) -> tuple[bool, float | None, float | None]:
    """Compile + validate a routed spec; return (passed, mass_g, solid_mass_g)."""
    program = interpret_layout(spec)  # type: ignore[arg-type]
    model = compile_program(program, WORK / tag)
    report = validate(model, program, material=material)
    mass = report.metrics.mass_g if report.metrics else None
    solid = (
        _solid_mass_g(spec.envelope_mm, material) if isinstance(spec, ComputedBodySpec) else None
    )
    return report.passed, mass, solid


def _run_full(prompt: str, tag: str) -> tuple[str, bool, bool, float | None, float | None, float]:
    cost = 0.0
    intent, u = _intent(prompt)
    cost += u.cost_usd
    blueprint, u = _blueprint(intent)
    cost += u.cost_usd
    spec, usages = _router(blueprint)
    cost += sum(x.cost_usd for x in usages)
    passed, mass, solid = _validate_spec(spec, tag, blueprint.material)
    return spec.pattern.value, _is_lattice(spec), passed, mass, solid, cost


def _lattice_blueprint() -> Blueprint:
    return Blueprint(
        name="light_member",
        summary="A lightweight load-bearing structural member: a long brace that spans a gap and "
        "carries force, weight-optimized with an internal lattice to remove as much mass as "
        "possible while staying printable. A bare member -- it holds no motor and no board.",
        envelope_mm=BlueprintVec3(x=120.0, y=40.0, z=25.0),
        print_orientation=BlueprintVec3(x=0, y=0, z=1),
        material="PLA",
        min_wall_mm=3.0,
        max_overhang_deg=45.0,
        features=[
            FeatureSpec(
                name="brace",
                kind="strut",
                purpose="span a gap and carry load at low weight",
                approx_size_mm=BlueprintVec3(x=120.0, y=40.0, z=25.0),
            )
        ],
        # The schema requires >=1 interface; a bare member just needs a mounting point. The
        # computed_body interpreter ignores interfaces (it reads envelope/material/min_wall/op).
        required_interfaces=[
            InterfaceSpec(
                name="mount", interface_type=InterfaceType.M5_TNUT, on_feature="brace", facing="-X"
            )
        ],
    )


def main() -> None:
    total = 0.0
    print("=== 3a GRAMMAR (real API) ===")
    for label, fits, detail in check_grammar():
        print(f"  {'FIT ' if fits else 'OVER'} {label:<24} {detail}")

    print("\n=== 3b LATTICE from a prompt -> computed_body / lattice, validated + lighter ===")
    prompt = (
        "A lightweight structural brace about 120mm long that spans a gap and carries load. "
        "Make it as light as possible using an internal lattice. PLA, 3mm min wall. "
        "It holds no motor and no circuit board."
    )
    for t in range(2):
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            return
        try:
            route, is_lat, passed, mass, solid, c = _run_full(prompt, f"lat-{t}")
            total += c
            lighter = mass is not None and solid is not None and mass < solid
            ok = (
                "OK"
                if (route == "computed_body" and is_lat and passed and lighter)
                else "(see per-path proof)"
            )
            saved = f"{100 * (1 - mass / solid):.0f}%" if lighter and mass and solid else "-"
            print(
                f"  t{t}: route={route:<14} lattice={is_lat} passed={passed} "
                f"mass={mass and round(mass, 1)}g solid={solid and round(solid, 1)}g saved={saved} {ok}"
            )
        except Exception as e:
            print(f"  t{t}: ERROR {type(e).__name__}: {e}")

    print("\n=== 3c REGRESSION (existing benches via router) ===")
    repo = Path.cwd()
    for bench, expected in [
        ("bench-001-nema17-mount-v1", "composable_plate"),
        ("bench-002-pi4-case-v1", "pi4_case"),
    ]:
        if total > BUDGET_STOP_USD:
            print(f"  BUDGET STOP at ${total:.2f}")
            return
        bp_prompt = (repo / "benchmarks" / bench / "prompt.txt").read_text().strip()
        try:
            route, _lat, passed, _m, _s, c = _run_full(bp_prompt, bench)
            total += c
            ok = "OK" if (route == expected and passed) else "*** REGRESSION ***"
            print(f"  {bench:<30} route={route:<16} passed={passed} {ok}")
        except Exception as e:
            print(f"  {bench:<30} ERROR {type(e).__name__}: {e}")

    print("\n=== per-path proof: generate_for_path(COMPUTED_BODY) on a lattice blueprint ===")
    if total <= BUDGET_STOP_USD:
        try:
            bp = _lattice_blueprint()
            spec, usage = _router.generate_for_path(GenerationPath.COMPUTED_BODY, bp)
            total += usage.cost_usd
            passed, mass, solid = _validate_spec(spec, "lat-direct", bp.material)
            lighter = mass is not None and solid is not None and mass < solid
            saved = f"{100 * (1 - mass / solid):.0f}%" if lighter and mass and solid else "-"
            print(
                f"  op.kind={'lattice' if _is_lattice(spec) else getattr(spec.op, 'kind', '?')}  "  # type: ignore[attr-defined]
                f"passed={passed} mass={mass and round(mass, 1)}g solid={solid and round(solid, 1)}g "
                f"saved={saved} {'LIGHTER' if lighter else ''}"
            )
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")

    print(f"\nTOTAL SPEND: ${total:.3f}  (budget stop ${BUDGET_STOP_USD})")


if __name__ == "__main__":
    main()
