"""Library Step-2: is the library ASKABLE? (PAID, <=$3, cost-tracked with a hard stop.)

A random sample across families: a natural-language prompt -> the REAL product path (the ADR-0020
router: intent -> blueprint -> classify -> per-path spec) -> the matching validated part. Confirms
a plain sentence reaches the same family of validated part the deterministic builder enumerated.

Reports routed (family == expected) + validated (report.passed) counts + actual spend. The
builder added NO new spec/union (it reuses the existing specs), so no grammar grew; the real-API
grammar smoke-check is included anyway as cheap insurance.

Run: uv run python scripts/ask_library_sample.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.compiler.compile import compile_program
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.router import RoutedLayoutGenerator, check_grammar
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="ask_library_"))
BUDGET_STOP_USD = 2.7

_intent = IntentParser()
_blueprint = BlueprintGenerator()
_router = RoutedLayoutGenerator()

# (prompt, expected family pattern value, a categorical hint to confirm, a predicate on the spec).
SAMPLE = [
    (
        "A flat plate that bolts a NEMA 17 motor on top and mounts to a 2020 aluminium extrusion underneath.",
        "composable_plate",
        "",
        lambda s: True,
    ),
    (
        "A stiff hexagonal structural strut that spans a gap and carries load. PLA, 4mm wall.",
        "computed_body",
        "faceted_strut",
        lambda s: getattr(s.op, "kind", "") == "faceted_strut",
    ),
    (
        "A lightweight lattice brace that spans a gap, as light as possible. PLA.",
        "computed_body",
        "lattice",
        lambda s: getattr(s.op, "kind", "") == "lattice",
    ),
    (
        "A drone arm that holds a NEMA 17 motor at its far tip.",
        "motor_arm",
        "solid",
        lambda s: getattr(s, "body", "") == "solid",
    ),
    (
        "A lightweight lattice drone arm that carries a NEMA 17 motor at the end, weight-optimized.",
        "motor_arm",
        "lattice",
        lambda s: getattr(s, "body", "") == "lattice",
    ),
    (
        "A Raspberry Pi 4 case: an open tray the board screws onto, with ventilation slots in the floor.",
        "pi4_case",
        "",
        lambda s: True,
    ),
]


def main() -> None:
    print("=== grammar (real API; no schema grew this task, checked anyway) ===")
    for label, fits, _detail in check_grammar():
        print(f"  {'FIT ' if fits else 'OVER'} {label}")

    print("\n=== ASKABLE sample: NL prompt -> product path -> matching validated part ===")
    total = 0.0
    routed = validated = 0
    for i, (prompt, expected, hint, ok_hint) in enumerate(SAMPLE):
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
            program = interpret_layout(spec)
            model = compile_program(program, WORK / f"s{i}")
            report = validate(model, program, material=blueprint.material)
            fam = spec.pattern.value
            fam_ok = fam == expected and ok_hint(spec)
            routed += int(fam_ok)
            validated += int(fam_ok and report.passed)
            mass = report.metrics.mass_g if report.metrics else None
            tag = "OK" if (fam_ok and report.passed) else "(miss)"
            print(
                f"  [{expected:16}{('/' + hint) if hint else '':>16}] routed={fam:<16} "
                f"validated={report.passed} mass={mass and round(mass, 1)} {tag}"
            )
        except Exception as e:
            print(f"  [{expected}] ERROR {type(e).__name__}: {e}")

    n = len(SAMPLE)
    print(f"\n  routed-to-expected-family: {routed}/{n}   routed+validated: {validated}/{n}")
    print(f"  TOTAL SPEND: ${total:.3f}  (cap ${BUDGET_STOP_USD + 0.3:.1f})")


if __name__ == "__main__":
    main()
