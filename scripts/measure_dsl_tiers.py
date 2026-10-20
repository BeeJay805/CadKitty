"""Measure the DSL stage across model tiers.

Runs DSLGenerator N times per model on the bench-001 Blueprint, compiles +
validates each output, and reports pass / compile-fail / error rates, cost, and
latency. Renders the first passing part per tier so the spread is inspectable.

    uv run python scripts/measure_dsl_tiers.py [N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.config import HAIKU, SONNET
from forge_mind.dsl.generator import DSLGenerator
from forge_mind.dsl.grader import grade_program
from forge_mind.viz.render import render_part_to_png

REPO = Path(__file__).resolve().parent.parent
BLUEPRINT = REPO / "tests" / "fixtures" / "blueprints" / "nema17_mount.blueprint.json"
RENDERS = REPO / "benchmarks" / "results" / "_renders"

OPUS_48 = "claude-opus-4-8"
MODELS = [HAIKU, SONNET, OPUS_48]

REQUIRED_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
    "interfaces.extrusion_face.two_m5_holes",
    "interfaces.extrusion_face.m5_spacing",
]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    blueprint = Blueprint.model_validate(json.loads(BLUEPRINT.read_text()))

    print(f"DSL stage across tiers, N={n} each\n")
    print(
        f"{'model':22} {'pass':>7} {'rate':>6} {'cmpfail':>8} {'err':>4} {'mean $':>9} {'mean s':>7}"
    )
    print("-" * 78)

    for model in MODELS:
        gen = DSLGenerator(model=model)
        passes = compile_fails = errors = 0
        cost = 0.0
        secs = 0.0
        first_pass_stl: Path | None = None
        fail_samples: list[str] = []
        for i in range(n):
            t0 = time.perf_counter()
            try:
                prog, usage = gen(blueprint)
            except Exception as exc:
                errors += 1
                secs += time.perf_counter() - t0
                if len(fail_samples) < 2:
                    fail_samples.append(f"ERR:{type(exc).__name__}")
                continue
            secs += time.perf_counter() - t0
            cost += usage.cost_usd
            grade = grade_program(prog, RENDERS / f"_dsl_{model}_{i}", REQUIRED_CHECKS)
            if grade.passed:
                passes += 1
                if first_pass_stl is None:
                    first_pass_stl = grade.stl_path
            else:
                if grade.compiled:
                    compile_fails += 1
                if len(fail_samples) < 2 and grade.failures:
                    fail_samples.append(grade.failures[0][:60])

        rate = passes / n if n else 0.0
        print(
            f"{model:22} {passes:>5}/{n} {rate * 100:>5.0f}% {compile_fails:>8} {errors:>4} "
            f"{cost / max(1, n - errors):>8.4f} {secs / n:>6.1f}"
        )
        if fail_samples:
            print(f"    sample failures: {' | '.join(fail_samples)}")
        if first_pass_stl is not None:
            png = RENDERS / f"dsl_{model.replace('claude-', '')}.png"
            render_part_to_png(first_pass_stl, png, title=f"DSL pass — {model}")
            print(f"    rendered: {png}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
