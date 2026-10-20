"""Measure the Blueprint stage across model tiers.

Runs the BlueprintGenerator N times per model on the bench-001 IntentSpec,
grades each output, and prints pass-rate / cost / latency per tier. This turns
"which model?" into data and directly tests the structure-beats-scale thesis:
can a cheaper model produce valid blueprints given the schema + grader?

    uv run python scripts/measure_blueprint_tiers.py [N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.grader import grade_blueprint
from forge_mind.config import HAIKU, SONNET
from forge_mind.intent.schema import IntentSpec

REPO = Path(__file__).resolve().parent.parent
INTENT = REPO / "tests" / "fixtures" / "intent" / "nema17_mount.intent.json"

OPUS_48 = "claude-opus-4-8"
MODELS = [HAIKU, SONNET, OPUS_48]

REQUIRED_TYPES = {"nema17_face", "extrusion_2020_slot"}
ENVELOPE_CAP = (60.0, 60.0, 40.0)
EXPECTED_WALL = 4.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    intent = IntentSpec.model_validate(json.loads(INTENT.read_text()))

    print(f"Blueprint stage across tiers, N={n} each\n")
    print(f"{'model':22} {'pass':>6} {'rate':>6} {'mean $':>9} {'mean s':>7}  failures")
    print("-" * 80)

    for model in MODELS:
        gen = BlueprintGenerator(model=model)
        passes = 0
        cost = 0.0
        secs = 0.0
        fail_samples: list[str] = []
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                bp, usage = gen(intent)
            except Exception as exc:
                fail_samples.append(f"ERROR:{type(exc).__name__}")
                secs += time.perf_counter() - t0
                continue
            secs += time.perf_counter() - t0
            cost += usage.cost_usd
            grade = grade_blueprint(
                bp,
                required_interface_types=REQUIRED_TYPES,
                envelope_cap_mm=ENVELOPE_CAP,
                expected_min_wall_mm=EXPECTED_WALL,
            )
            if grade.passed:
                passes += 1
            elif len(fail_samples) < 3:
                fail_samples.append("; ".join(grade.failures))

        rate = passes / n if n else 0.0
        sample = (" | ".join(fail_samples))[:40] if fail_samples else ""
        print(
            f"{model:22} {passes:>4}/{n} {rate * 100:>5.0f}% "
            f"{cost / n:>8.4f} {secs / n:>6.1f}  {sample}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
