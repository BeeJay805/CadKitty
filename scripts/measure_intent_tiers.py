"""Measure the Intent stage across model tiers over all 10 prompt variants.

Each variant is a different phrasing of the same NEMA 17 request; all must
extract to the same facts. Reports how many of the 10 each tier gets right.

    uv run python scripts/measure_intent_tiers.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from forge_mind.config import HAIKU, SONNET
from forge_mind.intent.grader import grade_intent
from forge_mind.intent.parser import IntentParser

REPO = Path(__file__).resolve().parent.parent
VARIANTS = REPO / "tests" / "fixtures" / "intent" / "prompt_variants.json"

OPUS_48 = "claude-opus-4-8"
MODELS = [HAIKU, SONNET, OPUS_48]


def main() -> int:
    data = json.loads(VARIANTS.read_text())
    variants = data["variants"]
    exp = data["expected"]
    required = set(exp["required_interface_types"])
    envelope = tuple(exp["envelope_mm"])

    print(f"Intent stage across tiers, {len(variants)} prompt variants each\n")
    print(f"{'model':22} {'pass':>7} {'rate':>6} {'mean $':>9} {'mean s':>7}")
    print("-" * 60)

    for model in MODELS:
        parser = IntentParser(model=model)
        passes = 0
        cost = 0.0
        secs = 0.0
        fails: list[str] = []
        for i, text in enumerate(variants):
            t0 = time.perf_counter()
            try:
                spec, usage = parser(text)
            except Exception as exc:
                secs += time.perf_counter() - t0
                fails.append(f"v{i}:ERR:{type(exc).__name__}")
                continue
            secs += time.perf_counter() - t0
            cost += usage.cost_usd
            grade = grade_intent(
                spec,
                required_interface_types=required,
                expected_min_wall_mm=exp["min_wall_mm"],
                expected_envelope_mm=envelope,
                expected_material=exp["material"],
            )
            if grade.passed:
                passes += 1
            elif len(fails) < 3:
                fails.append(f"v{i}:{grade.failures[0][:40]}")

        n = len(variants)
        print(
            f"{model:22} {passes:>5}/{n} {passes / n * 100:>5.0f}% {cost / n:>8.5f} {secs / n:>6.1f}"
        )
        if fails:
            print(f"    misses: {' | '.join(fails)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
