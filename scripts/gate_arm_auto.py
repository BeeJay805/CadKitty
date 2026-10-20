"""Gate A (PAID, <=$2): is the drone arm fully prompt-to-part automatic?

Runs a batch of DECONTAMINATED raw arm prompts (no worked numbers) through the real product
path -- run_pipeline(use_layoutspec=True) -- and measures how often a plain sentence becomes a
validated arm with ZERO hand-editing. A trial SUCCEEDS iff:

  - it routed through the deterministic path (gen_path == 'layoutspec', i.e. NOT the direct-DSL
    fallback -- a fallback is exactly the old "needs a nudge" failure), AND
  - the part is the ARM interpreter's output (a computed strut + a motor_face interface, which
    distinguishes motor_arm from a flat composable_plate mount), AND
  - it validated (report.passed).

Reports the Wilson 95% lower bound (bar: >= 80%) + actual spend. A hard budget stop protects
the <=$2 cap.

Run: uv run python scripts/gate_arm_auto.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from forge_mind.dsl.primitives import ComputedSolid
from forge_mind.pipeline import run_pipeline

BUDGET_STOP_USD = 1.8

PROMPTS = [
    "A drone arm that holds a NEMA 17 motor at its far end.",
    "A long-reach robot arm with a NEMA 17 stepper mounted at the tip.",
    "A compact arm that carries a NEMA 17 motor at the end.",
    "A structural arm spanning out to a NEMA 17 motor mount at its tip, printed in PLA.",
    "A quadcopter arm holding a NEMA 17 motor at the outboard end.",
    "An arm that reaches out and bolts a NEMA 17 stepper to its far tip.",
    "A standard drone arm carrying a NEMA 17 at the end.",
    "A cantilever arm with a NEMA 17 motor mount at the far end.",
    "A long arm that positions a NEMA 17 motor away from the body.",
    "A printed arm that holds a NEMA 17 stepper motor at its outer tip.",
    "A motor arm: a beam that carries a NEMA 17 at the end.",
    "A compact bracket-arm holding a NEMA 17 motor at the tip.",
    "An extended arm reaching to a NEMA 17 motor mount at its end.",
    "A drone frame arm with a NEMA 17 motor at the far end, PLA.",
    "A spanning arm that supports a NEMA 17 stepper at its tip.",
    "A long structural arm ending in a NEMA 17 motor mount.",
    "An arm that holds a stepper (NEMA 17) out at its far end.",
    "A robot link arm carrying a NEMA 17 motor at the tip.",
    "A standard-reach arm with a NEMA 17 motor at the far tip.",
    "A drone arm that bolts a NEMA 17 stepper at the spanning end.",
]


def _wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def _is_arm(result) -> bool:  # type: ignore[no-untyped-def]
    prog = result.program
    if prog is None:
        return False
    has_strut = any(isinstance(op, ComputedSolid) for op in prog.operations)
    has_motor_face = any(getattr(d, "name", "") == "motor_face" for d in prog.interfaces)
    return has_strut and has_motor_face


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="gate_arm_"))
    total = 0.0
    successes = 0
    completed = 0
    infra_errors = 0
    for i, prompt in enumerate(PROMPTS):
        if total > BUDGET_STOP_USD:
            print(f"\n  BUDGET STOP at ${total:.2f} after {completed} trials")
            break
        result = run_pipeline(prompt, work / f"t{i}", use_layoutspec=True)
        total += result.total_cost_usd
        # An intent/blueprint stage crash is infra (network), not the feature; record separately.
        if result.failed_stage in {"intent", "blueprint", "layoutspec"} and result.report is None:
            infra_errors += 1
            print(f"  t{i:<2} INFRA-ERR stage={result.failed_stage} {result.error}")
            continue
        completed += 1
        ok = result.passed and result.gen_path == "layoutspec" and _is_arm(result)
        successes += int(ok)
        mass = result.report.metrics.mass_g if (result.report and result.report.metrics) else None
        tag = "OK " if ok else "MISS"
        print(
            f"  t{i:<2} {tag} path={result.gen_path:<11} arm={_is_arm(result)} "
            f"passed={result.passed} mass={mass and round(mass, 1)}"
        )

    lb = _wilson_lb(successes, completed)
    print("\n=== GATE A ===")
    print(f"  successes: {successes}/{completed} completed ({infra_errors} infra-errors excluded)")
    print(
        f"  Wilson 95% lower bound: {lb * 100:.1f}%   (bar: >= 80%)   "
        f"{'PASS' if lb >= 0.80 else 'FAIL'}"
    )
    print(f"  spend: ${total:.3f}  (cap ${BUDGET_STOP_USD + 0.2:.1f})")


if __name__ == "__main__":
    main()
