"""Part A Step-0 de-risk: reproduce the arm SPAN-SIZING gap and pin where it lives.

The FIRST ARM caveat: from a raw prompt the routing is correct (-> motor_arm) but the upstream
BlueprintGenerator UNDER-SIZES the envelope (it sizes to the motor footprint, not the arm's
reach), so the arm's feasibility guard rejects and the pipeline falls back to direct-DSL. The
lattice did NOT have this gap (a bare member's envelope IS its stated size).

This shows, deterministically (offline) + with one cheap real blueprint call, that:
  1. nothing upstream DERIVES the arm span: Blueprint.envelope_mm is LLM-chosen, and interpret_arm
     just consumes envelope.x as the span (no derivation anywhere).
  2. a motor-footprint-sized envelope (what the LLM tends to emit) -> interpret_arm REJECTS
     (span too short for a strut + the tip mount) = the "needs a nudge" failure.
  3. a properly-sized envelope -> the SAME interpreter validates clean. So the geometry is fine;
     the ONLY gap is that the SPAN is not derived from categorical intent.

Run: uv run python scripts/derisk_arm_span.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.compiler.compile import compile_program
from forge_mind.intent.parser import IntentParser
from forge_mind.layout.arm_interpreter import ArmSpec, interpret_arm
from forge_mind.layout.interpreter import LayoutError
from forge_mind.llm.client import api_key_available
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="arm_span_derisk_"))
_ARM_PROMPT = "A drone arm that holds a NEMA 17 motor at its far end: a structural strut that spans out and carries the motor mount at the tip. PLA, 4mm min wall."


def _try_arm(envelope, tag: str) -> bool:  # type: ignore[no-untyped-def]
    """Return True iff interpret_arm produced a part that validated for this envelope."""
    spec = ArmSpec(
        name=f"arm_{tag}", envelope_mm=envelope, min_wall_mm=4.0, cross_section="hexagonal"
    )
    try:
        program = interpret_arm(spec)
        model = compile_program(program, WORK / tag)
        report = validate(model, program, material="PLA")
        mass = f", mass={report.metrics.mass_g:.1f}g" if report.metrics else ""
        print(
            f"  envelope=({envelope.x:.0f},{envelope.y:.0f},{envelope.z:.0f}) -> "
            f"VALIDATES (passed={report.passed}{mass})"
        )
        return bool(report.passed)
    except LayoutError as e:
        print(f"  envelope=({envelope.x:.0f},{envelope.y:.0f},{envelope.z:.0f}) -> REJECTED: {e}")
        return False


def main() -> None:
    print("=== 1. OFFLINE: the geometry is fine WHEN the span is right; the gap is the span ===")
    print("  (interpret_arm consumes envelope.x as the span; nothing upstream DERIVES it)")
    _try_arm(
        Vec3(x=47.0, y=47.0, z=47.0), "footprint"
    )  # motor-footprint envelope (the LLM's habit)
    _try_arm(Vec3(x=60.0, y=60.0, z=30.0), "borderline")
    _try_arm(Vec3(x=180.0, y=60.0, z=30.0), "reach")  # a real arm reach

    if not api_key_available():
        print("\n(no ANTHROPIC_API_KEY -> skipping the live blueprint reproduction)")
        return

    print("\n=== 2. LIVE: a raw arm prompt -> blueprint envelope is under-sized (the real gap) ===")
    intent, u1 = IntentParser()(_ARM_PROMPT)
    blueprint, u2 = BlueprintGenerator()(intent)
    cost = u1.cost_usd + u2.cost_usd
    env = blueprint.envelope_mm
    print(
        f"  blueprint.envelope_mm = ({env.x:.1f}, {env.y:.1f}, {env.z:.1f})  "
        f"(summary: {blueprint.summary[:70]}...)"
    )
    print("  -> feed that LLM-chosen envelope to interpret_arm:")
    live_ok = _try_arm(Vec3(x=env.x, y=env.y, z=env.z), "live")
    print(f"\n  spend: ${cost:.3f}")
    print(
        f"\n  GAP {'CONFIRMED' if not live_ok else 'not reproduced this draw'}: the LLM-sized "
        f"envelope {'does NOT yield a valid arm (rejected/fallback)' if not live_ok else 'happened to work'}. "
        "The LLM sizes the arm's mount-driven dims (span AND lateral) by guess; the fix = DERIVE "
        "the whole envelope from categorical intent (motor + reach class), no LLM numbers."
    )


if __name__ == "__main__":
    main()
