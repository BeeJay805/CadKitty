"""ADR-0039 spike demo: requirement-driven sizing + static deflection estimate, ONE bench part.

Runs the two deliverables on the bench-001 NEMA17 L-bracket, money-free and offline:

  1. FORWARD: "a NEMA17 + 2 kg payload hangs on the face" -> deflection RANGE + shown work.
  2. SIZING:  "...and the face must deflect <= 0.25 mm"   -> code sizes min_wall_mm, the part
     re-derives through the REAL interpreter, and the sized estimate is forward-verified.

REPORTED-ONLY: nothing here gates or touches the product paths. Run:

    uv run python scripts/static_load_spike_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.layout.interpreter import interpret as interpret_l_bracket
from forge_mind.layout.spec import LBracketMotorMountSpec
from forge_mind.physics.static_load import (
    StaticLoadCase,
    estimate_static_load,
    size_wall_for_deflection,
)

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "layout"


def main() -> None:
    spec = LBracketMotorMountSpec.model_validate(
        json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())
    )
    program = interpret_l_bracket(spec)
    load = StaticLoadCase(payload_kg=2.0)

    print("=== FORWARD ESTIMATE (NEMA17 + 2.0 kg, PETG) ===")
    est = estimate_static_load(program, load, "PETG")
    print(est.model_dump_json(indent=2))

    print("\n=== SIZING: face deflection <= 0.10 mm ===")
    # 0.10mm is past what the tightest legal dims reach at the current wall, so the wall
    # visibly GROWS (note: sizing re-derives dependent dims; the payload warning says so).
    sized = size_wall_for_deflection(
        program, load, "PETG", 0.10, interpret=interpret_l_bracket, spec=spec
    )
    print(sized.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
