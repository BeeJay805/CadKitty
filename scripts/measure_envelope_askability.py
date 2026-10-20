"""Envelope-sizing askability probe (PAID). Runs raw natural-language prompts end-to-end through
the PRODUCT path (run_pipeline use_layoutspec=True) and reports, per prompt: the routed gen_path,
whether it validated, and the fallback_reason (the observability added in the prior task).

Success = routed to the deterministic layout path AND validated, with NO silent fallback to the
direct-DSL path. That is the live-demo bar: a typed sentence -> a validated part, no fallback.

  uv run python scripts/measure_envelope_askability.py step0     # the cheap reproduction set
  uv run python scripts/measure_envelope_askability.py measure   # the larger varied batch
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.pipeline import run_pipeline

# (family label, prompt). Step 0: one per family, the task's list.
STEP0: list[tuple[str, str]] = [
    ("pi_case", "A Raspberry Pi 4 case with ventilation slots. PLA, 2mm walls."),
    ("lattice_brace", "A lightweight lattice brace about 130mm long. PLA, 4mm min wall."),
    ("fan_mount", "A 60mm axial fan mount that bolts onto a 2020 extrusion. PLA, 4mm walls."),
    (
        "nema23_adapter",
        "An adapter plate from a NEMA 23 stepper to 3030 extrusion. PLA, 4mm walls.",
    ),
    (
        "bare_strut",
        "A straight structural strut about 120mm long with a hexagonal cross-section. PLA, 4mm min wall.",
    ),
    (
        "plate_hardware",
        "A flat plate with a 608 bearing seat and four M3 clearance holes. PLA, 4mm walls.",
    ),
]

# Measure: a larger varied batch across families + sizes (the high-bar live-demo set).
MEASURE: list[tuple[str, str]] = [
    *STEP0,
    (
        "plate_hardware",
        "A mounting plate with a press-fit 625 bearing seat and two M5 counterbored screw holes. PLA, 4mm walls.",
    ),
    (
        "plate_hardware",
        "A flat PLA plate with an M4 heat-set insert boss and a 6x3 magnet pocket. 4mm walls.",
    ),
    (
        "plate_hardware",
        "A plate with a captive M5 nut pocket and two 4mm dowel locating holes. PLA, 4mm min wall.",
    ),
    ("fan_mount", "A 40mm fan mount plate on a 2020 extrusion. PLA, 4mm walls."),
    (
        "nema17_adapter",
        "A flat adapter plate from a NEMA 17 motor to a 2020 extrusion. PLA, 4mm walls.",
    ),
    ("lattice_brace", "A dense lattice brace roughly 90mm long. PLA, 4mm min wall."),
    ("bare_strut", "A rectangular structural strut about 200mm long. PLA, 4mm min wall."),
    ("lbracket", "A NEMA 17 L-bracket motor mount onto a 2020 extrusion. PLA, 4mm walls."),
    ("pi_case", "A generic PCB tray for an 80x60mm board with 3mm walls, no ventilation. PETG."),
    (
        "plate_hardware",
        "A bearing block hosting a 6800 thin-section bearing and three M3 clearance holes. PLA, 4mm walls.",
    ),
]


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "step0"
    prompts = {"step0": STEP0, "measure": MEASURE}.get(which, STEP0)
    out = Path(tempfile.mkdtemp(prefix=f"env_ask_{which}_"))
    total = 0.0
    ok = 0
    print(f"=== {which}: {len(prompts)} raw NL prompts -> product path ===")
    for i, (fam, prompt) in enumerate(prompts):
        try:
            r = run_pipeline(prompt, out / f"p{i}", use_layoutspec=True)
            total += r.total_cost_usd
            routed = r.gen_path == "layoutspec"
            validated = routed and r.passed
            ok += int(validated)
            flag = "OK " if validated else "MISS"
            print(f"  {flag} [{fam:<15}] route={r.gen_path:<11} passed={r.passed}")
            if not validated and r.refusal:
                print(f"        refusal: {r.refusal[:140]}")
            elif not validated and r.fallback_reason:
                print(f"        fallback: {r.fallback_reason[:140]}")
            elif not validated and r.error:
                print(f"        error@{r.failed_stage}: {r.error[:120]}")
        except Exception as exc:  # never let one prompt abort the batch
            print(f"  ERR  [{fam:<15}] {type(exc).__name__}: {str(exc)[:100]}")
    n = len(prompts)
    lb = wilson_lower_bound(ok, n) if n else 0.0
    print(f"\n  routed+validated WITHOUT fallback: {ok}/{n}  (Wilson 95% LB {lb:.3f})")
    print(f"  TOTAL SPEND: ${total:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
