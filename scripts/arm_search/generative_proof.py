"""The end-to-end generative proof, now THROUGH the dispatch loop (T2): spec -> author_part.

One reproducible command that exercises the WHOLE stack on a drone-arm:
  1. a typed RequirementSpec: envelope + a declared 0.5 kg load case + a deflection bound + a
     REAL MG996R servo placed near the tip (the founder's rule: pocket + mounts, the engine
     never invents the motor);
  2. ``author_part`` routes it (components present -> the TOPOPT author), runs the author
     (hardware-as-data carves the pocket + mount tabs; SIMP grows the lightest-at-bound
     structure via the volfrac auto-tune; each candidate: SIMP -> contour-vectorized B-rep ->
     REAL tet FEA), gates the artifact with the uniform ``verify_full``, and would fall through
     the monotone ladder on refusal -- except the agentic floor is INJECTED AS A REFUSAL here:
     a $0 physics proof never silently burns LLM calls (the refusal is honest and recorded);
  3. the report carries every attempt's per-check gate verdicts + the author's self-report
     (tune history, carved regions, grid).

The claim this earns (when claimable): "the engine generatively designed, FEA-verified and
slice-verified a drone-arm around a real MG996R at a declared load case and bound." No
"superhuman", no reference comparison -- the bound is a stated benchmark input, printed verbatim.

Usage:
  FORGE_FEA_TIMEOUT_S=500 uv run --no-sync python scripts/arm_search/generative_proof.py \
    --out C:/Users/brenn/work/arm_m1_runs/generative_proof
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

from forge_mind.structures.dispatch import author_part, refusing_author, topopt_author
from forge_mind.structures.loadcase import G_M_S2, LoadCase
from forge_mind.structures.spec import ComponentPlacement, RequirementSpec

# The drone-arm benchmark inputs (DECLARED, printed on the report verbatim). The grid is pinned
# to the row-76 claimable recipe (120 x 57 over 142 x 67 mm) so the proof reproduces the claim.
LENGTH, HEIGHT, DEPTH = 142.0, 67.0, 24.5  # mm; SIMP plane = LENGTH x HEIGHT, extruded DEPTH
PAYLOAD_KG = 0.5
BOUND_MM = 0.15
SERVO_AT = (112.0, 33.5)  # body center near the tip, mid-height
NELX, NELY = 120, 57


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end generative drone-arm proof.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fea-evals", type=int, default=6)
    ap.add_argument(
        "--rmin",
        type=float,
        default=2.4,
        help="SIMP filter radius (the min-member knob; the first run's 1.8 pinched a 0.36 mm "
        "sliver the min-wall gate refused)",
    )
    ap.add_argument("--extrude-thresh", type=float, default=0.45)
    ap.add_argument(
        "--open-radius-cells",
        type=int,
        default=1,
        help="morphological-opening radius (cells): the min-member GUARANTEE (deletes features "
        "thinner than ~2 cells; the FEA re-verifies the final geometry)",
    )
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = RequirementSpec(
        envelope_mm=(LENGTH, DEPTH, HEIGHT),
        material_id="pla",
        load=LoadCase(name="drone_arm_inplane_0p5kg", force_n=PAYLOAD_KG * G_M_S2),
        deflection_bound_mm=BOUND_MM,
        components=[ComponentPlacement(kind="mg996r_servo", at_mm=SERVO_AT, clearance_mm=0.5)],
    )
    report: dict[str, Any] = {
        "benchmark": {
            "envelope_mm": [LENGTH, DEPTH, HEIGHT],
            "payload_kg": PAYLOAD_KG,
            "deflection_bound_mm": BOUND_MM,
            "servo": {"kind": "mg996r_servo", "at_mm": list(SERVO_AT)},
        },
        "knobs": {
            "grid": [NELX, NELY],
            "rmin": args.rmin,
            "extrude_thresh": args.extrude_thresh,
            "open_radius_cells": args.open_radius_cells,
            "max_fea_evals": args.max_fea_evals,
        },
    }

    outcome = author_part(
        spec,
        out_dir,
        authors={
            "topopt": partial(
                topopt_author,
                nelx=NELX,
                nely=NELY,
                rmin=args.rmin,
                extrude_thresh=args.extrude_thresh,
                open_radius_cells=args.open_radius_cells,
                max_fea_evals=args.max_fea_evals,
            ),
            "agentic": refusing_author(
                "agentic", "the $0 physics proof never burns LLM calls (fallback disabled)"
            ),
        },
        part_name="drone arm",
    )
    final = outcome["final"]
    gate = outcome["attempts"][-1]["gate"]
    report["routed_to"] = outcome["chosen"]
    report["attempts"] = outcome["attempts"]
    report["self_reported"] = final.self_reported
    report["artifacts"] = {
        "stl": None if final.stl is None else str(final.stl),
        "step": None if final.step is None else str(final.step),
    }
    report["gate"] = gate
    report["verdict"] = (
        "claimable"
        if gate["claimable"]
        else ("no_candidate_met_the_bound" if final.stl is None else "not_claimable")
    )
    (out_dir / "proof_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if gate["claimable"] else 1


if __name__ == "__main__":
    sys.exit(main())
