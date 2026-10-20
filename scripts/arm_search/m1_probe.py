"""M1 driver: one superhuman-arm single-link LLM attempt (plan Ticket 4c; now on structures).

Authors a link through the agentic pipeline WITH the bounded mass_target (ADR-0096), then measures
the result against the reference bar with ``forge_mind.structures`` (mass + validated-FEA
deflection under the canonical case). Subscription only (FORGE_LLM_PROVIDER=claude_code,
FORGE_FORBID_API_SPEND=1); ``cost_usd`` surfaces the REAL LoopResult.total_cost_usd (the
spend-leak tell). FORGE_PARAM_OPT=1 enables the mass polish; FORGE_FEA_TIMEOUT_S must cover
arm-scale FEA (2-3 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from forge_mind.agentic.loop import run_agentic_pipeline
from forge_mind.agentic.param_opt import MassTarget
from forge_mind.agentic.params import param_opt_enabled
from forge_mind.materials import resolve_material
from forge_mind.structures.loadcase import canonical_link_case
from forge_mind.structures.objectives import score_stl


def run_m1(
    prompt: str,
    *,
    target_mass_g: float,
    ref_mass_g: float,
    ref_deflection_mm: float,
    material_id: str,
    out_dir: Path,
    max_iterations: int | None,
) -> dict[str, Any]:
    mat = resolve_material(material_id)
    mass_target = MassTarget(target_mass_g=target_mass_g, density_g_cm3=mat.density_g_cm3)
    t0 = time.monotonic()
    result = run_agentic_pipeline(
        prompt,
        out_dir,
        material=mat,
        mass_target=mass_target,
        max_iterations=max_iterations,
    )
    elapsed = round(time.monotonic() - t0, 1)

    stl = result.final_stl
    # Direct attribute access (LoopResult fields) -- NOT getattr-with-default, so a renamed field
    # fails loud instead of silently reading 0.0 (cost_usd is the subscription spend-leak tell).
    report: dict[str, Any] = {
        "prompt": prompt,
        "elapsed_s": elapsed,
        "cost_usd": round(float(result.total_cost_usd), 4),
        "disposition": result.stopped_reason,
        "geometry_passed": bool(result.geometry_passed),
        "built": stl is not None,
        "target_mass_g": target_mass_g,
        "reference_mass_g": ref_mass_g,
        "reference_deflection_mm": ref_deflection_mm,
    }
    if stl is None or not Path(stl).exists():
        report["verdict"] = "no_build"
        return report

    part = score_stl(stl, material=material_id, load_case=canonical_link_case)
    report["mass_g"] = part.mass_g
    report["watertight"] = part.watertight
    report["deflection_mm"] = part.deflection_mm
    report["von_mises_mpa"] = part.von_mises_mpa
    if part.mass_g is None:
        report["verdict"] = "unmeasurable_mass"
        return report
    report["lighter_than_reference"] = part.mass_g < ref_mass_g
    report["mass_vs_reference_pct"] = round(100.0 * (part.mass_g / ref_mass_g - 1.0), 1)
    if part.deflection_mm is None:
        report["verdict"] = "fea_unavailable"
        return report
    within = part.deflection_mm <= ref_deflection_mm
    report["within_deflection_bound"] = within
    report["verdict"] = (
        "superhuman_attempt_PASS"
        if (part.mass_g < ref_mass_g and within and bool(result.geometry_passed))
        else "did_not_beat_reference"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 single-link LLM attempt.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--target-mass-g", type=float, required=True)
    ap.add_argument("--ref-mass-g", type=float, required=True)
    ap.add_argument("--ref-deflection-mm", type=float, required=True)
    ap.add_argument("--material", default="pla")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iterations", type=int, default=None)
    args = ap.parse_args(argv)

    if not param_opt_enabled():
        print(
            json.dumps({"error": "FORGE_PARAM_OPT is not enabled -> the mass polish will not run"})
        )
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_m1(
        args.prompt,
        target_mass_g=args.target_mass_g,
        ref_mass_g=args.ref_mass_g,
        ref_deflection_mm=args.ref_deflection_mm,
        material_id=args.material,
        out_dir=out_dir,
        max_iterations=args.max_iterations,
    )
    (out_dir / "m1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
