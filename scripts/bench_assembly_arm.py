"""One R3+ assembly arm: frozen-plan multi-part briefs with registry-MAPPED hardware
through the full functional gauntlet (per-part loops -> seating -> mount patterns ->
motion sweep -> judge). The `bench_local_arm.py` sibling for the humanoid ladder's
assembly rungs (ADR-0124; design: docs/research/2026-07-22-r3-harness-design.md).

Usage:
  python scripts/bench_assembly_arm.py --briefs benchmarks/local_hard_v1/r3_pan_tilt_neck_candidate.json \
      --arm R3_pan_tilt_sonnet --out E:/bench/out [--max-iterations 4] [--n-drafts 1] \
      [--env K=V]... [--resume]

Brief rows carry `assembly.servos` (kind + machine pose {origin, z_axis, x_axis}) and
`assembly.plan` (a verbatim FunctionalPlan). The plan is FROZEN: it is written to the
run dir and loaded via FORGE_REUSE_PLAN so decomposition is never measured -- the R3
question is interfacing KNOWN hardware, and the qwen-vs-teacher A/B stays unconfounded.

Row contract (bench_local_arm-compatible + assembly columns): one row per (label,
sample), config-stamped (sha + FORGE_* env), --resume crash-safe, `approved` =
disposition=='shipped', `passed_final` = every part validator-green AND the motion
verdict articulates; `fit`/`motion`/per-part sub-records ride verbatim so a failure's
stage is a field read (the ADR-0124 identifiability condition)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent.parent,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _pose_transform(pose: dict) -> list[list[float]]:
    """origin + z_axis + x_axis -> the 4x4 world placement (y = z cross x)."""
    ox, oy, oz = (float(v) for v in pose["origin"])
    z = [float(v) for v in pose["z_axis"]]
    x = [float(v) for v in pose["x_axis"]]
    y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ]
    return [
        [x[0], y[0], z[0], ox],
        [x[1], y[1], z[1], oy],
        [x[2], y[2], z[2], oz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--n-drafts", type=int, default=1)
    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--env", action="append", default=[], help="extra K=V env overrides")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip (label, sample) pairs already in results.json",
    )
    args = ap.parse_args()

    for kv in args.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.json"
    rows: list[dict] = []
    if results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    done = {(r.get("label"), r.get("sample")) for r in rows}
    if args.resume and done:
        print(f"resume: {len(done)} runs already recorded")

    from forge_mind.agentic.brief import DesignBrief
    from forge_mind.agentic.functional import produce_functional_part
    from forge_mind.structures.servo_bridge import HardwareDecl

    corpus = [
        r for r in json.loads(Path(args.briefs).read_text(encoding="utf-8")) if "assembly" in r
    ]
    sha = _sha()
    fenv = {k: v for k, v in os.environ.items() if k.startswith("FORGE_")}

    for rec in corpus:
        label = rec["label"]
        asm = rec["assembly"]
        for sample in range(args.samples):
            if args.resume and (label, sample) in done:
                continue
            run_dir = out_root / f"{label}_s{sample}"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Freeze the plan: write it, point FORGE_REUSE_PLAN at it.
            plan_path = run_dir / "frozen_plan.json"
            plan_path.write_text(json.dumps(asm["plan"], indent=1), encoding="utf-8")
            os.environ["FORGE_REUSE_PLAN"] = str(plan_path)
            hardware = [
                HardwareDecl(
                    kind=s["kind"],
                    hw_id=s["id"],
                    transform=_pose_transform(s["pose"]),
                )
                for s in asm.get("servos", [])
                if "pose" in s
            ]
            t0 = time.monotonic()
            try:
                # Frozen-reference law (L84 judge incident): when the row carries a
                # whole-object brief, pass it -- the judge's reference must be the
                # FROZEN design, not a freshly-researched imagination of the prompt.
                frozen_brief = (
                    DesignBrief.model_validate(rec["brief"]) if rec.get("brief") else None
                )
                res = produce_functional_part(
                    rec["prompt"],
                    run_dir,
                    max_iterations=args.max_iterations,
                    n_drafts=args.n_drafts,
                    hardware=hardware,
                    brief=frozen_brief,
                )
                parts_sub = [
                    {
                        "name": name,
                        # geometry_passed is the RESUMPTION-SAFE verdict (the engine's own
                        # final predicate reads it): checkpoint-resumed parts carry
                        # validation=None, and reading validation.passed there minted the
                        # L83 false anomaly (accepted-but-"red" with zero issues).
                        "passed": bool(getattr(lr, "geometry_passed", False)),
                        "stopped": getattr(lr, "stopped_reason", None),
                        "critic_approved": bool(getattr(lr, "critic_approved", False)),
                        "resumed": getattr(lr, "validation", None) is None,
                        # L82 autopsy gap: the WHY was unpersisted -- carry the final
                        # validation issues verbatim (capped) so a red part names itself.
                        "issues": [
                            str(i) for i in (getattr(lr.validation, "issues", None) or [])[:6]
                        ]
                        if getattr(lr, "validation", None)
                        else [],
                    }
                    for name, lr in (res.parts or [])
                ]
                motion = res.motion or {}
                row = {
                    "label": label,
                    "sample": sample,
                    "arm": args.arm,
                    "approved": res.disposition in ("shipped", "shipped_with_caveats"),
                    "passed_final": bool(parts_sub)
                    and all(p["passed"] for p in parts_sub)
                    and motion.get("articulates") is True,
                    "stopped": res.disposition,
                    "failure": res.failure,
                    "seated_fit": res.fit,
                    "motion": motion,
                    "parts": parts_sub,
                    "hardware_declared": [s["id"] for s in asm.get("servos", [])],
                    "t_s": time.monotonic() - t0,
                    "config": {
                        "sha": sha,
                        "env": fenv,
                        "max_iterations": args.max_iterations,
                        "n_drafts": args.n_drafts,
                    },
                }
            except Exception as exc:  # kill-safe: a crashed run records honestly
                row = {
                    "label": label,
                    "sample": sample,
                    "arm": args.arm,
                    "approved": False,
                    "passed_final": False,
                    "stopped": "runner_error",
                    "failure": f"{type(exc).__name__}: {exc}",
                    "t_s": time.monotonic() - t0,
                    "config": {"sha": sha, "env": fenv},
                }
            rows.append(row)
            results_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            print(
                f"{label}_s{sample}: {row['stopped']} approved={row['approved']} "
                f"passed_final={row['passed_final']} ({row['t_s']:.0f}s)"
            )


if __name__ == "__main__":
    main()
