"""Go-live readiness for the invention lab (ADR-0048/0049) — ONE command, ONE verdict.

Run this the moment Fable is back, BEFORE starting RUNBOOK_WHEN_FABLE_RETURNS.md. It confirms the
OFFLINE scaffolding the live measurement rests on is sound, and reports the live-model status:

  1. preflight  -- forge_mind.invention.toolbelt.run_preflight: every offline-required tool present,
     and whether a model client is configured (live concept generation available vs blocked).
  2. offline bench -- forge_mind.invention.bench.offline_bench: the deterministic harness is green
     (pass-rate 100%, no spend, no model).
  3. baseline freeze guard -- forge_mind.invention.bench.freeze_baseline refuses to overwrite an
     existing baseline (the reference can never be clobbered by a tuning run). Exercised in a temp
     dir; also reports whether the real benchmarks/results/invention_BASELINE.json already exists.

Prints GO (offline scaffolding sound) or NOT-READY (with the exact blocking reason), plus the live
status. Exit 0 on GO, 1 on NOT-READY -- so an operator or CI can gate on it. Pure offline, no spend.

Run: uv run --no-sync python scripts/lab_readiness.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.invention.bench import BenchRun, freeze_baseline, offline_bench, run_to_dict
from forge_mind.invention.toolbelt import run_preflight

_BASELINE = Path("benchmarks/results/invention_BASELINE.json")


def _check_freeze_guard(run: BenchRun) -> tuple[bool, str]:
    """Prove freeze_baseline writes a baseline AND refuses to overwrite it -- in a temp dir, no repo
    writes. Returns (ok, detail)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        freeze_baseline(run, path)
        if not path.exists():
            return False, "freeze_baseline did not write the baseline"
        try:
            freeze_baseline(run, path)
        except FileExistsError:
            return True, "freeze_baseline writes a fresh baseline and refuses to overwrite it"
        return False, "freeze_baseline OVERWROTE an existing baseline (the guard is broken)"


def main() -> int:
    print("=== invention lab -- go-live readiness ===\n")
    blockers: list[str] = []

    # 1. preflight: tools present + model-client status (never pings a model).
    report = run_preflight()
    print("[1/3] preflight (toolbelt.run_preflight)")
    print(f"      offline build available: {report.offline_build_available}")
    print(f"      live concept generation: {report.live_concept_generation}")
    for check in report.checks:
        print(f"        - {check.tool_id}: {check.status.value} ({check.detail})")
    if report.blockers:
        print(f"      blockers: {report.blockers}")
        blockers.extend(report.blockers)
    else:
        print("      blockers: none")

    # 2. offline bench is green (the deterministic harness works with no model/spend).
    print("\n[2/3] offline bench (bench.offline_bench)")
    run = offline_bench()
    summary = run_to_dict(run)
    print(
        f"      pass_rate {run.pass_rate:.0%}  diversity {summary['mean_concept_diversity']}  "
        f"completeness {summary['mean_package_completeness']}  "
        f"honesty {summary['mean_feasibility_honesty']}  cost ${summary['total_cost_usd']}"
    )
    if run.pass_rate < 1.0:
        blockers.append(
            f"offline bench not green (pass_rate {run.pass_rate}); {summary['failures']}"
        )

    # 3. the baseline can be frozen and is overwrite-protected.
    print("\n[3/3] baseline freeze guard (bench.freeze_baseline)")
    guard_ok, guard_detail = _check_freeze_guard(run)
    print(f"      {guard_detail}")
    if _BASELINE.exists():
        print(
            f"      real baseline {_BASELINE}: present -- do NOT overwrite; one is already frozen"
        )
    else:
        print(f"      real baseline {_BASELINE}: not present -- runbook step 3 will create it")
    if not guard_ok:
        blockers.append(f"baseline freeze guard failed: {guard_detail}")

    # verdict
    print("\n" + "=" * 52)
    if blockers:
        print("VERDICT: NOT-READY")
        for blocker in blockers:
            print(f"  blocker: {blocker}")
        return 1
    print("VERDICT: GO -- offline scaffolding is sound.")
    if report.live_concept_generation == "available":
        print("  LIVE: a model client is configured -- you can run the live measurement now.")
    else:
        print(
            "  LIVE: blocked until a model is configured (expected while Fable is down). Set "
            "FORGE_LLM_PROVIDER=claude_code -- see RUNBOOK_WHEN_FABLE_RETURNS.md step 1."
        )
    print("  Next: docs/forge-mind/RUNBOOK_WHEN_FABLE_RETURNS.md (do the steps in order).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
