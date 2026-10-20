"""Analyze + compare invention-lab benchmark runs (ADR-0048).

Loads a ``BenchRun`` JSON (``forge_mind.invention.bench.load_run``) and reports the headline the
live runbook asks for: the pass-rate, the three score axes (concept_diversity,
package_completeness, feasibility_honesty), a per-tier breakdown, and FAILURES GROUPED BY STAGE
(intake / feasibility / concept / package / validation). Given a frozen baseline it also diffs the
run against it -- the pass-rate + per-axis deltas, plus which problems regressed or were fixed --
so a tuning loop can see exactly what a change moved (RUNBOOK_WHEN_FABLE_RETURNS.md, steps 5-7).

Pure analysis: it never calls a model and never touches the engine. Two input modes:

  - a run JSON produced by ``forge-mind invention-bench --out <path>`` (the positional arg), or
  - ``--fixtures <problems.json>`` to PRODUCE a deterministic OFFLINE run (mocked model, no spend)
    from any frozen problem set. The ``invention-bench`` CLI only runs the frozen invention_v1 set,
    so this is how the normal/hard tiers get exercised before Fable returns. (The offline stub
    always returns a fixed 2-concept plan, so a problem set with min_concepts > 2 -- e.g.
    invention_hard_v1 -- reports a 'concept' stage gap offline by design; the real measurement is
    the live run.)

Run:
  uv run --no-sync python scripts/analyze_invention_bench.py benchmarks/results/invention_live_run.json \\
      --baseline benchmarks/results/invention_BASELINE.json
  uv run --no-sync python scripts/analyze_invention_bench.py \\
      --fixtures benchmarks/invention_normal_v1/problems.json benchmarks/invention_hard_v1/problems.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from forge_mind.invention.bench import BenchRun, load_run, offline_bench

# The runbook freezes the protected reference here (benchmarks/results/ is gitignored, so it only
# exists once a live baseline has been captured -- used as the default baseline when present).
_DEFAULT_BASELINE = Path("benchmarks/results/invention_BASELINE.json")
_AXES = ("concept_diversity", "package_completeness", "feasibility_honesty")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _axis_means(run: BenchRun) -> dict[str, float]:
    return {axis: _mean([getattr(e.scores, axis) for e in run.entries]) for axis in _AXES}


def _failures_by_stage(run: BenchRun) -> dict[str, list[str]]:
    by_stage: dict[str, list[str]] = defaultdict(list)
    for e in run.entries:
        if e.failure_stage:
            by_stage[e.failure_stage].append(e.problem_id)
    return dict(by_stage)


def _report(run: BenchRun) -> None:
    n = len(run.entries)
    passed = sum(1 for e in run.entries if e.scores.rubric_passed)
    print(
        f"mode={run.mode}  model={run.model or '-'}  problems={n}  cost=${run.total_cost_usd:.4f}"
    )
    print(f"pass_rate: {passed}/{n} = {run.pass_rate * 100:.0f}%")
    means = _axis_means(run)
    for axis in _AXES:
        print(f"  mean {axis}: {means[axis]:.3f}")

    tiers = sorted({e.tier for e in run.entries})
    if len(tiers) > 1:
        print("\nby tier:")
        for tier in tiers:
            sub = [e for e in run.entries if e.tier == tier]
            tp = sum(1 for e in sub if e.scores.rubric_passed)
            cd = _mean([e.scores.concept_diversity for e in sub])
            pc = _mean([e.scores.package_completeness for e in sub])
            fh = _mean([e.scores.feasibility_honesty for e in sub])
            print(
                f"  {tier:7s} pass {tp}/{len(sub)}  diversity {cd:.2f}  "
                f"completeness {pc:.2f}  honesty {fh:.2f}"
            )

    fails = _failures_by_stage(run)
    total_failed = sum(len(v) for v in fails.values())
    print(f"\nfailures by stage ({total_failed}):")
    if not fails:
        print("  none -- every problem passed the rubric")
    for stage in sorted(fails):
        print(f"  {stage}: {', '.join(sorted(fails[stage]))}")
    if run.mode == "offline" and "concept" in fails:
        print(
            "  note: offline 'concept' failures are EXPECTED for tiers with min_concepts > 2 -- the "
            "offline stub returns a fixed 2-concept plan; measure concept divergence on the LIVE run."
        )


def _compare(run: BenchRun, baseline: BenchRun) -> None:
    print("\n=== vs baseline ===")
    print(
        f"baseline: mode={baseline.mode}  model={baseline.model or '-'}  "
        f"problems={len(baseline.entries)}"
    )
    d_pass = run.pass_rate - baseline.pass_rate
    print(
        f"pass_rate delta: {d_pass * 100:+.0f}%  "
        f"({baseline.pass_rate * 100:.0f}% -> {run.pass_rate * 100:.0f}%)"
    )
    run_means, base_means = _axis_means(run), _axis_means(baseline)
    for axis in _AXES:
        delta = run_means[axis] - base_means[axis]
        print(f"  {axis} delta: {delta:+.3f}  ({base_means[axis]:.3f} -> {run_means[axis]:.3f})")

    base_pass = {e.problem_id: e.scores.rubric_passed for e in baseline.entries}
    regressed, fixed = [], []
    for e in run.entries:
        was = base_pass.get(e.problem_id)
        if was is None:
            continue
        if was and not e.scores.rubric_passed:
            regressed.append(f"{e.problem_id} (now fails at {e.failure_stage})")
        elif not was and e.scores.rubric_passed:
            fixed.append(e.problem_id)
    new_ids = {e.problem_id for e in run.entries} - set(base_pass)

    if regressed:
        print(f"\nREGRESSED ({len(regressed)}):")
        for item in sorted(regressed):
            print(f"  {item}")
    if fixed:
        print(f"\nfixed ({len(fixed)}): {', '.join(sorted(fixed))}")
    if new_ids:
        print(f"\nnot in baseline ({len(new_ids)}): {', '.join(sorted(new_ids))}")
    if not (regressed or fixed or new_ids):
        print("\nno per-problem changes vs baseline")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze + compare invention-lab benchmark runs (ADR-0048)."
    )
    ap.add_argument(
        "run",
        nargs="?",
        type=Path,
        help="A BenchRun JSON (from `forge-mind invention-bench --out <path>`).",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"A frozen baseline BenchRun JSON to diff against (default {_DEFAULT_BASELINE} "
        "if it exists).",
    )
    ap.add_argument(
        "--fixtures",
        type=Path,
        nargs="+",
        default=None,
        help="Produce a deterministic OFFLINE run from these problems.json file(s) instead of loading "
        "a run (no model, no spend) -- the way to exercise the normal/hard tiers before Fable returns. "
        "Pass several to summarize them together (per-tier).",
    )
    args = ap.parse_args()

    if args.fixtures is not None and args.run is not None:
        ap.error("give a run JSON path OR --fixtures <problems.json ...>, not both")
    if args.fixtures is not None:
        entries = [e for fp in args.fixtures for e in offline_bench(fixtures_path=fp).entries]
        run = BenchRun(mode="offline", entries=entries)
    elif args.run is not None:
        run = load_run(args.run)
    else:
        ap.error("give a run JSON path or --fixtures <problems.json ...>")

    _report(run)

    baseline_path = args.baseline
    if baseline_path is None and _DEFAULT_BASELINE.exists():
        baseline_path = _DEFAULT_BASELINE
    if baseline_path is not None and baseline_path.exists():
        _compare(run, load_run(baseline_path))
    elif args.baseline is not None:
        print(f"\n(baseline {args.baseline} not found -- showing run summary only)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
