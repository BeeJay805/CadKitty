"""Run a benchmark with the handwritten runner and snapshot its report into
benchmarks/results/baselines/ for committing (regression reference).

    uv run python scripts/snapshot_baseline.py bench-001-nema17-mount-v1

The committed baseline is what `--compare-to-baseline` checks future runs
against. Re-run deliberately when the reference DSL or validator changes.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from forge_mind.bench.criteria import load_criteria
from forge_mind.bench.runner import run_benchmark
from forge_mind.bench.runners import HandwrittenRunner

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    bench_id = sys.argv[1] if len(sys.argv) > 1 else "bench-001-nema17-mount-v1"
    bench_dir = REPO / "benchmarks" / bench_id
    if not bench_dir.is_dir():
        print(f"No benchmark folder at {bench_dir}", file=sys.stderr)
        return 2

    criteria = load_criteria(bench_dir / "pass_criteria.yaml")
    assert criteria.reference_dsl is not None, "baseline needs a reference_dsl"

    output_root = REPO / "benchmarks" / "results"
    runner = HandwrittenRunner(REPO / criteria.reference_dsl)
    summary = run_benchmark(
        bench_dir=bench_dir,
        runner=runner,
        criteria=criteria,
        run_id="baseline-handwritten",
        output_root=output_root,
        reference_dsl=REPO / criteria.reference_dsl,
    )

    # Copy the report into the committed baselines/ dir.
    baselines = output_root / "baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    src_dir = output_root / bench_id / "baseline-handwritten"
    for ext in ("md", "json"):
        shutil.copyfile(src_dir / f"report.{ext}", baselines / f"{bench_id}-handwritten.{ext}")

    print(
        f"{'PASS' if summary.benchmark_passed else 'FAIL'}: "
        f"{summary.n_passed}/{summary.n_trials} -> snapshot in {baselines}"
    )
    return 0 if summary.benchmark_passed else 1


if __name__ == "__main__":
    sys.exit(main())
