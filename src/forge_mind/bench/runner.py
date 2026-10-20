"""run_benchmark — drive a Runner over N trials and assemble a BenchmarkSummary."""

from __future__ import annotations

from pathlib import Path

from forge_mind.bench.criteria import PassCriteria, evaluate_trial
from forge_mind.bench.report import append_history, write_reports
from forge_mind.bench.runners import Runner
from forge_mind.bench.types import BenchmarkSummary, TrialOutput, TrialResult
from forge_mind.validator.report import Tier, ValidationReport


def _tier_pass_counts(report: ValidationReport | None) -> dict[str, str]:
    if report is None:
        return {}
    counts: dict[str, str] = {}
    for tier in Tier:
        tr = report.tiers.get(tier)
        if tr is None:
            continue
        passed = sum(1 for c in tr.checks if c.status.value == "pass")
        counts[tier.value] = f"{passed}/{len(tr.checks)}"
    return counts


def _required_passes(criteria: PassCriteria, trials_run: int, *, overridden: bool) -> int:
    """The pass bar for ``trials_run`` trials. Default (no --trials override) = the configured
    ``min_passing_trials``. A custom trial count scales the bar to the configured pass RATE
    (min_passing_trials / trials) so the bar keeps its meaning -- otherwise `--trials 100` against a
    fixed min_passing_trials=9 would PASS at a 9% pass rate, and `--trials 5` could never pass
    (audit H7, ADR-0065)."""
    if not overridden or not criteria.trials:
        return criteria.min_passing_trials
    return -(-criteria.min_passing_trials * trials_run // criteria.trials)  # ceil division


def _render_trial(output: TrialOutput, trial_dir: Path, bench_id: str, idx: int) -> None:
    """Render a trial's STL to part.png (best-effort; never fails the trial)."""
    stl = output.artifacts.get("stl")
    if stl is None or not stl.exists():
        return
    try:
        from forge_mind.viz.render import render_part_to_png

        png = render_part_to_png(stl, trial_dir / "part.png", title=f"{bench_id} trial-{idx}")
        output.artifacts["render"] = png
    except Exception:
        pass


def _compile_reference(reference_dsl: Path | None, out_dir: Path) -> Path | None:
    """Compile a benchmark's reference DSL to an STL ONCE per run (best-effort).

    Returns the reference STL path, or None if there is no reference or it could not be
    compiled. Never raises — shape similarity is reported, not gated.
    """
    if reference_dsl is None or not Path(reference_dsl).exists():
        return None
    try:
        import json

        from forge_mind.compiler.compile import compile_program
        from forge_mind.dsl.program import PartProgram

        program = PartProgram.model_validate(json.loads(Path(reference_dsl).read_text()))
        out_dir.mkdir(parents=True, exist_ok=True)
        return compile_program(program, out_dir).stl_path
    except Exception:
        return None


def _trial_shape_similarity(
    generated_stl: Path | None, reference_stl: Path | None
) -> dict[str, float] | None:
    """compare_meshes(generated, reference) -> the similarity scores (chamfer_distance_mm /
    volumetric_iou / f1_at_threshold + params), best-effort. REPORTED, never a gate."""
    if generated_stl is None or reference_stl is None:
        return None
    if not Path(generated_stl).exists() or not Path(reference_stl).exists():
        return None
    try:
        from dataclasses import asdict

        from forge_mind.analysis.shape_similarity import compare_meshes

        return asdict(compare_meshes(generated_stl, reference_stl))
    except Exception:
        return None


def _wrap_trial(
    idx: int, output: TrialOutput, criteria: PassCriteria, reference_stl: Path | None = None
) -> TrialResult:
    passed, criteria_results = evaluate_trial(output, criteria)
    report = output.report
    return TrialResult(
        trial_index=idx,
        passed=passed,
        overall_status=(report.overall.value if report else "error"),
        first_failing_stage=output.first_failing_stage,
        failed_check_ids=(report.failed_check_ids() if report else []),
        criteria_results=criteria_results,
        tier_pass_counts=_tier_pass_counts(report),
        iterations=output.iterations,
        total_duration_ms=output.total_duration_ms,
        total_tokens=output.total_tokens,
        total_cost_usd=output.total_cost_usd,
        artifacts={k: str(v) for k, v in output.artifacts.items()},
        error=output.error,
        shape_similarity=_trial_shape_similarity(output.artifacts.get("stl"), reference_stl),
    )


def run_benchmark(
    bench_dir: Path,
    runner: Runner,
    criteria: PassCriteria,
    run_id: str,
    output_root: Path,
    n_trials: int | None = None,
    render: bool = False,
    reference_dsl: Path | None = None,
) -> BenchmarkSummary:
    """Run ``n_trials`` of ``runner`` against the benchmark in ``bench_dir``.

    Writes per-trial artifacts + a report under
    ``output_root/<bench_id>/<run_id>/`` and appends to that bench's history.
    If ``render`` is set, each trial's STL is rendered to ``part.png``.
    If ``reference_dsl`` is given (the benchmark's known-good DSL), it is compiled once and
    each trial's STL is compared to it; the shape-similarity scores are REPORTED on each
    TrialResult (never gated, never changing pass/fail).
    """
    bench_id = bench_dir.name
    prompt = (bench_dir / "prompt.txt").read_text().strip()
    trials_to_run = n_trials if n_trials is not None else criteria.trials
    required_passes = _required_passes(criteria, trials_to_run, overridden=n_trials is not None)

    run_dir = output_root / bench_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reference_stl = _compile_reference(reference_dsl, run_dir / "_reference")

    results: list[TrialResult] = []
    for i in range(trials_to_run):
        trial_dir = run_dir / f"trial-{i}"
        output = runner.run_trial(prompt, trial_dir, i)
        if render:
            _render_trial(output, trial_dir, bench_id, i)
        results.append(_wrap_trial(i, output, criteria, reference_stl))

    n_passed = sum(1 for r in results if r.passed)
    n = len(results)
    summary = BenchmarkSummary(
        bench_id=bench_id,
        runner=runner.name,
        run_id=run_id,
        n_trials=n,
        n_passed=n_passed,
        pass_rate=(n_passed / n if n else 0.0),
        min_passing_trials=required_passes,
        benchmark_passed=n_passed >= required_passes,
        mean_duration_ms=(sum(r.total_duration_ms for r in results) / n if n else 0.0),
        mean_tokens=(sum(r.total_tokens for r in results) / n if n else 0.0),
        mean_cost_usd=(sum(r.total_cost_usd for r in results) / n if n else 0.0),
        trials=results,
    )

    write_reports(summary, run_dir)
    append_history(summary, output_root / bench_id / "history.jsonl")
    return summary


__all__ = ["run_benchmark"]
