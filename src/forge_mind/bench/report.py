"""Render a BenchmarkSummary to markdown + JSON, and append to history."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge_mind.bench.types import BenchmarkSummary


def render_markdown(summary: BenchmarkSummary) -> str:
    s = summary
    lines: list[str] = []
    lines.append(f"# Benchmark: {s.bench_id}")
    lines.append("")
    lines.append(f"- **Runner:** {s.runner}")
    lines.append(f"- **Run:** {s.run_id}")
    verdict = "PASS" if s.benchmark_passed else "FAIL"
    lines.append(
        f"- **Result:** {verdict} ({s.n_passed}/{s.n_trials} trials, need {s.min_passing_trials})"
    )
    lines.append(f"- **Pass rate:** {s.pass_rate * 100:.0f}%")
    lines.append(f"- **Mean time:** {s.mean_duration_ms / 1000:.2f}s")
    lines.append(f"- **Mean tokens:** {s.mean_tokens:.0f}")
    lines.append(f"- **Mean cost:** ${s.mean_cost_usd:.4f}")
    lines.append("")
    lines.append("## Trials")
    lines.append("")
    lines.append(
        "| # | Result | Fail stage | Iters | Tiers (pass/total) | Time(s) | Tokens | Cost |"
    )
    lines.append(
        "|---|--------|-----------|-------|--------------------|---------|--------|------|"
    )
    for t in s.trials:
        result = "PASS" if t.passed else "FAIL"
        tiers = " ".join(f"{k}:{v}" for k, v in t.tier_pass_counts.items()) or "-"
        stage = t.first_failing_stage or "-"
        lines.append(
            f"| {t.trial_index} | {result} | {stage} | {t.iterations} | {tiers} | "
            f"{t.total_duration_ms / 1000:.2f} | {t.total_tokens} | ${t.total_cost_usd:.4f} |"
        )
    lines.append("")

    # Shape similarity vs the reference mesh (ADR-0019 follow-on) — REPORTED, not gated.
    # Only shown when a reference was available (otherwise the scores are absent).
    with_shape = [t for t in s.trials if t.shape_similarity]
    if with_shape:
        lines.append("## Shape similarity vs reference")
        lines.append("")
        lines.append(
            "Chamfer (mm, lower=better) / volumetric IoU / F1@threshold. Reported, not gated."
        )
        lines.append("")
        lines.append("| # | Chamfer (mm) | Volumetric IoU | F1@threshold |")
        lines.append("|---|--------------|----------------|--------------|")
        for t in with_shape:
            ss = t.shape_similarity or {}
            lines.append(
                f"| {t.trial_index} | {ss.get('chamfer_distance_mm', 0.0):.3f} | "
                f"{ss.get('volumetric_iou', 0.0):.3f} | {ss.get('f1_at_threshold', 0.0):.3f} |"
            )
        lines.append("")

    # Failure detail, if any.
    failed = [t for t in s.trials if not t.passed]
    if failed:
        lines.append("## Failures")
        lines.append("")
        for t in failed:
            crit = ", ".join(k for k, ok in t.criteria_results.items() if not ok) or "?"
            checks = ", ".join(t.failed_check_ids) or "-"
            err = f" error={t.error}" if t.error else ""
            lines.append(f"- Trial {t.trial_index}: unmet [{crit}]; failed checks: {checks}{err}")
        lines.append("")

    return "\n".join(lines)


def write_reports(summary: BenchmarkSummary, run_dir: Path) -> tuple[Path, Path]:
    """Write report.md + report.json into run_dir. Returns their paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    md_path = run_dir / "report.md"
    json_path = run_dir / "report.json"
    md_path.write_text(render_markdown(summary) + "\n")
    json_path.write_text(json.dumps(_summary_to_dict(summary), indent=2) + "\n")
    return md_path, json_path


def append_history(summary: BenchmarkSummary, history_path: Path) -> None:
    """Append a one-line summary to a benchmark's history.jsonl."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": summary.run_id,
        "runner": summary.runner,
        "n_passed": summary.n_passed,
        "n_trials": summary.n_trials,
        "pass_rate": summary.pass_rate,
        "benchmark_passed": summary.benchmark_passed,
        "mean_duration_ms": summary.mean_duration_ms,
        "mean_cost_usd": summary.mean_cost_usd,
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _summary_to_dict(summary: BenchmarkSummary) -> dict[str, Any]:
    return asdict(summary)


__all__ = ["append_history", "render_markdown", "write_reports"]
