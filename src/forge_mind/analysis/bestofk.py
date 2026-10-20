"""Best-of-k measurement: run N trials of run_best_of_k and compare
first-candidate vs best-of-k reliability.

Measurement only. Reuses load_expectations() and the existing pipeline stages;
changes no prompt, DSL primitive, or validator behavior.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge_mind.analysis.run import load_expectations
from forge_mind.candidates import BestOfKResult, run_best_of_k


@dataclass
class BestOfKTrial:
    trial_index: int
    first_candidate_passed: bool
    best_passed: bool
    winner_id: int | None
    candidate_scores: list[float]
    candidate_labels: list[str]
    first_failed_check_ids: list[str]
    winner_failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    error: str | None = None


@dataclass
class BestOfKDataset:
    bench_id: str
    run_id: str
    k: int
    n_trials: int
    first_candidate_pass_rate: float
    best_of_k_pass_rate: float
    mean_cost_usd: float
    mean_duration_ms: float
    score_summary: dict[str, float]
    candidate_label_counts: dict[str, int]
    top_failed_before: list[tuple[str, int]]
    top_failed_after: list[tuple[str, int]]
    trials: list[BestOfKTrial] = field(default_factory=list)


def _aggregate(bench_id: str, run_id: str, k: int, trials: list[BestOfKTrial]) -> BestOfKDataset:
    n = len(trials)
    first_pass = sum(1 for t in trials if t.first_candidate_passed)
    best_pass = sum(1 for t in trials if t.best_passed)

    all_scores = [s for t in trials for s in t.candidate_scores]
    score_summary = {
        "min": min(all_scores) if all_scores else 0.0,
        "median": statistics.median(all_scores) if all_scores else 0.0,
        "mean": statistics.fmean(all_scores) if all_scores else 0.0,
        "max": max(all_scores) if all_scores else 0.0,
    }

    label_counts = Counter(lbl for t in trials for lbl in t.candidate_labels)
    before = Counter(cid for t in trials for cid in t.first_failed_check_ids)
    after = Counter(cid for t in trials for cid in t.winner_failed_check_ids)

    return BestOfKDataset(
        bench_id=bench_id,
        run_id=run_id,
        k=k,
        n_trials=n,
        first_candidate_pass_rate=(first_pass / n if n else 0.0),
        best_of_k_pass_rate=(best_pass / n if n else 0.0),
        mean_cost_usd=(sum(t.cost_usd for t in trials) / n if n else 0.0),
        mean_duration_ms=(sum(t.duration_ms for t in trials) / n if n else 0.0),
        score_summary=score_summary,
        candidate_label_counts=dict(label_counts.most_common()),
        top_failed_before=[(c, n_) for c, n_ in before.most_common(10)],
        top_failed_after=[(c, n_) for c, n_ in after.most_common(10)],
        trials=trials,
    )


def render_markdown(d: BestOfKDataset) -> str:
    lines: list[str] = []
    lines.append(f"# Best-of-{d.k} analysis: {d.bench_id}")
    lines.append("")
    lines.append(f"- **Run:** {d.run_id}  (k={d.k}, {d.n_trials} trials)")
    first = d.first_candidate_pass_rate * 100
    best = d.best_of_k_pass_rate * 100
    lift = best - first
    lines.append(f"- **First-candidate pass rate:** {first:.0f}%")
    lines.append(f"- **Best-of-{d.k} pass rate:** {best:.0f}%  (+{lift:.0f} pts)")
    lines.append(f"- **Mean cost:** ${d.mean_cost_usd:.4f}/trial")
    lines.append(f"- **Mean latency:** {d.mean_duration_ms / 1000:.1f}s/trial")
    lines.append("")

    lines.append("## Candidate quality distribution (all k x trials candidates)")
    lines.append("")
    lines.append("| candidate label | count |")
    lines.append("|-----------------|-------|")
    for label, count in d.candidate_label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append(
        f"Score summary: min {d.score_summary['min']:.0f}, "
        f"median {d.score_summary['median']:.0f}, "
        f"mean {d.score_summary['mean']:.0f}, max {d.score_summary['max']:.0f}."
    )
    lines.append("")

    lines.append("## Top failed check_ids: first candidate vs best-of-k winner")
    lines.append("")
    lines.append("| check_id | before (first) | after (winner) |")
    lines.append("|----------|----------------|----------------|")
    before = dict(d.top_failed_before)
    after = dict(d.top_failed_after)
    for cid in sorted(set(before) | set(after), key=lambda c: -before.get(c, 0)):
        lines.append(f"| {cid} | {before.get(cid, 0)} | {after.get(cid, 0)} |")
    lines.append("")

    return "\n".join(lines)


def write_dataset(dataset: BestOfKDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(dataset) + "\n", encoding="utf-8")
    payload = asdict(dataset)
    payload.pop("trials", None)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8") as fh:
        for t in dataset.trials:
            fh.write(json.dumps(asdict(t)) + "\n")


_Runner = Callable[[str, Path, int, tuple[float, float, float]], BestOfKResult]


def run_best_of_k_analysis(
    bench_dir: Path,
    n_trials: int,
    k: int,
    output_root: Path,
    run_id: str,
    *,
    progress: bool = True,
    runner: _Runner = run_best_of_k,
) -> BestOfKDataset:
    """Run n_trials of a best-of-k path and aggregate. ``runner`` selects the
    generation path (direct DSL by default, or the LayoutSpec runner for the A/B).
    """
    prompt = (bench_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    expectations = load_expectations(bench_dir)
    run_dir = output_root / bench_dir.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trials: list[BestOfKTrial] = []
    for i in range(n_trials):
        trial_dir = run_dir / f"trial-{i}"
        result = runner(prompt, trial_dir, k, expectations.envelope_cap_mm)

        winner = result.candidates[result.winner_id] if result.winner_id is not None else None
        trial = BestOfKTrial(
            trial_index=i,
            first_candidate_passed=result.first_candidate_passed,
            best_passed=result.best_passed,
            winner_id=result.winner_id,
            candidate_scores=[c.score for c in result.candidates],
            candidate_labels=[c.primary_label for c in result.candidates],
            first_failed_check_ids=(
                result.candidates[0].failed_check_ids if result.candidates else []
            ),
            winner_failed_check_ids=(winner.failed_check_ids if winner else []),
            cost_usd=result.total_cost_usd,
            duration_ms=result.total_duration_ms,
            error=result.error,
        )
        trials.append(trial)
        if progress:
            print(
                f"  trial {i}: first={'P' if trial.first_candidate_passed else 'F'} "
                f"best={'P' if trial.best_passed else 'F'} "
                f"${trial.cost_usd:.4f}  {trial.duration_ms / 1000:.0f}s"
            )

    dataset = _aggregate(bench_dir.name, run_id, k, trials)
    write_dataset(dataset, run_dir)
    return dataset


__all__ = ["BestOfKDataset", "BestOfKTrial", "render_markdown", "run_best_of_k_analysis"]
