"""Best-of-k + revision measurement.

Per trial: run best-of-k, then if the winning candidate still fails, run the
patch-only LLM reviser on it. Reports three pass rates side by side --
first-candidate, best-of-k, and best-of-k + revision -- plus the cost and latency
revision adds and how the failed-check profile shifts after repair.

Measurement only: no prompt, DSL primitive, or validator threshold changes. The
reviser may only patch existing fields (enforced in revision/patch.py).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge_mind.analysis.run import load_expectations
from forge_mind.candidates import run_best_of_k
from forge_mind.dsl.program import PartProgram
from forge_mind.revision.llm_reviser import LLMReviser
from forge_mind.revision.reviser import revise
from forge_mind.validator.report import ValidationReport


@dataclass
class RevisionTrial:
    trial_index: int
    first_candidate_passed: bool
    best_passed: bool
    revised_passed: bool
    revision_attempted: bool
    revision_iterations: int
    winner_failed_check_ids: list[str]  # before revision
    revised_failed_check_ids: list[str]  # after revision
    bestofk_cost_usd: float
    revision_cost_usd: float
    total_cost_usd: float
    duration_ms: float
    error: str | None = None


@dataclass
class RevisionDataset:
    bench_id: str
    run_id: str
    k: int
    n_trials: int
    first_candidate_pass_rate: float
    best_of_k_pass_rate: float
    revised_pass_rate: float
    mean_cost_usd: float
    mean_revision_cost_usd: float
    mean_duration_ms: float
    n_revision_attempted: int
    n_revision_repaired: int
    mean_revision_iterations: float
    top_failed_before_revision: list[tuple[str, int]]
    top_failed_after_revision: list[tuple[str, int]]
    trials: list[RevisionTrial] = field(default_factory=list)


def _aggregate(bench_id: str, run_id: str, k: int, trials: list[RevisionTrial]) -> RevisionDataset:
    n = len(trials)
    first = sum(1 for t in trials if t.first_candidate_passed)
    best = sum(1 for t in trials if t.best_passed)
    revised = sum(1 for t in trials if t.revised_passed)
    attempted = [t for t in trials if t.revision_attempted]
    repaired = [t for t in attempted if t.revised_passed and not t.best_passed]

    before = Counter(cid for t in trials for cid in t.winner_failed_check_ids)
    after = Counter(cid for t in trials for cid in t.revised_failed_check_ids)

    return RevisionDataset(
        bench_id=bench_id,
        run_id=run_id,
        k=k,
        n_trials=n,
        first_candidate_pass_rate=(first / n if n else 0.0),
        best_of_k_pass_rate=(best / n if n else 0.0),
        revised_pass_rate=(revised / n if n else 0.0),
        mean_cost_usd=(sum(t.total_cost_usd for t in trials) / n if n else 0.0),
        mean_revision_cost_usd=(sum(t.revision_cost_usd for t in trials) / n if n else 0.0),
        mean_duration_ms=(sum(t.duration_ms for t in trials) / n if n else 0.0),
        n_revision_attempted=len(attempted),
        n_revision_repaired=len(repaired),
        mean_revision_iterations=(
            sum(t.revision_iterations for t in attempted) / len(attempted) if attempted else 0.0
        ),
        top_failed_before_revision=[(c, n_) for c, n_ in before.most_common(10)],
        top_failed_after_revision=[(c, n_) for c, n_ in after.most_common(10)],
        trials=trials,
    )


def render_markdown(d: RevisionDataset) -> str:
    first = d.first_candidate_pass_rate * 100
    best = d.best_of_k_pass_rate * 100
    revised = d.revised_pass_rate * 100
    lines: list[str] = [
        f"# Best-of-{d.k} + revision analysis: {d.bench_id}",
        "",
        f"- **Run:** {d.run_id}  (k={d.k}, {d.n_trials} trials)",
        "",
        "| stage | pass rate |",
        "|-------|-----------|",
        f"| first candidate | {first:.0f}% |",
        f"| best-of-{d.k} | {best:.0f}% |",
        f"| best-of-{d.k} + revision | {revised:.0f}% |",
        "",
        f"- **Lift from revision:** +{revised - best:.0f} pts over best-of-{d.k} "
        f"(+{revised - first:.0f} pts over first candidate).",
        f"- **Mean cost:** ${d.mean_cost_usd:.4f}/trial "
        f"(revision adds ${d.mean_revision_cost_usd:.4f}).",
        f"- **Mean latency:** {d.mean_duration_ms / 1000:.1f}s/trial.",
        f"- **Revision attempted:** {d.n_revision_attempted}/{d.n_trials} trials "
        f"(winner still failing); **repaired:** {d.n_revision_repaired}. "
        f"Mean {d.mean_revision_iterations:.1f} iters when attempted.",
        "",
        "## Failed check_ids: best-of-k winner vs after revision",
        "",
        "| check_id | before (winner) | after (revision) |",
        "|----------|-----------------|------------------|",
    ]
    before = dict(d.top_failed_before_revision)
    after = dict(d.top_failed_after_revision)
    for cid in sorted(set(before) | set(after), key=lambda c: -before.get(c, 0)):
        lines.append(f"| {cid} | {before.get(cid, 0)} | {after.get(cid, 0)} |")
    lines.append("")
    return "\n".join(lines)


def write_dataset(dataset: RevisionDataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(dataset) + "\n", encoding="utf-8")
    payload = asdict(dataset)
    payload.pop("trials", None)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8") as fh:
        for t in dataset.trials:
            fh.write(json.dumps(asdict(t)) + "\n")


def _revise_winner(
    winner_partdsl: str,
    winner_report: str,
    trial_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
) -> tuple[bool, int, list[str], float]:
    """Load the failing winner and run the LLM reviser. Returns
    (passed, iterations, failed_check_ids_after, cost). Robust to any failure.
    """
    try:
        program = PartProgram.model_validate_json(Path(winner_partdsl).read_text(encoding="utf-8"))
        report = ValidationReport.model_validate_json(
            Path(winner_report).read_text(encoding="utf-8")
        )
    except Exception:
        return False, 0, [], 0.0

    try:
        result = revise(
            program,
            report,
            LLMReviser(),
            output_dir=trial_dir / "revision",
            envelope_cap_mm=envelope_cap_mm,
            render=False,
        )
    except Exception:
        return False, 0, report.failed_check_ids(), 0.0

    return result.end_passed, result.iterations, result.end_failed_check_ids, result.cost_usd


def run_revision_analysis(
    bench_dir: Path,
    n_trials: int,
    k: int,
    output_root: Path,
    run_id: str,
    *,
    progress: bool = True,
) -> RevisionDataset:
    prompt = (bench_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    expectations = load_expectations(bench_dir)
    run_dir = output_root / bench_dir.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trials: list[RevisionTrial] = []
    for i in range(n_trials):
        trial_dir = run_dir / f"trial-{i}"
        t0 = time.perf_counter()
        bok = run_best_of_k(prompt, trial_dir, k, expectations.envelope_cap_mm)
        winner = bok.candidates[bok.winner_id] if bok.winner_id is not None else None

        winner_failed = winner.failed_check_ids if winner else []
        revised_passed = bok.best_passed
        revised_failed = winner_failed
        rev_iters = 0
        rev_cost = 0.0
        attempted = False

        if winner is not None and not winner.passed and winner.partdsl_path and winner.report_path:
            attempted = True
            revised_passed, rev_iters, revised_failed, rev_cost = _revise_winner(
                winner.partdsl_path, winner.report_path, trial_dir, expectations.envelope_cap_mm
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        trial = RevisionTrial(
            trial_index=i,
            first_candidate_passed=bok.first_candidate_passed,
            best_passed=bok.best_passed,
            revised_passed=revised_passed,
            revision_attempted=attempted,
            revision_iterations=rev_iters,
            winner_failed_check_ids=winner_failed,
            revised_failed_check_ids=revised_failed,
            bestofk_cost_usd=bok.total_cost_usd,
            revision_cost_usd=rev_cost,
            total_cost_usd=bok.total_cost_usd + rev_cost,
            duration_ms=duration_ms,
            error=bok.error,
        )
        trials.append(trial)
        if progress:
            print(
                f"  trial {i}: first={'P' if trial.first_candidate_passed else 'F'} "
                f"bok={'P' if trial.best_passed else 'F'} "
                f"rev={'P' if trial.revised_passed else 'F'} "
                f"(it{trial.revision_iterations}) "
                f"${trial.total_cost_usd:.4f}  {trial.duration_ms / 1000:.0f}s"
            )

    dataset = _aggregate(bench_dir.name, run_id, k, trials)
    write_dataset(dataset, run_dir)
    return dataset


__all__ = [
    "RevisionDataset",
    "RevisionTrial",
    "render_markdown",
    "run_revision_analysis",
]
