"""Phase 3 EXIT GATE.

Running bench-001 with the handwritten runner (LLM bypass) must produce 10/10
passing trials and a report containing every metric we care about. This proves
the measurement infrastructure works before any LLM stage exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.bench.criteria import load_criteria
from forge_mind.bench.runner import run_benchmark
from forge_mind.bench.runners import HandwrittenRunner

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "benchmarks" / "bench-001-nema17-mount-v1"


def _run(tmp_path: Path, n_trials: int = 10):  # type: ignore[no-untyped-def]
    criteria = load_criteria(BENCH_DIR / "pass_criteria.yaml")
    assert criteria.reference_dsl is not None
    runner = HandwrittenRunner(REPO / criteria.reference_dsl)
    return (
        run_benchmark(
            bench_dir=BENCH_DIR,
            runner=runner,
            criteria=criteria,
            run_id="test",
            output_root=tmp_path,
            n_trials=n_trials,
        ),
        criteria,
    )


def test_bench001_handwritten_passes_all_trials(tmp_path: Path) -> None:
    summary, _ = _run(tmp_path, n_trials=10)
    assert summary.n_trials == 10
    assert summary.n_passed == 10, (
        f"expected 10/10, got {summary.n_passed}; "
        f"first failure: {next((t for t in summary.trials if not t.passed), None)}"
    )
    assert summary.benchmark_passed
    assert summary.pass_rate == 1.0


def test_every_required_check_is_satisfied(tmp_path: Path) -> None:
    """The reference DSL must satisfy every check the benchmark requires."""
    summary, criteria = _run(tmp_path, n_trials=1)
    trial = summary.trials[0]
    # No required check should appear among the failed ids.
    for required in criteria.required_checks:
        assert required not in trial.failed_check_ids


def test_report_contains_all_metrics(tmp_path: Path) -> None:
    _run(tmp_path, n_trials=3)
    run_dir = tmp_path / "bench-001-nema17-mount-v1" / "test"
    md = (run_dir / "report.md").read_text()
    data = json.loads((run_dir / "report.json").read_text())

    # Markdown headline metrics.
    for token in ("Pass rate", "Mean time", "Mean tokens", "Mean cost", "Tiers (pass/total)"):
        assert token in md

    # JSON has per-trial telemetry.
    assert data["n_trials"] == 3
    assert len(data["trials"]) == 3
    t0 = data["trials"][0]
    for key in ("total_duration_ms", "total_tokens", "total_cost_usd", "tier_pass_counts"):
        assert key in t0


def test_history_appended(tmp_path: Path) -> None:
    _run(tmp_path, n_trials=2)
    history = tmp_path / "bench-001-nema17-mount-v1" / "history.jsonl"
    assert history.exists()
    rows = history.read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["n_trials"] == 2
