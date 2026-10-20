"""Unit tests for benchmark report rendering + history."""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.bench.report import append_history, render_markdown, write_reports
from forge_mind.bench.types import BenchmarkSummary, TrialResult


def _summary(passed: bool = True) -> BenchmarkSummary:
    trial = TrialResult(
        trial_index=0,
        passed=passed,
        overall_status="pass" if passed else "fail",
        first_failing_stage=None if passed else "validate",
        failed_check_ids=[] if passed else ["features.min_wall_thickness"],
        criteria_results={"validation_pass": passed},
        tier_pass_counts={"mesh": "6/6", "interfaces": "5/5"},
        iterations=1,
        total_duration_ms=210.0,
        total_tokens=0,
        total_cost_usd=0.0,
        artifacts={"stl": "x.stl"},
    )
    return BenchmarkSummary(
        bench_id="bench-001",
        runner="handwritten",
        run_id="r1",
        n_trials=1,
        n_passed=1 if passed else 0,
        pass_rate=1.0 if passed else 0.0,
        min_passing_trials=1,
        benchmark_passed=passed,
        mean_duration_ms=210.0,
        mean_tokens=0.0,
        mean_cost_usd=0.0,
        trials=[trial],
    )


def test_markdown_has_headline_and_table() -> None:
    md = render_markdown(_summary())
    assert "# Benchmark: bench-001" in md
    assert "PASS (1/1 trials" in md
    assert "| # | Result |" in md
    assert "mesh:6/6" in md


def test_markdown_lists_failures() -> None:
    md = render_markdown(_summary(passed=False))
    assert "## Failures" in md
    assert "features.min_wall_thickness" in md


def test_write_reports_creates_md_and_json(tmp_path: Path) -> None:
    md_path, json_path = write_reports(_summary(), tmp_path)
    assert md_path.exists() and json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["bench_id"] == "bench-001"
    assert data["trials"][0]["tier_pass_counts"]["mesh"] == "6/6"


def test_append_history_writes_one_line_per_run(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    append_history(_summary(), history)
    append_history(_summary(passed=False), history)
    lines = history.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["benchmark_passed"] is True
    assert json.loads(lines[1])["benchmark_passed"] is False
