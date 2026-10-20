"""Unit tests for the read-only research/recommendation engine.

Synthetic datasets only -- no LLM, no pipeline run, no real benchmarks.
Each test feeds a hand-built report.json (and optional records.jsonl) through
``analyze_next`` / ``diagnose`` / ``recommend`` and asserts the rule engine
picks the documented area.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from forge_mind.analysis.research import (
    Recommendation,
    ReportKind,
    analyze_next,
    diagnose,
    find_reports,
    load_report,
    recommend,
    render_next_action,
)

# ---------------------------------------------------------------------------
# Helpers: write the three on-disk report shapes we have to support
# ---------------------------------------------------------------------------


def _set_after(earlier: Path, later: Path, *, gap: float = 1.0) -> None:
    """Force ``later`` to have an mtime strictly after ``earlier``.

    Path.touch() is not enough on Windows when both files are created in the
    same filesystem tick (which can be ~15 ms wide). Set the mtime explicitly
    so find_reports() ordering is deterministic.
    """
    target = earlier.stat().st_mtime + gap
    os.utime(later, (target, target))


def _write_failure_report(
    dir_: Path,
    name: str,
    *,
    n_trials: int,
    n_passed: int,
    label_counts: dict[str, int],
    top_check_ids: list[tuple[str, int]],
    mean_cost_usd: float = 0.02,
    mean_duration_ms: float = 20000.0,
    stage_counts: dict[str, int] | None = None,
    bench_id: str = "bench-synth-v1",
    run_id: str = "rXX",
) -> Path:
    payload = {
        "bench_id": bench_id,
        "run_id": run_id,
        "n_trials": n_trials,
        "n_passed": n_passed,
        "pass_rate": n_passed / n_trials if n_trials else 0.0,
        "mean_cost_usd": mean_cost_usd,
        "mean_duration_ms": mean_duration_ms,
        "label_counts": label_counts,
        "stage_counts": stage_counts or {"validate": max(n_trials - n_passed, 0)},
        "top_check_ids": [list(p) for p in top_check_ids],
        "representative_artifacts": {},
    }
    path = dir_ / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_records(
    dir_: Path, name: str, validator_messages: list[str], errors: list[str] | None = None
) -> Path:
    """Write a records.jsonl with one record per validator message (and optional errors)."""
    path = dir_ / name
    errors = errors or []
    lines: list[str] = []
    for i, msg in enumerate(validator_messages):
        lines.append(
            json.dumps(
                {
                    "trial_index": i,
                    "passed": False,
                    "first_failing_stage": "validate",
                    "primary_label": "interface_failed",
                    "primary_check_id": "interfaces.board.placement",
                    "validator_message": msg,
                    "failed_check_ids": ["interfaces.board.placement"],
                    "cost_usd": 0.02,
                    "duration_ms": 20000.0,
                    "total_tokens": 9000,
                    "artifacts": {"dir": f"/runs/trial-{i}"},
                    "error": None,
                }
            )
        )
    for j, err in enumerate(errors):
        lines.append(
            json.dumps(
                {
                    "trial_index": 100 + j,
                    "passed": False,
                    "first_failing_stage": "compile",
                    "primary_label": "dsl_compile_failed",
                    "primary_check_id": None,
                    "validator_message": None,
                    "failed_check_ids": [],
                    "cost_usd": 0.02,
                    "duration_ms": 20000.0,
                    "total_tokens": 9000,
                    "artifacts": {"dir": f"/runs/trial-{100 + j}"},
                    "error": err,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_bestofk_report(
    dir_: Path,
    name: str,
    *,
    k: int,
    n_trials: int,
    first_pass_rate: float,
    best_pass_rate: float,
    candidate_label_counts: dict[str, int],
    top_failed_after: list[tuple[str, int]],
    bench_id: str = "bench-synth-v1",
    run_id: str = "k5",
) -> Path:
    payload = {
        "bench_id": bench_id,
        "run_id": run_id,
        "k": k,
        "n_trials": n_trials,
        "first_candidate_pass_rate": first_pass_rate,
        "best_of_k_pass_rate": best_pass_rate,
        "mean_cost_usd": 0.08,
        "mean_duration_ms": 60000.0,
        "score_summary": {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0},
        "candidate_label_counts": candidate_label_counts,
        "top_failed_before": [],
        "top_failed_after": [list(p) for p in top_failed_after],
    }
    path = dir_ / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_report normalises the three shapes
# ---------------------------------------------------------------------------


def test_load_report_failure_shape(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "n30-report.json",
        n_trials=30,
        n_passed=3,
        label_counts={"wall_too_thin": 20, "passed": 3, "envelope_failed": 7},
        top_check_ids=[("features.min_wall_thickness", 20), ("mesh.bbox_within_max", 8)],
    )
    report = load_report(path)
    assert report.kind is ReportKind.FAILURE
    assert report.n_trials == 30
    assert report.n_passed == 3
    assert report.pass_rate == 0.1
    assert report.label_counts["wall_too_thin"] == 20
    assert report.top_check_ids[0] == ("features.min_wall_thickness", 20)


def test_load_report_bestofk_shape(tmp_path: Path) -> None:
    path = _write_bestofk_report(
        tmp_path,
        "k5-report.json",
        k=5,
        n_trials=15,
        first_pass_rate=0.07,
        best_pass_rate=0.23,
        candidate_label_counts={"wall_too_thin": 50, "passed": 7},
        top_failed_after=[("features.min_wall_thickness", 10)],
    )
    report = load_report(path)
    assert report.kind is ReportKind.BESTOFK
    # best-of-k pass rate (winner column) is what gets reported.
    assert report.pass_rate == 0.23
    assert report.label_counts["wall_too_thin"] == 50  # candidate-level
    assert report.top_check_ids == [("features.min_wall_thickness", 10)]


def test_load_report_records_jsonl_is_mined(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rA-report.json",
        n_trials=2,
        n_passed=0,
        label_counts={"interface_failed": 2},
        top_check_ids=[("interfaces.board.placement", 2)],
    )
    _write_records(tmp_path, "rA-records.jsonl", ["Unknown component pi-zero in registry"])
    report = load_report(path)
    assert any("unknown component" in m.lower() for m in report.validator_messages)


# ---------------------------------------------------------------------------
# Rule engine: each documented signal -> correct recommendation area
# ---------------------------------------------------------------------------


def _recommend_for(report_path: Path) -> Recommendation:
    report = load_report(report_path)
    diag = diagnose(report)
    return recommend(report, diag)


def test_wall_failures_recommend_layout_helper(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rW-report.json",
        n_trials=30,
        n_passed=3,
        label_counts={"wall_too_thin": 22, "passed": 3, "envelope_failed": 5},
        top_check_ids=[("features.min_wall_thickness", 22)],
    )
    rec = _recommend_for(path)
    assert rec.area == "layout-helper"
    assert "wall" in rec.title.lower() or "min-wall" in rec.title.lower()


def test_overhang_failures_recommend_layout_helper(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rO-report.json",
        n_trials=20,
        n_passed=2,
        label_counts={"overhang_too_steep": 14, "passed": 2, "wall_too_thin": 4},
        top_check_ids=[("printability.overhang_max_angle", 14)],
    )
    rec = _recommend_for(path)
    assert rec.area == "layout-helper"
    assert "overhang" in rec.title.lower() or "printability" in rec.title.lower()


def test_envelope_failures_recommend_feasibility_rule(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rE-report.json",
        n_trials=20,
        n_passed=2,
        label_counts={"envelope_failed": 14, "passed": 2, "wall_too_thin": 4},
        top_check_ids=[("mesh.bbox_within_max", 14)],
    )
    rec = _recommend_for(path)
    assert rec.area == "feasibility-rule"


def test_interface_failures_recommend_interface_work(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rI-report.json",
        n_trials=20,
        n_passed=2,
        label_counts={"interface_failed": 15, "passed": 2, "wall_too_thin": 3},
        top_check_ids=[("interfaces.motor_face.bolt_pattern_31mm_square", 15)],
    )
    rec = _recommend_for(path)
    assert rec.area == "interface"
    assert "interface" in rec.title.lower()


def test_component_failures_recommend_component_registry(tmp_path: Path) -> None:
    # Label/check distribution alone looks interface-y, but validator messages
    # name unknown components -- the component-registry rule must trump.
    path = _write_failure_report(
        tmp_path,
        "rC-report.json",
        n_trials=10,
        n_passed=0,
        label_counts={"interface_failed": 10},
        top_check_ids=[("interfaces.board.placement", 10)],
    )
    _write_records(
        tmp_path,
        "rC-records.jsonl",
        [
            "Unknown component 'pi-zero' is not in registry",
            "Unknown component 'pi-zero' is not in registry",
        ],
        errors=["LayoutError: unsupported board 'pi-zero'"],
    )
    rec = _recommend_for(path)
    assert rec.area == "component-registry"
    assert "component" in rec.title.lower() or "registry" in rec.title.lower()


def test_repeated_unsupported_pattern_recommends_new_pattern(tmp_path: Path) -> None:
    # Dominant label is layoutspec_invalid / layout_infeasible -> the LLM is
    # asking for a shape family the union/interpreter cannot satisfy.
    path = _write_failure_report(
        tmp_path,
        "rP-report.json",
        n_trials=15,
        n_passed=0,
        label_counts={"layout_infeasible": 12, "wall_too_thin": 3},
        top_check_ids=[],
    )
    rec = _recommend_for(path)
    assert rec.area == "pattern-interpreter"
    assert "pattern" in rec.title.lower()


def test_no_dominant_failure_recommends_more_measurement(tmp_path: Path) -> None:
    # No label or check accounts for >= 30% of trials.
    path = _write_failure_report(
        tmp_path,
        "rN-report.json",
        n_trials=30,
        n_passed=15,
        label_counts={
            "passed": 15,
            "wall_too_thin": 3,
            "envelope_failed": 3,
            "overhang_too_steep": 3,
            "interface_failed": 3,
            "dsl_compile_failed": 3,
        },
        top_check_ids=[
            ("features.min_wall_thickness", 3),
            ("mesh.bbox_within_max", 3),
            ("printability.overhang_max_angle", 3),
        ],
    )
    rec = _recommend_for(path)
    assert rec.area == "measurement"


def test_saturated_pass_rate_recommends_new_benchmark(tmp_path: Path) -> None:
    path = _write_failure_report(
        tmp_path,
        "rS-report.json",
        n_trials=15,
        n_passed=15,
        label_counts={"passed": 15},
        top_check_ids=[],
    )
    rec = _recommend_for(path)
    assert rec.area == "benchmark-coverage"


def test_bestofk_winner_checkid_drives_recommendation(tmp_path: Path) -> None:
    # Best-of-k reports do not carry per-trial label_counts mapped to our
    # rule buckets, so the recommendation must come from the winner check_id.
    path = _write_bestofk_report(
        tmp_path,
        "kBO-report.json",
        k=5,
        n_trials=10,
        first_pass_rate=0.1,
        best_pass_rate=0.3,
        candidate_label_counts={"wall_too_thin": 40, "passed": 3},
        top_failed_after=[("features.min_wall_thickness", 7)],
    )
    rec = _recommend_for(path)
    assert rec.area == "layout-helper"


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


def test_regression_vs_baseline_is_detected(tmp_path: Path) -> None:
    # Baseline: 25/30 passed. Current: 5/30 passed.
    _write_failure_report(
        tmp_path,
        "01-report.json",
        n_trials=30,
        n_passed=25,
        label_counts={"passed": 25, "wall_too_thin": 5},
        top_check_ids=[("features.min_wall_thickness", 5)],
        run_id="baseline",
    )
    # Make sure mtimes order them; sleep is overkill -- touch the second file
    # after writing both.
    later = _write_failure_report(
        tmp_path,
        "02-report.json",
        n_trials=30,
        n_passed=5,
        label_counts={"wall_too_thin": 23, "passed": 5, "envelope_failed": 2},
        top_check_ids=[("features.min_wall_thickness", 23)],
        run_id="current",
    )
    _set_after(tmp_path / "01-report.json", later)

    reports = find_reports(tmp_path)
    assert len(reports) == 2
    current = load_report(reports[-1])
    baseline = load_report(reports[-2])
    assert baseline.run_id == "baseline"
    assert current.run_id == "current"

    diag = diagnose(current, baseline)
    assert diag.regression is True
    assert diag.improvement is False
    assert diag.delta_summary["pass_rate"] < -0.5


def test_improvement_vs_baseline_is_detected(tmp_path: Path) -> None:
    _write_failure_report(
        tmp_path,
        "01-report.json",
        n_trials=20,
        n_passed=2,
        label_counts={"wall_too_thin": 18, "passed": 2},
        top_check_ids=[("features.min_wall_thickness", 18)],
        run_id="before",
    )
    later = _write_failure_report(
        tmp_path,
        "02-report.json",
        n_trials=20,
        n_passed=18,
        label_counts={"passed": 18, "wall_too_thin": 2},
        top_check_ids=[("features.min_wall_thickness", 2)],
        run_id="after",
    )
    _set_after(tmp_path / "01-report.json", later)
    reports = find_reports(tmp_path)
    current = load_report(reports[-1])
    baseline = load_report(reports[-2])
    diag = diagnose(current, baseline)
    assert diag.improvement is True
    assert diag.regression is False


# ---------------------------------------------------------------------------
# analyze_next end-to-end: writes a structured next_action.md
# ---------------------------------------------------------------------------


def test_analyze_next_writes_next_action_md(tmp_path: Path) -> None:
    _write_failure_report(
        tmp_path,
        "rR-report.json",
        n_trials=30,
        n_passed=3,
        label_counts={"wall_too_thin": 20, "passed": 3, "envelope_failed": 7},
        top_check_ids=[("features.min_wall_thickness", 20)],
    )
    out = analyze_next(tmp_path)
    assert out.name == "next_action.md"
    body = out.read_text(encoding="utf-8")
    # Required sections.
    for required in (
        "# Next action",
        "## Diagnosis",
        "## Evidence",
        "## Recommendation",
        "## Implementation prompt",
        "## Constraints",
        "## Stop conditions",
    ):
        assert required in body
    # The implementation prompt is fenced and references the bench.
    assert "```" in body
    assert "bench-synth-v1" in body


def test_analyze_next_raises_when_no_report(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        analyze_next(tmp_path)


def test_render_includes_baseline_section_when_present(tmp_path: Path) -> None:
    _write_failure_report(
        tmp_path,
        "01-report.json",
        n_trials=10,
        n_passed=8,
        label_counts={"passed": 8, "wall_too_thin": 2},
        top_check_ids=[("features.min_wall_thickness", 2)],
        run_id="prev",
    )
    later = _write_failure_report(
        tmp_path,
        "02-report.json",
        n_trials=10,
        n_passed=2,
        label_counts={"wall_too_thin": 8, "passed": 2},
        top_check_ids=[("features.min_wall_thickness", 8)],
        run_id="now",
    )
    _set_after(tmp_path / "01-report.json", later)
    reports = find_reports(tmp_path)
    current = load_report(reports[-1])
    baseline = load_report(reports[-2])
    diag = diagnose(current, baseline)
    rec = recommend(current, diag)
    body = render_next_action(current, diag, rec)
    assert "Comparison vs baseline" in body
    assert "regression" in body.lower()
