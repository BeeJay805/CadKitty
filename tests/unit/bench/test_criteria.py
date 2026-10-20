"""Unit tests for pass-criteria evaluation."""

from __future__ import annotations

from forge_mind.bench.criteria import PassCriteria, evaluate_trial
from forge_mind.bench.types import StageTiming, TrialOutput
from forge_mind.validator.report import Check, CheckStatus, Tier, build_report


def _report(*statuses: tuple[str, CheckStatus]):  # type: ignore[no-untyped-def]
    checks = [Check(id=cid, tier=Tier.MESH, status=st, message="") for cid, st in statuses]
    return build_report(checks, [])


def test_passing_trial_meets_all_criteria() -> None:
    report = _report(("mesh.watertight", CheckStatus.PASS))
    output = TrialOutput(
        report=report,
        stage_timings=[StageTiming("compile", 100.0), StageTiming("validate", 50.0)],
    )
    criteria = PassCriteria(required_checks=["mesh.watertight"])
    passed, results = evaluate_trial(output, criteria)
    assert passed
    assert all(results.values())


def test_fails_when_required_check_absent() -> None:
    report = _report(("mesh.watertight", CheckStatus.PASS))
    output = TrialOutput(report=report)
    criteria = PassCriteria(required_checks=["features.min_wall_thickness"])
    passed, results = evaluate_trial(output, criteria)
    assert not passed
    assert results["required_checks_pass"] is False


def test_fails_on_validation_failure() -> None:
    report = _report(("mesh.watertight", CheckStatus.FAIL))
    output = TrialOutput(report=report)
    passed, results = evaluate_trial(output, PassCriteria())
    assert not passed
    assert results["validation_pass"] is False


def test_fails_over_time_budget() -> None:
    report = _report(("mesh.watertight", CheckStatus.PASS))
    output = TrialOutput(report=report, stage_timings=[StageTiming("compile", 120_000.0)])
    criteria = PassCriteria(max_wall_seconds=60)
    passed, results = evaluate_trial(output, criteria)
    assert not passed
    assert results["within_time"] is False


def test_fails_over_cost_budget() -> None:
    report = _report(("mesh.watertight", CheckStatus.PASS))
    output = TrialOutput(
        report=report,
        stage_timings=[StageTiming("dsl", 10.0, cost_usd=0.5)],
    )
    criteria = PassCriteria(max_cost_usd_per_trial=0.30)
    passed, results = evaluate_trial(output, criteria)
    assert not passed
    assert results["within_cost"] is False


def test_error_trial_fails() -> None:
    output = TrialOutput(report=None, error="compile blew up")
    passed, results = evaluate_trial(output, PassCriteria(required_checks=[]))
    assert not passed
    assert results["no_error"] is False
