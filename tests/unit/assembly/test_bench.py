"""Assembly benchmark harness math (no LLM): Wilson bound + result tallying +
honest warn/required-check scoring (the P2a actuated-assembly fix)."""

from __future__ import annotations

from pathlib import Path

import forge_mind.assembly.bench as bench_mod
from forge_mind.assembly.bench import (
    BenchResult,
    PromptResult,
    TrialResult,
    run_one_trial,
    to_detail_dict,
    wilson_lower_bound,
)
from forge_mind.llm.cost import LLMUsage
from forge_mind.validator.report import Check, CheckStatus, Tier, build_report


def test_wilson_lower_bound_is_conservative_and_bounded() -> None:
    assert wilson_lower_bound(0, 0) == 0.0
    assert 0.0 <= wilson_lower_bound(0, 10) < 0.4
    # A perfect run still has a lower bound below 1 (honest small-n).
    lb = wilson_lower_bound(12, 12)
    assert 0.7 < lb < 1.0
    # More samples at 100% -> tighter (higher) lower bound.
    assert wilson_lower_bound(24, 24) > wilson_lower_bound(12, 12)


def test_result_aggregation() -> None:
    trials = [
        TrialResult("hinge", True, "pass", [], 0.01),
        TrialResult("hinge", False, "validate", ["assembly.range_of_motion"], 0.01),
    ]
    pr = PromptResult(prompt_id="hinge", trials=trials)
    assert pr.n == 2 and pr.passed == 1 and pr.pass_rate == 0.5
    assert abs(pr.cost_usd - 0.02) < 1e-9
    bench = BenchResult(prompts=[pr])
    assert bench.n == 2 and bench.passed == 1
    assert abs(bench.cost_usd - 0.02) < 1e-9


def test_to_detail_dict_keeps_per_trial_stage_and_failed_checks() -> None:
    # The console summary only shows stage counts; the detail dump must preserve
    # which stage each trial failed at and which checks failed, so a weak prompt
    # can be diagnosed without re-running the LLM.
    trials = [
        TrialResult("hinge_door", True, "pass", [], 0.01),
        TrialResult("hinge_door", False, "validate", ["assembly.range_of_motion"], 0.02),
    ]
    bench = BenchResult(prompts=[PromptResult(prompt_id="hinge_door", trials=trials)])

    detail = to_detail_dict(bench)

    assert detail["overall"]["passed"] == 1 and detail["overall"]["n"] == 2
    door = detail["prompts"][0]
    assert door["prompt_id"] == "hinge_door"
    trial_detail = door["trials"]
    assert [t["stage"] for t in trial_detail] == ["pass", "validate"]
    assert trial_detail[1]["failed_checks"] == ["assembly.range_of_motion"]


# ---------------------------------------------------------------------------
# Honest warn/required-check scoring (the P2a actuated-assembly fix).
# ---------------------------------------------------------------------------


def _check(cid: str, status: CheckStatus) -> Check:
    return Check(id=cid, tier=Tier.ASSEMBLY, status=status, message=cid)


def _trial_with_report(monkeypatch, checks, **kwargs):  # type: ignore[no-untyped-def]
    """Run run_one_trial with the pipeline stubbed to yield a synthetic report."""
    from forge_mind.assembly.joints import AssemblyGraph, PartNode

    report = build_report(checks, [])
    usage = LLMUsage(model="test", input_tokens=0, output_tokens=0)
    # A minimal real graph so run_one_trial's topology capture (graph.nodes/.joints) works.
    graph = AssemblyGraph(nodes=[PartNode(id="x", archetype="rod")], joints=[])
    fake_gen = lambda _prompt: (graph, usage)  # noqa: E731  (graph, usage)
    monkeypatch.setattr(bench_mod, "solve", lambda _graph: object())
    monkeypatch.setattr(bench_mod, "compile_assembly", lambda _prog, _out: object())
    monkeypatch.setattr(bench_mod, "validate_assembly", lambda _model: report)
    return run_one_trial(fake_gen, "p", "prompt", Path("/tmp/x"), **kwargs)


def test_accepted_warn_scores_as_pass_and_is_recorded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A static-actuated joint: hardware_coaxial PASS + range_of_motion WARN (deferred).
    checks = [
        _check("assembly.hardware_coaxial", CheckStatus.PASS),
        _check("assembly.range_of_motion", CheckStatus.WARN),
    ]
    t = _trial_with_report(
        monkeypatch,
        checks,
        required_checks=["assembly.hardware_coaxial"],
        accepted_warns=["assembly.range_of_motion"],
    )
    assert t.passed and t.stage == "pass"
    assert t.failed_checks == []
    # The deferred warn is recorded honestly (not hidden, not a failure).
    assert t.accepted_warns == ["assembly.range_of_motion"]


def test_unaccepted_warn_fails_and_is_named(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    checks = [_check("assembly.range_of_motion", CheckStatus.WARN)]
    t = _trial_with_report(monkeypatch, checks)  # no acceptance
    assert not t.passed
    assert t.failed_checks == ["assembly.range_of_motion"]  # never phantom-empty


def test_missing_required_check_fails_and_is_named(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # No FAIL, no unaccepted warn, but a required check is absent -> must FAIL with a
    # named reason (Codex P2: never a phantom-empty failure list).
    checks = [_check("assembly.body.hub.single_body", CheckStatus.PASS)]
    t = _trial_with_report(monkeypatch, checks, required_checks=["assembly.hardware_coaxial"])
    assert not t.passed
    assert t.failed_checks == ["missing_required:assembly.hardware_coaxial"]


def test_plain_all_pass_still_passes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    checks = [_check("assembly.hardware_coaxial", CheckStatus.PASS)]
    t = _trial_with_report(monkeypatch, checks)
    assert t.passed and t.failed_checks == [] and t.accepted_warns == []
