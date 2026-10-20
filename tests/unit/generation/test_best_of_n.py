"""Offline tests for best-of-N generation + auto-repair.

NO real LLM / paid calls: ``run_pipeline`` is monkeypatched on the best_of_n module
with a fake that returns a scripted sequence of real ``PipelineResult`` objects and
captures the exact prompt each attempt was called with. We assert the selection and
repair LOGIC, not geometry.

Constructing results:
  - ``PipelineResult.passed`` is a property (report exists AND report.passed), so a
    WIN is simulated by attaching a minimal PASS ``ValidationReport`` built from a
    single passing ``Check`` via ``build_report``.
  - ``PipelineResult.total_cost_usd`` is a property summed from ``usages``, so cost is
    set by attaching one ``LLMUsage`` with the desired ``cost_usd``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.generation import best_of_n as bon
from forge_mind.generation import generate_best_of_n
from forge_mind.llm.cost import LLMUsage
from forge_mind.pipeline import PipelineResult
from forge_mind.validator.report import (
    Check,
    CheckStatus,
    Tier,
    ValidationReport,
    build_report,
)


def _usage(cost: float) -> LLMUsage:
    return LLMUsage(
        model="fake",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=cost,
    )


def _passing_report() -> ValidationReport:
    check = Check(
        id="mesh.watertight",
        tier=Tier.MESH,
        status=CheckStatus.PASS,
        message="ok",
    )
    return build_report([check], [])


def _failing_report(*check_ids: str) -> ValidationReport:
    checks = [
        Check(id=cid, tier=Tier.FEATURES, status=CheckStatus.FAIL, message="nope")
        for cid in check_ids
    ]
    return build_report(checks, [])


def _winner(cost: float = 0.10) -> PipelineResult:
    return PipelineResult(
        prompt="ignored",
        gen_path="layoutspec",
        report=_passing_report(),
        usages=[_usage(cost)],
    )


def _refusal(message: str, cost: float = 0.05) -> PipelineResult:
    return PipelineResult(
        prompt="ignored",
        gen_path="infidelity",
        refusal=message,
        usages=[_usage(cost)],
    )


def _validate_fail(*check_ids: str, cost: float = 0.05) -> PipelineResult:
    return PipelineResult(
        prompt="ignored",
        gen_path="layoutspec",
        failed_stage="validate",
        report=_failing_report(*check_ids),
        usages=[_usage(cost)],
    )


def _fallback(reason: str, cost: float = 0.05) -> PipelineResult:
    return PipelineResult(
        prompt="ignored",
        gen_path="direct-dsl",
        fallback_reason=reason,
        usages=[_usage(cost)],
    )


class _FakePipeline:
    """Returns a scripted sequence of results and records every (prompt, output_dir)."""

    def __init__(self, results: list[PipelineResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, Path]] = []

    def __call__(
        self, prompt: str, output_dir: Path, *, use_layoutspec: bool = False
    ) -> PipelineResult:
        assert use_layoutspec is True, "best-of-N must always request the deterministic path"
        self.calls.append((prompt, output_dir))
        return self._results[len(self.calls) - 1]


def _install(monkeypatch: pytest.MonkeyPatch, results: list[PipelineResult]) -> _FakePipeline:
    fake = _FakePipeline(results)
    monkeypatch.setattr(bon, "run_pipeline", fake)
    return fake


# ---------------------------------------------------------------------------
# Early-stop on the first winner
# ---------------------------------------------------------------------------


def test_keeps_first_winning_candidate_and_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Attempt 1 fails, attempt 2 wins; attempt 3 would also win but must NOT run.
    fake = _install(monkeypatch, [_refusal("dropped insert"), _winner(), _winner()])

    out = generate_best_of_n("make a bracket", tmp_path, n=3)

    assert out.won is True
    assert out.winning_attempt == 2
    assert out.result.gen_path == "layoutspec" and out.result.passed
    assert len(fake.calls) == 2, "third attempt must not run after an early win"
    assert len(out.attempts) == 2
    # The winning attempt carries no forward hint.
    assert out.attempts[-1].repair_hint is None
    assert out.attempts[-1].passed is True


def test_attempt_1_uses_raw_prompt_and_outputs_to_attempt_1_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(monkeypatch, [_winner()])

    generate_best_of_n("raw prompt", tmp_path, n=3)

    prompt0, out0 = fake.calls[0]
    assert prompt0 == "raw prompt"
    assert out0 == tmp_path / "attempt_1"


# ---------------------------------------------------------------------------
# Targeted repair-hint derivation
# ---------------------------------------------------------------------------


def test_refusal_drives_next_prompt_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refusal_msg = "You asked for heat-set insert(s), but I couldn't place that."
    fake = _install(monkeypatch, [_refusal(refusal_msg), _winner()])

    out = generate_best_of_n("plate with insert", tmp_path, n=2)

    # The SECOND attempt's prompt is original + hint, and the hint quotes the refusal.
    second_prompt = fake.calls[1][0]
    assert second_prompt.startswith("plate with insert\n\n")
    assert refusal_msg in second_prompt
    assert "INCLUDE every requested feature exactly" in second_prompt
    # And the first attempt logged that exact forward hint.
    assert out.attempts[0].repair_hint is not None
    assert refusal_msg in out.attempts[0].repair_hint


def test_validate_failure_hint_lists_failed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        monkeypatch,
        [_validate_fail("features.bore_diameter", "interfaces.bolt_pattern"), _winner()],
    )

    generate_best_of_n("a mount", tmp_path, n=2)

    second_prompt = fake.calls[1][0]
    assert "failed validation checks" in second_prompt
    assert "features.bore_diameter" in second_prompt
    assert "interfaces.bolt_pattern" in second_prompt


def test_fallback_reason_drives_routing_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(monkeypatch, [_fallback("LLMError: router declined"), _winner()])

    generate_best_of_n("an adapter", tmp_path, n=2)

    second_prompt = fake.calls[1][0]
    assert "did not route through the deterministic path" in second_prompt
    assert "LLMError: router declined" in second_prompt
    assert "interface named" in second_prompt


# ---------------------------------------------------------------------------
# repair=False -> pure resampling (original prompt unchanged)
# ---------------------------------------------------------------------------


def test_repair_false_reuses_original_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install(
        monkeypatch,
        [_refusal("dropped a feature"), _refusal("dropped again"), _winner()],
    )

    out = generate_best_of_n("original request", tmp_path, n=3, repair=False)

    # Every attempt sees the IDENTICAL original prompt.
    assert [c[0] for c in fake.calls] == ["original request"] * 3
    assert out.won is True
    # A hint is still COMPUTED and recorded for observability, just not applied.
    assert out.attempts[0].repair_hint is not None


# ---------------------------------------------------------------------------
# All-fail after N
# ---------------------------------------------------------------------------


def test_all_fail_returns_geometrically_valid_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None win (none are gen_path==layoutspec AND passed), but attempt 2 at least
    # validated geometrically (a passing report). It must be selected over the
    # refusal and the last attempt.
    geom_ok = PipelineResult(
        prompt="ignored",
        gen_path="infidelity",  # refused for fidelity, yet carries a passing report
        refusal="mismapped a feature",
        report=_passing_report(),
        usages=[_usage(0.05)],
    )
    fake = _install(
        monkeypatch,
        [_refusal("dropped"), geom_ok, _refusal("dropped again")],
    )

    out = generate_best_of_n("hard part", tmp_path, n=3)

    assert out.won is False
    assert out.winning_attempt is None
    assert out.result is geom_ok  # the geometrically-valid one, not the last
    assert len(fake.calls) == 3


def test_all_fail_no_valid_returns_last_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = _refusal("still dropping", cost=0.05)
    fake = _install(monkeypatch, [_refusal("dropped"), _refusal("dropped"), last])

    out = generate_best_of_n("hard part", tmp_path, n=3)

    assert out.won is False
    assert out.result is last  # no attempt validated -> the last one
    # The final attempt has no forward hint (no attempt remained).
    assert out.attempts[-1].repair_hint is None
    assert len(fake.calls) == 3


def test_all_fail_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [_fallback("x"), _fallback("y")])
    # Simply reaching the assertion without an exception is the contract.
    out = generate_best_of_n("p", tmp_path, n=2)
    assert out.won is False


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


def test_total_cost_is_summed_across_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        [_refusal("a", cost=0.03), _refusal("b", cost=0.04), _winner(cost=0.10)],
    )

    out = generate_best_of_n("p", tmp_path, n=3)

    assert out.total_cost_usd == pytest.approx(0.17)
    assert [a.cost_usd for a in out.attempts] == pytest.approx([0.03, 0.04, 0.10])


def test_cost_stops_summing_after_early_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The would-be third attempt is expensive but never runs, so it never counts.
    _install(monkeypatch, [_refusal("a", cost=0.03), _winner(cost=0.10), _winner(cost=99.0)])

    out = generate_best_of_n("p", tmp_path, n=3)

    assert out.total_cost_usd == pytest.approx(0.13)
