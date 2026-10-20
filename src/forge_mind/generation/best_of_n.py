"""Best-of-N generation with targeted auto-repair.

The pipeline is reliable per-attempt but not perfect: a single generation can drop
a requested interface (refused as ``gen_path == "infidelity"``), fail a validation
check, or decline the deterministic route and fall back to direct-DSL. This wrapper
turns one shot into up to ``n`` shots and — crucially — feeds each failure's own
diagnostics back into the NEXT prompt as a targeted repair hint, so re-prompting is
informed rather than blind resampling.

Selection rule (the heart):
    A candidate WINS iff ``result.gen_path == "layoutspec" and result.passed``.
    That single condition already encodes BOTH guarantees the pipeline offers:
      - geometric validity (``passed`` -> a Validationreport exists and passed), and
      - intent fidelity (a dropped/mismapped feature makes ``gen_path`` "infidelity"
        with ``passed`` False, so it can never satisfy the win condition).
    The FIRST winner stops the loop immediately; remaining attempts are not run.

Repair-hint derivation (priority order, most-specific first):
    1. ``result.refusal`` set  -> the fidelity/infeasible/unsupported message already
       NAMES the dropped/mismapped feature (or why infeasible). Re-attempt while
       insisting every requested feature be included exactly. We still re-attempt on
       an infeasible/unsupported refusal (a later sample may land on a feasible
       variant) but never instruct the model to fabricate.
    2. ``failed_stage == "validate"`` with a report -> hint from the failed check ids.
    3. ``fallback_reason`` set (routing declined / direct-DSL) -> ask for a clearer
       re-statement that names each interface so the router can classify it.
    4. otherwise (e.g. an upstream stage error) -> a generic "regenerate" hint.

The validator, interpreter, compiler, and every existing geometry path are untouched:
each candidate goes through ``run_pipeline(..., use_layoutspec=True)`` normally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.pipeline import PipelineResult, run_pipeline


@dataclass
class AttemptLog:
    """A per-attempt record: what was sent, what came back, and the hint (if any)
    that WILL be appended to the next attempt's prompt."""

    attempt: int
    prompt: str
    gen_path: str
    passed: bool
    refusal: str | None
    fallback_reason: str | None
    repair_hint: str | None
    cost_usd: float


@dataclass
class BestOfNResult:
    """The outcome of a best-of-N run.

    ``result`` is the winner when ``won`` is True; otherwise it is the "best" attempt
    (preferring one that at least validated geometrically, else the last attempt).
    ``winning_attempt`` is 1-based, or None if no attempt won.
    """

    result: PipelineResult
    won: bool
    attempts: list[AttemptLog] = field(default_factory=list)
    total_cost_usd: float = 0.0
    winning_attempt: int | None = None


def _is_winner(result: PipelineResult) -> bool:
    """A winning candidate: routed through the deterministic path AND validated AND
    intent-faithful. ``passed`` already requires a passing report; ``gen_path`` being
    "layoutspec" already excludes an "infidelity" refusal (a dropped/mismapped
    feature)."""
    return result.gen_path == "layoutspec" and result.passed


def _repair_hint(result: PipelineResult) -> str:
    """Derive a TARGETED repair hint from a non-winning result's own diagnostics.

    Priority is most-specific first so a named refusal beats a generic check list,
    and a check list beats a routing message. Always returns a non-empty hint; the
    caller decides whether to actually apply it (``repair`` flag) and on which
    attempts (only when one remains)."""
    if result.refusal is not None:
        return (
            f'Your previous attempt was rejected: "{result.refusal}". '
            "Regenerate and INCLUDE every requested feature exactly; "
            "do not drop or substitute any."
        )

    if result.failed_stage == "validate" and result.report is not None:
        ids = ", ".join(result.report.failed_check_ids())
        return (
            f"Your previous attempt failed validation checks: {ids}. "
            "Regenerate a version that satisfies them."
        )

    if result.fallback_reason is not None:
        return (
            "Your previous attempt did not route through the deterministic path "
            f"({result.fallback_reason}). Re-state the request clearly with each "
            "interface named."
        )

    # An upstream stage error (intent/blueprint/compile) or any other non-winning
    # state with no specific signal: a plain retry is still worth attempting.
    return "Your previous attempt did not produce a valid part. Regenerate it."


def _select_best(results: list[PipelineResult]) -> PipelineResult:
    """When no attempt won, pick the best fallback to report.

    Prefer the FIRST attempt that at least validated geometrically (``report.passed``)
    — that is closer to shippable than a refusal or stage crash — otherwise return the
    LAST attempt (the most-repaired one, which carries the freshest diagnostics)."""
    for r in results:
        if r.report is not None and r.report.passed:
            return r
    return results[-1]


def generate_best_of_n(
    prompt: str,
    output_dir: Path,
    *,
    n: int = 3,
    repair: bool = True,
) -> BestOfNResult:
    """Run the pipeline up to ``n`` times, keeping the FIRST winning candidate.

    Args:
        prompt: the original request. Attempt 1 always uses it verbatim.
        output_dir: each attempt writes under ``output_dir / f"attempt_{i}"`` (1-based).
        n: maximum number of attempts. ``n <= 0`` is treated as a single attempt.
        repair: when True, a non-winning attempt's diagnostics are turned into a hint
            that is appended to the ORIGINAL prompt for the next attempt
            (``original + "\\n\\n" + hint``). When False, every attempt re-uses the
            original prompt unchanged (pure best-of-N resampling); a hint is still
            COMPUTED and recorded for observability but never applied.

    Returns:
        ``BestOfNResult``. ``won`` True with ``winning_attempt`` set on the first win;
        otherwise ``won`` False with ``result`` = the best/last attempt (never raises).
    """
    n_attempts = max(1, n)

    attempts: list[AttemptLog] = []
    results: list[PipelineResult] = []
    total_cost = 0.0
    attempt_prompt = prompt

    for i in range(1, n_attempts + 1):
        result = run_pipeline(attempt_prompt, output_dir / f"attempt_{i}", use_layoutspec=True)
        results.append(result)
        total_cost += result.total_cost_usd

        if _is_winner(result):
            # Record the winning attempt with no forward hint, then stop early —
            # remaining attempts are deliberately not run.
            attempts.append(
                AttemptLog(
                    attempt=i,
                    prompt=attempt_prompt,
                    gen_path=result.gen_path,
                    passed=result.passed,
                    refusal=result.refusal,
                    fallback_reason=result.fallback_reason,
                    repair_hint=None,
                    cost_usd=result.total_cost_usd,
                )
            )
            return BestOfNResult(
                result=result,
                won=True,
                attempts=attempts,
                total_cost_usd=total_cost,
                winning_attempt=i,
            )

        # Non-winning: derive a hint for the NEXT attempt only if one remains.
        has_next = i < n_attempts
        hint = _repair_hint(result) if has_next else None
        attempts.append(
            AttemptLog(
                attempt=i,
                prompt=attempt_prompt,
                gen_path=result.gen_path,
                passed=result.passed,
                refusal=result.refusal,
                fallback_reason=result.fallback_reason,
                repair_hint=hint,
                cost_usd=result.total_cost_usd,
            )
        )

        if has_next:
            # repair=True  -> re-prompt with the original + targeted hint.
            # repair=False -> pure resampling: the original prompt, unchanged.
            attempt_prompt = f"{prompt}\n\n{hint}" if (repair and hint is not None) else prompt

    # No winner after all attempts.
    return BestOfNResult(
        result=_select_best(results),
        won=False,
        attempts=attempts,
        total_cost_usd=total_cost,
        winning_attempt=None,
    )


__all__ = ["AttemptLog", "BestOfNResult", "generate_best_of_n"]
