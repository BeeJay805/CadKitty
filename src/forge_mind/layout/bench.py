"""LayoutSpec generation path for the A/B benchmark (ADR-0004).

One candidate = one LayoutSpecGenerator call -> interpret -> compile -> validate
-> score, sharing best-of-k's scorer and CandidateResult so the LayoutSpec path
and the direct-DSL path are scored identically. The only difference between the
two paths is the generation step; intent, blueprint, selection, compile, and
validate are shared.
"""

from __future__ import annotations

import time
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.candidates import (
    DEFAULT_WEIGHTS,
    BestOfKResult,
    CandidateResult,
    ScoreWeights,
    finalize_candidate,
    run_best_of_k,
    score_candidate,
)
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout


def evaluate_layoutspec_candidate(
    blueprint: Blueprint,
    candidate_id: int,
    candidate_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
    *,
    layoutspec_model: str | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
) -> CandidateResult:
    """Generate ONE LayoutSpec, interpret it to a PartProgram, then compile +
    validate + score via the shared finalizer. Robust: failures are captured.
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cost = 0.0

    # LayoutSpec generation (the LLM chooses sizes + constraints only).
    try:
        spec, usage = LayoutSpecGenerator(model=layoutspec_model)(blueprint)
        cost += usage.cost_usd
        (candidate_dir / "layoutspec.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            passed=False,
            compiled=False,
            score=score_candidate(
                compiled=False, report=None, op_count=0, bbox=None, envelope_cap_mm=envelope_cap_mm
            ),
            primary_label="layoutspec_invalid",
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Deterministic interpretation. An infeasible spec is rejected here (the LLM
    # chose sizes that cannot hold the interfaces with legal lands).
    try:
        program = interpret_layout(spec)
    except LayoutError as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            passed=False,
            compiled=False,
            score=score_candidate(
                compiled=False, report=None, op_count=0, bbox=None, envelope_cap_mm=envelope_cap_mm
            ),
            primary_label="layout_infeasible",
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"LayoutError: {exc}",
        )

    return finalize_candidate(
        program,
        candidate_id,
        candidate_dir,
        envelope_cap_mm,
        cost_usd=cost,
        t0=t0,
        weights=weights,
        render=render,
    )


def run_layoutspec_best_of_k(
    prompt: str,
    output_dir: Path,
    k: int,
    envelope_cap_mm: tuple[float, float, float],
    *,
    intent_model: str | None = None,
    blueprint_model: str | None = None,
    layoutspec_model: str | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
) -> BestOfKResult:
    """Shared intent + blueprint, then k LayoutSpec candidates; keep the best.

    Reuses run_best_of_k for everything except the per-candidate generator, so the
    A/B comparison differs only in the generation step.
    """

    def _evaluate(
        bp: Blueprint, cid: int, cdir: Path, env: tuple[float, float, float]
    ) -> CandidateResult:
        return evaluate_layoutspec_candidate(
            bp, cid, cdir, env, layoutspec_model=layoutspec_model, weights=weights, render=render
        )

    return run_best_of_k(
        prompt,
        output_dir,
        k,
        envelope_cap_mm,
        intent_model=intent_model,
        blueprint_model=blueprint_model,
        weights=weights,
        render=render,
        evaluate_fn=_evaluate,
    )


__all__ = ["evaluate_layoutspec_candidate", "run_layoutspec_best_of_k"]
