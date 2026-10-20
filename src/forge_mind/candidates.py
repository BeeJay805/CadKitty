"""Best-of-k DSL candidate generation + deterministic scoring.

The bench-001 failure dataset showed the DSL stage produces a valid part only
~10% of the time, with thin-wall and overhang body-layout defects dominating and
interfaces almost always correct. That profile is ideal for best-of-k: generate
k independent DSL candidates for the same blueprint, score them deterministically,
and keep the best. No revision -- just sampling + selection.

Scoring is tiered so the hard priorities always hold regardless of penalties:

    validator pass  >>  compile success  >  interface pass  >  (penalties)

The tier gaps exceed the maximum possible penalty sum, so penalties only ever
re-order candidates *within* a tier (e.g. two compiled, interface-passing
candidates are separated by how thin/steep/oversized they are).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.analysis.taxonomy import FailureLabel, label_for_check, primary_failed_check
from forge_mind.blueprint.schema import Blueprint
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.generator import DSLGenerator
from forge_mind.dsl.program import PartProgram
from forge_mind.types import BBox
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


@dataclass(frozen=True)
class ScoreWeights:
    """Scoring weights. Tier bonuses are separated by more than the max penalty
    sum, so the hard priorities (pass > compile > interface) are inviolable.
    """

    # Hard-priority tier bonuses.
    pass_bonus: float = 1_000_000.0
    compile_bonus: float = 10_000.0
    interface_bonus: float = 1_000.0
    # Penalties (max realistic sum stays under interface_bonus).
    wall_penalty: float = 250.0  # features.min_wall_thickness
    overhang_penalty: float = 250.0  # printability.overhang_max_angle
    bbox_penalty: float = 80.0  # mesh.bbox_within_max
    compile_fail_penalty: float = 150.0  # dsl_compile_failed
    op_count_soft_limit: int = 14
    op_count_penalty_per_op: float = 2.0
    bbox_drift_penalty_per_mm: float = 1.0  # exceedance beyond the envelope


DEFAULT_WEIGHTS = ScoreWeights()


def _bbox_exceedance_mm(bbox: BBox, envelope_cap_mm: tuple[float, float, float]) -> float:
    """Total mm by which the part's (sorted) dimensions exceed the (sorted)
    envelope. Zero if the part fits under some orientation.
    """
    size = bbox.size
    part = sorted((size.x, size.y, size.z))
    cap = sorted(envelope_cap_mm)
    return sum(max(0.0, p - c) for p, c in zip(part, cap, strict=True))


def score_candidate(
    *,
    compiled: bool,
    report: ValidationReport | None,
    op_count: int,
    bbox: BBox | None,
    envelope_cap_mm: tuple[float, float, float],
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """Deterministic score; higher is better. Pure function of its inputs."""
    score = 0.0

    if compiled:
        score += weights.compile_bonus
    else:
        score -= weights.compile_fail_penalty

    if report is not None:
        failed = set(report.failed_check_ids())
        interface_failed = any(cid.startswith("interfaces.") for cid in failed)
        if not interface_failed:
            score += weights.interface_bonus
        if report.passed:
            score += weights.pass_bonus
        if "features.min_wall_thickness" in failed:
            score -= weights.wall_penalty
        if "printability.overhang_max_angle" in failed:
            score -= weights.overhang_penalty
        if "mesh.bbox_within_max" in failed:
            score -= weights.bbox_penalty

    # Small structural penalties.
    over = max(0, op_count - weights.op_count_soft_limit)
    score -= over * weights.op_count_penalty_per_op
    if bbox is not None:
        score -= _bbox_exceedance_mm(bbox, envelope_cap_mm) * weights.bbox_drift_penalty_per_mm

    return score


def candidate_label(passed: bool, compiled: bool, report: ValidationReport | None) -> str:
    """A lightweight label for one candidate (intent/blueprint are shared and
    assumed correct, so we only classify the DSL/compile/validate outcome).
    """
    if passed:
        return FailureLabel.PASSED.value
    if not compiled:
        return FailureLabel.DSL_COMPILE_FAILED.value
    if report is not None:
        check = primary_failed_check(report)
        if check is not None:
            return label_for_check(check.id).value
    return FailureLabel.UNKNOWN.value


@dataclass
class CandidateResult:
    candidate_id: int
    passed: bool
    compiled: bool
    score: float
    primary_label: str
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    partdsl_path: str | None = None
    stl_path: str | None = None
    step_path: str | None = None
    report_path: str | None = None
    render_path: str | None = None
    error: str | None = None


@dataclass
class BestOfKResult:
    prompt: str
    k: int
    candidates: list[CandidateResult] = field(default_factory=list)
    winner_id: int | None = None
    intent_ok: bool = False
    blueprint_ok: bool = False
    error: str | None = None
    shared_cost_usd: float = 0.0  # intent + blueprint
    shared_duration_ms: float = 0.0

    @property
    def first_candidate_passed(self) -> bool:
        return bool(self.candidates) and self.candidates[0].passed

    @property
    def best_passed(self) -> bool:
        if self.winner_id is None:
            return False
        return self.candidates[self.winner_id].passed

    @property
    def total_cost_usd(self) -> float:
        return self.shared_cost_usd + sum(c.cost_usd for c in self.candidates)

    @property
    def total_duration_ms(self) -> float:
        return self.shared_duration_ms + sum(c.duration_ms for c in self.candidates)


def pick_best(candidates: list[CandidateResult]) -> int | None:
    """Index of the highest-scoring candidate; ties broken by lowest id."""
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c.score, -c.candidate_id))
    return best.candidate_id


def evaluate_candidate(
    blueprint: Blueprint,
    candidate_id: int,
    candidate_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
    *,
    dsl_model: str | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
) -> CandidateResult:
    """Generate ONE DSL candidate, compile + validate it, score it, and save its
    artifacts. Robust: any failure is captured into the result, never raised.
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cost = 0.0

    # DSL generation.
    try:
        program, usage = DSLGenerator(model=dsl_model)(blueprint)
        cost += usage.cost_usd
    except Exception as exc:
        return CandidateResult(
            candidate_id=candidate_id,
            passed=False,
            compiled=False,
            score=score_candidate(
                compiled=False, report=None, op_count=0, bbox=None, envelope_cap_mm=envelope_cap_mm
            ),
            primary_label=FailureLabel.DSL_SCHEMA_INVALID.value,
            failed_check_ids=[],
            cost_usd=cost,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
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


def finalize_candidate(
    program: PartProgram,
    candidate_id: int,
    candidate_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
    *,
    cost_usd: float,
    t0: float,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
) -> CandidateResult:
    """Compile + validate + score an already-generated PartProgram, save its
    artifacts, and build the CandidateResult. Shared by every generation path
    (direct DSL and LayoutSpec) so candidates are scored identically. Robust: any
    compile/validate/render failure is captured, never raised.
    """
    p = candidate_dir / "part.partdsl.json"
    p.write_text(program.model_dump_json(indent=2), encoding="utf-8")
    partdsl_path = str(p)
    stl_path: str | None = None
    step_path: str | None = None
    report_path: str | None = None
    render_path: str | None = None

    model = None
    compiled = False
    compile_error: str | None = None
    try:
        model = compile_program(program, candidate_dir)
        compiled = True
        stl_path = str(model.stl_path)
        step_path = str(model.step_path)
    except Exception as exc:
        compile_error = f"{type(exc).__name__}: {exc}"

    report: ValidationReport | None = None
    if compiled and model is not None:
        try:
            report = validate(model, program)
            rp = candidate_dir / "report.json"
            rp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            report_path = str(rp)
        except Exception as exc:
            compile_error = f"validate: {type(exc).__name__}: {exc}"

    if render and model is not None:
        try:
            from forge_mind.viz.render import render_part_to_png

            png = render_part_to_png(
                model.stl_path, candidate_dir / "part.png", title=f"candidate {candidate_id}"
            )
            render_path = str(png)
        except Exception:
            pass

    bbox = model.bounding_box if model is not None else None
    score = score_candidate(
        compiled=compiled,
        report=report,
        op_count=len(program.operations),
        bbox=bbox,
        envelope_cap_mm=envelope_cap_mm,
        weights=weights,
    )
    passed = report.passed if report is not None else False
    failed_ids = report.failed_check_ids() if report is not None else []

    return CandidateResult(
        candidate_id=candidate_id,
        passed=passed,
        compiled=compiled,
        score=score,
        primary_label=candidate_label(passed, compiled, report),
        failed_check_ids=failed_ids,
        cost_usd=cost_usd,
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        partdsl_path=partdsl_path,
        stl_path=stl_path,
        step_path=step_path,
        report_path=report_path,
        render_path=render_path,
        error=compile_error,
    )


# A candidate evaluator: (blueprint, candidate_id, candidate_dir, envelope) -> result.
CandidateEvaluator = Callable[[Blueprint, int, Path, tuple[float, float, float]], CandidateResult]


def run_best_of_k(
    prompt: str,
    output_dir: Path,
    k: int,
    envelope_cap_mm: tuple[float, float, float],
    *,
    intent_model: str | None = None,
    blueprint_model: str | None = None,
    dsl_model: str | None = None,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
    evaluate_fn: CandidateEvaluator | None = None,
) -> BestOfKResult:
    """Shared intent + blueprint, then k independent candidates; keep the best.

    ``evaluate_fn`` is the per-candidate generator+scorer. It defaults to the
    direct-DSL path; the LayoutSpec A/B path passes its own evaluator so intent +
    blueprint + selection are identical and only the generation step differs.
    Saves EVERY candidate's artifacts (not just the winner).
    """
    if evaluate_fn is None:

        def evaluate_fn(
            bp: Blueprint, cid: int, cdir: Path, env: tuple[float, float, float]
        ) -> CandidateResult:
            return evaluate_candidate(
                bp, cid, cdir, env, dsl_model=dsl_model, weights=weights, render=render
            )

    # Imported here to keep the module import-light and avoid any cycle.
    from forge_mind.blueprint.generator import BlueprintGenerator
    from forge_mind.intent.parser import IntentParser

    result = BestOfKResult(prompt=prompt, k=k)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        intent, u = IntentParser(model=intent_model)(prompt)
        result.intent_ok = True
        result.shared_cost_usd += u.cost_usd
        (output_dir / "intent.json").write_text(intent.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:
        result.error = f"intent: {type(exc).__name__}: {exc}"
        result.shared_duration_ms = (time.perf_counter() - t0) * 1000.0
        return result

    try:
        blueprint, u = BlueprintGenerator(model=blueprint_model)(intent)
        result.blueprint_ok = True
        result.shared_cost_usd += u.cost_usd
        (output_dir / "blueprint.json").write_text(
            blueprint.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as exc:
        result.error = f"blueprint: {type(exc).__name__}: {exc}"
        result.shared_duration_ms = (time.perf_counter() - t0) * 1000.0
        return result

    result.shared_duration_ms = (time.perf_counter() - t0) * 1000.0

    for i in range(k):
        cand = evaluate_fn(blueprint, i, output_dir / f"candidate-{i}", envelope_cap_mm)
        result.candidates.append(cand)

    result.winner_id = pick_best(result.candidates)
    return result


__all__ = [
    "DEFAULT_WEIGHTS",
    "BestOfKResult",
    "CandidateEvaluator",
    "CandidateResult",
    "ScoreWeights",
    "candidate_label",
    "evaluate_candidate",
    "finalize_candidate",
    "pick_best",
    "run_best_of_k",
    "score_candidate",
]
