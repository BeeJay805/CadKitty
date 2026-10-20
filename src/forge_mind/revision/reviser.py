"""Patch-only revision loop.

Given a failing PartProgram and its ValidationReport, ask a ``propose`` callable
for a list of field patches, apply them, recompile + revalidate, and keep going
up to ``max_iters`` times. The loop is deterministic and conservative:

- It scores every candidate with the SAME deterministic scorer best-of-k uses, so
  "better" means the same thing in both stages.
- It only ever adopts a strictly-better program. If a patched program does not
  improve the score, or regresses a check that was passing, the loop stops and
  returns the best program seen so far (never worse than the input).
- ``propose`` is injectable. The no-LLM tests pass a hand-authored function; the
  LLM reviser is just one implementation of the same signature, gated behind a
  flag. The loop itself never calls an LLM.

Every iteration's artifacts (patched DSL, STL, report, PNG) are saved under
``output_dir/iter-N`` so a revision can be inspected step by step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.candidates import DEFAULT_WEIGHTS, ScoreWeights, candidate_label, score_candidate
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.llm.cost import LLMUsage
from forge_mind.revision.patch import Patch, PatchError, apply_patches
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


@dataclass
class ProposeResult:
    """What a reviser returns: the patches to try, and (if it used an LLM) usage."""

    patches: list[Patch]
    usage: LLMUsage | None = None


# A reviser is any function from (program, its report) to proposed patches.
ProposeFn = Callable[[PartProgram, ValidationReport], ProposeResult]


@dataclass
class _Evaluated:
    """Internal: the outcome of compiling+validating one program."""

    program: PartProgram
    compiled: bool
    report: ValidationReport | None
    score: float
    passed: bool
    failed_check_ids: list[str]
    label: str
    error: str | None
    partdsl_path: str | None = None
    stl_path: str | None = None
    report_path: str | None = None
    render_path: str | None = None


@dataclass
class RevisionStep:
    iteration: int
    patches: list[Patch]
    applied: bool
    compiled: bool
    passed: bool
    score: float
    failed_check_ids: list[str]
    label: str
    adopted: bool
    stop_reason: str | None
    cost_usd: float
    error: str | None = None
    partdsl_path: str | None = None
    stl_path: str | None = None
    report_path: str | None = None
    render_path: str | None = None


@dataclass
class RevisionResult:
    start_passed: bool
    end_passed: bool
    iterations: int
    start_failed_check_ids: list[str]
    end_failed_check_ids: list[str]
    start_label: str
    end_label: str
    final_program: PartProgram
    final_report: ValidationReport | None
    cost_usd: float
    steps: list[RevisionStep] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """True if revision turned a failing part into a passing one."""
        return self.end_passed and not self.start_passed


def _evaluate_program(
    program: PartProgram,
    work_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
    *,
    weights: ScoreWeights,
    render: bool,
) -> _Evaluated:
    """Compile + validate + score a program. Robust: failures are captured."""
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / "part.partdsl.json"
    p.write_text(program.model_dump_json(indent=2), encoding="utf-8")

    model = None
    compiled = False
    error: str | None = None
    try:
        model = compile_program(program, work_dir)
        compiled = True
    except Exception as exc:
        error = f"compile: {type(exc).__name__}: {exc}"

    report: ValidationReport | None = None
    report_path: str | None = None
    if compiled and model is not None:
        try:
            report = validate(model, program)
            rp = work_dir / "report.json"
            rp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            report_path = str(rp)
        except Exception as exc:
            error = f"validate: {type(exc).__name__}: {exc}"

    render_path: str | None = None
    if render and model is not None:
        try:
            from forge_mind.viz.render import render_part_to_png

            png = render_part_to_png(model.stl_path, work_dir / "part.png", title="revision")
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
    return _Evaluated(
        program=program,
        compiled=compiled,
        report=report,
        score=score,
        passed=passed,
        failed_check_ids=report.failed_check_ids() if report is not None else [],
        label=candidate_label(passed, compiled, report),
        error=error,
        partdsl_path=str(p),
        stl_path=str(model.stl_path) if model is not None else None,
        report_path=report_path,
        render_path=render_path,
    )


def _regresses(prev_failed: set[str], new_failed: set[str]) -> bool:
    """True if a check that was passing before is now failing (anti-thrashing)."""
    return bool(new_failed - prev_failed)


def revise(
    program: PartProgram,
    report: ValidationReport,
    propose: ProposeFn,
    *,
    output_dir: Path,
    envelope_cap_mm: tuple[float, float, float],
    max_iters: int = 3,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
    render: bool = True,
) -> RevisionResult:
    """Iteratively patch ``program`` toward passing. Returns the best result seen.

    Never returns a program worse than the input: the loop only adopts strictly
    improving, non-regressing patches and otherwise stops.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    start_failed = report.failed_check_ids()
    start_label = candidate_label(report.passed, True, report)
    # Score the starting point so we only ever move uphill from it.
    start_eval = _evaluate_program(
        program, output_dir / "iter-0-start", envelope_cap_mm, weights=weights, render=False
    )

    best = start_eval
    current = start_eval
    steps: list[RevisionStep] = []
    cost = 0.0

    for it in range(1, max_iters + 1):
        if current.report is None or current.report.passed:
            break

        proposal = propose(current.program, current.report)
        if proposal.usage is not None:
            cost += proposal.usage.cost_usd
        step_cost = proposal.usage.cost_usd if proposal.usage is not None else 0.0

        if not proposal.patches:
            steps.append(
                _make_step(
                    it,
                    [],
                    applied=False,
                    ev=None,
                    adopted=False,
                    cost=step_cost,
                    stop_reason="no patches proposed",
                    error=None,
                )
            )
            break

        try:
            patched = apply_patches(current.program, proposal.patches)
        except PatchError as exc:
            steps.append(
                _make_step(
                    it,
                    proposal.patches,
                    applied=False,
                    ev=None,
                    adopted=False,
                    cost=step_cost,
                    stop_reason="patch rejected",
                    error=str(exc),
                )
            )
            break

        ev = _evaluate_program(
            patched, output_dir / f"iter-{it}", envelope_cap_mm, weights=weights, render=render
        )

        # A state we could not validate (compile or validator crash -> no report)
        # is never an improvement: its score lacks the failing-report penalties, so
        # it would otherwise look better than a clean-but-failing part. Refuse to
        # adopt it and never treat it as the best seen.
        validatable = ev.report is not None
        regressed = _regresses(set(current.failed_check_ids), set(ev.failed_check_ids))
        improved = ev.score > current.score
        adopt = ev.passed or (validatable and improved and not regressed)

        stop_reason: str | None = None
        if not adopt:
            if not validatable:
                stop_reason = "patched program did not validate"
            elif regressed:
                stop_reason = "regression"
            else:
                stop_reason = "no score improvement"

        steps.append(
            _make_step(
                it,
                proposal.patches,
                applied=True,
                ev=ev,
                adopted=adopt,
                cost=step_cost,
                stop_reason=stop_reason,
                error=ev.error,
            )
        )

        if validatable and ev.score > best.score:
            best = ev
        if not adopt:
            break
        current = ev
        if ev.passed:
            break

    return RevisionResult(
        start_passed=start_eval.passed,
        end_passed=best.passed,
        iterations=len([s for s in steps if s.applied]),
        start_failed_check_ids=start_failed,
        end_failed_check_ids=best.failed_check_ids,
        start_label=start_label,
        end_label=best.label,
        final_program=best.program,
        final_report=best.report,
        cost_usd=cost,
        steps=steps,
    )


def _make_step(
    iteration: int,
    patches: list[Patch],
    *,
    applied: bool,
    ev: _Evaluated | None,
    adopted: bool,
    cost: float,
    stop_reason: str | None,
    error: str | None,
) -> RevisionStep:
    return RevisionStep(
        iteration=iteration,
        patches=patches,
        applied=applied,
        compiled=ev.compiled if ev is not None else False,
        passed=ev.passed if ev is not None else False,
        score=ev.score if ev is not None else 0.0,
        failed_check_ids=ev.failed_check_ids if ev is not None else [],
        label=ev.label if ev is not None else "not_applied",
        adopted=adopted,
        stop_reason=stop_reason,
        cost_usd=cost,
        error=error,
        partdsl_path=ev.partdsl_path if ev is not None else None,
        stl_path=ev.stl_path if ev is not None else None,
        report_path=ev.report_path if ev is not None else None,
        render_path=ev.render_path if ev is not None else None,
    )


__all__ = [
    "ProposeFn",
    "ProposeResult",
    "RevisionResult",
    "RevisionStep",
    "revise",
]
