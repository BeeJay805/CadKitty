"""Invention benchmark runner (ADR-0048, P7) — the live-run harness, completed without calling Fable.

The engine's ``arbitrary_v1`` measures only the CAD half. This bench measures the WHOLE lifecycle
(problem -> package): three named scores (concept diversity, package completeness, feasibility
honesty), the rubric pass, a cost field, and FAILURE ATTRIBUTION to the first failing stage. Two modes:

  - ``offline`` — deterministic fixtures + MOCKED model outputs (built-in stubs). Always available,
    no spend, fully reproducible. This is what proves the harness works while Fable is down.
  - ``live`` — the real front door + engine. It first asserts the model is available (the preflight)
    and FAILS CLEARLY (:class:`LiveModelUnavailableError`) if it is not. No fake fallback is ever produced.

Baselines are preserved (:func:`freeze_baseline` refuses to overwrite) and runs resume/replay by
problem id, so a tuning loop never clobbers the reference numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.concept import ConceptBrief, ConceptPlan, Mechanism
from forge_mind.invention.labels import FeasibilityVerdict
from forge_mind.invention.lifecycle import GeneratedPart, run_lifecycle
from forge_mind.invention.package import InventionPackage
from forge_mind.invention.rubric import LifecycleProblem, load_problems, score_lifecycle
from forge_mind.invention.toolbelt import run_preflight
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

# Map a failing rubric check to the lifecycle STAGE it belongs to (failure attribution).
_STAGE_BY_CHECK: dict[str, str] = {
    "problem_carried": "intake",
    "feasibility_advisory": "feasibility",
    "concept_divergence": "concept",
    "recommendation_valid": "concept",
    "concept_in_package": "package",
    "disposition_honest": "validation",
    "banner_present": "package",
}

_REQUIRED_SECTIONS = (
    "banner",
    "feasibility",
    "concept",
    "bom_summary",
    "disposition",
    "confidence",
)


class BenchScores(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rubric_passed: bool
    concept_diversity: float  # distinct mechanisms / concepts (0..1)
    package_completeness: float  # fraction of required sections present (0..1)
    feasibility_honesty: float  # 1.0 iff feasibility is advisory (disclaimer present, no pass)


def score_bench(
    package: InventionPackage, plan: ConceptPlan, *, problem: str, min_concepts: int = 2
) -> BenchScores:
    """Deterministically score one lifecycle output on the three bench axes + the rubric pass.

    ``problem`` is the BENCHMARK's problem statement (the reference the lifecycle had to carry). It
    must come from the bench problem, NOT from ``package.problem_statement``: passing the package's
    own statement made the rubric's ``problem_carried`` check (``package.problem_statement == problem``)
    a tautology that always passed (audit M9, ADR-0065)."""
    rubric = score_lifecycle(problem, package, plan, min_concepts=min_concepts)
    mechanisms = {c.primary_mechanism for c in plan.concepts}
    diversity = len(mechanisms) / len(plan.concepts) if plan.concepts else 0.0
    present = {
        "banner": bool(package.physical_validation.banner),
        "feasibility": bool(package.feasibility.disclaimer),
        "concept": package.concept is not None,
        "bom_summary": bool(package.bom_summary),
        "disposition": bool(package.disposition),
        "confidence": package.confidence is not None,
    }
    completeness = sum(1 for s in _REQUIRED_SECTIONS if present[s]) / len(_REQUIRED_SECTIONS)
    honest = (
        "not a guarantee" in package.feasibility.disclaimer
        and package.feasibility.verdict in set(FeasibilityVerdict)
    )
    return BenchScores(
        rubric_passed=rubric.passed,
        concept_diversity=round(diversity, 3),
        package_completeness=round(completeness, 3),
        feasibility_honesty=1.0 if honest else 0.0,
    )


def _failure_stage(
    package: InventionPackage, plan: ConceptPlan, *, problem: str, min_concepts: int = 2
) -> str | None:
    """The first failing rubric check, mapped to its lifecycle stage. None when everything passes.

    ``problem`` is the benchmark reference (see ``score_bench``); passing the package's own statement
    made ``problem_carried`` vacuous (audit M9, ADR-0065)."""
    rubric = score_lifecycle(problem, package, plan, min_concepts=min_concepts)
    for check in rubric.checks:
        if not check.passed:
            return _STAGE_BY_CHECK.get(check.id, check.id)
    return None


class BenchEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    problem_id: str
    tier: str
    scores: BenchScores
    cost_usd: float = 0.0
    failure_stage: str | None = None


class BenchRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["offline", "live"]
    model: str | None = None
    entries: list[BenchEntry] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(1 for e in self.entries if e.scores.rubric_passed) / len(self.entries)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.entries), 4)


class LiveModelUnavailableError(RuntimeError):
    """Raised when a LIVE bench is requested but no model is available -- fail clearly, never fake."""


def assert_live_model_available() -> None:
    """The live guard: a model must be configured + available, else raise. No fake fallback."""
    report = run_preflight()
    if report.live_concept_generation != "available":
        raise LiveModelUnavailableError(
            "live invention-bench needs a configured model; the preflight reports it blocked. "
            + " ".join(report.notes)
        )


def run_bench(
    problems: list[LifecycleProblem],
    *,
    client: LLMClientLike,
    produce: Callable[[str], GeneratedPart],
    mode: Literal["offline", "live"],
    model: str | None = None,
    cost_per_problem: float = 0.0,
    resume_from: BenchRun | None = None,
) -> BenchRun:
    """Run the bench over the problems, scoring each. ``client`` (front door) + ``produce`` (engine)
    are injectable: offline passes deterministic stubs, live passes the real seams. Resume skips any
    problem already present in ``resume_from`` (so a baseline is never recomputed)."""
    done = {e.problem_id for e in resume_from.entries} if resume_from is not None else set()
    entries = list(resume_from.entries) if resume_from is not None else []
    for problem in problems:
        if problem.id in done:
            continue
        package, plan = run_lifecycle(problem.problem, produce=produce, client=client, model=model)
        entries.append(
            BenchEntry(
                problem_id=problem.id,
                tier=problem.tier,
                scores=score_bench(
                    package, plan, problem=problem.problem, min_concepts=problem.min_concepts
                ),
                cost_usd=cost_per_problem,
                failure_stage=_failure_stage(
                    package, plan, problem=problem.problem, min_concepts=problem.min_concepts
                ),
            )
        )
    return BenchRun(mode=mode, model=model, entries=entries)


# --- persistence: baseline preservation + resume ---------------------------------------------


def save_run(run: BenchRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def load_run(path: Path) -> BenchRun:
    return BenchRun.model_validate_json(path.read_text(encoding="utf-8"))


def freeze_baseline(run: BenchRun, baseline_path: Path) -> None:
    """Write the reference baseline, REFUSING to overwrite an existing one (preserve the baseline)."""
    if baseline_path.exists():
        raise FileExistsError(
            f"baseline {baseline_path} already exists -- not overwriting (preserve the baseline; "
            "save tuning runs to a separate results file)"
        )
    save_run(run, baseline_path)


# --- built-in deterministic offline stubs (mocked model outputs) ------------------------------


def _default_fixture() -> Path:
    return Path(__file__).parents[3] / "benchmarks" / "invention_v1" / "problems.json"


def default_problems() -> list[LifecycleProblem]:
    """The frozen invention_v1 problem set (the bench's default)."""
    return load_problems(_default_fixture())


class _OfflineFrontDoor:
    """A deterministic front-door client: two distinct concepts per problem (full diversity)."""

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[Any],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[Any, LLMUsage]:
        plan = ConceptPlan(
            problem_restated=f"first-principles restatement of: {user[:60]}",
            feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
            feasibility_note="a routine FDM build for a typical maker",
            concepts=[
                ConceptBrief(
                    concept_id="c1",
                    name="compliant clip",
                    primary_mechanism=Mechanism.SNAP_FIT,
                    approach="a flexure that snaps over the edge",
                    obvious_baseline="a screwed bracket",
                    difference_from_obvious="no fasteners",
                ),
                ConceptBrief(
                    concept_id="c2",
                    name="friction mount",
                    primary_mechanism=Mechanism.FRICTION_FIT,
                    approach="a weighted body held by friction",
                    obvious_baseline="adhesive",
                    difference_from_obvious="removable",
                ),
            ],
            recommended_concept_id="c1",
            recommendation_reason="simplest and needs no hardware",
        )
        return plan, LLMUsage(model="offline-stub", input_tokens=0, output_tokens=0)

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[Any],
        model: str,
        max_tokens: int,
    ) -> tuple[Any, LLMUsage]:
        raise AssertionError("the offline bench does not use images")


def _offline_produce(prompt: str) -> GeneratedPart:
    from forge_mind.agentic.confidence import standard_lane_label
    from forge_mind.dsl.primitives import Box, OpMode
    from forge_mind.dsl.program import Constraints, Metadata, PartProgram
    from forge_mind.kit import build_kit
    from forge_mind.types import Vec3

    program = PartProgram(
        metadata=Metadata(
            name="bench-part", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return GeneratedPart(
        part_name="bench-part",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(1),
        artifacts={},
    )


def offline_bench(
    problems: list[LifecycleProblem] | None = None, *, fixtures_path: Path | None = None
) -> BenchRun:
    """Run the bench fully OFFLINE: deterministic fixtures + built-in mocked model outputs. No spend,
    no model, fully reproducible -- the harness that works while Fable is down."""
    probs = problems if problems is not None else load_problems(fixtures_path or _default_fixture())
    return run_bench(probs, client=_OfflineFrontDoor(), produce=_offline_produce, mode="offline")


def run_to_dict(run: BenchRun) -> dict[str, object]:
    """A compact summary dict for the CLI (pass rate + the mean of each score axis)."""
    n = len(run.entries) or 1
    return {
        "mode": run.mode,
        "model": run.model,
        "problems": len(run.entries),
        "pass_rate": round(run.pass_rate, 3),
        "mean_concept_diversity": round(
            sum(e.scores.concept_diversity for e in run.entries) / n, 3
        ),
        "mean_package_completeness": round(
            sum(e.scores.package_completeness for e in run.entries) / n, 3
        ),
        "mean_feasibility_honesty": round(
            sum(e.scores.feasibility_honesty for e in run.entries) / n, 3
        ),
        "total_cost_usd": run.total_cost_usd,
        "failures": {e.problem_id: e.failure_stage for e in run.entries if e.failure_stage},
    }


__all__ = [
    "BenchEntry",
    "BenchRun",
    "BenchScores",
    "LiveModelUnavailableError",
    "assert_live_model_available",
    "default_problems",
    "freeze_baseline",
    "load_run",
    "offline_bench",
    "run_bench",
    "run_to_dict",
    "save_run",
    "score_bench",
]
