"""The lifecycle-level rubric (ADR-0048) — does problem -> package actually do its job?

The engine's ``arbitrary_v1`` benchmark measures only the CAD half (a prompt -> a valid part). The
pivot adds a concept phase BEFORE the engine, so it needs its OWN measure: a PROBLEM in -> a
coherent, honestly-labelled :class:`InventionPackage` out, with real concept divergence and the moat
gates intact. This module is the DETERMINISTIC scorer + a thin runner; PRODUCING the outputs to
score needs Fable (the deferred live run), but the scoring is pure and offline-testable, so the
rubric is frozen and ready now.

Frozen fixture: ``benchmarks/invention_v1/problems.json`` (easy/normal tiers). :func:`run_rubric` is
the testable orchestration (both seams injectable); the live runner that wires the real engine
``produce`` adapter + the Fable client is the deferred live step (it needs Fable to author parts).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.concept import ConceptPlan
from forge_mind.invention.lifecycle import GeneratedPart, run_lifecycle
from forge_mind.invention.package import InventionPackage
from forge_mind.llm.provider import LLMClientLike

# "not_assemblable" / "motion_blocked" are HONEST outcomes like "rejected": the engine refusing
# to call an unbuildable assembly, or a mechanism that cannot move, shippable. Omitting either
# would mark the strictest verdicts dishonest.
_HONEST_DISPOSITIONS = {
    "shipped",
    "shipped_with_caveats",
    "rejected",
    "no_build",
    "not_assemblable",
    "motion_blocked",
}


class LifecycleProblem(BaseModel):
    """One frozen benchmark problem: a physical PROBLEM (not a part request) + its expectations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    tier: str  # easy | normal | hard
    problem: str
    min_concepts: int = 2


class RubricCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    passed: bool
    detail: str


class RubricResult(BaseModel):
    """The per-problem verdict: the lifecycle-honesty checks + a pass/score derived from them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problem: str
    checks: list[RubricCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)


def score_lifecycle(
    problem: str,
    package: InventionPackage,
    plan: ConceptPlan,
    *,
    min_concepts: int = 2,
) -> RubricResult:
    """Deterministically score one lifecycle output against the moat + divergence criteria. Pure:
    no model call, no I/O — so the rubric is testable today and only the inputs wait for Fable."""
    checks: list[RubricCheck] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append(RubricCheck(id=check_id, passed=passed, detail=detail))

    add(
        "banner_present",
        "NOT PHYSICALLY VALIDATED" in package.physical_validation.banner,
        "the conceptual-model banner is shown (digital-only honesty)",
    )
    add(
        "feasibility_advisory",
        "not a guarantee" in package.feasibility.disclaimer,
        "feasibility carries its advisory disclaimer, never a pass",
    )
    mechanisms = {concept.primary_mechanism for concept in plan.concepts}
    add(
        "concept_divergence",
        len(plan.concepts) >= min_concepts and len(mechanisms) >= 2,
        f"{len(plan.concepts)} concepts / {len(mechanisms)} distinct mechanisms "
        f"(want >= {min_concepts} / 2)",
    )
    ids = {concept.concept_id for concept in plan.concepts}
    add(
        "recommendation_valid",
        plan.recommended_concept_id in ids,
        "the recommendation names a real concept",
    )
    add(
        "concept_in_package",
        package.concept is not None and package.concept.concept_id == plan.recommended_concept_id,
        "the chosen concept rides the package",
    )
    add(
        "disposition_honest",
        package.disposition in _HONEST_DISPOSITIONS,
        f"disposition '{package.disposition}' is one of the honest set",
    )
    add(
        "problem_carried",
        package.problem_statement == problem,
        "the user's problem rode through verbatim",
    )
    return RubricResult(problem=problem, checks=checks)


def load_problems(path: Path) -> list[LifecycleProblem]:
    """Load the frozen problem set from JSON. Deterministic; the fixture is SHA-pinned in git."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [LifecycleProblem.model_validate(p) for p in data["problems"]]


def run_rubric(
    problems: list[LifecycleProblem],
    *,
    produce: Callable[[str], GeneratedPart],
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> list[RubricResult]:
    """Run the full lifecycle on each problem and score it. Both seams (the front-door ``client``
    and the engine ``produce``) are injectable, so this is testable offline; the live bench wires
    the real engine + Fable. REPORTED — scoring gates nothing."""
    results: list[RubricResult] = []
    for problem in problems:
        package, plan = run_lifecycle(problem.problem, produce=produce, client=client, model=model)
        results.append(
            score_lifecycle(problem.problem, package, plan, min_concepts=problem.min_concepts)
        )
    return results


__all__ = [
    "LifecycleProblem",
    "RubricCheck",
    "RubricResult",
    "load_problems",
    "run_rubric",
    "score_lifecycle",
]
