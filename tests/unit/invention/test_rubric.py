"""The lifecycle-level rubric (ADR-0048): a deterministic problem->package scorer, testable today
even though producing the outputs to score needs Fable. A good lifecycle passes; a degenerate one
(no concept divergence) fails — so the measure can actually discriminate when Fable returns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    GeneratedPart,
    Mechanism,
    build_invention_package,
)
from forge_mind.invention.package import InventionPackage
from forge_mind.invention.rubric import (
    load_problems,
    run_rubric,
    score_lifecycle,
)
from forge_mind.kit import build_kit
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)
_FIXTURE = Path(__file__).parents[3] / "benchmarks" / "invention_v1" / "problems.json"


def _concepts(distinct: bool = True) -> list[ConceptBrief]:
    second = Mechanism.FRICTION_FIT if distinct else Mechanism.SNAP_FIT
    return [
        ConceptBrief(
            concept_id="c1",
            name="cantilever C-clip",
            primary_mechanism=Mechanism.SNAP_FIT,
            approach="a C-shaped jaw that flexes over the desk edge",
            obvious_baseline="a screwed-on bracket",
            difference_from_obvious="no fasteners",
        ),
        ConceptBrief(
            concept_id="c2",
            name="friction saddle",
            primary_mechanism=second,
            approach="a weighted saddle resting on the edge",
            obvious_baseline="an adhesive clip",
            difference_from_obvious="removable",
        ),
    ]


def _plan(concepts: list[ConceptBrief]) -> ConceptPlan:
    return ConceptPlan(
        problem_restated="route two flexible conductors along a panel edge without fasteners",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="a routine FDM print",
        concepts=concepts,
        recommended_concept_id=concepts[0].concept_id,
        recommendation_reason="simplest",
    )


def _kit() -> Any:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


def _package(problem: str, plan: ConceptPlan) -> InventionPackage:
    return build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=_kit(),
        confidence=standard_lane_label(0),
        problem_statement=problem,
        feasibility=plan.feasibility(),
        concept=plan.recommended(),
    )


def test_a_good_lifecycle_passes_every_check() -> None:
    problem = "cables fall off my desk edge"
    plan = _plan(_concepts(distinct=True))
    result = score_lifecycle(problem, _package(problem, plan), plan)
    assert result.passed is True
    assert result.score == 1.0


def test_no_concept_divergence_fails_the_rubric() -> None:
    """One concept = no divergence; the lifecycle did not act like an invention lab."""
    problem = "cables fall off my desk edge"
    one = _concepts()[:1]
    plan = _plan(one)
    result = score_lifecycle(problem, _package(problem, plan), plan)
    assert result.passed is False
    failed = {c.id for c in result.checks if not c.passed}
    assert "concept_divergence" in failed


def test_identical_mechanisms_are_not_real_divergence() -> None:
    problem = "cables fall off my desk edge"
    plan = _plan(_concepts(distinct=False))  # two concepts, SAME mechanism
    result = score_lifecycle(problem, _package(problem, plan), plan)
    assert result.passed is False
    assert "concept_divergence" in {c.id for c in result.checks if not c.passed}


def test_frozen_fixture_loads() -> None:
    problems = load_problems(_FIXTURE)
    assert len(problems) >= 4
    ids = {p.id for p in problems}
    assert "cable_clip" in ids
    assert all(p.problem.strip() for p in problems)


class _StubClient:
    def __init__(self, plan: ConceptPlan) -> None:
        self.plan = plan

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
        return self.plan, LLMUsage(model=model, input_tokens=1, output_tokens=1)

    def generate_structured_with_image(self, **_: Any) -> tuple[Any, LLMUsage]:
        raise AssertionError("no image")


def test_run_rubric_orchestrates_lifecycle_and_scores_offline() -> None:
    """The full bench, run offline: both seams stubbed, every frozen problem produces a scored,
    passing result — so when Fable returns only the inputs change, not the measurement."""
    plan = _plan(_concepts(distinct=True))

    def produce(prompt: str) -> GeneratedPart:
        return GeneratedPart(
            part_name="cable-clip",
            disposition="shipped",
            kit=_kit(),
            confidence=standard_lane_label(0),
        )

    problems = load_problems(_FIXTURE)
    results = run_rubric(problems, produce=produce, client=_StubClient(plan))
    assert len(results) == len(problems)
    assert all(r.passed for r in results)
