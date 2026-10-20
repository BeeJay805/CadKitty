"""The invention lifecycle orchestrator (ADR-0048): problem -> concept -> part -> package, run
fully offline with both seams mocked (front-door client + the engine ``produce`` callable). The
live measurement waits for Fable; the wiring is proven now."""

from __future__ import annotations

from typing import Any

import pytest

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    GeneratedPart,
    Mechanism,
    concept_to_prompt,
    invention_enabled,
    run_lifecycle,
)
from forge_mind.kit import build_kit
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _plan() -> ConceptPlan:
    return ConceptPlan(
        problem_restated="route two flexible conductors along a panel edge without fasteners",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="a snap-fit clip is a routine FDM print",
        concepts=[
            ConceptBrief(
                concept_id="c1",
                name="cantilever C-clip",
                primary_mechanism=Mechanism.SNAP_FIT,
                approach="a C-shaped jaw that flexes over the desk edge",
                obvious_baseline="a screwed-on bracket",
                difference_from_obvious="no fasteners; the jaw flexes to grip",
            ),
            ConceptBrief(
                concept_id="c2",
                name="friction saddle",
                primary_mechanism=Mechanism.FRICTION_FIT,
                approach="a weighted saddle resting on the edge",
                obvious_baseline="an adhesive clip",
                difference_from_obvious="removable, leaves no residue",
            ),
        ],
        recommended_concept_id="c1",
        recommendation_reason="simplest, needs no hardware",
    )


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
        raise AssertionError("the lifecycle does not use images")


def _kit() -> Any:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


def _produce_factory(record: list[str]) -> Any:
    def produce(prompt: str) -> GeneratedPart:
        record.append(prompt)
        return GeneratedPart(
            part_name="cable-clip",
            disposition="shipped",
            kit=_kit(),
            confidence=standard_lane_label(0),
            artifacts={"stl": "/tmp/part.stl"},
        )

    return produce


def test_run_lifecycle_composes_package_with_the_chosen_concept() -> None:
    prompts: list[str] = []
    package, plan = run_lifecycle(
        "cables fall off my desk edge",
        produce=_produce_factory(prompts),
        client=_StubClient(_plan()),
    )
    # the recommended concept rides the package, closed + attributed
    assert package.concept is not None
    assert package.concept.concept_id == "c1"
    assert package.concept.primary_mechanism is Mechanism.SNAP_FIT
    # feasibility is advisory, carried from the plan, never a pass
    assert package.feasibility.verdict is FeasibilityVerdict.LIKELY_FEASIBLE
    assert "not a guarantee" in package.feasibility.disclaimer
    # the banner is always present (digital-only honesty)
    assert "NOT PHYSICALLY VALIDATED" in package.physical_validation.banner
    # the user's problem rode through verbatim; the empty BOM is honest
    assert package.problem_statement == "cables fall off my desk edge"
    assert package.bom_summary == "no registry-backed hardware"
    # produce received the concept-derived prompt
    assert len(prompts) == 1
    assert "cantilever C-clip" in prompts[0]
    assert "cables fall off my desk edge" in prompts[0]
    # the full plan (the alternatives) comes back too
    assert [c.concept_id for c in plan.concepts] == ["c1", "c2"]


def test_concept_to_prompt_is_deterministic() -> None:
    concept = _plan().concepts[0]
    assert concept_to_prompt("a problem", concept) == concept_to_prompt("a problem", concept)
    prompt = concept_to_prompt("a problem", concept)
    assert "cantilever C-clip" in prompt
    assert "a problem" in prompt


def test_invention_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_INVENTION", raising=False)
    assert invention_enabled() is False


def test_invention_enabled_only_when_flag_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_INVENTION", "1")
    assert invention_enabled() is True
    monkeypatch.setenv("FORGE_INVENTION", "0")
    assert invention_enabled() is False
