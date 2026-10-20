"""The problem -> concept front door (ADR-0048): one brain pass, fully testable offline via the
injectable client seam (no spend, no live model). The default brain is BRAIN_MODEL (Opus today,
ADR-0055; FORGE_BRAIN_MODEL swaps it)."""

from __future__ import annotations

from typing import Any

from forge_mind.config import BRAIN_MODEL
from forge_mind.invention import ConceptBrief, ConceptPlan, FeasibilityVerdict, Mechanism
from forge_mind.invention.frontdoor import run_front_door
from forge_mind.llm.cost import LLMUsage


def _plan(recommended: str = "c1") -> ConceptPlan:
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
        recommended_concept_id=recommended,
        recommendation_reason="simplest, needs no hardware",
    )


class _StubClient:
    """A canned structured-output client: records each call and returns a pre-built plan. The front
    door's ``client`` seam is exactly this shape, so no live model is needed to test the wiring."""

    def __init__(self, plan: ConceptPlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {"system": system, "user": user, "output_model": output_model, "model": model}
        )
        return self.plan, LLMUsage(model=model, input_tokens=10, output_tokens=20)

    def generate_structured_with_image(self, **_: Any) -> tuple[Any, LLMUsage]:
        raise AssertionError("the front door does not use images")


def test_front_door_returns_the_plan_via_the_injected_client() -> None:
    stub = _StubClient(_plan())
    plan, usage = run_front_door("cables fall off my desk edge", client=stub)
    assert isinstance(plan, ConceptPlan)
    assert [c.concept_id for c in plan.concepts] == ["c1", "c2"]
    assert usage.model == BRAIN_MODEL


def test_front_door_uses_the_brain_model_by_default() -> None:
    stub = _StubClient(_plan())
    run_front_door("a problem", client=stub)
    assert stub.calls[0]["model"] == BRAIN_MODEL
    assert stub.calls[0]["output_model"] is ConceptPlan


def test_front_door_threads_the_problem_into_the_prompt() -> None:
    stub = _StubClient(_plan())
    run_front_door("cables keep sliding off the desk edge", client=stub)
    assert "cables keep sliding off the desk edge" in stub.calls[0]["user"]


def test_dangling_recommendation_is_coerced_to_a_real_concept() -> None:
    """A brain that names a recommended id not in the list is silently corrected to the first
    concept, so the stored plan is internally consistent."""
    stub = _StubClient(_plan(recommended="c9"))
    plan, _ = run_front_door("a problem", client=stub)
    assert plan.recommended_concept_id == "c1"
    assert plan.recommended().concept_id == "c1"


def test_valid_recommendation_is_preserved() -> None:
    stub = _StubClient(_plan(recommended="c2"))
    plan, _ = run_front_door("a problem", client=stub)
    assert plan.recommended_concept_id == "c2"
    assert plan.recommended().name == "friction saddle"


def test_feasibility_is_advisory_and_never_a_pass() -> None:
    stub = _StubClient(_plan())
    plan, _ = run_front_door("a problem", client=stub)
    feas = plan.feasibility()
    assert feas.verdict is FeasibilityVerdict.LIKELY_FEASIBLE
    assert "not a guarantee" in feas.disclaimer
