"""The ConceptBrief / ConceptPlan contracts (ADR-0048): closed mechanism vocabulary + attributed
proposal prose, no fitness field, and a feasibility disclaimer that code controls (not the model)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.invention import ConceptBrief, ConceptPlan, FeasibilityVerdict, Mechanism


def _brief(concept_id: str = "c1", mechanism: Mechanism = Mechanism.SNAP_FIT) -> ConceptBrief:
    return ConceptBrief(
        concept_id=concept_id,
        name="cantilever C-clip",
        primary_mechanism=mechanism,
        approach="a C-shaped jaw that flexes over the desk edge and grips it",
        obvious_baseline="a bracket screwed to the desk",
        difference_from_obvious="no fasteners — the jaw itself provides the grip",
    )


def _plan(recommended: str = "c1") -> ConceptPlan:
    return ConceptPlan(
        problem_restated="route two flexible conductors along a panel edge without fasteners",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="a snap-fit clip is a routine FDM print",
        concepts=[_brief("c1"), _brief("c2", Mechanism.FRICTION_FIT)],
        recommended_concept_id=recommended,
        recommendation_reason="simplest, needs no hardware",
    )


def test_concept_brief_is_closed() -> None:
    with pytest.raises(ValidationError):
        ConceptBrief(
            concept_id="c1",
            name="x",
            primary_mechanism=Mechanism.SNAP_FIT,
            approach="a",
            obvious_baseline="b",
            difference_from_obvious="c",
            rated_for_kg=5,  # type: ignore[call-arg]
        )


def test_concept_plan_is_closed() -> None:
    with pytest.raises(ValidationError):
        ConceptPlan(
            problem_restated="p",
            feasibility_verdict=FeasibilityVerdict.NOT_ASSESSED,
            feasibility_note="n",
            concepts=[_brief()],
            recommended_concept_id="c1",
            recommendation_reason="r",
            passes_validation=True,  # type: ignore[call-arg]
        )


def test_concept_has_no_fitness_field() -> None:
    """The brief carries a mechanism + attributed proposal prose, never a performance/fitness field —
    the same structural guard confidence.py applies ('no quantified performance claims, ever')."""
    assert set(ConceptBrief.model_fields) == {
        "concept_id",
        "name",
        "primary_mechanism",
        "secondary_mechanisms",
        "approach",
        "obvious_baseline",
        "difference_from_obvious",
        "key_risks",
    }


def test_primary_mechanism_must_be_in_the_closed_vocabulary() -> None:
    with pytest.raises(ValidationError):
        ConceptBrief(
            concept_id="c1",
            name="x",
            primary_mechanism="teleportation",  # type: ignore[arg-type]
            approach="a",
            obvious_baseline="b",
            difference_from_obvious="c",
        )
    # A real mechanism from the handoff's library is accepted.
    assert _brief(mechanism=Mechanism.FLEXURE).primary_mechanism is Mechanism.FLEXURE


def test_recommended_falls_back_to_first_when_id_dangles() -> None:
    plan = _plan(recommended="does-not-exist")
    assert plan.recommended().concept_id == "c1"


def test_recommended_returns_the_named_concept() -> None:
    assert _plan(recommended="c2").recommended().concept_id == "c2"


def test_feasibility_disclaimer_is_code_controlled_never_a_pass() -> None:
    """The plan carries a categorical verdict + an attributed note; the assembled assessment's
    disclaimer is the canned constant, so feasibility can never read as the validator's pass."""
    feas = _plan().feasibility()
    assert feas.verdict is FeasibilityVerdict.LIKELY_FEASIBLE
    assert feas.note == "a snap-fit clip is a routine FDM print"
    assert "not a guarantee" in feas.disclaimer
    assert "never as a passed validation gate" in feas.disclaimer
