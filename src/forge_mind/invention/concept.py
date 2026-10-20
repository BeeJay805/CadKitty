"""The ConceptBrief / ConceptPlan contracts — the front door's typed output (ADR-0047, ADR-0048).

The council's resolution for the problem -> concept front door: it MUST do real concept divergence
(2-3 distinct approaches — that is what makes it an invention lab, not a renamed generate flow), but
it emits CLOSED, table-backed structured briefs, never free-text fitness claims. So:

  - ``primary_mechanism`` is a closed :class:`Mechanism` enum (effects/mechanisms, not products —
    the handoff's "mechanism library, not product library"). It is the structural backbone an
    unknown value cannot pass.
  - the prose fields (``approach``, ``difference_from_obvious``, ...) are the brain's DESIGN
    PROPOSAL, clearly attributed — a description of HOW it would work, never a claim that it DOES
    work or is rated for anything. There is no field that can carry "holds X kg / passes Y".

The feasibility verdict is a categorical enum; the brain's reasoning is an attributed ``note``. Code
(not the model) assembles the :class:`FeasibilityAssessment` with the mandatory disclaimer, so the
front door can never dress feasibility up as a pass (``ConceptPlan.feasibility``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.labels import FeasibilityAssessment, FeasibilityVerdict


class Mechanism(StrEnum):
    """A closed vocabulary of physical mechanisms/effects (not products) the concept may use —
    the handoff's "retrieve mechanisms and effects, not just products". An unknown mechanism cannot
    pass the schema, which keeps the concept anchored to a real working principle."""

    SNAP_FIT = "snap_fit"
    FRICTION_FIT = "friction_fit"
    PRESS_FIT = "press_fit"
    FLEXURE = "flexure"
    LIVING_HINGE = "living_hinge"
    COMPLIANT_HINGE = "compliant_hinge"
    HINGE = "hinge"
    CAM = "cam"
    WEDGE = "wedge"
    LATCH = "latch"
    DETENT = "detent"
    HOOK = "hook"
    CLIP = "clip"
    CLAMP = "clamp"
    LEVER = "lever"
    LINKAGE = "linkage"
    OVER_CENTER_TOGGLE = "over_center_toggle"
    RATCHET = "ratchet"
    SPRING = "spring"
    DAMPER = "damper"
    TUNED_MASS = "tuned_mass"
    COMPLIANT_LATTICE = "compliant_lattice"
    DIAPHRAGM = "diaphragm"
    BELLOWS = "bellows"
    HEAT_SINK_FIN = "heat_sink_fin"
    MAGNETIC_COUPLING = "magnetic_coupling"
    THREADED_FASTENER = "threaded_fastener"
    DOVETAIL = "dovetail"
    CHANNEL = "channel"
    STANDOFF = "standoff"
    GUSSET = "gusset"


class ConceptBrief(BaseModel):
    """One candidate concept. Closed mechanism + attributed design-proposal prose; no fitness field."""

    model_config = ConfigDict(extra="forbid")

    concept_id: str = Field(..., description="Short stable id within this plan, e.g. 'c1'.")
    name: str = Field(..., description="A short concept name, e.g. 'cantilever C-clip'.")
    primary_mechanism: Mechanism = Field(
        ..., description="The dominant working principle (closed vocabulary)."
    )
    secondary_mechanisms: list[Mechanism] = Field(
        default_factory=list, max_length=4, description="Supporting mechanisms, if any."
    )
    approach: str = Field(
        ...,
        description="The design approach: HOW this concept works (an attributed PROPOSAL, never a "
        "claim that it is proven or rated for anything).",
    )
    obvious_baseline: str = Field(
        ...,
        description="The obvious/known solution this concept is measured against (anti-obviousness).",
    )
    difference_from_obvious: str = Field(
        ...,
        description="The constraint-specific adaptation that distinguishes it from the baseline.",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="The brain's stated risks/unknowns for this concept (attributed, advisory).",
    )


class ConceptPlan(BaseModel):
    """The front door's whole output from ONE brain pass: a first-principles restatement, an
    advisory feasibility read, 2-3 distinct concepts, and a recommendation. REPORTED — the chosen
    concept feeds the proven CAD path; nothing here is a verdict."""

    model_config = ConfigDict(extra="forbid")

    problem_restated: str = Field(
        ...,
        description="The problem restated in first principles (de-familiarized, no product names) — "
        "the handoff's problem-de-familiarization step.",
    )
    feasibility_verdict: FeasibilityVerdict = Field(
        ..., description="The advisory feasibility read (categorical; never a pass)."
    )
    feasibility_note: str = Field(
        ..., description="The brain's reasoning for the verdict (attributed, advisory)."
    )
    concepts: list[ConceptBrief] = Field(
        ..., min_length=1, max_length=4, description="2-3 distinct concepts (different mechanisms)."
    )
    recommended_concept_id: str = Field(
        ..., description="The concept_id the brain recommends (coerced to a real one by code)."
    )
    recommendation_reason: str = Field(
        ..., description="Why this concept over the others (attributed)."
    )

    def feasibility(self) -> FeasibilityAssessment:
        """Assemble the advisory feasibility assessment — the disclaimer is code-controlled, so the
        brain's verdict + note can never read as a passed gate."""
        return FeasibilityAssessment.advisory(self.feasibility_verdict, self.feasibility_note)

    def recommended(self) -> ConceptBrief:
        """The recommended concept, with a deterministic fallback to the first if the id dangles."""
        for concept in self.concepts:
            if concept.concept_id == self.recommended_concept_id:
                return concept
        return self.concepts[0]


__all__ = ["ConceptBrief", "ConceptPlan", "Mechanism"]
