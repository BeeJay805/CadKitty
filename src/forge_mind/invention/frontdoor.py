"""The problem -> concept front door (ADR-0048) — the lifecycle's first brain pass.

ONE brain pass turns a physical PROBLEM into a :class:`ConceptPlan`: a first-principles
restatement, an advisory feasibility read, 2-3 distinct concepts (different mechanisms), and a
recommendation. This is the organ that makes the lab "not a renamed generate flow" — yet it stays
inside the moat: concepts are closed/attributed structures (``concept.py``), feasibility is advisory
(the disclaimer is code-controlled), and NOTHING here issues a verdict. The chosen concept feeds the
proven CAD path downstream; the validator remains the sole verdict-issuer.

The ``client`` seam is injectable (every engine stage accepts one), so this is fully testable
offline with a mock — no spend, no live model required. Default model is :data:`BRAIN_MODEL` (Opus
today; ``FORGE_BRAIN_MODEL`` swaps it — ADR-0046/0055, the one brain holds the whole plan); cost is
one premium pass, not one call per phase.
"""

from __future__ import annotations

from forge_mind.config import BRAIN_MODEL
from forge_mind.invention.concept import ConceptPlan
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_FRONTDOOR_MAX_TOKENS = 6000

_SYSTEM = """You are the lead inventor of a physical-invention lab. You turn a real-world PROBLEM \
into a small set of buildable CONCEPTS — not a single answer, and not a finished part. A separate, \
deterministic engine will model and VALIDATE the chosen concept; your job is the reasoning before \
that, expressed as structure.

Produce a ConceptPlan:
- problem_restated: restate the problem in FIRST PRINCIPLES, stripped of familiar product names \
(e.g. instead of "anti-vibration feet" say "reduce periodic mechanical energy transfer from a \
moving machine into a flexible support surface while keeping it stable"). This forces real \
problem-solving instead of retrieving a known product.
- feasibility_verdict: an HONEST advisory read — likely_feasible / feasible_with_changes / \
likely_infeasible (or not_assessed if you genuinely cannot tell). This is an ASSESSMENT, never a \
guarantee; a downstream disclaimer always accompanies it, so do not overstate.
- feasibility_note: one or two sentences of reasoning for that verdict (what makes it buildable or \
not for a typical maker), attributed as your judgment.
- concepts: 2-3 GENUINELY DISTINCT concepts that solve the same physics differently. For EACH: a \
short name; a primary_mechanism from the allowed vocabulary (a real mechanism/effect — snap_fit, \
flexure, cam, wedge, latch, lever, spring, clamp, magnetic_coupling, ...); optional secondary \
mechanisms; an `approach` describing HOW it works (a proposal — never claim it is proven or rated); \
the `obvious_baseline` (the known/obvious solution); `difference_from_obvious` (the \
constraint-specific adaptation that makes yours non-obvious); and key_risks (what is uncertain). \
Give each a short concept_id like c1, c2, c3. Do NOT make the concepts trivial variants of one idea.
- recommended_concept_id: the one you would build first, by concept_id.
- recommendation_reason: why that one over the others (buildability, reliability, simplicity — \
attributed reasoning, not a guarantee).

Rules: describe mechanisms and geometry, never numeric performance claims ("holds 5 kg", "rated \
for") — the engine measures, you propose. Prefer concepts a normal maker can actually build. Be \
honest about risk; an unproven concept clearly stated is worth more than a confident wrong one."""


def _coerce_recommendation(plan: ConceptPlan) -> ConceptPlan:
    """Deterministic post-validation: make ``recommended_concept_id`` point at a REAL concept. A
    brain that names a dangling id is silently corrected to the first concept, so the stored plan is
    internally consistent (the data never claims a recommendation that isn't in the list)."""
    ids = {concept.concept_id for concept in plan.concepts}
    if plan.recommended_concept_id in ids:
        return plan
    return plan.model_copy(update={"recommended_concept_id": plan.concepts[0].concept_id})


def run_front_door(
    problem: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    context: str = "",
    max_tokens: int = _FRONTDOOR_MAX_TOKENS,
) -> tuple[ConceptPlan, LLMUsage]:
    """Run the problem -> concept front door (one brain pass) and return a validated ConceptPlan.

    Pure of side effects beyond the LLM call; never gates anything. ``client`` is injectable
    (mock it in tests — no spend). ``model`` defaults to :data:`BRAIN_MODEL` (Opus today; swap via
    ``FORGE_BRAIN_MODEL``).
    """
    client = client or make_client()
    context_block = f"{context.strip()}\n\n" if context.strip() else ""
    user = (
        f"Physical problem:\n{problem.strip()}\n\n"
        f"{context_block}"
        "Produce the ConceptPlan: restate the problem in first principles, give an advisory "
        "feasibility read, propose 2-3 genuinely distinct concepts (different mechanisms), and "
        "recommend one to build first."
    )
    plan, usage = client.generate_structured(
        system=_SYSTEM,
        user=user,
        output_model=ConceptPlan,
        model=model or BRAIN_MODEL,
        max_tokens=max_tokens,
    )
    return _coerce_recommendation(plan), usage


__all__ = ["run_front_door"]
