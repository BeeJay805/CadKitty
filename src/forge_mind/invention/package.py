"""The InventionPackage — the council-vetted v0 envelope (ADR-0047).

A thin, REPORTED, DETERMINISTIC view that COMPOSES a finished part's existing outputs (the
geometry artifacts, the validation disposition, the ``Kit``/BOM, the confidence label) into one
"invention package" document — the lifecycle frame the pivot wraps around the proven engine.

Three laws, inherited from the moat (ADR-0035 / 0040 / 0041) and enforced STRUCTURALLY (the
:class:`PhysicalValidation` banner + the advisory :class:`FeasibilityAssessment` live in
``invention.labels``) so a polished lifecycle narrative can never out-talk the validator:

  1. COMPOSED FROM CLOSED FIELDS ONLY. Every field is a typed sub-object, a closed enum, or a
     table-selected canned string — exactly as ``confidence.py`` forbids "rated for X kg". No LLM
     free text about fitness can enter this envelope. The one verbatim string is the user's own
     ``problem_statement``, attributed as THE REQUEST, never an AI claim.
  2. A STRUCTURAL "NOT PHYSICALLY VALIDATED" BANNER (``physical_validation``).
  3. FEASIBILITY IS ADVISORY, NEVER A PASS (``feasibility``).

REPORTED, never gated: :func:`build_invention_package` builds nothing, changes no geometry, and
issues no verdict. The deferred live problem -> concept front door (``invention.frontdoor``) will
populate ``feasibility`` (still advisory) and a future ``concept`` field; until then both stay
honestly minimal.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.confidence import ConfidenceLabel
from forge_mind.invention.concept import ConceptBrief
from forge_mind.invention.labels import (
    FeasibilityAssessment,
    FeasibilityVerdict,
    PhysicalValidation,
    PhysicalValidationStatus,
)
from forge_mind.kit import Kit

# ---------------------------------------------------------------------------------------------
# The InventionPackage — the composed document
# ---------------------------------------------------------------------------------------------


class InventionPackage(BaseModel):
    """The composed invention document for ONE part. REPORTED, never gated (ADR-0047).

    Every field is a typed sub-object, a closed enum, or a table-selected string. The single
    verbatim string is ``problem_statement`` — the user's own request, attributed as the ask.
    There is structurally no field that can carry a generated fitness/performance claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part_name: str = Field(..., description="The part's name, from its metadata.")
    problem_statement: str | None = Field(
        default=None,
        description="The user's request, VERBATIM — attributed as the ask, never an AI claim.",
    )
    disposition: str = Field(
        ...,
        description="The engine's honest disposition: shipped / shipped_with_caveats / rejected / "
        "no_build (the validator's verdict, copied — never softened here).",
    )
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="The existing artifact paths (stl / step / glb / render / report / ...).",
    )
    kit: Kit = Field(
        ..., description="The buildable kit (BOM + make-instructions), composed as-is."
    )
    bom_summary: str = Field(
        ...,
        description="Honest one-line BOM headline; 'no registry-backed hardware' for a part with "
        "no purchased components (the empty BOM is the correct answer, not a gap).",
    )
    confidence: ConfidenceLabel | None = Field(
        default=None,
        description="The computed confidence/fitness label (ADR-0041); None when no coherent solid "
        "earned one (a no_build / hard-fail).",
    )
    concept: ConceptBrief | None = Field(
        default=None,
        description="The chosen concept from the front door (closed mechanism + attributed "
        "design-proposal prose); None when the part was produced without a concept phase.",
    )
    feasibility: FeasibilityAssessment = Field(
        ..., description="The ADVISORY feasibility lens — never a passed gate."
    )
    physical_validation: PhysicalValidation = Field(
        ..., description="The structural 'NOT PHYSICALLY VALIDATED' banner — always present."
    )


def build_invention_package(
    *,
    part_name: str,
    disposition: str,
    kit: Kit,
    confidence: ConfidenceLabel | None = None,
    artifacts: dict[str, str] | None = None,
    problem_statement: str | None = None,
    feasibility: FeasibilityAssessment | None = None,
    concept: ConceptBrief | None = None,
) -> InventionPackage:
    """Compose a finished part's existing outputs into one REPORTED invention document.

    Pure + additive: reads already-built objects, builds nothing, issues no verdict. The
    physical-validation banner is ALWAYS present (the engine has no physical-test evidence to
    claim otherwise); ``feasibility`` defaults to an advisory 'not assessed' until the deferred
    front door populates it. ``bom_summary`` is read straight from the Kit's BOM, so the honest
    empty-BOM line ('no registry-backed hardware') surfaces by construction.
    """
    return InventionPackage(
        part_name=part_name,
        problem_statement=problem_statement,
        disposition=disposition,
        artifacts=dict(artifacts or {}),
        kit=kit,
        bom_summary=kit.bom.summary(),
        confidence=confidence,
        concept=concept,
        feasibility=feasibility or FeasibilityAssessment.not_assessed(),
        physical_validation=PhysicalValidation.not_validated(),
    )


__all__ = [
    "FeasibilityAssessment",
    "FeasibilityVerdict",
    "InventionPackage",
    "PhysicalValidation",
    "PhysicalValidationStatus",
    "build_invention_package",
]
