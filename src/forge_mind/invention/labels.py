"""The invention-lab moat-label types (ADR-0047) — shared by the package + the front door.

Factored out of ``package.py`` so the front door (``concept.py``) and the assembler (``package.py``)
can both depend on these without a circular import (``labels`` <- ``concept`` <- ``package``).

Two structural gates the council surfaced live here:

  - :class:`PhysicalValidation` — the "NOT PHYSICALLY VALIDATED" banner a package cannot render
    without (CAD-valid is never mission success; the engine has no physical-test loop).
  - :class:`FeasibilityAssessment` — feasibility as an ADVISORY lens that can never read as a PASS.
    The disclaimer is ALWAYS the canned constant, never LLM-authored: the front door's model picks a
    categorical ``verdict`` + an attributed ``note``, and code assembles the assessment via
    :meth:`FeasibilityAssessment.advisory` so the disclaimer cannot be softened or dropped.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------------------------
# Physical validation — the structural banner
# ---------------------------------------------------------------------------------------------


class PhysicalValidationStatus(StrEnum):
    """How far the part has been PHYSICALLY proven. Today only one value is reachable: the engine
    has no physical-test loop, so a digital lifecycle is always ``not_physically_validated``.
    Higher tiers (printed / bench-tested / field-tested) are EARNED by real test evidence only —
    they are deliberately absent until that evidence can exist (no false maturity)."""

    NOT_VALIDATED = "not_physically_validated"


# Table-selected banner text per status (reviewed in git, never LLM-generated).
_PHYSICAL_BANNER: dict[PhysicalValidationStatus, str] = {
    PhysicalValidationStatus.NOT_VALIDATED: (
        "NOT PHYSICALLY VALIDATED — conceptual model only. This package was produced and checked "
        "DIGITALLY; the part has not been printed, assembled, or tested. CAD-valid is not mission "
        "success — verify with a real build before relying on it."
    ),
}


class PhysicalValidation(BaseModel):
    """The structural banner. A package CANNOT be assembled without one, so the honesty about
    physical proof can never be dropped from the document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: PhysicalValidationStatus
    banner: str = Field(..., description="Table-selected from ``status``; never free text.")

    @classmethod
    def not_validated(cls) -> PhysicalValidation:
        """The only status the digital engine can honestly claim today."""
        status = PhysicalValidationStatus.NOT_VALIDATED
        return cls(status=status, banner=_PHYSICAL_BANNER[status])


# ---------------------------------------------------------------------------------------------
# Feasibility — advisory, never a gate
# ---------------------------------------------------------------------------------------------


class FeasibilityVerdict(StrEnum):
    """The feasibility lens's verdict. ``not_assessed`` is the default (no front door has run yet).
    No value here is a certifying PASS — feasibility has no deterministic checker behind it, so the
    disclaimer always rides along."""

    NOT_ASSESSED = "not_assessed"
    LIKELY_FEASIBLE = "likely_feasible"
    FEASIBLE_WITH_CHANGES = "feasible_with_changes"
    LIKELY_INFEASIBLE = "likely_infeasible"


_FEASIBILITY_DISCLAIMER = (
    "Feasibility is an engineering ASSESSMENT, not a guarantee — no deterministic check stands "
    "behind it. Treat it as a hypothesis to verify, never as a passed validation gate."
)


class FeasibilityAssessment(BaseModel):
    """The ungated feasibility lens, surfaced as ADVISORY. The disclaimer is mandatory and the
    verdict is never a PASS — it cannot be mistaken for the validator's certifying verdict. The
    ``note`` carries the brain's attributed reasoning (advisory, never a guarantee)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: FeasibilityVerdict = FeasibilityVerdict.NOT_ASSESSED
    note: str | None = Field(
        default=None,
        description="The brain's advisory reasoning for the verdict (attributed; never a guarantee). "
        "None until a front door assesses it.",
    )
    disclaimer: str = Field(
        default=_FEASIBILITY_DISCLAIMER,
        description="Always present: feasibility is an assessment, never a certifying gate.",
    )

    @classmethod
    def not_assessed(cls) -> FeasibilityAssessment:
        """The v0 default: the problem -> concept front door that assesses feasibility is deferred."""
        return cls(verdict=FeasibilityVerdict.NOT_ASSESSED, disclaimer=_FEASIBILITY_DISCLAIMER)

    @classmethod
    def advisory(
        cls, verdict: FeasibilityVerdict, note: str | None = None
    ) -> FeasibilityAssessment:
        """Build an advisory assessment from a categorical verdict + optional attributed reasoning.
        The disclaimer is ALWAYS the canned constant — never taken from a model — so feasibility can
        never be dressed up as a pass even when a brain authored the verdict and the note."""
        return cls(verdict=verdict, note=note, disclaimer=_FEASIBILITY_DISCLAIMER)


# ---------------------------------------------------------------------------------------------
# Evidence tiers + maturity — the explicit "what is actually proven" labels (handoff 25.3 / 12)
# ---------------------------------------------------------------------------------------------


class EvidenceTier(StrEnum):
    """How strongly a claim is backed. Downstream code must NEVER auto-upgrade a tier — promotion
    requires NEW evidence an artifact carries, exactly like the dimensional-provenance anchor
    invariant (ADR-0040). The labels are the handoff's T0-T5 ladder, spelled out."""

    AI_HYPOTHESIS = "ai_hypothesis"  # model judgment / design intuition only
    SOURCED_SPEC = "sourced_spec"  # backed by a datasheet / standard / known equation
    DETERMINISTIC_CHECK = "deterministic_check"  # CAD/DSL/compiler/validator confirmed geometry/fit
    SIMPLIFIED_SIMULATION = "simplified_simulation"  # a basic physics / torque / stress estimate
    HIGH_FIDELITY_SIMULATION = "high_fidelity_simulation"  # FEA / CFD / multibody
    PHYSICAL_TEST = "physical_test"  # real-world measurement from a prototype


_EVIDENCE_RANK: dict[EvidenceTier, int] = {
    EvidenceTier.AI_HYPOTHESIS: 0,
    EvidenceTier.SOURCED_SPEC: 1,
    EvidenceTier.DETERMINISTIC_CHECK: 2,
    EvidenceTier.SIMPLIFIED_SIMULATION: 3,
    EvidenceTier.HIGH_FIDELITY_SIMULATION: 4,
    EvidenceTier.PHYSICAL_TEST: 5,
}


def evidence_rank(tier: EvidenceTier) -> int:
    """Strength order (higher = stronger). Use to take the WEAKEST tier across a claim set — never
    to auto-promote one (promotion needs new evidence, not arithmetic)."""
    return _EVIDENCE_RANK[tier]


class MaturityLabel(StrEnum):
    """How far an invention has been PROVEN. Caps at ``digitally_validated`` today: the engine has
    no physical-test loop, so ``physically_tested`` is unreachable without real test evidence (no
    false maturity — the same discipline as the physical-validation banner)."""

    EXPLORE = "explore"
    CONCEPT = "concept"
    PROTOTYPE_CANDIDATE = "prototype_candidate"
    DIGITALLY_VALIDATED = "digitally_validated"
    PHYSICALLY_TESTED = "physically_tested"


def derive_maturity(
    confidence_basis: str | None, physical: PhysicalValidationStatus
) -> MaturityLabel:
    """Deterministic maturity from the COMPUTED confidence basis (ADR-0041) + the physical-validation
    status. Never reaches ``physically_tested`` unless real physical evidence exists (today: never,
    so the top reachable label is ``digitally_validated``)."""
    if physical != PhysicalValidationStatus.NOT_VALIDATED:
        return MaturityLabel.PHYSICALLY_TESTED
    if confidence_basis == "validated":
        return MaturityLabel.DIGITALLY_VALIDATED
    if confidence_basis == "approximate":
        return MaturityLabel.PROTOTYPE_CANDIDATE
    return MaturityLabel.CONCEPT


def derive_package_evidence(confidence_basis: str | None) -> EvidenceTier:
    """The package-level evidence tier: a coherent, gate-checked solid is a ``deterministic_check``;
    a part that failed its gates (basis None) is only an ``ai_hypothesis``. Higher tiers (simulation,
    physical) require evidence this engine does not yet produce — they are never claimed here."""
    if confidence_basis is None:
        return EvidenceTier.AI_HYPOTHESIS
    return EvidenceTier.DETERMINISTIC_CHECK


__all__ = [
    "EvidenceTier",
    "FeasibilityAssessment",
    "FeasibilityVerdict",
    "MaturityLabel",
    "PhysicalValidation",
    "PhysicalValidationStatus",
    "derive_maturity",
    "derive_package_evidence",
    "evidence_rank",
]
