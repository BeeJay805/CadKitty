"""Deterministic lifecycle reports (ADR-0048, P4) — the company-style reports the user sees.

The user should see resolved engineering REPORTS, not raw agent thinking (handoff 10/14). These are
DETERMINISTIC projections over typed inputs (the mission state, the concept plan, the package): every
field is a value copied from an evidence-bearing object or a closed table-selected phrase. No model
call, no generated prose, no roleplay -- so the same inputs always yield the same report.

Each report carries its evidence tier honestly: feasibility/concept reviews are ``ai_hypothesis``
(reasoning lenses with no deterministic checker), the validation summary is ``deterministic_check``
(the validator ran), and the package-ready summary names what is PROVEN vs UNPROVEN explicitly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.concept import ConceptPlan
from forge_mind.invention.labels import (
    EvidenceTier,
    FeasibilityAssessment,
    MaturityLabel,
    derive_maturity,
    derive_package_evidence,
)
from forge_mind.invention.mission import (
    EventType,
    MissionEvent,
    MissionState,
    RejectedPath,
)
from forge_mind.invention.package import InventionPackage

# ---------------------------------------------------------------------------------------------
# Feasibility review
# ---------------------------------------------------------------------------------------------


class FeasibilityReview(BaseModel):
    """The feasibility read, surfaced as a review. ALWAYS ``ai_hypothesis`` evidence -- there is no
    deterministic checker behind 'this is buildable', and the disclaimer rides along."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problem: str
    verdict: str
    note: str | None
    disclaimer: str
    evidence_tier: str = EvidenceTier.AI_HYPOTHESIS.value


def build_feasibility_review(problem: str, feasibility: FeasibilityAssessment) -> FeasibilityReview:
    return FeasibilityReview(
        problem=problem,
        verdict=feasibility.verdict.value,
        note=feasibility.note,
        disclaimer=feasibility.disclaimer,
    )


# ---------------------------------------------------------------------------------------------
# Concept review
# ---------------------------------------------------------------------------------------------


class ConceptOption(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_id: str
    name: str
    primary_mechanism: str


class RejectedConceptView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_id: str
    reason: str
    reopen_conditions: list[str] = Field(default_factory=list)


class ConceptReview(BaseModel):
    """What was considered, what was chosen and why, and what was rejected (with reopen conditions).
    Concepts are proposals, so the review is ``ai_hypothesis`` evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problem: str
    considered: list[ConceptOption]
    selected: ConceptOption | None
    selection_reason: str
    rejected: list[RejectedConceptView] = Field(default_factory=list)
    evidence_tier: str = EvidenceTier.AI_HYPOTHESIS.value


def build_concept_review(
    plan: ConceptPlan,
    *,
    selected_concept_id: str | None = None,
    rejected: list[RejectedPath] | None = None,
) -> ConceptReview:
    selected_id = selected_concept_id or plan.recommended_concept_id
    options = [
        ConceptOption(
            concept_id=c.concept_id, name=c.name, primary_mechanism=c.primary_mechanism.value
        )
        for c in plan.concepts
    ]
    selected = next((o for o in options if o.concept_id == selected_id), None)
    return ConceptReview(
        problem=plan.problem_restated,
        considered=options,
        selected=selected,
        selection_reason=plan.recommendation_reason,
        rejected=[
            RejectedConceptView(
                concept_id=r.concept_id,
                reason=r.reason,
                reopen_conditions=list(r.reopen_conditions),
            )
            for r in (rejected or [])
        ],
    )


# ---------------------------------------------------------------------------------------------
# Validation + revision summaries (over the mission timeline)
# ---------------------------------------------------------------------------------------------


class ValidationSummary(BaseModel):
    """The validation outcomes folded from the mission. The validator IS a deterministic check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int
    passed: int
    failed: int
    latest_passed: bool | None
    failed_checks: list[str] = Field(default_factory=list)
    evidence_tier: str = EvidenceTier.DETERMINISTIC_CHECK.value


def build_validation_summary(state: MissionState) -> ValidationSummary:
    vals = state.validations
    passed = sum(1 for v in vals if v.passed)
    latest = vals[-1] if vals else None
    return ValidationSummary(
        total=len(vals),
        passed=passed,
        failed=len(vals) - passed,
        latest_passed=(latest.passed if latest is not None else None),
        failed_checks=(
            list(latest.failed_checks) if latest is not None and not latest.passed else []
        ),
    )


class RevisionEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    event_type: str
    summary: str
    stage: str | None = None


class RevisionSummary(BaseModel):
    """The revision/validation timeline: each revision and the validations around it, as structured
    entries (the event summaries -- never raw reasoning)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revisions: int
    timeline: list[RevisionEntry] = Field(default_factory=list)


_TIMELINE_EVENTS = (
    EventType.REVISION_CREATED,
    EventType.VALIDATION_PASSED,
    EventType.VALIDATION_FAILED,
)


def build_revision_summary(events: list[MissionEvent]) -> RevisionSummary:
    ordered = sorted(events, key=lambda e: e.seq)
    timeline = [
        RevisionEntry(seq=e.seq, event_type=e.event_type.value, summary=e.summary, stage=e.stage)
        for e in ordered
        if e.event_type in _TIMELINE_EVENTS
    ]
    revisions = sum(1 for e in ordered if e.event_type is EventType.REVISION_CREATED)
    return RevisionSummary(revisions=revisions, timeline=timeline)


# ---------------------------------------------------------------------------------------------
# Package-ready summary (proven vs unproven, named explicitly)
# ---------------------------------------------------------------------------------------------

_PROVEN_VALIDATED = (
    "geometry deterministically validated (mesh / feature / printability / interface tiers)"
)
_PROVEN_BOM = "every BOM line cites a standard or is flagged provisional (no fabricated parts)"
_UNPROVEN_PHYSICAL = "not physically built, assembled, or tested"
_UNPROVEN_FEASIBILITY = "feasibility is an engineering assessment, not a guarantee"

# Closed next-step table, keyed on maturity (reviewed in git, never generated).
_NEXT_BY_MATURITY: dict[MaturityLabel, list[str]] = {
    MaturityLabel.EXPLORE: ["pick a concept to develop"],
    MaturityLabel.CONCEPT: ["refine the concept into a validated CAD model"],
    MaturityLabel.PROTOTYPE_CANDIDATE: [
        "verify the estimated dimensions against the real parts",
        "print and test-fit",
    ],
    MaturityLabel.DIGITALLY_VALIDATED: [
        "print the part",
        "test-fit against the real hardware",
        "run a physical load / function test",
    ],
    MaturityLabel.PHYSICALLY_TESTED: ["document the physical test results"],
}


class PackageReadySummary(BaseModel):
    """The final package read: maturity + evidence + what is PROVEN vs UNPROVEN, the not-fit-for
    list, the mandatory banner, and the closed next steps. Every field is copied or table-selected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part_name: str
    disposition: str
    maturity: str
    evidence_tier: str
    proven: list[str] = Field(default_factory=list)
    unproven: list[str] = Field(default_factory=list)
    not_fit_for: list[str] = Field(default_factory=list)
    banner: str
    next_steps: list[str] = Field(default_factory=list)


def build_package_ready_summary(package: InventionPackage) -> PackageReadySummary:
    basis = package.confidence.basis if package.confidence is not None else None
    maturity = derive_maturity(basis, package.physical_validation.status)
    proven: list[str] = []
    if basis == "validated":
        proven.append(_PROVEN_VALIDATED)
    if package.bom_summary != "no registry-backed hardware":
        proven.append(_PROVEN_BOM)
    return PackageReadySummary(
        part_name=package.part_name,
        disposition=package.disposition,
        maturity=maturity.value,
        evidence_tier=derive_package_evidence(basis).value,
        proven=proven,
        unproven=[_UNPROVEN_PHYSICAL, _UNPROVEN_FEASIBILITY],
        not_fit_for=(
            list(package.confidence.not_fit_for) if package.confidence is not None else []
        ),
        banner=package.physical_validation.banner,
        next_steps=_NEXT_BY_MATURITY.get(maturity, []),
    )


__all__ = [
    "ConceptOption",
    "ConceptReview",
    "FeasibilityReview",
    "PackageReadySummary",
    "RejectedConceptView",
    "RevisionEntry",
    "RevisionSummary",
    "ValidationSummary",
    "build_concept_review",
    "build_feasibility_review",
    "build_package_ready_summary",
    "build_revision_summary",
    "build_validation_summary",
]
