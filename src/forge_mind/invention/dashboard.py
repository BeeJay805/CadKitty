"""Minimal dashboard data API (ADR-0048, P8) — the backend read-model for a future lab dashboard.

NOT a frontend redesign and no polish: just the typed data a dashboard would surface, COMPOSED
deterministically from the mission state + the timeline + the package (the pieces P2-P5 already
produce). It is backend-only; surfacing it is a later flag-gated (``FORGE_INVENTION``) step. Every
field is copied or table-derived -- no model call, no generated prose.

Surfaces: mission summary, current stage, the latest important finding, artifact previews, validation
state, the maturity label, unresolved risks, the revision timeline, and support-KB readiness.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.labels import derive_maturity, derive_package_evidence
from forge_mind.invention.mission import MissionEvent, MissionState
from forge_mind.invention.package import InventionPackage
from forge_mind.invention.reports import (
    RevisionEntry,
    build_revision_summary,
    build_validation_summary,
)


class ArtifactPreview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    ref: str


class ValidationState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: int
    failed: int
    latest_passed: bool | None


class FindingView(BaseModel):
    """The latest important finding -- the most recent event's structured summary, never raw
    reasoning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str | None
    event_type: str
    summary: str


class MissionDashboard(BaseModel):
    """The composed read-model. ``maturity`` / ``evidence_tier`` / ``banner`` are present only once a
    package exists; everything else folds from the mission timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_id: str
    problem: str
    current_stage: str
    maturity: str | None = None
    evidence_tier: str | None = None
    banner: str | None = None
    latest_finding: FindingView | None = None
    validation_state: ValidationState
    artifact_previews: list[ArtifactPreview] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    revision_timeline: list[RevisionEntry] = Field(default_factory=list)
    support_kb_ready: bool = False


def _latest_finding(events: list[MissionEvent]) -> FindingView | None:
    if not events:
        return None
    latest = max(events, key=lambda e: e.seq)
    return FindingView(
        stage=latest.stage, event_type=latest.event_type.value, summary=latest.summary
    )


def _unresolved_risks(state: MissionState, package: InventionPackage | None) -> list[str]:
    risks = ["Not physically validated -- CAD-valid is not physical proof."]
    if package is not None and package.confidence is not None:
        risks.extend(f"Not fit for: {nf}" for nf in package.confidence.not_fit_for)
    # the latest failed validation's checks are open risks
    if state.validations and not state.validations[-1].passed:
        risks.extend(f"Failed check: {c}" for c in state.validations[-1].failed_checks)
    # a rejected path's reopen conditions are open items to track
    for rejected in state.rejected_paths:
        for condition in rejected.reopen_conditions:
            risks.append(f"Reopen {rejected.concept_id} if: {condition}")
    return risks


def _artifact_previews(
    state: MissionState, package: InventionPackage | None
) -> list[ArtifactPreview]:
    previews = {a.ref: ArtifactPreview(kind=a.kind, ref=a.ref) for a in state.artifacts}
    if package is not None:
        for kind, ref in package.artifacts.items():
            previews.setdefault(ref, ArtifactPreview(kind=kind, ref=ref))
    return [previews[ref] for ref in sorted(previews)]


def build_dashboard(
    state: MissionState,
    events: list[MissionEvent],
    *,
    package: InventionPackage | None = None,
) -> MissionDashboard:
    """Compose the dashboard read-model deterministically. ``package`` (when the part exists) adds the
    maturity / evidence / banner and enables the support KB; otherwise the dashboard reflects the
    in-progress mission. REPORTED -- nothing here gates."""
    validation = build_validation_summary(state)
    maturity = evidence = banner = None
    if package is not None:
        basis = package.confidence.basis if package.confidence is not None else None
        maturity = derive_maturity(basis, package.physical_validation.status).value
        evidence = derive_package_evidence(basis).value
        banner = package.physical_validation.banner
    return MissionDashboard(
        mission_id=state.mission_id,
        problem=state.problem,
        current_stage=state.current_stage,
        maturity=maturity,
        evidence_tier=evidence,
        banner=banner,
        latest_finding=_latest_finding(events),
        validation_state=ValidationState(
            passed=validation.passed,
            failed=validation.failed,
            latest_passed=validation.latest_passed,
        ),
        artifact_previews=_artifact_previews(state, package),
        unresolved_risks=_unresolved_risks(state, package),
        revision_timeline=build_revision_summary(events).timeline,
        support_kb_ready=package is not None,
    )


__all__ = [
    "ArtifactPreview",
    "FindingView",
    "MissionDashboard",
    "ValidationState",
    "build_dashboard",
]
