"""Persistent mission state v0 (ADR-0048, P3) — an append-only event log + a deterministic replay.

One invention run is an ORDERED, APPEND-ONLY list of :class:`MissionEvent`s; the latest
:class:`MissionState` is a pure fold over them (:func:`replay`). Same events -> same state, every
time (ordering is by ``seq``, never wall-clock), so the state is reconstructable and auditable.

The discipline (the handoff's Memory-OS laws, enforced in types):

  - **No raw chain-of-thought.** Events carry a structured ``summary`` + typed records, never a
    free-form "thoughts" dump -- there is no field for it.
  - **No model claim authoritative without evidence.** A :class:`DecisionRecord` with no
    ``evidence_refs`` cannot claim an evidence tier above ``ai_hypothesis`` (a model validator
    rejects it) -- the same anchor discipline as ADR-0040.
  - **Intent is locked; geometry is revisable.** ``MissionState.problem`` (the locked intent) only
    changes through an ACCEPTED :class:`ChangeRequest` targeting ``problem``; nothing else mutates it.
  - **Rejected paths are retained, not erased**, with the reason + what would make them worth
    reopening (a failure becomes memory).

Storage is a thin seam: events are JSON-serializable (:meth:`MissionLog.to_records` /
:meth:`MissionLog.from_records`) and every event carries its ``mission_id``, so a real binding can
append them to the existing ADR-0023 ``generations`` persistence (one row per event) -- the in-memory
log here is the source of truth the binding mirrors.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_mind.invention.labels import EvidenceTier


class EventType(StrEnum):
    """The closed lifecycle event vocabulary (shared with the reports in ``mission_reports``)."""

    MISSION_CREATED = "mission_created"
    CONCEPT_SELECTED = "concept_selected"
    CONCEPT_REJECTED = "concept_rejected"
    ARTIFACT_CREATED = "artifact_created"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_PASSED = "validation_passed"
    REVISION_CREATED = "revision_created"
    PACKAGE_ASSEMBLED = "package_assembled"
    DECISION_RECORDED = "decision_recorded"
    CHANGE_REQUESTED = "change_requested"


class ArtifactRecord(BaseModel):
    """One produced artifact, by REFERENCE (a path / storage key) -- never the content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    kind: str  # stl | step | glb | render | report | concept | package | ...
    ref: str
    description: str | None = None


class ValidationRecord(BaseModel):
    """A validation outcome. The validator is itself a deterministic check, so a passing validation
    is ``deterministic_check`` evidence (never higher -- no simulation/physical claim here)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    failed_checks: list[str] = Field(default_factory=list)
    evidence_tier: EvidenceTier = EvidenceTier.DETERMINISTIC_CHECK
    report_ref: str | None = None


class DecisionRecord(BaseModel):
    """A decision tied to EVIDENCE. A decision with no ``evidence_refs`` cannot claim a tier above
    ``ai_hypothesis`` -- a model's say-so is never authoritative on its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: str
    rationale: str
    evidence_refs: list[str] = Field(
        default_factory=list, description="Artifact / validation ids this decision rests on."
    )
    evidence_tier: EvidenceTier = EvidenceTier.AI_HYPOTHESIS

    @model_validator(mode="after")
    def _no_authority_without_evidence(self) -> DecisionRecord:
        if not self.evidence_refs and self.evidence_tier is not EvidenceTier.AI_HYPOTHESIS:
            raise ValueError(
                "a decision with no evidence_refs cannot claim an evidence tier above "
                "ai_hypothesis (no model claim is authoritative without evidence)"
            )
        return self


class RejectedPath(BaseModel):
    """A rejected concept, RETAINED with its reason + what would make it worth reopening. A failure
    becomes memory, not a deletion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_id: str
    idea: str
    reason: str
    evidence_tier: EvidenceTier = EvidenceTier.AI_HYPOTHESIS
    reopen_conditions: list[str] = Field(default_factory=list)
    physically_tested: bool = False


class ChangeRequest(BaseModel):
    """A proposed change to a LOCKED contract field. Silent interface drift is forbidden; the change
    only takes effect when ``status`` is ``accepted``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str  # the locked field, e.g. "problem"
    from_value: str
    to_value: str
    reason: str
    status: Literal["proposed", "accepted", "rejected"] = "proposed"


class MissionEvent(BaseModel):
    """One append-only event. ``seq`` orders the log (replay never depends on ``at``). ``summary``
    is a structured one-liner -- there is deliberately no raw-reasoning field. On MISSION_CREATED,
    ``summary`` carries the locked problem statement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_id: str
    seq: int
    event_type: EventType
    summary: str
    at: str = ""  # caller-provided ISO timestamp (metadata only; never used for ordering)
    stage: str | None = None  # the lifecycle stage this event advances to, if any
    concept_id: str | None = None
    artifact: ArtifactRecord | None = None
    validation: ValidationRecord | None = None
    decision: DecisionRecord | None = None
    rejected_path: RejectedPath | None = None
    change_request: ChangeRequest | None = None


class MissionState(BaseModel):
    """The latest state, FOLDED from the events. A pure function of the log (deterministic)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_id: str
    problem: str  # the LOCKED intent (changes only via an accepted ChangeRequest)
    current_stage: str
    selected_concept_id: str | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    validations: list[ValidationRecord] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    rejected_paths: list[RejectedPath] = Field(default_factory=list)
    change_requests: list[ChangeRequest] = Field(default_factory=list)


def replay(events: list[MissionEvent]) -> MissionState:
    """Fold the append-only events into the latest state. Deterministic: events are processed in
    ``seq`` order, so the same set always yields the same state regardless of insertion order."""
    ordered = sorted(events, key=lambda e: e.seq)
    if not ordered or ordered[0].event_type is not EventType.MISSION_CREATED:
        raise ValueError("a mission log must begin with a MISSION_CREATED event")
    head = ordered[0]
    problem = head.summary
    stage = head.stage or "intake"
    selected: str | None = None
    artifacts: list[ArtifactRecord] = []
    validations: list[ValidationRecord] = []
    decisions: list[DecisionRecord] = []
    rejected: list[RejectedPath] = []
    changes: list[ChangeRequest] = []

    for event in ordered[1:]:
        if event.stage is not None:
            stage = event.stage
        if event.event_type is EventType.CONCEPT_SELECTED:
            selected = event.concept_id
        elif event.event_type is EventType.CONCEPT_REJECTED and event.rejected_path is not None:
            rejected.append(event.rejected_path)
        elif event.event_type is EventType.ARTIFACT_CREATED and event.artifact is not None:
            artifacts.append(event.artifact)
        elif (
            event.event_type in (EventType.VALIDATION_PASSED, EventType.VALIDATION_FAILED)
            and event.validation is not None
        ):
            validations.append(event.validation)
        elif event.event_type is EventType.DECISION_RECORDED and event.decision is not None:
            decisions.append(event.decision)
        elif event.event_type is EventType.CHANGE_REQUESTED and event.change_request is not None:
            change = event.change_request
            changes.append(change)
            if change.target == "problem" and change.status == "accepted":
                problem = change.to_value  # the ONLY path that mutates the locked intent

    return MissionState(
        mission_id=head.mission_id,
        problem=problem,
        current_stage=stage,
        selected_concept_id=selected,
        artifacts=artifacts,
        validations=validations,
        decisions=decisions,
        rejected_paths=rejected,
        change_requests=changes,
    )


class MissionLog:
    """An append-only event log for one mission. ``append`` assigns the next ``seq`` and stamps the
    ``mission_id``; history is never edited in place. ``replay``/``state`` fold it deterministically."""

    def __init__(self, mission_id: str, problem: str, *, at: str = "") -> None:
        self.mission_id = mission_id
        self._events: list[MissionEvent] = [
            MissionEvent(
                mission_id=mission_id,
                seq=0,
                event_type=EventType.MISSION_CREATED,
                summary=problem,
                at=at,
                stage="intake",
            )
        ]

    def append(
        self,
        event_type: EventType,
        *,
        summary: str,
        at: str = "",
        stage: str | None = None,
        concept_id: str | None = None,
        artifact: ArtifactRecord | None = None,
        validation: ValidationRecord | None = None,
        decision: DecisionRecord | None = None,
        rejected_path: RejectedPath | None = None,
        change_request: ChangeRequest | None = None,
    ) -> MissionEvent:
        """Append one event, assigning the next sequence number. Append-only: never edits history."""
        event = MissionEvent(
            mission_id=self.mission_id,
            seq=len(self._events),
            event_type=event_type,
            summary=summary,
            at=at,
            stage=stage,
            concept_id=concept_id,
            artifact=artifact,
            validation=validation,
            decision=decision,
            rejected_path=rejected_path,
            change_request=change_request,
        )
        self._events.append(event)
        return event

    def events(self) -> list[MissionEvent]:
        """A copy of the append-only history (callers cannot mutate the log through it)."""
        return list(self._events)

    def state(self) -> MissionState:
        return replay(self._events)

    def to_records(self) -> list[dict[str, object]]:
        """JSON-ready rows for persistence (one per event) -- the seam onto the generations store."""
        return [event.model_dump(mode="json") for event in self._events]

    @classmethod
    def from_records(cls, records: list[dict[str, object]]) -> MissionLog:
        """Reconstruct a log from persisted rows (the inverse of :meth:`to_records`)."""
        events = sorted((MissionEvent.model_validate(r) for r in records), key=lambda e: e.seq)
        log = cls.__new__(cls)
        log.mission_id = events[0].mission_id if events else ""
        log._events = events
        return log


__all__ = [
    "ArtifactRecord",
    "ChangeRequest",
    "DecisionRecord",
    "EventType",
    "MissionEvent",
    "MissionLog",
    "MissionState",
    "RejectedPath",
    "ValidationRecord",
    "replay",
]
