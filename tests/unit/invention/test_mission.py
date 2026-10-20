"""Persistent mission state v0 (ADR-0048, P3): an append-only log with deterministic replay, a
locked intent that changes only via an accepted ChangeRequest, retained rejected paths, and a
decision that cannot claim authority without evidence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge_mind.invention.labels import EvidenceTier
from forge_mind.invention.mission import (
    ArtifactRecord,
    ChangeRequest,
    DecisionRecord,
    EventType,
    MissionEvent,
    MissionLog,
    RejectedPath,
    ValidationRecord,
    replay,
)


def test_replay_is_deterministic_regardless_of_insertion_order() -> None:
    log = MissionLog("m1", "route two cables")
    log.append(EventType.CONCEPT_SELECTED, summary="pick c1", concept_id="c1", stage="concept")
    log.append(
        EventType.ARTIFACT_CREATED,
        summary="stl written",
        artifact=ArtifactRecord(artifact_id="a1", kind="stl", ref="part.stl"),
    )
    events = log.events()
    assert replay(events) == log.state()
    # order-independence: a reversed event list folds to the SAME state (seq, not arrival, orders it)
    assert replay(list(reversed(events))) == log.state()
    assert log.state().selected_concept_id == "c1"
    assert log.state().current_stage == "concept"


def test_replay_requires_a_mission_created_head() -> None:
    with pytest.raises(ValueError, match="MISSION_CREATED"):
        replay([])


def test_intent_is_locked_until_an_accepted_change_request() -> None:
    log = MissionLog("m1", "route two cables")
    log.append(
        EventType.CHANGE_REQUESTED,
        summary="propose wider scope",
        change_request=ChangeRequest(
            target="problem",
            from_value="route two cables",
            to_value="route four cables",
            reason="more cables now",
            status="proposed",
        ),
    )
    assert log.state().problem == "route two cables"  # a PROPOSED change does not mutate intent
    log.append(
        EventType.CHANGE_REQUESTED,
        summary="accept wider scope",
        change_request=ChangeRequest(
            target="problem",
            from_value="route two cables",
            to_value="route four cables",
            reason="more cables now",
            status="accepted",
        ),
    )
    assert log.state().problem == "route four cables"  # only an ACCEPTED change does
    assert len(log.state().change_requests) == 2


def test_rejected_path_is_retained_with_reopen_conditions() -> None:
    log = MissionLog("m1", "route two cables")
    log.append(
        EventType.CONCEPT_REJECTED,
        summary="reject magnet clip",
        rejected_path=RejectedPath(
            concept_id="c2",
            idea="magnetic two-piece clip",
            reason="hobby magnets too weak for the load",
            reopen_conditions=["a stronger N52 magnet is sourced and confirmed"],
        ),
    )
    rejected = log.state().rejected_paths
    assert len(rejected) == 1
    assert rejected[0].concept_id == "c2"
    assert rejected[0].reopen_conditions == ["a stronger N52 magnet is sourced and confirmed"]
    assert rejected[0].physically_tested is False  # a conceptual rejection, not a physical one


def test_decision_cannot_claim_authority_without_evidence() -> None:
    # a higher tier with NO evidence refs is rejected
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision="ship the C-clip",
            rationale="it looks fine",
            evidence_tier=EvidenceTier.DETERMINISTIC_CHECK,
        )
    # ai_hypothesis with no refs is allowed (an honest hunch)
    assert DecisionRecord(decision="explore", rationale="hunch").evidence_tier is (
        EvidenceTier.AI_HYPOTHESIS
    )
    # with evidence refs, a stronger tier is allowed
    backed = DecisionRecord(
        decision="ship",
        rationale="geometry validated",
        evidence_refs=["val:geometry"],
        evidence_tier=EvidenceTier.DETERMINISTIC_CHECK,
    )
    assert backed.evidence_tier is EvidenceTier.DETERMINISTIC_CHECK


def test_records_round_trip_through_persistence() -> None:
    log = MissionLog("m1", "route two cables", at="2026-06-13T00:00:00Z")
    log.append(
        EventType.VALIDATION_PASSED,
        summary="geometry passed all tiers",
        validation=ValidationRecord(passed=True),
        stage="validation",
    )
    log.append(EventType.PACKAGE_ASSEMBLED, summary="package assembled", stage="package")
    rebuilt = MissionLog.from_records(log.to_records())
    assert rebuilt.mission_id == "m1"
    assert rebuilt.state() == log.state()
    assert rebuilt.state().current_stage == "package"
    assert rebuilt.state().validations[0].evidence_tier is EvidenceTier.DETERMINISTIC_CHECK


def test_event_has_no_raw_reasoning_field() -> None:
    """No raw chain-of-thought: events carry a structured summary + typed records only."""
    fields = set(MissionEvent.model_fields)
    assert "summary" in fields
    assert not (fields & {"reasoning", "thoughts", "chain_of_thought", "raw", "transcript"})


def test_log_is_append_only_and_seq_is_sequential() -> None:
    log = MissionLog("m1", "p")
    first = log.append(
        EventType.DECISION_RECORDED,
        summary="chose snap-fit",
        decision=DecisionRecord(decision="snap-fit", rationale="no hardware"),
    )
    second = log.append(
        EventType.ARTIFACT_CREATED,
        summary="stl",
        artifact=ArtifactRecord(artifact_id="a1", kind="stl", ref="r"),
    )
    assert first.seq == 1
    assert second.seq == 2
    # events() returns a COPY -- mutating it cannot corrupt the log
    snapshot = log.events()
    snapshot.clear()
    assert len(log.events()) == 3
    assert isinstance(log.events()[0], MissionEvent)
