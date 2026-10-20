"""The MissionLog <-> mission_events adapter (ADR-0050): a row round-trips back to the exact event,
the whole append-only log maps one-row-per-event, and a fetch reconstructs the same state by
mission_id regardless of arrival order (folded by seq). Pure -- no I/O."""

from __future__ import annotations

import pytest

from forge_mind.invention.labels import EvidenceTier
from forge_mind.invention.mission import (
    ArtifactRecord,
    DecisionRecord,
    EventType,
    MissionLog,
    ValidationRecord,
)
from forge_mind.learning.mission_records import (
    MissionEventRecord,
    log_from_rows,
    rows_for_log,
)


def _log() -> MissionLog:
    log = MissionLog("m1", "route two cables", at="2026-06-14T00:00:00Z")
    log.append(EventType.CONCEPT_SELECTED, summary="pick c1", concept_id="c1", stage="concept")
    log.append(
        EventType.ARTIFACT_CREATED,
        summary="stl written",
        artifact=ArtifactRecord(artifact_id="a1", kind="stl", ref="part.stl"),
    )
    log.append(
        EventType.VALIDATION_PASSED,
        summary="geometry passed",
        validation=ValidationRecord(passed=True),
        stage="validation",
    )
    log.append(
        EventType.DECISION_RECORDED,
        summary="ship",
        decision=DecisionRecord(
            decision="ship",
            rationale="validated",
            evidence_refs=["val:geometry"],
            evidence_tier=EvidenceTier.DETERMINISTIC_CHECK,
        ),
    )
    return log


def test_event_record_round_trips_through_a_row() -> None:
    event = _log().events()[2]  # the ARTIFACT_CREATED event (seq 2)
    rec = MissionEventRecord.from_event(event)
    assert rec.mission_id == "m1"
    assert rec.seq == event.seq
    assert rec.event_type == EventType.ARTIFACT_CREATED.value
    row = rec.to_row()
    # promoted scalars are real columns; the full typed event rides the jsonb payload
    assert row["mission_id"] == "m1" and row["seq"] == event.seq
    assert row["event_type"] == "artifact_created"
    assert row["payload"]["artifact"]["ref"] == "part.stl"
    assert "created_at" not in row  # unset -> dropped so the table default (now()) applies
    # a fetched row folds back to the EXACT event (no information loss)
    assert MissionEventRecord.from_row(row).to_event() == event


def test_to_row_keeps_created_at_when_set() -> None:
    rec = MissionEventRecord.from_event(_log().events()[0]).model_copy(
        update={"created_at": "2026-06-14T01:00:00Z"}
    )
    assert rec.to_row()["created_at"] == "2026-06-14T01:00:00Z"


def test_rows_for_log_is_one_row_per_event_with_the_genesis_first() -> None:
    log = _log()
    rows = rows_for_log(log)
    assert len(rows) == len(log.events())
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    # seq 0 is always MISSION_CREATED and carries the locked problem (the replay invariant)
    assert rows[0]["event_type"] == EventType.MISSION_CREATED.value
    assert rows[0]["payload"]["summary"] == "route two cables"


def test_log_from_rows_reconstructs_state_order_independently() -> None:
    log = _log()
    rebuilt = log_from_rows(list(reversed(rows_for_log(log))))  # arrival order must not matter
    assert rebuilt.mission_id == "m1"
    assert rebuilt.state() == log.state()
    assert rebuilt.state().current_stage == "validation"
    assert rebuilt.state().selected_concept_id == "c1"
    assert rebuilt.state().decisions[0].evidence_tier is EvidenceTier.DETERMINISTIC_CHECK


def test_log_from_rows_rejects_rows_from_multiple_missions() -> None:
    rows = rows_for_log(_log()) + rows_for_log(MissionLog("m2", "another problem"))
    with pytest.raises(ValueError, match="multiple missions"):
        log_from_rows(rows)
