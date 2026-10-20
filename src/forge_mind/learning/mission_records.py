"""The MissionLog <-> Supabase adapter (ADR-0050).

Binds the append-only invention :class:`~forge_mind.invention.mission.MissionLog` to the durable
``mission_events`` table (supabase/migrations/0003_mission_events.sql), one row per event. A mission
is reconstructed by selecting its rows ``where mission_id = ? order by seq`` and folding them with the
existing :func:`~forge_mind.invention.mission.replay`. This is the ONLY durable event store for
missions -- ruflo coordination memory stays recall-only, never authoritative.

Two pieces, mirroring the ``records.py`` pattern (a typed pydantic mirror of a table + to_row/from_row):

  - :class:`MissionEventRecord` -- the typed row. Promotes ``mission_id``/``seq``/``event_type`` to real
    columns (indexing + append-only idempotency via the ``(mission_id, seq)`` primary key) and keeps the
    full typed event in a single jsonb ``payload`` (``MissionEvent.model_dump(mode="json")`` round-trips
    losslessly, and :func:`replay` never queries the event's sub-objects -- so normalizing them would buy
    no query power and cost a migration on every schema change).
  - :class:`MissionEventStore` -- the thin durable binding over a service-role Supabase client (the
    lifecycle has no authenticated user in v0, exactly like ``design_sessions``). Injectable, so it is
    fully testable with a fake client and is NOT wired into ``/generate``; wiring it into the live
    lifecycle is the v1 step, with this seam ready.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from forge_mind.invention.mission import MissionEvent, MissionLog

_TABLE = "mission_events"


class MissionEventRecord(BaseModel):
    """One row of the ``mission_events`` table -- the typed mirror of a persisted :class:`MissionEvent`.

    Keep this in sync with the migration by hand: the promoted columns are ``mission_id``/``seq``/
    ``event_type``/``payload`` (+ the table-defaulted ``created_at``). ``to_row`` produces a Supabase
    upsert payload; ``from_row`` parses a fetched row; ``payload`` is the full event dump so a row
    round-trips back to an exact :class:`MissionEvent` with no information loss.
    """

    model_config = ConfigDict(extra="forbid")

    mission_id: str
    seq: int
    event_type: str
    payload: dict[str, Any]
    created_at: str | None = None

    @classmethod
    def from_event(cls, event: MissionEvent) -> MissionEventRecord:
        """Project a :class:`MissionEvent` into its persisted row (promoted scalars + full payload)."""
        return cls(
            mission_id=event.mission_id,
            seq=event.seq,
            event_type=event.event_type.value,
            payload=event.model_dump(mode="json"),
        )

    def to_row(self) -> dict[str, Any]:
        """Flatten to a ``mission_events`` upsert payload. ``created_at`` is dropped when unset so the
        table default (``now()``) applies, mirroring ``GenerationRecord.to_row`` dropping None."""
        row: dict[str, Any] = {
            "mission_id": self.mission_id,
            "seq": self.seq,
            "event_type": self.event_type,
            "payload": self.payload,
        }
        if self.created_at is not None:
            row["created_at"] = self.created_at
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> MissionEventRecord:
        """Parse a fetched ``mission_events`` row back into a record."""
        return cls(
            mission_id=row["mission_id"],
            seq=int(row["seq"]),
            event_type=row["event_type"],
            payload=row["payload"],
            created_at=row.get("created_at"),
        )

    def to_event(self) -> MissionEvent:
        """Reconstruct the exact :class:`MissionEvent` from the stored payload."""
        return MissionEvent.model_validate(self.payload)


def rows_for_log(log: MissionLog) -> list[dict[str, Any]]:
    """One upsert row per event in the log (the whole append-only history)."""
    return [MissionEventRecord.from_event(event).to_row() for event in log.events()]


def log_from_rows(rows: list[dict[str, Any]]) -> MissionLog:
    """Reconstruct a :class:`MissionLog` from fetched rows.

    Order-independent: :meth:`MissionLog.from_records` re-sorts by ``seq``. Raises ``ValueError`` if
    the rows span more than one ``mission_id`` (one log per mission) -- a fetch is expected to be
    scoped to a single mission, so a mixed set is a caller error, not silently merged.
    """
    records = [MissionEventRecord.from_row(row) for row in rows]
    mission_ids = {record.mission_id for record in records}
    if len(mission_ids) > 1:
        raise ValueError(
            f"rows span multiple missions, cannot fold into one log: {sorted(mission_ids)}"
        )
    return MissionLog.from_records([record.payload for record in records])


class MissionEventStore:
    """Durable append-only mission-event store over the ``mission_events`` table.

    ``client`` is a SERVICE-ROLE Supabase client (the lifecycle has no authenticated user in v0, so --
    like ``design_sessions`` -- the table is service-role-managed and RLS denies everyone else). The
    upsert is idempotent against the ``(mission_id, seq)`` primary key, so re-persisting an event or a
    whole log is safe (a retry never duplicates). Injectable for tests; NOT wired into ``/generate``.
    """

    def __init__(self, client: Any) -> None:
        self._sb = client

    def append_event(self, event: MissionEvent) -> None:
        """Persist ONE event (idempotent upsert on ``(mission_id, seq)``)."""
        self._sb.table(_TABLE).upsert(MissionEventRecord.from_event(event).to_row()).execute()

    def append_log(self, log: MissionLog) -> None:
        """Persist (upsert) every event in a log -- the whole append-only history at once."""
        self._sb.table(_TABLE).upsert(rows_for_log(log)).execute()

    def load(self, mission_id: str) -> MissionLog | None:
        """Reconstruct a mission's log from its rows (``order by seq``). ``None`` when no rows exist."""
        res = self._sb.table(_TABLE).select("*").eq("mission_id", mission_id).order("seq").execute()
        rows = res.data or []
        if not rows:
            return None
        return log_from_rows(rows)


__all__ = [
    "MissionEventRecord",
    "MissionEventStore",
    "log_from_rows",
    "rows_for_log",
]
