"""MissionEventStore over a FAKE service-role Supabase client (ADR-0050): append a log, load it back
by mission_id, and prove the upsert is idempotent on (mission_id, seq) and scoped per mission. The
fake mimics the supabase-py chain (table().upsert().execute() / table().select().eq().order().execute())
so the binding is proven money-free, no live Supabase."""

from __future__ import annotations

from typing import Any

from forge_mind.invention.mission import EventType, MissionLog
from forge_mind.learning.mission_records import MissionEventStore


class _Result:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class _Query:
    """The read chain: .select(...).eq('mission_id', id).order('seq').execute()."""

    def __init__(self, table: _FakeTable) -> None:
        self._table = table
        self._mission_id: str | None = None

    def eq(self, col: str, val: str) -> _Query:
        if col == "mission_id":
            self._mission_id = val
        return self

    def order(self, _col: str) -> _Query:
        return self

    def execute(self) -> _Result:
        rows = [r for r in self._table.rows if r["mission_id"] == self._mission_id]
        return _Result(sorted(rows, key=lambda r: r["seq"]))


class _FakeTable:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []

    def upsert(self, payload: Any) -> _FakeTable:
        self._pending = payload if isinstance(payload, list) else [payload]
        return self

    def select(self, _cols: str) -> _Query:
        return _Query(self)

    def execute(self) -> _Result:
        # the upsert path: apply pending rows idempotently on the (mission_id, seq) primary key
        for item in self._pending:
            key = (item["mission_id"], item["seq"])
            self.rows = [r for r in self.rows if (r["mission_id"], r["seq"]) != key]
            self.rows.append(item)
        self._pending = []
        return _Result(list(self.rows))


class _FakeClient:
    def __init__(self) -> None:
        self._tables: dict[str, _FakeTable] = {}

    def table(self, name: str) -> _FakeTable:
        return self._tables.setdefault(name, _FakeTable())


def test_append_log_then_load_round_trips() -> None:
    store = MissionEventStore(_FakeClient())
    log = MissionLog("m1", "route two cables")
    log.append(EventType.CONCEPT_SELECTED, summary="pick c1", concept_id="c1", stage="concept")
    log.append(EventType.PACKAGE_ASSEMBLED, summary="assembled", stage="package")
    store.append_log(log)

    loaded = store.load("m1")
    assert loaded is not None
    assert loaded.state() == log.state()
    assert loaded.state().current_stage == "package"


def test_append_event_is_idempotent_on_mission_seq() -> None:
    store = MissionEventStore(_FakeClient())
    genesis = MissionLog("m1", "p").events()[0]
    store.append_event(genesis)
    store.append_event(
        genesis
    )  # re-emit the same (mission, seq) -> a no-op upsert, not a duplicate

    loaded = store.load("m1")
    assert loaded is not None
    assert len(loaded.events()) == 1


def test_load_unknown_mission_is_none() -> None:
    assert MissionEventStore(_FakeClient()).load("nope") is None


def test_load_scopes_to_one_mission() -> None:
    store = MissionEventStore(_FakeClient())
    store.append_log(MissionLog("m1", "first"))
    store.append_log(MissionLog("m2", "second"))

    m1 = store.load("m1")
    assert m1 is not None
    assert m1.mission_id == "m1"
    assert m1.state().problem == "first"
