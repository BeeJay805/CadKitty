"""Crash recovery must not impose a duration limit on healthy work."""

import sqlite3
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest

from forge_mind.workspace.store import Store


def claimed(store: Store, **payload: object) -> dict:
    sid = store.create("owner", "part", None)["id"]
    store.enqueue(sid, "owner", payload)
    job = store.claim()
    assert job is not None
    return job


@pytest.mark.parametrize("seconds", [None, 1])
def test_live_owner_survives_old_deadline(tmp_path: Path, seconds: int | None) -> None:
    store = Store(tmp_path)
    job = claimed(store, max_seconds=seconds)
    with store.connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone()
        assert (row["deadline"] is None) == (seconds is None)
        con.execute("UPDATE jobs SET deadline=0 WHERE id=?", (job["id"],))
    sid = store.create("second", "queued part", None)["id"]
    store.enqueue(sid, "second", {})
    assert Store(tmp_path).claim() is None
    assert not store.cancelled(job["id"])


@pytest.mark.parametrize("cause", ["missing", "reused", "zombie"])
def test_dead_owner_releases_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    store = Store(tmp_path)
    job = claimed(store)
    sid = store.create("second", "queued part", None)["id"]
    store.enqueue(sid, "second", {})
    real_process = psutil.Process
    with store.connect() as con:
        con.execute("UPDATE jobs SET worker_pid=999999 WHERE id=?", (job["id"],))

    def process(pid: int) -> object:
        if pid != 999999:
            return real_process(pid)
        if cause == "missing":
            raise psutil.NoSuchProcess(pid)
        fake = Mock()
        with store.connect() as con:
            created = con.execute(
                "SELECT worker_created FROM jobs WHERE id=?", (job["id"],)
            ).fetchone()[0]
        fake.create_time.return_value = created + (1 if cause == "reused" else 0)
        fake.is_running.return_value = True
        fake.status.return_value = (
            psutil.STATUS_ZOMBIE if cause == "zombie" else psutil.STATUS_RUNNING
        )
        return fake

    monkeypatch.setattr(psutil, "Process", process)
    next_job = store.claim()
    assert next_job and next_job["session"] == sid
    assert store.get(job["session"], "owner")["state"] == "interrupted"


@pytest.mark.parametrize("cause", ["foreign", "partial", "denied"])
def test_unknown_owner_is_not_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    store = Store(tmp_path)
    job = claimed(store)
    with store.connect() as con:
        con.execute("UPDATE jobs SET deadline=0 WHERE id=?", (job["id"],))
        if cause == "foreign":
            con.execute("UPDATE jobs SET worker_host='other-host' WHERE id=?", (job["id"],))
        if cause == "partial":
            con.execute("UPDATE jobs SET worker_created=NULL WHERE id=?", (job["id"],))
    if cause == "denied":
        monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.AccessDenied()))
    assert store.claim() is None
    assert not store.cancelled(job["id"])


def test_dead_preview_owner_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    job = claimed(store, preview_only=True)
    store.update(job["session"], selected_revision="candidate", instructions=["saved work"])
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=psutil.NoSuchProcess(123)))
    assert store.claim() is None
    session = store.get(job["session"], "owner")
    assert session["state"] == "interrupted"
    assert session["selected_revision"] is None
    assert session["resume_candidate"]["revision_id"] == "candidate"
    assert session["resume_candidate"]["instructions"] == ["saved work"]


def test_existing_database_migrates_additively(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "workspace.sqlite3") as con:
        con.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY,session TEXT,state TEXT,payload TEXT,"
            "started REAL,deadline REAL,cancel INTEGER DEFAULT 0)"
        )
        con.execute("INSERT INTO jobs(id,state,payload) VALUES ('old','finished','{}')")
    store = Store(tmp_path)
    Store(tmp_path)
    with store.connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id='old'").fetchone()
        assert row["state"] == "finished"
        assert row["worker_pid"] is None
        assert row["worker_created"] is None
        assert row["worker_host"] is None
