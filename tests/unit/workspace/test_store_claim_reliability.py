"""Database job identity must survive internal or historical payload collisions."""

import json
from pathlib import Path

import pytest

from forge_mind.workspace.store import Store


@pytest.mark.parametrize("field", ["id", "session", "owner"])
def test_claim_keeps_database_identity_for_payload_collisions(tmp_path: Path, field: str) -> None:
    store = Store(tmp_path)
    sid = store.create("alice", "plate", None)["id"]
    other = store.create("bob", "ring", None)["id"]
    forged = {"id": "f" * 32, "session": other, "owner": "bob"}
    jid = store.enqueue(sid, "alice", {field: forged[field], "max_seconds": 60})

    job = store.claim()

    assert job is not None
    assert {key: job[key] for key in forged} == {"id": jid, "session": sid, "owner": "alice"}
    store.finish(job["id"], job["session"], "needs_review")
    with store.connect() as con:
        assert con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] == (
            "needs_review"
        )
    assert store.session_record(sid, "alice")["state"] == "needs_review"
    assert store.session_record(other, "bob")["state"] == "idle"


def test_reopened_claim_preserves_work_payload_and_cancellation_identity(tmp_path: Path) -> None:
    store = Store(tmp_path)
    sid = store.create("alice", "plate", None)["id"]
    work = {
        "max_seconds": 60,
        "max_actions": 4,
        "parameters": {"width": 20},
        "instruction": "Widen",
    }
    jid = store.enqueue(sid, "alice", work)
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
        payload.update(id="f" * 32, session="e" * 32, owner="bob")
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))

    reopened = Store(tmp_path)
    job = reopened.claim()

    assert job is not None
    assert {key: job[key] for key in work} == work
    assert (job["id"], job["session"], job["owner"]) == (jid, sid, "alice")
    reopened.cancel(sid, "alice")
    assert reopened.cancelled(job["id"])
    reopened.finish(job["id"], job["session"], "needs_review")
    assert reopened.session_record(sid, "alice")["state"] == "cancelled"
