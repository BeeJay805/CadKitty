from __future__ import annotations

import sqlite3

import pytest

from forge_mind.workspace.hosting import HostingError
from forge_mind.workspace.hosting_budget import grant, initialize, reserve, settle


def ledger(tmp_path):
    con = sqlite3.connect(tmp_path / "allowances.sqlite3")
    initialize(con)
    con.execute("BEGIN IMMEDIATE")
    grant(con, "owner", "operator-grant", 100)
    con.commit()
    return con


def test_reservation_and_grant_replays_never_double_spend(tmp_path):
    con = ledger(tmp_path)
    con.execute("BEGIN IMMEDIATE")
    grant(con, "owner", "operator-grant", 100)
    reserve(con, "owner", "job", 80)
    reserve(con, "owner", "job", 80)
    con.commit()
    assert con.execute("SELECT count(*) FROM host_grants").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM host_reservations").fetchone()[0] == 1
    con.execute("BEGIN IMMEDIATE")
    with pytest.raises(HostingError, match="concurrency"):
        reserve(con, "owner", "second", 10)
    with pytest.raises(HostingError, match="conflicts"):
        reserve(con, "other-owner", "job", 80)
    con.rollback()
    con.close()


def test_continuation_retains_original_ceiling_and_unused_release(tmp_path):
    con = ledger(tmp_path)
    con.execute("BEGIN IMMEDIATE")
    reserve(con, "owner", "job", 80)
    settle(con, "owner", "job", 30)
    settle(con, "owner", "job", 30)
    reserve(con, "owner", "continuation", 50, lineage_id="job")
    con.commit()
    con.execute("BEGIN IMMEDIATE")
    settle(con, "owner", "continuation", 40)
    with pytest.raises(HostingError, match="original compute"):
        reserve(con, "owner", "third", 11, lineage_id="job")
    reserve(con, "owner", "third", 10, lineage_id="job")
    con.commit()
    assert con.execute("SELECT count(*) FROM host_grants").fetchone()[0] == 1
    con.close()


def test_failure_can_rollback_queue_and_reservation_together(tmp_path):
    con = ledger(tmp_path)
    con.execute("CREATE TABLE queue (id TEXT)")
    con.execute("BEGIN IMMEDIATE")
    con.execute("INSERT INTO queue VALUES ('job')")
    with pytest.raises(HostingError, match="Insufficient"):
        reserve(con, "owner", "job", 101)
    con.rollback()
    assert con.execute("SELECT count(*) FROM queue").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM host_reservations").fetchone()[0] == 0
    with pytest.raises(HostingError, match="transaction"):
        reserve(con, "owner", "job", 1)
    con.close()


def test_settlement_cannot_release_an_uncertain_native_job_implicitly(tmp_path):
    con = ledger(tmp_path)
    con.execute("BEGIN IMMEDIATE")
    reserve(con, "owner", "uncertain-job", 100)
    con.commit()
    # A lost runner response has no automatic settlement entrypoint.
    con.execute("BEGIN IMMEDIATE")
    with pytest.raises(HostingError, match="concurrency"):
        reserve(con, "owner", "retry", 1)
    con.rollback()
    assert con.execute("SELECT used_seconds FROM host_reservations").fetchone()[0] is None
    con.close()
