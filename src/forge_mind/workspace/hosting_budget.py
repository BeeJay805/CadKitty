"""Transactional operator compute allowances, independent of money and Stripe."""

from __future__ import annotations

import sqlite3
from typing import Any

from forge_mind.workspace.hosting import HostingError


def initialize(con: sqlite3.Connection) -> None:
    # Do not executescript here: it commits an existing admission transaction.
    con.execute(
        "CREATE TABLE IF NOT EXISTS host_grants (grant_id TEXT PRIMARY KEY, owner TEXT NOT NULL, seconds INTEGER NOT NULL CHECK(seconds>0))"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS host_lineages (lineage_id TEXT PRIMARY KEY, owner TEXT NOT NULL, seconds INTEGER NOT NULL CHECK(seconds>0))"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS host_reservations (job_id TEXT PRIMARY KEY, owner TEXT NOT NULL, seconds INTEGER NOT NULL CHECK(seconds>0), used_seconds INTEGER, lineage_id TEXT NOT NULL, CHECK(used_seconds IS NULL OR (used_seconds>=0 AND used_seconds<=seconds)))"
    )


def _transaction(con: sqlite3.Connection) -> None:
    if not con.in_transaction:
        raise HostingError("Compute authority requires the caller's admission transaction")


def _positive(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise HostingError("Compute allowance must be positive integer seconds")


def grant(con: sqlite3.Connection, owner: str, grant_id: str, seconds: int) -> None:
    """Operator-only grant; no public route may call this function."""
    _transaction(con)
    _positive(seconds)
    row = con.execute(
        "SELECT owner,seconds FROM host_grants WHERE grant_id=?", (grant_id,)
    ).fetchone()
    if row is not None:
        if tuple(row) != (owner, seconds):
            raise HostingError("Compute grant replay conflicts with original authority")
        return
    con.execute("INSERT INTO host_grants VALUES (?,?,?)", (grant_id, owner, seconds))


def reserve(
    con: sqlite3.Connection,
    owner: str,
    job_id: str,
    seconds: int,
    *,
    active_limit: int = 1,
    lineage_id: str | None = None,
) -> dict[str, Any]:
    _transaction(con)
    _positive(seconds)
    _positive(active_limit)
    lineage_id = job_id if lineage_id is None else lineage_id
    row = con.execute(
        "SELECT owner,seconds,used_seconds,lineage_id FROM host_reservations WHERE job_id=?",
        (job_id,),
    ).fetchone()
    if row is not None:
        if tuple(row)[:2] != (owner, seconds) or row[3] != lineage_id:
            raise HostingError("Compute reservation replay conflicts with original authority")
        return {"job_id": job_id, "seconds": seconds, "settled": row[2] is not None}
    active = con.execute(
        "SELECT count(*) FROM host_reservations WHERE owner=? AND used_seconds IS NULL", (owner,)
    ).fetchone()[0]
    if active >= active_limit:
        raise HostingError("Owner compute concurrency limit reached")
    granted = con.execute(
        "SELECT coalesce(sum(seconds),0) FROM host_grants WHERE owner=?", (owner,)
    ).fetchone()[0]
    used = con.execute(
        "SELECT coalesce(sum(coalesce(used_seconds,seconds)),0) FROM host_reservations WHERE owner=?",
        (owner,),
    ).fetchone()[0]
    if granted - used < seconds:
        raise HostingError("Insufficient operator compute allowance")
    lineage = con.execute(
        "SELECT owner,seconds FROM host_lineages WHERE lineage_id=?", (lineage_id,)
    ).fetchone()
    if lineage is None:
        if lineage_id != job_id:
            raise HostingError("Continuation requires its original compute lineage")
        con.execute("INSERT INTO host_lineages VALUES (?,?,?)", (lineage_id, owner, seconds))
    else:
        consumed = con.execute(
            "SELECT coalesce(sum(coalesce(used_seconds,seconds)),0) FROM host_reservations WHERE lineage_id=?",
            (lineage_id,),
        ).fetchone()[0]
        if lineage[0] != owner or seconds > lineage[1] - consumed:
            raise HostingError("Continuation exceeds its original compute authority")
    con.execute(
        "INSERT INTO host_reservations VALUES (?,?,?,NULL,?)", (job_id, owner, seconds, lineage_id)
    )
    return {"job_id": job_id, "lineage_id": lineage_id, "seconds": seconds, "settled": False}


def settle(con: sqlite3.Connection, owner: str, job_id: str, used_seconds: int) -> None:
    _transaction(con)
    row = con.execute(
        "SELECT owner,seconds,used_seconds FROM host_reservations WHERE job_id=?", (job_id,)
    ).fetchone()
    if row is None or row[0] != owner:
        raise HostingError("Compute reservation does not belong to this owner/job")
    if type(used_seconds) is not int or not 0 <= used_seconds <= row[1]:
        raise HostingError("Consumed compute exceeds its reservation")
    if row[2] is not None and row[2] != used_seconds:
        raise HostingError("Settled compute cannot be rewritten")
    con.execute(
        "UPDATE host_reservations SET used_seconds=? WHERE job_id=?", (used_seconds, job_id)
    )
