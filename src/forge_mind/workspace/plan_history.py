"""Archive server-owned evidence pointers without making historical proof current."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from typing import Any


def request_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def plan_history_entry(
    con: sqlite3.Connection,
    sid: str,
    session: dict[str, Any],
    *,
    head_revision_id: str | None,
) -> dict[str, Any]:
    """Use the caller's transaction; archived receipts never supply a current verdict."""
    prior = session.get("completion_report") or {}
    entry = {
        "plan": deepcopy(session["design_plan"]),
        "plan_version": session.get("plan_version", 0),
        "plan_hash": session.get("plan_hash"),
        "contract_hash": session.get("plan_contract_hash"),
        "revision_id": prior.get("revision_id", head_revision_id),
        "receipt_ids": list(prior.get("receipt_ids", [])),
        "status": prior.get("status", "unverified"),
    }
    definitions = {
        check["id"]: check
        for subsystem in session["design_plan"].get("subsystems", [])
        for check in subsystem.get("checks", [])
    }
    pointers: dict[str, set[str]] = {check_id: set() for check_id in definitions}
    for check_id, cached in (session.get("prepared_check_cache") or {}).items():
        if check_id in pointers and cached.get("receipt_id"):
            pointers[check_id].add(cached["receipt_id"])
    for report in (prior, session.get("author_checkpoint") or {}):
        for subsystem in report.get("subsystems", []):
            for check in subsystem.get("checks", []):
                if check.get("id") in pointers and check.get("receipt_id"):
                    pointers[check["id"]].add(check["receipt_id"])
    if not any(pointers.values()):
        return entry
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='product_checks'"
    ).fetchone()
    if not exists:
        return entry
    records = []
    for check_id, receipt_ids in pointers.items():
        for rid in sorted(receipt_ids):
            row = con.execute(
                "SELECT document,digest FROM product_checks WHERE session=? AND id=?", (sid, rid)
            ).fetchone()
            if row is None or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                continue
            try:
                receipt = json.loads(row[0])
                source = con.execute(
                    "SELECT document FROM revisions WHERE session=? AND id=?",
                    (sid, receipt.get("revision_id")),
                ).fetchone()
                revision = json.loads(source[0]) if source else {}
                if (
                    receipt.get("id") != rid
                    or request_hash(receipt.get("request"))
                    != request_hash(definitions[check_id]["request"])
                    or receipt.get("contract") != revision.get("contract")
                    or receipt.get("artifact_hashes")
                    != revision.get("measurement", {}).get("artifact_hashes")
                    or receipt.get("engine_fingerprint")
                    != revision.get("measurement", {}).get("engine_fingerprint")
                    or not revision
                ):
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            records.append(
                {
                    "check_id": check_id,
                    "request_hash": request_hash(definitions[check_id]["request"]),
                    "receipt_id": rid,
                    "receipt_digest": row[1],
                    "revision_id": receipt["revision_id"],
                    "verification_scope": "historical_receipt_identity; not current completion proof",
                }
            )
    if records:
        entry["prior_check_receipts"] = records
        entry["receipt_ids"] = list(
            dict.fromkeys(
                [
                    *entry["receipt_ids"],
                    *(record["receipt_id"] for record in records),
                ]
            )
        )
    return entry


def historical_check_context(
    con: sqlite3.Connection,
    sid: str,
    session: dict[str, Any],
    plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read archived criterion receipts without supplying any current verdict.

    Archived pointers must still match their literal immutable receipt and saved
    source identity. A broken pointer remains an explicit historical uncertainty.
    Only currently amended check IDs enter this compact customer projection.
    """
    active = {
        check["id"]: check
        for subsystem in (plan or session.get("design_plan") or {}).get("subsystems", [])
        for check in subsystem.get("checks", [])
        if check.get("criterion_amendment")
    }
    if not active:
        return []
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='product_checks'"
    ).fetchone()
    result = []
    seen: set[tuple[str, str, str | None]] = set()
    successors = {
        (check_id, check["criterion_amendment"]["previous_request_hash"]): {
            "amendment": deepcopy(check["criterion_amendment"]),
            "successor_request_hash": request_hash(check["request"]),
        }
        for check_id, check in active.items()
    }
    for entry in reversed(session.get("design_plan_history", [])):
        definitions = {
            check["id"]: check
            for subsystem in (entry.get("plan") or {}).get("subsystems", [])
            for check in subsystem.get("checks", [])
            if check.get("id") in active
        }
        pointers: dict[str, list[dict[str, Any]]] = {}
        for pointer in entry.get("prior_check_receipts", []):
            if pointer.get("check_id") in definitions:
                pointers.setdefault(pointer["check_id"], []).append(pointer)
        for check_id, definition in definitions.items():
            old_hash = request_hash(definition["request"])
            for pointer in pointers.get(check_id) or [{}]:
                receipt_id = pointer.get("receipt_id")
                key = (check_id, old_hash, receipt_id)
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "check_id": check_id,
                    "request_hash": old_hash,
                    "tool": definition["request"].get("tool"),
                    "receipt_id": receipt_id,
                    "receipt_digest": pointer.get("receipt_digest"),
                    "revision_id": pointer.get("revision_id"),
                    "historical_status": "unverified",
                    "verification": "unverified_historical",
                    "reason": "Archived receipt is missing or no longer matches its saved identity",
                    **successors.get((check_id, old_hash), {"amendment": None}),
                    "active_request_hash": request_hash(active[check_id]["request"]),
                    "scope": "Historical criterion and receipt only; never current completion proof or physical approval.",
                }
                saved = (
                    con.execute(
                        "SELECT document,digest FROM product_checks WHERE session=? AND id=?",
                        (sid, receipt_id),
                    ).fetchone()
                    if exists and receipt_id
                    else None
                )
                try:
                    if saved is None or not (
                        hashlib.sha256(saved[0].encode()).hexdigest()
                        == saved[1]
                        == pointer.get("receipt_digest")
                    ):
                        raise ValueError("Historical receipt digest changed")
                    receipt = json.loads(saved[0])
                    source = con.execute(
                        "SELECT document FROM revisions WHERE session=? AND id=?",
                        (sid, receipt.get("revision_id")),
                    ).fetchone()
                    revision = json.loads(source[0]) if source else {}
                    if (
                        not revision
                        or receipt.get("id") != receipt_id
                        or receipt.get("revision_id") != pointer.get("revision_id")
                        or request_hash(receipt.get("request")) != old_hash
                        or pointer.get("request_hash") != old_hash
                        or receipt.get("contract") != revision.get("contract")
                        or receipt.get("artifact_hashes")
                        != revision.get("measurement", {}).get("artifact_hashes")
                        or receipt.get("engine_fingerprint")
                        != revision.get("measurement", {}).get("engine_fingerprint")
                    ):
                        raise ValueError("Historical receipt source changed")
                    measured = receipt.get("result") or {}
                    status = measured.get("status")
                    if status not in {"pass", "fail", "unverified"}:
                        raise ValueError("Historical receipt status is unsupported")
                    row.update(
                        historical_status=status,
                        verification="verified_historical",
                        algorithm_scope=measured.get("algorithm_scope"),
                        reason="Verified archived receipt; the current corrected check has a separate request and result",
                    )
                except (ValueError, KeyError, TypeError):
                    pass
                result.append(row)
        for check_id, definition in definitions.items():
            amendment = definition.get("criterion_amendment")
            if amendment:
                successors[(check_id, amendment["previous_request_hash"])] = {
                    "amendment": deepcopy(amendment),
                    "successor_request_hash": request_hash(definition["request"]),
                }
    return list(reversed(result))
