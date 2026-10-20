"""Immutable, owner-scoped investigation records with explicit evidence ancestry."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.document import digest
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.product_sources import source_fingerprint
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

RESERVED = {
    "id",
    "kind",
    "project_id",
    "binding",
    "sequence",
    "created_at",
    "parents",
    "receipt_sha256",
    "stale",
    "matches_current_design",
    "unique_key",
}


def _table(service: WorkspaceService) -> None:
    with service.store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS lab_records ("
            "sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, "
            "session TEXT NOT NULL, kind TEXT NOT NULL, document TEXT NOT NULL, "
            "digest TEXT NOT NULL, unique_key TEXT, "
            "UNIQUE(session,kind,unique_key), FOREIGN KEY(session) REFERENCES sessions(id))"
        )
        con.execute("CREATE INDEX IF NOT EXISTS lab_session ON lab_records(session,sequence)")


def _snapshot(con: Any, sid: str, owner: str) -> tuple[dict[str, Any], dict[str, Any]]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Workspace not found")
    session = json.loads(row[0])
    inputs = con.execute(
        "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    engineering = {
        "input_version": inputs[0] if inputs else 0,
        "input_hash": digest(json.loads(inputs[1]) if inputs else EngineeringInputs().model_dump()),
        "source_fingerprint": source_fingerprint(con, sid),
    }
    return session, engineering


def _head(session: dict[str, Any]) -> str | None:
    return (session.get("proposal") or {}).get("revision_id") or session.get("selected_revision")


def capture_binding(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    input_version: int | None = None,
    require_current: bool = True,
) -> dict[str, Any]:
    service.store.session_record(sid, owner)
    revision = service._load_revision(sid, rid)
    if not revision.get("document") or not revision.get("measurement", {}).get("geometry_valid"):
        raise WorkspaceError("Engineering Lab requires a valid saved editable CAD revision")
    with service.store.connect() as con:
        session, inputs = _snapshot(con, sid, owner)
    if require_current and _head(session) != rid:
        raise WorkspaceError("Design changed; refresh before recording this investigation")
    if input_version is not None and inputs["input_version"] != input_version:
        raise WorkspaceError("Engineering inputs changed; refresh before recording")
    return {
        "revision_id": rid,
        **inputs,
        "geometry_hash": revision["measurement"].get("artifact_hashes", {}).get("part.step"),
        "document_hash": digest(revision["document"]),
        "contract_hash": digest(revision.get("contract") or {}),
        "engine_fingerprint": engine_fingerprint(),
    }


def _verify_row(
    con: Any,
    sid: str,
    identifier: str,
    seen: set[str],
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache = {} if cache is None else cache
    if identifier in seen or len(seen) >= 128:
        raise WorkspaceError("Investigation ancestry is cyclic or exceeds the supported depth")
    if identifier in cache:
        return cache[identifier]
    if len(cache) >= 1024:
        raise WorkspaceError("Investigation ancestry exceeds 1024 distinct records")
    seen.add(identifier)
    row = con.execute(
        "SELECT sequence,document,digest FROM lab_records WHERE session=? AND id=?",
        (sid, identifier),
    ).fetchone()
    if row is None:
        raise WorkspaceError("Investigation record not found")
    result: dict[str, Any] = json.loads(row[1])
    if (
        digest(result) != row[2]
        or result.get("id") != identifier
        or result.get("sequence") != row[0]
    ):
        raise WorkspaceError("Investigation receipt integrity check failed")
    for parent in result["parents"]:
        previous = _verify_row(con, sid, parent["id"], seen.copy(), cache)
        if previous["receipt_sha256"] != parent["receipt_sha256"] or previous["sequence"] >= row[0]:
            raise WorkspaceError("Investigation parent evidence changed")
    cache[identifier] = {**result, "receipt_sha256": row[2]}
    return cache[identifier]


def _binding_current(
    con: Any,
    sid: str,
    owner: str,
    binding: dict[str, Any],
    *,
    require_current: bool,
) -> None:
    session, inputs = _snapshot(con, sid, owner)
    if require_current and (
        _head(session) != binding["revision_id"]
        or any(binding.get(key) != value for key, value in inputs.items())
    ):
        raise WorkspaceError(
            "Investigation source or engineering inputs changed; recompute before saving"
        )
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, binding["revision_id"])
    ).fetchone()
    if row is None:
        raise WorkspaceError("Investigation source revision not found")
    revision = json.loads(row[0])
    if (
        digest(revision.get("document")) != binding["document_hash"]
        or digest(revision.get("contract") or {}) != binding["contract_hash"]
        or revision["measurement"].get("artifact_hashes", {}).get("part.step")
        != binding["geometry_hash"]
    ):
        raise WorkspaceError("Investigation source geometry or contract changed")


def save_record(
    service: WorkspaceService,
    sid: str,
    owner: str,
    kind: str,
    binding: dict[str, Any],
    payload: dict[str, Any],
    parents: Sequence[str] = (),
    require_current: bool = True,
    *,
    unique_key: str | None = None,
    expected_sequence: int | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z][a-z_]{0,47}", kind) or RESERVED.intersection(payload):
        raise WorkspaceError("Invalid investigation kind or reserved record fields")
    if len(parents) > 32 or len(set(parents)) != len(parents):
        raise WorkspaceError("Investigation parents must be distinct and bounded")
    service.store.session_record(sid, owner)
    # Verify source bytes outside the DB lock, then check DB identities inside it.
    service._load_revision(sid, binding["revision_id"])
    if require_current and binding["engine_fingerprint"] != engine_fingerprint():
        raise WorkspaceError("Analysis implementation changed; recompute before saving")
    _table(service)
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _binding_current(con, sid, owner, binding, require_current=require_current)
        last = con.execute(
            "SELECT coalesce(max(sequence),0) FROM lab_records WHERE session=?", (sid,)
        ).fetchone()[0]
        if expected_sequence is not None and expected_sequence != last:
            raise WorkspaceError("Investigation changed while computing; refresh and retry")
        if (
            unique_key is not None
            and con.execute(
                "SELECT 1 FROM lab_records WHERE session=? AND kind=? AND unique_key=?",
                (sid, kind, unique_key),
            ).fetchone()
        ):
            raise WorkspaceError("This immutable investigation outcome has already been recorded")
        cache: dict[str, dict[str, Any]] = {}
        ancestry = [_verify_row(con, sid, key, set(), cache) for key in parents]
        identifier = uuid.uuid4().hex
        cursor = con.execute(
            "INSERT INTO lab_records(id,session,kind,document,digest,unique_key) VALUES (?,?,?,'','',?)",
            (identifier, sid, kind, unique_key),
        )
        result = {
            "id": identifier,
            "kind": kind,
            "project_id": sid,
            "sequence": cursor.lastrowid,
            "created_at": time.time(),
            "binding": binding,
            "parents": [{"id": p["id"], "receipt_sha256": p["receipt_sha256"]} for p in ancestry],
            **payload,
        }
        data = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(data.encode()) > 4 * 1024 * 1024:
            raise WorkspaceError("Investigation record exceeds 4 MiB; split the dataset")
        sha = digest(result)
        con.execute(
            "UPDATE lab_records SET document=?,digest=? WHERE id=?", (data, sha, identifier)
        )
    return {**result, "receipt_sha256": sha, "stale": False, "matches_current_design": True}


def read_record(
    service: WorkspaceService,
    sid: str,
    owner: str,
    identifier: str,
    kind: str | None = None,
) -> dict[str, Any]:
    service.store.session_record(sid, owner)
    _table(service)
    with service.store.connect() as con:
        result = _verify_row(con, sid, identifier, set())
        if kind is not None and result["kind"] != kind:
            raise WorkspaceError(f"Expected {kind} investigation record")
        _binding_current(con, sid, owner, result["binding"], require_current=False)
        session, inputs = _snapshot(con, sid, owner)
    service._load_revision(sid, result["binding"]["revision_id"])
    return _decorate(result, session, inputs, engine_fingerprint())


def _decorate(
    result: dict[str, Any],
    session: dict[str, Any],
    inputs: dict[str, Any],
    engine: str,
) -> dict[str, Any]:
    binding = result["binding"]
    current = _head(session) == binding["revision_id"]
    return {
        **result,
        "matches_current_design": current,
        "stale": not current
        or binding["engine_fingerprint"] != engine
        or any(binding.get(key) != value for key, value in inputs.items()),
    }


def list_records(
    service: WorkspaceService,
    sid: str,
    owner: str,
    kind: str | None = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    service.store.session_record(sid, owner)
    _table(service)
    with service.store.connect() as con:
        ids = [
            r[0]
            for r in con.execute(
                "SELECT id FROM lab_records WHERE session=? AND (? IS NULL OR kind=?) "
                "ORDER BY sequence DESC LIMIT ?",
                (sid, kind, kind, min(200, max(1, limit))),
            )
        ]
        session, inputs = _snapshot(con, sid, owner)
        cache: dict[str, dict[str, Any]] = {}
        records = [_verify_row(con, sid, identifier, set(), cache) for identifier in ids]
        for record in records:
            _binding_current(con, sid, owner, record["binding"], require_current=False)
    for rid in {record["binding"]["revision_id"] for record in records}:
        service._load_revision(sid, rid)
    engine = engine_fingerprint()
    return [_decorate(record, session, inputs, engine) for record in records]


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Polling must not retransmit imported CSV, sampled curves or feature documents."""
    keys = RESERVED | {
        "status",
        "title",
        "name",
        "question",
        "scope",
        "provenance",
        "physical_qualification",
    }
    return {key: value for key, value in record.items() if key in keys}
