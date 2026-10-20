"""Owned physical pilot preparation and append-only reported outcomes."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.definition_meshes import MANIFEST_NAME, validate_definition_mesh
from forge_mind.workspace.delivery_guidance import _locked_context
from forge_mind.workspace.document import digest
from forge_mind.workspace.physical_pilot import SCOPE, build_physical_pilot
from forge_mind.workspace.physical_pilot_models import PhysicalPilotOutcomeRequest
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3

    from forge_mind.workspace.agent import WorkspaceService


def _saved_print_setup(
    service: WorkspaceService,
    con: sqlite3.Connection,
    sid: str,
    owner: str,
    rid: str,
    session: dict[str, Any],
) -> dict[str, Any] | None:
    records = session.get("print_setup_records", {})
    if not isinstance(records, dict):
        raise WorkspaceError("Saved print setup records are invalid")
    if rid not in records:
        return None
    # Geometry lane supplies a read-only receipt getter. It must check derivative
    # bytes and source identity; physical preparation never guesses its hashes.
    from forge_mind.workspace.print_setup_service import _locked_print_setup

    context = _locked_print_setup(service, con, sid, owner, rid)
    if context["status"] != "current" or not context.get("record"):
        raise WorkspaceError("Saved print setup is stale; refresh it before physical preparation")
    record = context["record"]
    return {
        "record_hash": context["record_hash"],
        **{
            key: record[key]
            for key in (
                "basis",
                "basis_hash",
                "selected_candidates",
                "files",
                "package",
                "result_sha256",
            )
        },
    }


def _prepare_locked(
    service: WorkspaceService,
    con: sqlite3.Connection,
    sid: str,
    owner: str,
    rid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(rid, str) or re.fullmatch(r"[a-f0-9]{32}", rid) is None:
        raise WorkspaceError("Physical pilot requires an exact saved revision identifier")
    session, guide = _locked_context(service, con, sid, owner, rid)
    if (guide["facts"].get("print_setup") or {}).get("status", "current") != "current":
        raise WorkspaceError("Saved print setup is stale; refresh it before physical preparation")
    if guide["design_revision"] != rid:
        raise WorkspaceError("Physical pilot requires the current design revision")
    if guide["status"] != "current":
        raise WorkspaceError("Refresh all four current instructions before physical preparation")
    if guide["basis"]["contract_hash"] != guide["basis"]["session_contract_hash"]:
        raise WorkspaceError("Current physical pilot requirements differ from captured CAD")
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Physical pilot revision is unavailable")
    revision = json.loads(row[0])
    setup = _saved_print_setup(service, con, sid, owner, rid, session)
    try:
        prepared = build_physical_pilot(revision, guide, setup)
        if revision["document"].get("assembly"):
            directory = service.store.directory(sid, rid)
            for part in prepared["protocol"]["made_parts"]:
                stem = "definition_" + part["part_id"]
                validate_definition_mesh(
                    (directory / "document.json").read_bytes(),
                    (directory / MANIFEST_NAME).read_bytes(),
                    rid,
                    part["part_id"],
                    revision["measurement"]["artifact_hashes"],
                    (directory / (stem + ".step")).read_bytes(),
                    (directory / (stem + ".stl")).read_bytes(),
                )
    except (ValueError, OSError) as exc:
        raise WorkspaceError(str(exc)) from exc
    return session, prepared


def prepare_physical_pilot(
    service: WorkspaceService, sid: str, owner: str, rid: str
) -> dict[str, Any]:
    """Read a deferred bench packet without adding jobs or physical observations."""
    with service.store.connect() as con:
        con.execute("BEGIN")
        _, packet = _prepare_locked(service, con, sid, owner, rid)
    return packet


def _verified_records(session: dict[str, Any], rid: str) -> list[dict[str, Any]]:
    mapping = session.get("physical_pilot_records", {})
    if not isinstance(mapping, dict):
        raise WorkspaceError("Physical pilot records are invalid")
    rows = mapping.get(rid, [])
    if not isinstance(rows, list):
        raise WorkspaceError("Physical pilot history is invalid")
    previous = None
    seen: set[str] = set()
    for sequence, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise WorkspaceError("Physical pilot record is invalid")
        payload = {key: value for key, value in row.items() if key != "receipt_sha256"}
        if (
            row.get("project_id") != session["id"]
            or row.get("revision_id") != rid
            or row.get("sequence") != sequence
            or row.get("previous_receipt_sha256") != previous
            or not isinstance(row.get("id"), str)
            or row["id"] in seen
            or digest(payload) != row.get("receipt_sha256")
        ):
            raise WorkspaceError("Physical pilot history integrity check failed")
        try:
            report = PhysicalPilotOutcomeRequest.model_validate(row["report"])
        except (ValueError, KeyError) as exc:
            raise WorkspaceError("Physical pilot report is invalid") from exc
        if (
            report.revision_id != rid
            or report.protocol_hash != row.get("protocol_hash")
            or row.get("binding", {}).get("revision_id") != rid
            or not isinstance(row.get("protocol"), dict)
            or digest(row["protocol"]) != report.protocol_hash
            or row["protocol"].get("binding") != row.get("binding")
        ):
            raise WorkspaceError("Physical pilot report binding is invalid")
        seen.add(row["id"])
        previous = row["receipt_sha256"]
    return rows


def record_outcome(
    service: WorkspaceService,
    sid: str,
    owner: str,
    request: PhysicalPilotOutcomeRequest,
) -> dict[str, Any]:
    """Append owner declarations for the current protocol; never accept a part."""
    request = PhysicalPilotOutcomeRequest.model_validate(request.model_dump())
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        session, packet = _prepare_locked(service, con, sid, owner, request.revision_id)
        if packet["protocol_hash"] != request.protocol_hash:
            raise WorkspaceError("Physical protocol changed; refresh before recording observations")
        known = {item["part_id"] for item in packet["protocol"]["hardware"]}
        if any(item.part_id is not None and item.part_id not in known for item in request.hardware):
            raise WorkspaceError("Reported hardware refers to an unknown bought definition")
        if (
            known
            and not request.hardware
            and any(
                check.check_id in {"hardware_fit", "assembly_fit", "removal"}
                and check.status.startswith("observed_")
                for check in request.checks
            )
        ):
            raise WorkspaceError(
                "Observed hardware or assembly checks need actual hardware declarations"
            )
        earlier = _verified_records(session, request.revision_id)
        reported = any(check.status != "not_run" for check in request.checks)
        row = {
            "id": uuid.uuid4().hex,
            "project_id": sid,
            "revision_id": request.revision_id,
            "sequence": len(earlier) + 1,
            "created_at": time.time(),
            "previous_receipt_sha256": earlier[-1]["receipt_sha256"] if earlier else None,
            "protocol_hash": packet["protocol_hash"],
            "protocol": packet["protocol"],
            "binding": packet["protocol"]["binding"],
            "report": request.model_dump(mode="json"),
            "physical_status": "reported" if reported else "not_run",
            "physical_qualification": "unverified",
            "acquisition_scope": "Authenticated owner's report; no authenticated instrument acquisition or independent physical confirmation.",
            "scope": SCOPE,
        }
        row["receipt_sha256"] = digest(row)
        mapping = dict(session.get("physical_pilot_records", {}))
        mapping[request.revision_id] = [*earlier, row]
        session["physical_pilot_records"] = mapping
        con.execute(
            "UPDATE sessions SET document=? WHERE id=? AND owner=?",
            (json.dumps(session, allow_nan=False), sid, owner),
        )
    return row


def read_outcomes(service: WorkspaceService, sid: str, owner: str, rid: str) -> dict[str, Any]:
    """Read immutable past reports, including after a revision is superseded."""
    if not isinstance(rid, str) or re.fullmatch(r"[a-f0-9]{32}", rid) is None:
        raise WorkspaceError("Physical pilot requires an exact saved revision identifier")
    with service.store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        session = json.loads(row[0])
        records = _verified_records(session, rid)
        head = (session.get("proposal") or {}).get("revision_id") or session.get(
            "selected_revision"
        )
        current_hash = None
        if head == rid:
            try:
                _, current = _prepare_locked(service, con, sid, owner, rid)
                current_hash = current["protocol_hash"]
            except WorkspaceError:
                pass  # Old reports remain readable when current inputs are unavailable.
    return {
        "revision_id": rid,
        "records": records,
        "historical": head != rid,
        "current_protocol_hash": current_hash,
        "record_context": {
            record["id"]: {
                "matches_current_protocol": current_hash is not None
                and record["protocol_hash"] == current_hash,
            }
            for record in records
        },
        "physical_qualification": "unverified",
        "scope": SCOPE,
    }
