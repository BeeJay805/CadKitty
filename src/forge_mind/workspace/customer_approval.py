"""Requested owner plan review, separate from CAD acceptance and final proof."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic import Field

from forge_mind.workspace.customer_projects import customer_project
from forge_mind.workspace.document import digest
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.input_application import source
from forge_mind.workspace.models import Record
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.store import Store

SCOPE = "Approval authorizes construction of the displayed design intent only. It proves no geometry, manufacture or physical performance."


class ApprovalRequest(Record):
    proposal_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    basis: EditBasis


class ProposalRevisionRequest(ApprovalRequest):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    instruction: str = Field(min_length=1, max_length=10000)


class CustomerApprovalRequiredError(WorkspaceError):
    pass


def _owned(con: Any, sid: str, owner: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    return dict(json.loads(row[0]))


def _valid(record: Any) -> TypeGuard[dict[str, Any]]:
    keys: tuple[str, ...] = ("basis", "plan", "contract", "source")
    if isinstance(record, dict) and record.get("version") == 2:
        keys = (*keys, "command_hash")
    return (
        isinstance(record, dict)
        and all(isinstance(key, str) for key in record)
        and all(key in record for key in keys)
        and record.get("proposal_hash") == digest({key: record[key] for key in keys})
    )


def _save_proposal(
    store: Store,
    con: Any,
    sid: str,
    doc: dict[str, Any],
    jid: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    store._check_plan_writer(con, sid, doc, jid)
    record = {
        "basis": store._edit_basis(con, sid, doc).model_dump(),
        "plan": deepcopy(snapshot["plan"]),
        "contract": deepcopy(snapshot["contract"]),
        "source": source(doc),
    }
    record.update(proposal_hash=digest(record), status="pending", job_id=jid, scope=SCOPE)
    previous = doc.get("customer_proposal")
    if previous:
        doc.setdefault("customer_proposal_history", []).append(previous)
    doc["customer_proposal"] = record
    con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
    return record


def pause_after_plan(service: WorkspaceService, sid: str, owner: str, jid: str) -> bool:
    """Pause the first saved plan only when the owner requested review."""
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        if not customer_project(doc) or doc.get("approval_required") is not True:
            return False
        record = doc.get("customer_proposal")
        if _valid(record) and record.get("approved_job_id") == jid:
            return False  # The common build boundary checks the actual target snapshot.
        snapshot = doc.get("operation_plan_snapshot")
        if not snapshot:
            return False
        _save_proposal(service.store, con, sid, doc, jid, snapshot)
        return True


def guard_construction(
    service: WorkspaceService,
    sid: str,
    owner: str,
    snapshot: dict[str, Any] | None,
) -> None:
    """Every typed build uses this guard, including atomic successor construction."""
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        if not customer_project(doc) or doc.get("approval_required") is not True:
            return
        service.store._check_plan_writer(con, sid, doc, doc.get("job_id"))
        record = doc.get("customer_proposal")
        target = snapshot or doc.get("operation_plan_snapshot")
        if not target:
            raise CustomerApprovalRequiredError(
                "Save a design plan before requesting owner approval"
            )
        if _valid(record) and record.get("version") == 2:
            from forge_mind.workspace.customer_construction import construction_authorized

            if construction_authorized(service.store, con, sid, owner, doc, target):
                return
            raise CustomerApprovalRequiredError("Current exact construction is not authorized")
        if (
            _valid(record)
            and record.get("status") == "approved"
            and record.get("approved_job_id") == doc.get("job_id")
            and record["source"] == source(doc)
            and digest(record["plan"]) == digest(target["plan"])
            and digest(record["contract"]) == digest(target["contract"])
        ):
            # Source and engineering changes must not become authority through a replay.
            basis = record["basis"]
            actual = service.store._edit_basis(con, sid, doc)
            if (
                actual.input_version == basis["input_version"]
                and actual.source_fingerprint == basis["source_fingerprint"]
                and actual.contract_hash == basis["contract_hash"]
            ):
                return
        _save_proposal(service.store, con, sid, doc, doc["job_id"], target)
    raise CustomerApprovalRequiredError("Review the current saved design plan before construction")


def preserve_budget(
    store: Store, sid: str, owner: str, jid: str, seconds: float | None, actions: int | None
) -> None:
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        if not _valid(record) or record.get("job_id") != jid or record.get("status") != "pending":
            raise WorkspaceError("Owner proposal changed before pausing its work budget")
        record["remaining_seconds"] = None if seconds is None else max(0, math.floor(seconds))
        record["remaining_actions"] = actions
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))


def read_proposal(store: Store, sid: str, owner: str) -> dict[str, Any]:
    with store.connect() as con:
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        if not doc.get("approval_required"):
            return {"status": "not_requested", "scope": SCOPE}
        if not _valid(record):
            return {"status": "stale", "scope": SCOPE}
        status = record["status"]
        if status == "pending":
            try:
                store._compare_edit_basis(con, sid, doc, record["basis"])
                if record["source"] != source(doc):
                    status = "stale"
            except WorkspaceError:
                status = "stale"
        return {**record, "status": status}


def approve_proposal(store: Store, sid: str, owner: str, request: ApprovalRequest) -> str:
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        if (
            not customer_project(doc)
            or not doc.get("approval_required")
            or not _valid(record)
            or request.proposal_hash != record["proposal_hash"]
            or request.basis.model_dump() != record["basis"]
        ):
            raise WorkspaceError("Owner proposal changed; read the current plan before approval")
        if record.get("approved_job_id"):
            return str(record["approved_job_id"])
        store._compare_edit_basis(con, sid, doc, request.basis)
        if record["source"] != source(doc) or doc.get("state") != "needs_approval":
            raise WorkspaceError("Owner proposal source or state changed before approval")
        seconds, actions = record.get("remaining_seconds"), record.get("remaining_actions")
        if (seconds is not None and seconds <= 0) or (actions is not None and actions <= 0):
            raise WorkspaceError("Original work budget is exhausted; approval cannot extend it")
        from forge_mind.workspace.customer_construction import JOB_BINDING, approval_binding

        binding = approval_binding(store, con, sid, owner, doc, record)
        jid = store._enqueue_with_connection(
            con,
            sid,
            owner,
            {
                "resume": True,
                "preview_only": False,
                "delivery_requested": True,
                "max_seconds": seconds,
                "max_actions": actions,
            },
            construction_approval=binding is not None,
        )
        queued = _owned(con, sid, owner)
        record.update(status="approved", approved_job_id=jid)
        if binding is not None:
            row = con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()
            payload = json.loads(row[0])
            payload[JOB_BINDING] = binding
            con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
            record["execution"].update(phase="approved", approved_at=time.time())
        queued["customer_proposal"] = record
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(queued), sid))
        return jid


def revise_proposal(store: Store, sid: str, owner: str, request: ProposalRevisionRequest) -> str:
    """A new explicit owner instruction is new work, with its own durable key."""
    from forge_mind.workspace.input_application import pending

    if not request.instruction.strip():
        raise WorkspaceError("A proposal revision needs a written instruction")
    payload_hash = digest({"session": sid, "request": request.model_dump(mode="json")})
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        prior = con.execute(
            "SELECT session,payload_hash,job FROM customer_operations WHERE owner=? AND request_id=?",
            (owner, request.request_id),
        ).fetchone()
        if prior:
            if prior["session"] != sid or prior["payload_hash"] != payload_hash:
                raise WorkspaceError(
                    "Request identity already belongs to a different proposal revision"
                )
            return str(prior["job"])
        record = doc.get("customer_proposal")
        if (
            not customer_project(doc)
            or not _valid(record)
            or record["proposal_hash"] != request.proposal_hash
            or record["basis"] != request.basis.model_dump()
            or doc.get("state") != "needs_approval"
            or record["source"] != source(doc)
        ):
            raise WorkspaceError("Owner proposal changed; read it before requesting a revision")
        store._compare_edit_basis(con, sid, doc, request.basis)
        snapshots = (doc, doc.get("resume_candidate") or {}, doc.get("proposal") or {})
        item = next((pending(value) for value in snapshots if pending(value) is not None), None)
        if item:
            store._abandon_input_with_connection(
                con, sid, owner, item["input_id"], request.basis, item["source_hash"]
            )
            doc = _owned(con, sid, owner)
        doc.setdefault("customer_proposal_history", []).append(deepcopy(record))
        doc["customer_proposal"] = {**record, "status": "stale"}
        if doc.get("resume_candidate"):
            doc.setdefault("abandoned_customer_drafts", []).append(doc["resume_candidate"])
            doc["resume_candidate"] = None
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc), sid))
        jid = store._enqueue_with_connection(
            con,
            sid,
            owner,
            {"instruction": request.instruction, "preview_only": False, "delivery_requested": True},
        )
        con.execute(
            "INSERT INTO customer_operations VALUES (?,?,?,?,?)",
            (owner, request.request_id, sid, payload_hash, jid),
        )
        return jid
