"""Command-bound customer review at the existing native construction boundary."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeGuard

from forge_mind.workspace.customer_approval import (
    SCOPE,
    CustomerApprovalRequiredError,
    _owned,
    _valid,
)
from forge_mind.workspace.customer_construction_claims import ConstructionClaim
from forge_mind.workspace.customer_projects import customer_project
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.input_application import source
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.construction_transaction import PreparedConstruction
    from forge_mind.workspace.store import Store

JOB_BINDING = "customer_construction_binding"


def _required(doc: dict[str, Any]) -> bool:
    return customer_project(doc) and doc.get("approval_required") is True


def _binding(store: Store, con: Any, sid: str, owner: str, doc: dict[str, Any]) -> dict[str, Any]:
    from forge_mind.workspace.customer_references import reference_image
    from forge_mind.workspace.document_service import owned_asset_paths
    from forge_mind.workspace.engine_identity import engine_fingerprint

    # Recheck real source bytes, rather than trusting metadata in the proposal.
    owned_asset_paths(store, sid, owner)
    if doc.get("customer_reference"):
        reference_image(store, sid, owner)
    return {
        "owner": owner,
        "session": sid,
        "basis": store._edit_basis(con, sid, doc).model_dump(),
        "source": source(doc),
        "engine_fingerprint": engine_fingerprint(),
        "assets_hash": digest(doc.get("assets", [])),
        "reference_hash": digest(doc.get("customer_reference")),
        "authority": store._native_author_binding(doc),
        "selection_version": doc.get("selection_version", 0),
        "acceptance": store._preview_acceptance_identity(doc),
    }


def _command_valid(record: Any) -> TypeGuard[dict[str, Any]]:
    return bool(
        _valid(record)
        and record.get("version") == 2
        and isinstance(record.get("command"), dict)
        and record["command"].get("version") == 1
        and record["command"].get("kind") == "document"
        and digest(record["command"]) == record.get("command_hash")
    )


def _job(con: Any, sid: str, jid: str) -> Any:
    return con.execute(
        "SELECT payload,state,cancel FROM jobs WHERE id=? AND session=?", (jid, sid)
    ).fetchone()


def _seal_matches(con: Any, sid: str, jid: str, record: dict[str, Any]) -> bool:
    row = _job(con, sid, jid)
    return bool(
        row
        and json.loads(row["payload"]).get(JOB_BINDING)
        == {"proposal_hash": record["proposal_hash"], "command_hash": record["command_hash"]}
    )


def _live(
    store: Store, con: Any, sid: str, owner: str, doc: dict[str, Any], record: dict[str, Any]
) -> None:
    if (
        not _command_valid(record)
        or _binding(store, con, sid, owner, doc) != record["command"]["binding"]
    ):
        raise WorkspaceError(
            "Approved construction source, basis, engine, assets or command changed"
        )


def _published_current(
    store: Store, con: Any, sid: str, owner: str, doc: dict[str, Any], record: dict[str, Any]
) -> bool:
    from forge_mind.workspace.customer_construction_publication import publication_current

    try:
        return publication_current(
            store, con, sid, doc, record, _binding(store, con, sid, owner, doc)
        )
    except (WorkspaceError, ValueError, OSError):
        return False


def stage_construction(
    service: WorkspaceService,
    sid: str,
    owner: str,
    jid: str,
    document: DesignDocument,
    contract: Contract,
    parent: str | None,
    explanation: str,
    *,
    prepared: PreparedConstruction | None = None,
    feature_basis: EditBasis | None = None,
    remaining_seconds: float | None = None,
    remaining_actions: int | None = None,
) -> None:
    from forge_mind.workspace.construction_transaction import validated_successor
    from forge_mind.workspace.document_service import preflight_document

    current = service.store.session_record(sid, owner)
    if not _required(current):
        return
    target = (
        validated_successor(prepared)
        if prepared is not None
        else service.store.plan_snapshot(sid, owner)
    )
    if prepared is not None and prepared.document != document:
        raise WorkspaceError("Prepared construction document changed")
    if feature_basis is not None and service.store.edit_basis(sid, owner) != feature_basis:
        raise WorkspaceError("Feature construction basis changed")
    _, target, _, _ = preflight_document(service, sid, owner, document, contract, parent, target)
    if target is None:
        raise WorkspaceError("Construction requires a saved validated target plan")
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        service.store._check_plan_writer(con, sid, doc, jid)
        binding = _binding(service.store, con, sid, owner, doc)
        row = _job(con, sid, jid)
        assert row is not None
        policy = json.loads(row["payload"])
        command = {
            "version": 1,
            "kind": "document",
            "document": document.model_dump(mode="json"),
            "contract": contract.model_dump(mode="json"),
            "parent": parent,
            "target_snapshot": target,
            "plan_patch": prepared.patch.model_dump(mode="json", exclude_unset=True)
            if prepared
            else None,
            "feature_basis": feature_basis.model_dump() if feature_basis else None,
            "explanation": explanation,
            "binding": binding,
            "originating_job_id": jid,
            "original_budget": {key: policy.get(key) for key in ("max_seconds", "max_actions")},
        }
        record = doc.get("customer_proposal")
        if _command_valid(record) and _seal_matches(con, sid, jid, record):
            execution = record.get("execution") or {}
            retained = record["command"]
            if (
                execution.get("phase") == "claimed"
                and execution.get("job_id") == jid
                and all(
                    command[key] == retained[key]
                    for key in (
                        "document",
                        "contract",
                        "parent",
                        "target_snapshot",
                        "plan_patch",
                        "feature_basis",
                        "explanation",
                        "binding",
                    )
                )
            ):
                return
            if execution.get("phase") != "published":
                raise WorkspaceError("Claimed construction cannot be replaced or replayed")
            if (
                command["document"] == retained["document"]
                and command["target_snapshot"] == retained["target_snapshot"]
            ):
                raise WorkspaceError(
                    "Approved construction is already published; inspect its saved revision"
                )
        if _command_valid(record) and record.get("execution", {}).get("phase") in {
            "claimed",
            "uncertain",
        }:
            raise WorkspaceError("Construction outcome is uncertain; automatic replay is forbidden")
        record = {
            "version": 2,
            "basis": binding["basis"],
            "plan": deepcopy(target["plan"]),
            "contract": deepcopy(target["contract"]),
            "source": binding["source"],
            "command_hash": digest(command),
            "command": command,
        }
        record["proposal_hash"] = digest(
            {key: record[key] for key in ("basis", "plan", "contract", "source", "command_hash")}
        )
        record.update(
            status="pending",
            job_id=jid,
            scope=SCOPE,
            remaining_seconds=remaining_seconds,
            remaining_actions=remaining_actions,
            execution={"phase": "pending", "staged_at": time.time()},
        )
        if doc.get("customer_proposal"):
            doc.setdefault("customer_proposal_history", []).append(doc["customer_proposal"])
        doc["customer_proposal"] = record
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc, allow_nan=False), sid)
        )
    raise CustomerApprovalRequiredError(
        "Review the exact validated construction command before CAD is built"
    )


def approval_binding(
    store: Store, con: Any, sid: str, owner: str, doc: dict[str, Any], record: dict[str, Any]
) -> dict[str, str] | None:
    if record.get("version") != 2:
        return None
    _live(store, con, sid, owner, doc, record)
    if record.get("execution", {}).get("phase") != "pending":
        raise WorkspaceError("Construction proposal already has an execution outcome")
    return {"proposal_hash": record["proposal_hash"], "command_hash": record["command_hash"]}


def resume_binding(
    con: Any, doc: dict[str, Any], previous: dict[str, Any], payload: dict[str, Any]
) -> None:
    binding = previous.get(JOB_BINDING)
    if binding is None:
        return
    record = doc.get("customer_proposal")
    if not _command_valid(record) or binding != {
        "proposal_hash": record["proposal_hash"],
        "command_hash": record["command_hash"],
    }:
        raise WorkspaceError("Saved construction approval changed before resume")
    phase = record.get("execution", {}).get("phase")
    if phase in {"claimed", "uncertain", "not_applied"}:
        raise WorkspaceError("Construction outcome is uncertain; automatic replay is forbidden")
    for key in ("max_seconds", "max_actions"):
        remaining = record.get("remaining_seconds" if key == "max_seconds" else "remaining_actions")
        if remaining is not None and (payload.get(key) is None or payload[key] > remaining):
            raise WorkspaceError("Construction resume cannot extend its approved remaining budget")
    payload[JOB_BINDING] = deepcopy(binding)


def claim_construction(store: Store, sid: str, owner: str, jid: str) -> ConstructionClaim | None:
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        row = _job(con, sid, jid)
        if row is None or not json.loads(row["payload"]).get(JOB_BINDING):
            return None
        store._check_plan_writer(con, sid, doc, jid)
        if not _command_valid(record) or not _seal_matches(con, sid, jid, record):
            raise WorkspaceError("Construction approval binding changed")
        phase = record.get("execution", {}).get("phase")
        if phase == "published":
            if not _published_current(store, con, sid, owner, doc, record):
                raise WorkspaceError("Published construction changed before continuation")
            return {
                "mode": "continue_published",
                "revision_id": record["execution"]["revision_id"],
                "execution_job_id": record["execution"]["job_id"],
                "command_hash": record["command_hash"],
            }
        if phase != "approved" or record.get("status") != "approved":
            raise WorkspaceError("Construction outcome is uncertain; automatic replay is forbidden")
        _live(store, con, sid, owner, doc, record)
        command = record["command"]
        record["execution"].update(
            phase="claimed", job_id=jid, claimed_at=time.time(), build_started=False
        )
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc, allow_nan=False), sid)
        )
        return {
            "mode": "execute_saved_command",
            "action": {
                "kind": "document",
                "document_json": json.dumps(command["document"], allow_nan=False),
                "plan_patch_json": json.dumps(command["plan_patch"], allow_nan=False)
                if command["plan_patch"]
                else "",
                "explanation": command["explanation"],
            },
            "feature_basis": deepcopy(command["feature_basis"]),
            "command_hash": record["command_hash"],
        }


def construction_authorized(
    store: Store, con: Any, sid: str, owner: str, doc: dict[str, Any], target: dict[str, Any] | None
) -> bool:
    record = doc.get("customer_proposal")
    if not _command_valid(record) or record.get("status") != "approved" or not target:
        return False
    execution = record.get("execution") or {}
    if execution.get("phase") != "claimed" or execution.get("job_id") != doc.get("job_id"):
        return False
    try:
        store._check_plan_writer(con, sid, doc, doc["job_id"])
        _live(store, con, sid, owner, doc, record)
    except (WorkspaceError, ValueError, OSError):
        return False
    return (
        _seal_matches(con, sid, doc["job_id"], record)
        and record["command"]["target_snapshot"] == target
    )


def guard_document(
    service: WorkspaceService,
    sid: str,
    owner: str,
    document: DesignDocument,
    contract: Contract,
    parent: str | None,
    explanation: str,
    target: dict[str, Any] | None,
) -> None:
    from forge_mind.workspace.customer_approval import guard_construction

    doc = service.store.session_record(sid, owner)
    if not _required(doc):
        return
    record = doc.get("customer_proposal")
    if not _command_valid(record):
        stage_construction(
            service, sid, owner, doc["job_id"], document, contract, parent, explanation
        )
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        if not _command_valid(record) or not construction_authorized(
            service.store, con, sid, owner, doc, target
        ):
            raise CustomerApprovalRequiredError(
                "Current exact construction requires approval and a durable execution claim"
            )
        command = record["command"]
        if (
            document.model_dump(mode="json") != command["document"]
            or contract.model_dump(mode="json") != command["contract"]
            or parent != command["parent"]
            or record["execution"].get("build_started")
        ):
            raise WorkspaceError("Approved construction command differs or has already started")
        record["execution"].update(build_started=True, build_started_at=time.time())
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc, allow_nan=False), sid)
        )
    guard_construction(service, sid, owner, target)


def complete_construction(
    store: Store,
    sid: str,
    owner: str,
    jid: str,
    revision_id: str | None,
    *,
    publication: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _owned(con, sid, owner)
        record = doc.get("customer_proposal")
        if not _command_valid(record) or not _seal_matches(con, sid, jid, record):
            return
        execution = record.get("execution") or {}
        if execution.get("phase") == "published" and execution.get("revision_id") == revision_id:
            return
        store._check_plan_writer(con, sid, doc, jid)
        if execution.get("phase") != "claimed" or execution.get("job_id") != jid:
            raise WorkspaceError("Construction has no current durable execution claim")
        phase = "uncertain" if execution.get("build_started") else "not_applied"
        if revision_id is not None and error is None:
            row = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?", (sid, revision_id)
            ).fetchone()
            revision = json.loads(row[0]) if row else {}
            current = _binding(store, con, sid, owner, doc)
            original = record["command"]["binding"]
            if (
                execution.get("build_started") is True
                and doc.get("selected_revision") == revision_id
                and all(
                    current[key] == original[key]
                    for key in (
                        "owner",
                        "session",
                        "source",
                        "engine_fingerprint",
                        "assets_hash",
                        "reference_hash",
                    )
                )
                and all(
                    current["basis"][key] == original["basis"][key]
                    for key in ("input_version", "source_fingerprint", "contract_hash")
                )
                and store._edit_target_snapshot(doc, current["basis"]["contract_hash"])
                == record["command"]["target_snapshot"]
                and revision.get("document") == record["command"]["document"]
                and revision.get("contract") == record["command"]["contract"]
                and revision.get("parent_id") == record["command"]["parent"]
                and (revision.get("measurement") or {}).get("contract_hash")
                == current["basis"]["contract_hash"]
                and (revision.get("operation_record") or {}).get("plan_snapshot")
                == record["command"]["target_snapshot"]
                and (revision.get("measurement") or {}).get("geometry_valid") is True
                and (publication is None or publication.get("applied") is True)
            ):
                phase = "published"
                execution["publication_binding"] = current
            elif row and (revision.get("measurement") or {}).get("geometry_valid") is False:
                phase = "not_applied"
            else:
                raise WorkspaceError("Construction publication differs from its approved command")
        execution.update(
            phase=phase,
            revision_id=revision_id,
            publication=deepcopy(publication),
            error=error,
            completed_at=time.time(),
        )
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?", (json.dumps(doc, allow_nan=False), sid)
        )


def author_workflow_context(store: Store, sid: str, owner: str, jid: str) -> dict[str, Any] | None:
    with store.connect() as con:
        doc = _owned(con, sid, owner)
        if not customer_project(doc):
            return None
        record = doc.get("customer_proposal")
        result: dict[str, Any] = {
            "approval_required": _required(doc),
            "status": "not_requested" if not _required(doc) else "not_prepared",
            "construction_authorized": False,
            "scope": SCOPE,
        }
        if _valid(record):
            result.update(
                status=record.get("status"),
                proposal_hash=record["proposal_hash"],
                approved_basis=record["basis"],
                approved_job_id=record.get("approved_job_id"),
            )
            if record.get("version") != 2:
                result.update(
                    status="inventory_only",
                    next_action="Prepare a complete validated construction command for review",
                )
            else:
                result.update(
                    command_hash=record["command_hash"], execution=deepcopy(record.get("execution"))
                )
                if record.get("execution", {}).get("phase") == "published":
                    if _published_current(store, con, sid, owner, doc, record):
                        result["status"] = "published"
                        result["next_action"] = (
                            "Continue current checks, print preparation and guidance; approval is not physical proof"
                        )
                    else:
                        result["status"] = "stale"
                else:
                    result["construction_authorized"] = doc.get(
                        "job_id"
                    ) == jid and construction_authorized(
                        store, con, sid, owner, doc, record["command"]["target_snapshot"]
                    )
                    try:
                        _live(store, con, sid, owner, doc, record)
                    except (WorkspaceError, ValueError, OSError):
                        result.update(status="stale", construction_authorized=False)
                if result["status"] != "stale":
                    try:
                        store._check_plan_writer(con, sid, doc, jid)
                    except WorkspaceError:
                        result.update(status="inactive", construction_authorized=False)
                        result.pop("next_action", None)
        return result


def construction_finalization_allowed(store: Store, sid: str, owner: str, jid: str) -> bool:
    """Only an exact published command may enter downstream automatic work."""
    with store.connect() as con:
        doc = _owned(con, sid, owner)
        row = _job(con, sid, jid)
        if row is None:
            return False
        if not json.loads(row["payload"]).get(JOB_BINDING):
            return True
        try:
            store._check_plan_writer(con, sid, doc, jid)
        except WorkspaceError:
            return False
        record = doc.get("customer_proposal")
        return bool(
            _command_valid(record)
            and _seal_matches(con, sid, jid, record)
            and record.get("execution", {}).get("phase") == "published"
            and _published_current(store, con, sid, owner, doc, record)
        )
