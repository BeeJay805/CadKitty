"""Explicit fresh attempts for narrowly allowlisted, immutable-input operations."""

from __future__ import annotations

import json
import socket
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.document import digest
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.product_sources import source_fingerprint
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.store import Store

OPERATIONS = {"review", "motion_check"}
SCOPE = (
    "A fresh attempt with the same frozen inputs and per-attempt budget. Original work is "
    "preserved; this does not continue an interrupted calculation or accept a design."
)


def operation(payload: dict[str, Any]) -> str | None:
    return payload.get("engineering_operation") or (
        "document" if payload.get("document") is not None else None
    )


def _snapshot(con: sqlite3.Connection, sid: str, doc: dict[str, Any], rid: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Original operation revision is unavailable")
    inputs = con.execute(
        "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    return {
        "revision_id": rid,
        "revision_digest": digest(json.loads(row[0])),
        "design_head": (doc.get("proposal") or {}).get("revision_id")
        or doc.get("selected_revision"),
        "accepted_revision": doc.get("accepted_revision", doc.get("selected_revision")),
        "contract_digest": digest(doc.get("contract")),
        "input_version": inputs[0] if inputs else 0,
        "input_digest": digest(
            json.loads(inputs[1]) if inputs else EngineeringInputs().model_dump()
        ),
        "source_fingerprint": source_fingerprint(con, sid),
        "engine_fingerprint": engine_fingerprint(),
    }


def bind_admission(
    con: sqlite3.Connection, sid: str, doc: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Only original admission may define the frozen dependencies of a rerun."""
    if operation(payload) not in OPERATIONS:
        return payload
    if any(
        key in payload
        for key in ("document", "resume", "restore_only", "lab_binding", "lab_realization")
    ) or payload.get("preview_only"):
        return payload
    rid = payload.get("revision_id")
    if not rid:
        return payload
    binding = _snapshot(con, sid, doc, rid)
    if operation(payload) == "review":
        binding["inspection_plan_digest"] = digest(doc.get("design_plan"))
    return {
        **payload,
        "operation_binding": {
            "version": 1,
            "worker_registration": "sqlite_before_compute_v1",
            "payload_hash": digest(payload),
            **binding,
        },
    }


def _check_artifacts(store: Store, con: sqlite3.Connection, sid: str, rid: str) -> None:
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    revision = json.loads(row[0])
    directory = store.directory(sid, rid)
    hashes = revision.get("measurement", {}).get("artifact_hashes", {})
    if not hashes or "part.step" not in hashes:
        raise WorkspaceError("Original operation has no verified geometry artifacts")
    for name, expected in hashes.items():
        path = directory / name
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or not regular_file(path)
            or file_sha256(path) != expected
        ):
            raise WorkspaceError("Original operation revision artifacts changed")


def check_binding(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    doc: dict[str, Any],
    payload: dict[str, Any],
    *,
    artifacts: bool,
) -> None:
    binding = payload.get("operation_binding")
    if (
        operation(payload) not in OPERATIONS
        or not isinstance(binding, dict)
        or binding.get("version") != 1
        or binding.get("worker_registration") != "sqlite_before_compute_v1"
    ):
        raise WorkspaceError("This operation has no supported frozen rerun binding")
    original = {k: v for k, v in payload.items() if k not in {"operation_binding", "recovery_of"}}
    if digest(original) != binding.get("payload_hash"):
        raise WorkspaceError("Original operation payload changed")
    current = _snapshot(con, sid, doc, payload["revision_id"])
    if operation(payload) == "review":
        if "inspection_plan_digest" in binding:
            current["inspection_plan_digest"] = digest(doc.get("design_plan"))
        elif (doc.get("design_plan") or {}).get("inspection"):
            raise WorkspaceError(
                "Historical review has no frozen inspection plan; start a new review"
            )
    if any(binding.get(k) != v for k, v in current.items()):
        raise WorkspaceError(
            "Operation revision, design, engineering inputs, sources or instruments changed; start a new operation"
        )
    if artifacts:
        _check_artifacts(store, con, sid, payload["revision_id"])


def check_previous_worker(store: Store, row: sqlite3.Row) -> None:
    if not store._owner_dead(row):
        raise WorkspaceError("Original worker is live or its ownership is unknown")
    for identity in json.loads(row["native_workers"] or "[]"):
        if not store._owner_dead(identity):
            raise WorkspaceError("A native worker is live or its ownership is unknown")
    if row["operation_directory"]:
        directory = Path(row["operation_directory"])
        if not directory.resolve().is_relative_to(store.root):
            raise WorkspaceError("Original operation directory is invalid")
        # Registered children are checked above. A child still importing cannot
        # register or compute once its original job has become interrupted.


def retry_payload(
    store: Store, con: sqlite3.Connection, sid: str, doc: dict[str, Any], jid: str
) -> tuple[dict[str, Any] | None, str | None]:
    row = con.execute("SELECT * FROM jobs WHERE id=? AND session=?", (jid, sid)).fetchone()
    if row is None:
        raise WorkspaceError("Interrupted operation not found")
    previous = con.execute(
        "SELECT id FROM jobs WHERE recovery_of=? AND session=?", (jid, sid)
    ).fetchone()
    if previous:
        return None, str(previous[0])
    if doc.get("job_id") != jid or row["state"] != "interrupted" or row["cancel"]:
        raise WorkspaceError("Only the current interrupted operation can be rerun")
    check_previous_worker(store, row)
    payload = json.loads(row["payload"])
    check_binding(store, con, sid, doc, payload, artifacts=True)
    return {**payload, "recovery_of": jid}, None


def descriptor(
    store: Store, con: sqlite3.Connection, sid: str, doc: dict[str, Any]
) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT * FROM jobs WHERE id=? AND session=?", (doc.get("job_id"), sid)
    ).fetchone()
    if row is None or row["state"] not in {"interrupted", "cancelled", "error"}:
        return None
    payload = json.loads(row["payload"])
    kind = operation(payload)
    if kind is None:
        return None
    answer = {
        "job_id": row["id"],
        "operation": kind,
        "revision_id": payload.get("revision_id"),
        "available": False,
        "reason": None,
        "action_label": "Run review again"
        if kind == "review"
        else "Run motion check again"
        if kind == "motion_check"
        else "Rerun unavailable",
        "scope": SCOPE,
    }
    try:
        if row["state"] != "interrupted" or row["cancel"]:
            raise WorkspaceError("Start a new explicit operation after cancellation or failure")
        check_previous_worker(store, row)
        check_binding(store, con, sid, doc, payload, artifacts=False)
        answer["available"] = True
    except (WorkspaceError, OSError, ValueError) as exc:
        answer["reason"] = str(exc)
    return answer


def _guard(con: sqlite3.Connection, store: Store, job: dict[str, Any]) -> dict[str, Any]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (job["session"], job["owner"])
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    current = con.execute(
        "SELECT payload,state,cancel FROM jobs WHERE id=? AND session=?",
        (job["id"], job["session"]),
    ).fetchone()
    if current is None or current["state"] != "running" or current["cancel"]:
        raise WorkspaceError("Operation is no longer running")
    payload = json.loads(current["payload"])
    actual = {k: v for k, v in job.items() if k not in {"id", "session", "owner"}}
    if digest(actual) != digest(payload):
        raise WorkspaceError("Dispatched operation differs from the frozen job payload")
    doc: dict[str, Any] = json.loads(row[0])
    if doc.get("job_id") != job["id"]:
        raise WorkspaceError("Operation no longer owns this project")
    check_binding(store, con, job["session"], doc, payload, artifacts=True)
    return doc


def _requires_binding(service: WorkspaceService, job: dict[str, Any]) -> bool:
    if job.get("operation_binding") or job.get("recovery_of"):
        return True
    if not job.get("id"):
        return False
    with service.store.connect() as con:
        row = con.execute(
            "SELECT payload FROM jobs WHERE id=? AND session=?", (job["id"], job["session"])
        ).fetchone()
    return bool(row and json.loads(row[0]).get("operation_binding"))


def guard_job(service: WorkspaceService, job: dict[str, Any]) -> None:
    """Repeat admission checks before computation and before publishing evidence."""
    if not _requires_binding(service, job):
        return  # Historical jobs remain executable but have no generic recovery route.
    with service.store.connect() as con:
        _guard(con, service.store, job)


def publish(service: WorkspaceService, job: dict[str, Any], **fields: Any) -> None:
    """Freshness, cancellation and session publication share one write transaction."""
    if not _requires_binding(service, job):
        service.store.update(job["session"], **fields)
        return
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        doc = _guard(con, service.store, job)
        doc.update(fields)
        con.execute(
            "UPDATE sessions SET document=? WHERE id=?",
            (json.dumps(doc, allow_nan=False), job["session"]),
        )


def worker_arguments(
    service: WorkspaceService, job: dict[str, Any], directory: Path
) -> dict[str, Any]:
    """Persist a launch intent; the child must register before doing native work."""
    if not _requires_binding(service, job):
        return {}
    if not directory.resolve().is_relative_to(service.store.root):
        raise WorkspaceError("Operation directory is outside the workspace")
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        _guard(con, service.store, job)
        token = uuid.uuid4().hex
        con.execute(
            "UPDATE jobs SET operation_directory=?,operation_launch_token=? WHERE id=?",
            (str(directory.resolve()), token, job["id"]),
        )
    return {
        "operation_job": {
            "root": str(service.store.root),
            "session": job["session"],
            "job_id": job["id"],
            "launch_token": token,
            "payload_hash": digest(
                {k: v for k, v in job.items() if k not in {"id", "session", "owner"}}
            ),
        }
    }


def register_child(directory: Path, context: dict[str, Any]) -> None:
    """Serialize late-child startup against recovery; no process scans are needed."""
    from forge_mind.workspace.store import Store

    store = Store(Path(context["root"]))
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM jobs WHERE id=? AND session=?", (context["job_id"], context["session"])
        ).fetchone()
        if row is None or row["state"] != "running" or row["cancel"]:
            raise WorkspaceError("Original operation ended before native worker startup")
        if (
            row["operation_directory"] != str(directory.resolve())
            or row["operation_launch_token"] != context.get("launch_token")
            or digest(json.loads(row["payload"])) != context["payload_hash"]
        ):
            raise WorkspaceError("Native worker does not match the admitted operation")
        child = psutil.Process()
        child_created = child.create_time()
        ancestors = child.parents()
        intermediaries = []
        owner = None
        younger_created = child_created
        for ancestor in ancestors:
            created = ancestor.create_time()
            if created > younger_created:
                raise WorkspaceError("Native worker ancestry changed during registration")
            younger_created = created
            if ancestor.pid == row["worker_pid"] and created == row["worker_created"]:
                owner = ancestor
                break
            intermediaries.append(
                {
                    "worker_pid": ancestor.pid,
                    "worker_created": created,
                    "worker_host": socket.gethostname(),
                }
            )
        if owner is None or row["worker_host"] != socket.gethostname() or store._owner_dead(row):
            raise WorkspaceError("Native worker has no matching live queue owner")
        identities = json.loads(row["native_workers"] or "[]")
        if identities:
            raise WorkspaceError("This operation already registered its native worker")
        identity = {
            "worker_pid": child.pid,
            "worker_created": child_created,
            "worker_host": socket.gethostname(),
        }
        con.execute(
            "UPDATE jobs SET native_workers=? WHERE id=?",
            (json.dumps([identity, *intermediaries]), row["id"]),
        )
