"""Revision-bound product tool receipts; author probes never rewrite acceptance."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import process_revision
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _table(service: WorkspaceService) -> None:
    with service.store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS product_checks (id TEXT PRIMARY KEY, session TEXT NOT NULL, document TEXT NOT NULL, digest TEXT NOT NULL, FOREIGN KEY(session) REFERENCES sessions(id))"
        )


def run_tool(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str | None,
    request: dict[str, Any],
    timeout: float | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    *,
    check_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, rid) if rid else None
    sources = read_sources(service.store, sid, owner)
    process = (
        process_revision
        if service.executor.mode == "trusted_local"
        else service.executor.process_observation
    )
    with tempfile.TemporaryDirectory(prefix="forge-product-tool-") as scratch:
        result = process(
            Path(scratch),
            {
                "operation": "product_tool",
                "source": str(service.store.directory(sid, rid)) if rid else None,
                "request": request,
            },
            timeout,
            cancelled,
        )
    if cancelled():
        raise WorkspaceError("Product check cancelled before publishing evidence")
    receipt = {
        "id": check_id or uuid.uuid4().hex,
        "job_id": job_id,
        "revision_id": rid,
        "at": time.time(),
        "engine_fingerprint": engine_fingerprint(),
        "source_fingerprint": sources["fingerprint"],
        "contract": revision["contract"] if revision else session.get("contract"),
        "artifact_hashes": revision["measurement"]["artifact_hashes"] if revision else {},
        "request": request,
        "result": result,
        "acceptance_scope": "Exploratory engineering evidence. Probe limits do not replace frozen acceptance requirements; physical qualification remains unverified.",
    }
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    if len(encoded) > 2 * 1024 * 1024:
        raise WorkspaceError("Product tool receipt exceeds its size budget")
    _table(service)
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        if job_id:
            job = con.execute(
                "SELECT state,cancel FROM jobs WHERE id=? AND session=?", (job_id, sid)
            ).fetchone()
            if job is None or job[0] != "running" or job[1]:
                raise WorkspaceError("Product check cancelled before publishing evidence")
        con.execute(
            "INSERT INTO product_checks VALUES (?,?,?,?)",
            (receipt["id"], sid, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
    return receipt


def read_tool_receipt(
    service: WorkspaceService, sid: str, owner: str, check_id: str
) -> dict[str, Any]:
    """Read one integrity-checked receipt without expanding other checks or history."""
    session = service.store.session_record(sid, owner)
    _table(service)
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,digest FROM product_checks WHERE id=? AND session=?", (check_id, sid)
        ).fetchone()
    if row is None:
        raise WorkspaceError("Product check is not available; poll its job before reading")
    if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
        raise WorkspaceError("Engineering tool receipt integrity check failed")
    receipt = json.loads(row[0])
    rid = receipt["revision_id"]
    revision = service._load_revision(sid, rid) if rid else None
    receipt["stale"] = (
        receipt["engine_fingerprint"] != engine_fingerprint()
        or receipt["source_fingerprint"] != read_sources(service.store, sid, owner)["fingerprint"]
        or receipt["contract"] != (revision["contract"] if revision else session.get("contract"))
    )
    current_design = (session.get("proposal") or {}).get("revision_id") or session.get(
        "selected_revision"
    )
    receipt["current_design_revision"] = current_design
    receipt["matches_current_design"] = rid is not None and rid == current_design
    return dict(receipt)


def run_tool_job(
    service: WorkspaceService, job: dict[str, Any], cancelled: Callable[[], bool]
) -> None:
    receipt = run_tool(
        service,
        job["session"],
        job["owner"],
        job.get("revision_id"),
        job["request"],
        timeout=job.get("max_seconds"),
        cancelled=cancelled,
        check_id=job["check_id"],
        job_id=job["id"],
    )
    service.store.event(
        job["session"],
        "product_tool",
        "Engineering probe finished: " + job["request"]["tool"],
        check_id=receipt["id"],
        job_id=job["id"],
        revision_id=job.get("revision_id"),
    )


def product_review(
    service: WorkspaceService, sid: str, owner: str, rid: str | None
) -> dict[str, Any]:
    session = service.store.get(sid, owner)
    sources = read_sources(service.store, sid, owner)
    revision = service._load_revision(sid, rid) if rid else None
    _table(service)
    checks = []
    fingerprint = engine_fingerprint()
    with service.store.connect() as con:
        for row in con.execute(
            "SELECT document,digest FROM product_checks WHERE session=? ORDER BY rowid DESC LIMIT 100",
            (sid,),
        ):
            if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                raise WorkspaceError("Engineering tool receipt integrity check failed")
            item = json.loads(row[0])
            item["stale"] = (
                item["revision_id"] != rid
                or item["source_fingerprint"] != sources["fingerprint"]
                or item["engine_fingerprint"] != fingerprint
                or item["contract"]
                != (revision["contract"] if revision else session.get("contract"))
            )
            checks.append(item)
    production = None
    if revision and revision.get("document"):
        from forge_mind.workspace.document import DesignDocument
        from forge_mind.workspace.production import production_report

        production = production_report(
            DesignDocument.model_validate(revision["document"]),
            revision["measurement"].get("facts", {}).get("assembly"),
        )
        from forge_mind.workspace.production import resolve_sources

        production = resolve_sources(production, sources)
    contract = (revision or {}).get("contract") or session.get("contract") or {}
    from forge_mind.workspace.product_interfaces import read_interfaces

    interfaces = read_interfaces(service.store, sid, owner, rid)
    from forge_mind.workspace.completion import completion_context

    findings = {
        f["requirement_id"]: f for f in (revision or {}).get("measurement", {}).get("findings", [])
    }
    obligations = [
        {
            **r,
            "finding": findings.get(r["id"]),
            "status": findings.get(r["id"], {}).get("status", "unverified"),
            "interface_evidence": [
                {
                    "interface_id": record["id"],
                    "status": record["check_status"],
                    "stale": record["stale"],
                    "binding_scope": record["binding_scope"],
                }
                for record in interfaces["records"]
                if r["id"] in record["requirement_ids"]
            ],
        }
        for r in contract.get("requirements", [])
    ]
    return {
        "revision_id": rid,
        "production": production,
        "sources": sources,
        "checks": checks,
        "interfaces": interfaces,
        "completion": completion_context(service, sid, owner, rid),
        "obligations": obligations,
        "physical_readiness": "unverified",
        "scope": "Frozen requirements remain visible, including unsupported functional and appearance requirements. Passing exploratory tools alone does not close them.",
    }
