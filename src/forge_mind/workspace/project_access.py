"""Focused project access for assistants: bounded context and guarded named edits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit, authoring_schema
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context, update_guidance
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.input_application_api import input_application_summary
from forge_mind.workspace.intent_attribution_service import attribution_summary
from forge_mind.workspace.models import Record
from forge_mind.workspace.project_assets import list_project_assets
from forge_mind.workspace.store import Store, WorkspaceError


class ScopedAssemblyEdit(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    edit: AssemblyEdit
    explanation: str = Field(default="Scoped assembly edit", min_length=1, max_length=1000)


def revision_summary(revision: dict[str, Any]) -> dict[str, Any]:
    measurement = revision.get("measurement") or {}
    facts = measurement.get("facts") or {}
    findings = measurement.get("findings", [])
    return {
        "id": revision["id"],
        "parent_id": revision.get("parent_id"),
        "at": revision.get("at"),
        "explanation": str(revision.get("explanation", ""))[:500],
        "geometry_valid": measurement.get("geometry_valid"),
        "requirements_satisfied": measurement.get("requirements_satisfied"),
        "findings": dict(Counter(f["status"] for f in findings)),
        "errors": [str(e)[:300] for e in measurement.get("errors", [])[:3]],
        "extents_mm": facts.get("extents_mm"),
        "volume_mm3": facts.get("volume_mm3"),
        "solid_count": facts.get("solid_count"),
        "engine_fingerprint": measurement.get("engine_fingerprint"),
        "scope": "Measured CAD facts; full findings are available in revision evidence.",
    }


def project_overview(store: Store, sid: str, owner: str) -> dict[str, Any]:
    from forge_mind.workspace.product_sources import source_fingerprint
    from forge_mind.workspace.requirement_verification import (
        completion_projection,
        verification_summary,
    )

    session = store.session_record(sid, owner)
    with store.connect() as con:
        count = con.execute("SELECT count(*) FROM revisions WHERE session=?", (sid,)).fetchone()[0]
        row = con.execute(
            "SELECT version FROM engineering_inputs WHERE session=?", (sid,)
        ).fetchone()
        sources_hash = source_fingerprint(con, sid)
    rid = session.get("selected_revision")
    candidate = session.get("proposal") or {}
    design_head = candidate.get("revision_id") or rid
    final_proof = verification_summary(store, sid, owner)

    def evidence(revision_id: str | None) -> dict[str, Any] | None:
        if revision_id is None:
            return None
        result = revision_summary(store.revision(sid, revision_id))
        if final_proof.get("revision_id") == revision_id:
            result.update(
                evidence_stage="build_time_before_final_verification",
                scope=(
                    "Earlier immutable build-time geometry evidence. "
                    "Read contract_verification for the current postcapture frozen requirement proof."
                ),
            )
        return result

    return {
        "id": sid,
        "title": str(session.get("title", session["prompt"]))[:160],
        "prompt": session["prompt"],
        **completion_projection(session, final_proof),
        "job_id": session.get("job_id"),
        "selected_revision": rid,
        "design_revision": design_head,
        "accepted_revision": session.get("accepted_revision"),
        "revision_count": count,
        "input_version": row[0] if row else 0,
        **(
            {
                "authoring_mode": session.get("authoring_mode", "hybrid"),
                "plan_version": session.get("plan_version", 0),
                "plan_hash": session.get("plan_hash"),
                "design_plan": session.get("design_plan"),
            }
            if "authoring_mode" in session or "plan_version" in session
            else {}
        ),
        "source_fingerprint": sources_hash,
        "candidate": {
            k: candidate.get(k)
            for k in ("revision_id", "based_on", "eligible", "input_version", "engine_fingerprint")
        }
        if candidate
        else None,
        "selected": evidence(rid),
        "design": evidence(design_head),
        "recent_events": [
            {"kind": e["kind"], "message": str(e["message"])[:500], "at": e["at"]}
            for e in session.get("events", [])[-6:]
        ],
        "report_ids": [r["id"] for r in session.get("reports", [])[-5:]],
        "motion_check_ids": [r["id"] for r in session.get("motion_checks", [])[-5:]],
        "has_resume_candidate": bool(session.get("resume_candidate")),
        "operation_recovery": session.get("operation_recovery"),
        "last_product_probe": session.get("last_product_probe"),
        "last_structural_study": session.get("last_structural_study"),
        "last_edit_result": session.get("last_edit_result"),
        "intent_attribution": attribution_summary(session.get("intent_attribution")),
        "input_application": input_application_summary(session),
        "contract_verification": final_proof,
        "scope": "Compact live status. Read specific revisions and evidence before judging or accepting.",
    }


def design_relationships(document: DesignDocument) -> dict[str, Any]:
    """Direct explicit dependencies; retrieve transitive impact for selected variables."""
    from forge_mind.workspace.inspection import design_graph

    return design_graph(document)


def assembly_structure(document: DesignDocument) -> dict[str, Any]:
    """Return editable placement and motion records without definition feature bodies."""
    assembly = document.assembly
    if assembly is None:
        raise WorkspaceError("Revision has no assembly structure")
    return {
        "document_hash": digest(document.model_dump()),
        "name": document.name,
        "variables": document.variables,
        "relations": [r.model_dump() for r in document.relations],
        "values": document.values(),
        "parts": [
            {
                "id": p.id,
                "name": p.name,
                "feature_count": len(p.features),
                "ports": [port.name for port in p.ports],
            }
            for p in assembly.parts
        ],
        "roots": [r.model_dump() for r in assembly.roots],
        "subassemblies": [s.model_dump() for s in assembly.subassemblies],
        "mates": [m.model_dump() for m in assembly.mates],
        "motion_rig": document.motion_rig.model_dump() if document.motion_rig else None,
        "mechanism": document.mechanism.model_dump() if document.mechanism else None,
        "scope": "Revision-bound declared placements and prescribed motion. Read part definitions separately. No physical relationship or force validation.",
    }


def part_definition(document: DesignDocument, part_id: str) -> dict[str, Any]:
    """Exact part and resolved direct bindings, curve points and port references."""
    selected = (
        next((part for part in document.assembly.parts if part.id == part_id), None)
        if document.assembly
        else None
    )
    if selected is None:
        raise WorkspaceError("Part definition not found")
    bound = {value for feature in selected.features for value in feature.bindings.values()}
    bound.update(
        value
        for feature in selected.features
        for point in feature.points
        for value in point
        if isinstance(value, str)
    )
    bound.update(
        value
        for port in selected.ports
        for value in (*port.origin, port.diameter)
        if isinstance(value, str)
    )
    values = document.values()
    return {
        "part": selected.model_dump(),
        "values": {value: values[value] for value in sorted(bound)},
    }


def project_access_router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    routes = APIRouter(prefix="/workspace")

    @routes.get("/projects")
    def projects(owner: str = Depends(owner_dependency)) -> Any:
        return {
            "projects": [
                {
                    "id": s["id"],
                    "title": str(s.get("title", s["prompt"]))[:160],
                    "state": s["state"],
                    "selected_revision": s.get("selected_revision"),
                }
                for s in store.list_sessions(owner)
            ],
            "limit": 100,
        }

    @routes.get("/authoring-schema")
    def schema(query: str = "assembly_edit", owner: str = Depends(owner_dependency)) -> Any:
        return authoring_schema(query)

    @routes.get("/sessions/{sid}/overview")
    def overview(sid: str, owner: str = Depends(owner_dependency)) -> Any:
        return project_overview(store, sid, owner)

    @routes.get("/sessions/{sid}/guidance")
    def guidance(
        sid: str,
        revision_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"),
        owner: str = Depends(owner_dependency),
    ) -> Any:
        try:
            return guidance_context(service, sid, owner, revision_id)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/guidance")
    def refresh_guidance(
        sid: str, request: GuidanceUpdate, owner: str = Depends(owner_dependency)
    ) -> Any:
        try:
            return update_guidance(service, sid, owner, request)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/assets")
    def assets(sid: str, owner: str = Depends(owner_dependency)) -> Any:
        try:
            return list_project_assets(store, sid, owner)
        except WorkspaceError as exc:
            raise HTTPException(404, "Owned imported asset inventory unavailable") from exc

    @routes.get("/sessions/{sid}/history")
    def history(
        sid: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10, ge=1, le=30),
        owner: str = Depends(owner_dependency),
    ) -> Any:
        import json

        store.session_record(sid, owner)
        with store.connect() as con:
            rows = con.execute(
                "SELECT document FROM revisions WHERE session=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (sid, limit + 1, offset),
            ).fetchall()
        return {
            "items": [revision_summary(json.loads(r[0])) for r in rows[:limit]],
            "next_offset": offset + limit if len(rows) > limit else None,
        }

    def load(sid: str, rid: str, owner: str) -> DesignDocument:
        store.session_record(sid, owner)
        revision = service._load_revision(sid, rid)
        if not revision.get("document"):
            raise WorkspaceError("Revision has no editable feature document")
        return DesignDocument.model_validate(revision["document"])

    @routes.get("/sessions/{sid}/design/{rid}/relationships")
    def relationships(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> Any:
        return {"revision_id": rid, **design_relationships(load(sid, rid, owner))}

    @routes.get("/sessions/{sid}/design/{rid}/structure")
    def structure(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> Any:
        return {"revision_id": rid, **assembly_structure(load(sid, rid, owner))}

    @routes.get("/sessions/{sid}/design/{rid}/parts/{part_id}")
    def part(sid: str, rid: str, part_id: str, owner: str = Depends(owner_dependency)) -> Any:
        document = load(sid, rid, owner)
        try:
            return {"revision_id": rid, **part_definition(document, part_id)}
        except WorkspaceError as exc:
            raise HTTPException(404, str(exc)) from exc

    @routes.post("/sessions/{sid}/assembly-edit", status_code=202)
    def edit(sid: str, body: ScopedAssemblyEdit, owner: str = Depends(owner_dependency)) -> Any:
        current = load(sid, body.revision_id, owner)
        try:
            merged = apply_assembly_edit(current, body.edit)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        job = store.enqueue(
            sid,
            owner,
            {
                "document": merged.model_dump(),
                "revision_id": body.revision_id,
                "expected_current_revision": body.revision_id,
                "expected_input_version": body.input_version,
                "preview_only": True,
                "explanation": body.explanation,
            },
        )
        service.start()
        return {"id": sid, "job_id": job, "state": "queued"}

    return routes
