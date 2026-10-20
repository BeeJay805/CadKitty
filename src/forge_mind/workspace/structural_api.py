"""Owner-scoped component studies and derived visual qualification evidence."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.qualification import analysis_geometry, qualification_review
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.structural_models import (
    STRUCTURAL_SCOPE,
    StructuralRequest,
    StructuralStudy,
)
from forge_mind.workspace.structural_service import field_path, list_studies, read_study


def structural_router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    routes = APIRouter(prefix="/workspace")

    @routes.get("/structural-schema")
    def schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return {"study": StructuralStudy.model_json_schema(), "scope": STRUCTURAL_SCOPE}

    @routes.post("/sessions/{sid}/structural-studies", status_code=202)
    def create_study(
        sid: str, body: StructuralRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        store.session_record(sid, owner)
        revision = service._load_revision(sid, body.revision_id)
        if not revision["measurement"]["geometry_valid"]:
            raise WorkspaceError("Structural study requires valid exported geometry")
        study_id = uuid.uuid4().hex
        job_id = store.enqueue(
            sid,
            owner,
            {
                **body.model_dump(),
                "study_id": study_id,
                "expected_current_revision": body.revision_id,
                "expected_input_version": body.input_version,
                "engineering_operation": "structural_study",
            },
        )
        service.start()
        return {"id": sid, "job_id": job_id, "study_id": study_id, "state": "queued"}

    @routes.get("/sessions/{sid}/structural-studies")
    def studies(
        sid: str,
        revision_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"),
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        return {"studies": list_studies(service, sid, owner, revision_id)}

    @routes.get("/sessions/{sid}/structural-studies/{study_id}")
    def receipt(sid: str, study_id: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return read_study(service, sid, owner, study_id)

    @routes.get("/sessions/{sid}/structural-studies/{study_id}/field.json")
    def field(sid: str, study_id: str, owner: str = Depends(owner_dependency)) -> FileResponse:
        return FileResponse(
            field_path(service, sid, owner, study_id), media_type="application/json"
        )

    @routes.get("/sessions/{sid}/qualification/{rid}")
    def qualification(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return qualification_review(service, sid, owner, rid)

    @routes.get("/sessions/{sid}/analysis-geometry/{rid}")
    def geometry(
        sid: str,
        rid: str,
        body_index: int = Query(ge=0),
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        return analysis_geometry(service, sid, owner, rid, body_index)

    return routes
