"""The same recoverable design experiment lifecycle for people and assistant tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from forge_mind.workspace.action_payloads import (
    DesignCollection,
    ReadDesignArguments,
    ReadDesignQuery,
    read_design_arguments,
)
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_evaluators import evaluator_discovery
from forge_mind.workspace.design_study_models import (
    AcceptStudyCandidate,
    AddStudyCandidate,
    CreateDesignStudy,
    DesignStudySpec,
    RunDesignStudy,
    SearchDesignStudy,
)
from forge_mind.workspace.design_study_progress import study_progress
from forge_mind.workspace.design_study_service import (
    add_candidate,
    create_study,
    enqueue_study,
    field_path,
    list_studies,
    propose_candidate,
    read_study,
)
from forge_mind.workspace.lab_api import LabRoute


def study_schema() -> dict[str, Any]:
    from forge_mind.workspace.capacity import optional_limit
    from forge_mind.workspace.processing import resource_policy

    return {
        "spec": DesignStudySpec.model_json_schema(),
        "create": CreateDesignStudy.model_json_schema(),
        "candidate": AddStudyCandidate.model_json_schema(),
        "read_design": ReadDesignArguments.model_json_schema(),
        "read_design_query": ReadDesignQuery.model_json_schema(),
        "evaluators": evaluator_discovery(),
        "resource_policy": resource_policy(),
        "design_limits": {
            name.lower(): optional_limit(name)
            for name in (
                "OCCURRENCE_LIMIT",
                "BODY_LIMIT",
                "FEATURE_FACE_LIMIT",
                "FEATURE_EDGE_LIMIT",
                "ASSEMBLY_FACE_LIMIT",
                "ASSEMBLY_EDGE_LIMIT",
                "PATTERN_COUNT_LIMIT",
                "PATTERN_INSTANCE_LIMIT",
                "PICK_TRIANGLE_LIMIT",
                "MCP_RESPONSE_BYTES",
                "MCP_EXPORT_BYTES",
            )
        },
    }


def design_study_router(
    service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    routes = APIRouter(prefix="/workspace", route_class=LabRoute)

    @routes.get("/design-study-schema")
    def schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return study_schema()

    @routes.get("/sessions/{sid}/design-studies")
    def listing(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return list_studies(service, sid, owner)

    @routes.post("/sessions/{sid}/design-studies", status_code=201)
    def create(
        sid: str, body: CreateDesignStudy, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return create_study(service, sid, owner, body)

    @routes.get("/sessions/{sid}/design-studies/{study_id}")
    def read(sid: str, study_id: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return read_study(service, sid, owner, study_id)

    @routes.get("/sessions/{sid}/design-studies/{study_id}/progress")
    def progress(sid: str, study_id: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return study_progress(service, sid, owner, study_id)

    @routes.post("/sessions/{sid}/design-studies/{study_id}/candidates", status_code=201)
    def candidate(
        sid: str, study_id: str, body: AddStudyCandidate, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return add_candidate(service, sid, owner, study_id, body)

    @routes.post("/sessions/{sid}/design-studies/{study_id}/run", status_code=202)
    def run(
        sid: str, study_id: str, body: RunDesignStudy, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return enqueue_study(service, sid, owner, study_id, retry_failed=body.retry_failed)

    @routes.post("/sessions/{sid}/design-studies/{study_id}/search", status_code=202)
    def search(
        sid: str, study_id: str, body: SearchDesignStudy, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return enqueue_study(service, sid, owner, study_id, search=body)

    @routes.post("/sessions/{sid}/design-studies/{study_id}/accept", status_code=202)
    def accept(
        sid: str, study_id: str, body: AcceptStudyCandidate, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return propose_candidate(service, sid, owner, study_id, body)

    @routes.get("/sessions/{sid}/design-studies/{study_id}/candidates/{candidate_id}/field")
    def field(
        sid: str, study_id: str, candidate_id: str, owner: str = Depends(owner_dependency)
    ) -> FileResponse:
        path = field_path(service, sid, owner, study_id, candidate_id)
        return FileResponse(
            path, media_type="application/json", headers={"Cache-Control": "private, no-store"}
        )

    @routes.get("/sessions/{sid}/design/{revision_id}/page")
    def page(
        sid: str,
        revision_id: str,
        request: Request,
        collection: DesignCollection,
        scope: str | None = None,
        cursor: str | None = None,
        feature_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        from forge_mind.workspace.document import DesignDocument
        from forge_mind.workspace.inspection import query_document

        values: dict[str, Any] = {
            "scope": scope,
            "cursor": cursor,
            "feature_id": feature_id,
            "limit": limit,
        }
        values.update(
            {
                key: value
                for key, value in request.query_params.items()
                if key not in {"collection", "scope", "cursor", "feature_id", "limit"}
            }
        )
        arguments = read_design_arguments(collection, values)
        service.store.session_record(sid, owner)
        revision = service._load_revision(sid, revision_id)
        return query_document(
            DesignDocument.model_validate(revision.get("document")),
            revision_id,
            arguments.collection,
            **arguments.model_dump(exclude={"collection"}),
        )

    return routes
