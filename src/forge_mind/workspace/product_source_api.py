"""Source research and observations, with separate user-only claim confirmation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import Field

from forge_mind.workspace.models import Record
from forge_mind.workspace.product_sources import (
    DimensionalClaim,
    confirm_claim,
    fetch_source,
    propose_claim,
    read_sources,
    research_sources,
)
from forge_mind.workspace.source_observations import (
    SourceObservationError,
    observation_image,
    observe_source_page,
    read_source_observation,
)
from forge_mind.workspace.source_page_models import SourcePageRequest, source_page_schema
from forge_mind.workspace.store import Store, WorkspaceError

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class FetchRequest(Record):
    url: str = Field(min_length=1, max_length=2048)


class SearchRequest(Record):
    query: str = Field(min_length=1, max_length=2000)


class Confirmation(Record):
    version: int = Field(ge=0)


def source_router(store: Store, service: Any, owner_dependency: Callable[..., Any]) -> APIRouter:
    router = APIRouter(prefix="/workspace")

    def observation_access(sid: str, owner: str) -> None:
        try:
            store.get(sid, owner)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace not found") from exc

    def observation_call(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except SourceObservationError as exc:
            status = (
                404
                if exc.code in {"source_not_found", "observation_not_found"}
                else 429
                if exc.code == "backend_busy"
                else 400
                if exc.code in {"invalid_request", "page_out_of_range"}
                else 409
            )
            raise HTTPException(status, detail={"code": exc.code, "message": str(exc)}) from exc
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/source-page-schema")
    def page_schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return source_page_schema()

    @router.post("/sessions/{sid}/sources/{artifact_id}/pages")
    def observe_page(
        sid: Identifier,
        artifact_id: Identifier,
        body: SourcePageRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        observation_access(sid, owner)
        return dict(
            observation_call(
                observe_source_page,
                service,
                sid,
                owner,
                artifact_id,
                body.page_number,
                scale=body.scale,
                timeout=30,
            )
        )

    @router.get("/sessions/{sid}/source-observations/{observation_id}")
    def source_observation(
        sid: Identifier, observation_id: Identifier, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        observation_access(sid, owner)
        return dict(observation_call(read_source_observation, store, sid, owner, observation_id))

    @router.get("/sessions/{sid}/source-observations/{observation_id}/image.png")
    def source_image(
        sid: Identifier, observation_id: Identifier, owner: str = Depends(owner_dependency)
    ) -> Response:
        observation_access(sid, owner)
        content = observation_call(observation_image, store, sid, owner, observation_id)
        return Response(
            content=content,
            media_type="image/png",
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    def invoke(call: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        try:
            return call(*args)
        except WorkspaceError as exc:
            raise HTTPException(
                409 if "changed" in str(exc) or "active operation" in str(exc) else 400, str(exc)
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise HTTPException(400, "Public source could not be retrieved") from exc

    @router.get("/sessions/{sid}/sources")
    def sources(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return invoke(read_sources, store, sid, owner)

    @router.post("/sessions/{sid}/sources/fetch")
    def fetch(
        sid: str, body: FetchRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return invoke(fetch_source, service, sid, owner, body.url)

    @router.post("/sessions/{sid}/sources/search")
    def search(
        sid: str, body: SearchRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return invoke(research_sources, service, sid, owner, body.query)

    @router.post("/sessions/{sid}/source-claims")
    def claim(
        sid: str, body: DimensionalClaim, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return invoke(propose_claim, store, sid, owner, body)

    @router.post("/sessions/{sid}/source-claims/{claim_id}/confirm")
    def confirm(
        sid: str, claim_id: str, body: Confirmation, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return invoke(confirm_claim, store, sid, owner, claim_id, body.version)

    return router
