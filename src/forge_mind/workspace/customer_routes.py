"""Minimum authenticated hosted customer delivery surface over existing services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.routing import APIRoute

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_projects import customer_capabilities
from forge_mind.workspace.document_service import read_topology
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.project_access import project_access_router
from forge_mind.workspace.store import Store, WorkspaceError

READ_ROUTES = {
    "/workspace/projects",
    "/workspace/sessions/{sid}/overview",
    "/workspace/sessions/{sid}/guidance",
    "/workspace/sessions/{sid}/history",
    "/workspace/sessions/{sid}/assets",
    "/workspace/sessions/{sid}/product",
    "/workspace/sessions/{sid}/product-checks/{check_id}",
    "/workspace/sessions/{sid}/print-exports/{package_id}/{sha256}",
    "/workspace/sessions/{sid}/project-exports/{package_id}/{sha256}.zip",
}
WRITE_ROUTES = {
    "/workspace/sessions/{sid}/print-exports",
    "/workspace/sessions/{sid}/project-exports",
}


def customer_delivery_router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    def qualified_owner(request: Request, owner: str = Depends(owner_dependency)) -> str:
        if not customer_capabilities(service.executor)["enabled"]:
            raise HTTPException(503, "Hosted customer delivery is not qualified")
        if request.path_params.get("sid"):
            try:
                store.session_record(request.path_params["sid"], owner)
            except WorkspaceError as exc:
                raise HTTPException(404, "workspace not found") from exc
        return owner

    result = APIRouter()

    @result.get("/workspace/sessions/{sid}/topology/{rid}")
    def topology(sid: str, rid: str, owner: str = Depends(qualified_owner)) -> dict[str, Any]:
        """Read the existing hash-checked owned viewer data without native execution."""
        try:
            return read_topology(service, sid, rid)
        except (WorkspaceError, ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    for source in (
        project_access_router(store, service, qualified_owner),
        product_router(store, service, qualified_owner),
    ):
        for route in source.routes:
            if isinstance(route, APIRoute) and (
                (route.path in READ_ROUTES and route.methods == {"GET"})
                or (route.path in WRITE_ROUTES and route.methods == {"POST"})
            ):
                result.routes.append(route)
    return result
