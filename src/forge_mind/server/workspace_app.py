"""Private local workspace or authenticated container-worker service.

Local mode is explicit, accepts only loopback clients, and checks browser origins.
Run on 127.0.0.1. Never put the private local mode behind a reverse proxy.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.store import Store, WorkspaceError


def create_app(
    root: Path | None = None, *, local: bool | None = None, service: WorkspaceService | None = None
) -> FastAPI:
    local = os.environ.get("FORGE_WORKSPACE_LOCAL") == "1" if local is None else local
    mode = os.environ.get("FORGE_WORKSPACE_EXECUTOR", "docker")
    if mode == "trusted_local" and not local:
        raise RuntimeError("trusted_local execution requires private loopback workspace mode")
    if service is None:
        root = root or Path(
            os.environ.get("FORGE_WORKSPACE_DIR", str(Path.home() / ".forge-workspace"))
        )
        service = WorkspaceService(Store(root), Executor(mode))
    if service.executor.mode == "trusted_local" and not local:
        raise RuntimeError("trusted_local execution requires private loopback workspace mode")
    origins = os.environ.get(
        "FORGE_WORKSPACE_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"
    ).split(",")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if not local and os.environ.get("FORGE_WORKSPACE_QUALIFY_CUSTOMER") == "1":
            # This requests real probes; the setting itself grants no capability.
            app.state.worker_qualification = service.executor.qualify_typed_container()
            app.state.customer_qualification = service.executor.qualify_customer_container()
        service.start()
        yield
        service.stop.set()
        if service.worker:
            service.worker.join(timeout=30)

    app = FastAPI(title="CadKitty workspace", lifespan=lifespan)

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request: Request, exc: WorkspaceError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Filename"],
    )

    def owner(request: Request, authorization: str | None = Header(default=None)) -> str:
        if local:
            if not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
                raise HTTPException(403, "Private workspace accepts loopback clients only")
            if request.headers.get("origin") and request.headers["origin"] not in origins:
                raise HTTPException(403, "Browser origin is not allowed")
            host = request.url.hostname
            if host not in {"127.0.0.1", "localhost", "::1", "testserver"}:
                raise HTTPException(403, "Private workspace requires a localhost hostname")
            return "local-owner"
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Sign in to use the workspace")
        url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_ANON_KEY", "")
        if not url or not key:
            raise HTTPException(503, "Workspace authentication is not configured")
        try:
            response = httpx.get(
                f"{url.rstrip('/')}/auth/v1/user",
                headers={"Authorization": authorization, "apikey": key},
                timeout=10,
            )
            if response.status_code != 200 or not response.json().get("id"):
                raise HTTPException(401, "Invalid session")
            subject = str(response.json()["id"])
            host_config = getattr(service.store, "host_config", None)
            if host_config is not None:
                from forge_mind.workspace.hosting_identity import canonical_principal

                if f"{url.rstrip('/')}/auth/v1" != host_config.issuer:
                    raise HTTPException(
                        503, "Workspace owner issuer differs from host configuration"
                    )
                return canonical_principal(host_config.issuer, subject)
            return subject
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Authentication service unavailable") from exc

    app.include_router(router(service.store, service, owner))
    from forge_mind.workspace.engineering_api import engineering_router

    if local:
        app.include_router(engineering_router(service.store, service, owner))
        from forge_mind.workspace.project_access import project_access_router

        app.include_router(project_access_router(service.store, service, owner))
        from forge_mind.workspace.product_api import product_router
        from forge_mind.workspace.product_source_api import source_router

        app.include_router(product_router(service.store, service, owner))
        app.include_router(source_router(service.store, service, owner))
        from forge_mind.workspace.structural_api import structural_router

        app.include_router(structural_router(service.store, service, owner))
        from forge_mind.workspace.lab_api import lab_router

        app.include_router(lab_router(service, owner))
        from forge_mind.workspace.design_study_api import design_study_router

        app.include_router(design_study_router(service, owner))
    else:
        from forge_mind.workspace.customer_routes import customer_delivery_router

        app.include_router(customer_delivery_router(service.store, service, owner))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "private_local" if local else "authenticated"}

    return app


app = create_app()
