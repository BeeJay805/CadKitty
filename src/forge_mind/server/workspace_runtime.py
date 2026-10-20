"""Private candidate for the modern authenticated customer service.

Run exactly one ASGI process. Offline maintenance holds the same OS lock. A
configured candidate serves no mutating request and starts no job until its real
runner, credentials, provider exercise, and recovery evidence are current.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from forge_mind.workspace.hosting import (
    HostConfig,
    HostingError,
    HostLock,
    provider_credentials,
    qualification_receipt,
    readiness,
)


def create_app(config: HostConfig | None = None) -> FastAPI:
    config = HostConfig.from_env() if config is None else config
    state: dict[str, Any] = {
        "service": None,
        "engine": "",
        "credentials": False,
        "qualification": {"provider": False, "recovery": False},
    }

    def report(*, prospective_worker: bool = False) -> dict[str, Any]:
        service = state["service"]
        if service is None:
            return {"ready": False, "release": config.release, "checks": {"startup": False}}
        worker = service.worker
        qualification = state["qualification"]
        return readiness(
            config,
            current_engine=state["engine"],
            executor=service.executor,
            worker_alive=prospective_worker or bool(worker and worker.is_alive()),
            provider_qualified=state["credentials"] and qualification["provider"],
            recovery_qualified=qualification["recovery"],
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        with HostLock(config.root):
            if (config.root / ".restore-required.json").exists() or (
                config.root / ".restore-incomplete.json"
            ).exists():
                raise HostingError(
                    "Offline fenced recovery must finish before serving this restore"
                )
            from forge_mind.workspace.agent import WorkspaceService
            from forge_mind.workspace.engine_identity import engine_fingerprint
            from forge_mind.workspace.hosting_runner import RunnerClient
            from forge_mind.workspace.hosting_runner_executor import BrokerExecutor
            from forge_mind.workspace.hosting_store import HostingStore

            class GuardedService(WorkspaceService):
                def start(self) -> None:
                    if report(prospective_worker=True)["ready"]:
                        super().start()

                def run_job(self, job: dict[str, Any]) -> None:
                    with store.active_job(job["id"]):
                        super().run_job(job)

            store = HostingStore(config)
            executor = BrokerExecutor(
                RunnerClient(config.runner_socket, config.spool, config.runner_fence),
                config.image,
                on_uncertain=store.hold_uncertain,
            )
            service = GuardedService(store, executor)
            state.update(
                service=service,
                engine=engine_fingerprint(),
                credentials=provider_credentials(),
                qualification=qualification_receipt(config),
            )
            # No settings grant worker qualification. These are actual restricted-runner probes.
            try:
                executor.qualify_typed_container()
                executor.qualify_customer_container()
            except (OSError, ValueError):
                executor._typed_container_verified = False
                executor._customer_container_verified = False
            from forge_mind.server.workspace_app import create_app as modern_app

            modern = modern_app(local=False, service=service)
            app.mount("/", modern)
            try:
                async with modern.router.lifespan_context(modern):
                    yield
            finally:
                # Retain the OS lock while an owned daemon thread can publish.
                service.stop.set()
                if service.worker and service.worker.is_alive():
                    service.worker.join()

    app = FastAPI(title="CadKitty customer host candidate", lifespan=lifespan)

    @app.middleware("http")
    async def admission(request: Request, call_next: Any) -> Any:
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not report()["ready"]:
            return JSONResponse(
                status_code=503, content={"detail": "Customer host qualification is incomplete"}
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "scope": "process_liveness_only"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        result = report()
        return JSONResponse(status_code=200 if result["ready"] else 503, content=result)

    @app.get("/version")
    def version() -> dict[str, str]:
        return {
            "release": config.release,
            "engine": state["engine"],
            "scope": "private_candidate",
            "authoring_policy": "staged_native_operations"
            if os.environ.get("FORGE_NATIVE_STAGED_PLANNING") == "1"
            else "legacy_unqualified",
        }

    return app
