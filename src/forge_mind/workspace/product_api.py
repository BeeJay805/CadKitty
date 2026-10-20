"""Owner-scoped product engineering tools and manufacturing handoff."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import Field

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.models import Record
from forge_mind.workspace.physical_pilot_models import PhysicalPilotOutcomeRequest
from forge_mind.workspace.product_service import product_review, read_tool_receipt, run_tool
from forge_mind.workspace.product_tools import Request as ProductToolRequest
from forge_mind.workspace.store import Store, WorkspaceError


class ToolRequest(Record):
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    request: dict[str, Any]


class InterfaceConfirmation(Record):
    version: int = Field(ge=0)


class QueuedToolRequest(Record):
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    request: ProductToolRequest


class InterfaceValidation(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class ProjectExportRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    include_stl: bool = Field(default=False, strict=True)
    include_import_assets: bool = Field(default=False, strict=True)


class ProjectPhotoRequest(Record):
    preset: Literal["plastic", "metal"] = "plastic"
    part_id: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    include_alternate: bool = Field(default=False, strict=True)


class PrintPreparationRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")


def product_router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    routes = APIRouter(prefix="/workspace")

    @routes.get("/sessions/{sid}/design/{rid}/parts/{part_id}/print-setup")
    def print_setup(
        sid: str, rid: str, part_id: str, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.print_setup_service import inspect_print_setup

        try:
            return inspect_print_setup(service, sid, owner, rid, part_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/print-exports")
    def print_exports(
        sid: str, body: PrintPreparationRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.print_setup_service import prepare_print_files

        try:
            return prepare_print_files(service, sid, owner, body.revision_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/print-exports/{package_id}/{sha256}")
    def print_export_bytes(
        sid: str, package_id: str, sha256: str, owner: str = Depends(owner_dependency)
    ) -> Response:
        from forge_mind.workspace.print_setup_service import read_print_export

        try:
            data, _ = read_print_export(service, sid, owner, package_id, sha256)
            return Response(
                data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="print-files.zip"',
                    "Cache-Control": "private, no-store",
                },
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/design/{rid}/physical-pilot")
    def physical_pilot(
        sid: str, rid: str, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.physical_pilot_service import prepare_physical_pilot

        try:
            return prepare_physical_pilot(service, sid, owner, rid)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/physical-pilot-outcomes")
    def physical_outcome(
        sid: str, body: PhysicalPilotOutcomeRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.physical_pilot_service import record_outcome

        try:
            return record_outcome(service, sid, owner, body)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/design/{rid}/physical-pilot-outcomes")
    def physical_outcomes(
        sid: str, rid: str, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.physical_pilot_service import read_outcomes

        try:
            return read_outcomes(service, sid, owner, rid)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/product-schema")
    def schema(query: str = "", owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        from forge_mind.workspace.product_models import PartProduction
        from forge_mind.workspace.product_tools import product_tool_schema

        if query == "interface":
            from forge_mind.workspace.product_interfaces import InterfaceProposal

            return {"interface": InterfaceProposal.model_json_schema()}
        return {
            "production": PartProduction.model_json_schema(),
            "tools": product_tool_schema(query or None),
        }

    @routes.get("/sessions/{sid}/product")
    def review(
        sid: str, revision_id: str | None = None, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        try:
            return product_review(service, sid, owner, revision_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/product-tools")
    def tool(sid: str, body: ToolRequest, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        try:
            session = store.get(sid, owner)
            if session["state"] in {"queued", "running", "cancelling"}:
                raise WorkspaceError(
                    "Wait for the active operation before running a manual engineering tool"
                )
            return run_tool(service, sid, owner, body.revision_id, body.request)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/product-tool-jobs", status_code=202)
    def queue_tool(
        sid: str, body: QueuedToolRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        store.session_record(sid, owner)
        if body.revision_id:
            service._load_revision(sid, body.revision_id)
        check_id = uuid.uuid4().hex
        job_id = store.enqueue(
            sid,
            owner,
            {
                "engineering_operation": "product_tool",
                "revision_id": body.revision_id,
                "request": body.request.model_dump(),
                "check_id": check_id,
            },
        )
        return {"id": sid, "job_id": job_id, "check_id": check_id, "state": "queued"}

    @routes.get("/sessions/{sid}/product-checks/{check_id}")
    def tool_receipt(
        sid: str, check_id: str, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return read_tool_receipt(service, sid, owner, check_id)

    @routes.get("/sessions/{sid}/manufacturing/{rid}")
    def manufacturing(
        sid: str,
        rid: str,
        include_stl: bool = False,
        include_import_assets: bool = False,
        owner: str = Depends(owner_dependency),
    ) -> FileResponse:
        from pathlib import Path

        from forge_mind.workspace.project_delivery import build_manufacturing_handoff

        output: Path | None = None
        try:
            store.session_record(sid, owner)
            target = store.root / "manufacturing" / sid
            target.mkdir(parents=True, exist_ok=True)
            output = target / f"{uuid.uuid4().hex}.zip"
            build_manufacturing_handoff(
                service,
                sid,
                owner,
                rid,
                output,
                include_stl=include_stl,
                include_import_assets=include_import_assets,
            )
            return FileResponse(
                output, media_type="application/zip", filename="manufacturing-handoff.zip"
            )
        except (ValueError, OSError) as exc:
            if output is not None:
                output.unlink(missing_ok=True)
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/project-exports")
    def project_export(
        sid: str, body: ProjectExportRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.project_delivery import create_project_export

        try:
            return create_project_export(
                service,
                sid,
                owner,
                body.revision_id,
                include_stl=body.include_stl,
                include_import_assets=body.include_import_assets,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/project-exports/{package_id}/{sha256}.zip")
    def project_export_bytes(
        sid: str, package_id: str, sha256: str, owner: str = Depends(owner_dependency)
    ) -> Response:
        from forge_mind.workspace.project_delivery import read_project_export

        try:
            data, _ = read_project_export(service, sid, owner, package_id, sha256)
            return Response(
                data,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="project-review.zip"'},
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/project-exports/{package_id}/{sha256}/photo")
    def project_photo(
        sid: str,
        package_id: str,
        sha256: str,
        body: ProjectPhotoRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Render saved made geometry with a separate availability outcome."""
        from forge_mind.workspace.project_photo import (
            create_project_photo,
            create_project_photo_pair,
        )

        try:
            operation = (
                create_project_photo_pair if body.include_alternate else create_project_photo
            )
            return operation(
                service,
                sid,
                owner,
                package_id,
                sha256,
                preset=body.preset,
                part_id=body.part_id,
                timeout=15,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get(
        "/sessions/{sid}/project-exports/{package_id}/{sha256}/photos/{photo_id}/{image_sha256}.png"
    )
    def project_photo_bytes(
        sid: str,
        package_id: str,
        sha256: str,
        photo_id: str,
        image_sha256: str,
        owner: str = Depends(owner_dependency),
    ) -> Response:
        """Read a verified photo through its owned captured package identity."""
        from forge_mind.workspace.project_photo import read_project_photo

        try:
            data, _ = read_project_photo(
                service, sid, owner, package_id, sha256, photo_id, image_sha256
            )
            return Response(
                data,
                media_type="image/png",
                headers={
                    "Content-Disposition": 'inline; filename="part-preview.png"',
                    "Cache-Control": "private, no-store",
                },
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.delete(
        "/sessions/{sid}/project-exports/{package_id}/{sha256}/photos/{photo_id}", status_code=204
    )
    def discard_unpublished_photo(
        sid: str,
        package_id: str,
        sha256: str,
        photo_id: str,
        owner: str = Depends(owner_dependency),
    ) -> Response:
        """Remove only an identified owned photo after interrupted publication."""
        from forge_mind.workspace.project_delivery import read_project_export
        from forge_mind.workspace.project_photo import discard_project_photo

        try:
            read_project_export(service, sid, owner, package_id, sha256)
            discard_project_photo(service, sid, owner, package_id, photo_id)
            return Response(status_code=204)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/interfaces")
    def interfaces(
        sid: str, revision_id: str | None = None, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.product_interfaces import read_interfaces

        return read_interfaces(store, sid, owner, revision_id)

    @routes.post("/sessions/{sid}/interfaces")
    def propose(
        sid: str, body: dict[str, Any], owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.product_interfaces import propose_interface

        try:
            return propose_interface(store, sid, owner, body)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/interfaces/{interface_id}/confirm")
    def confirm(
        sid: str,
        interface_id: str,
        body: InterfaceConfirmation,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        from forge_mind.workspace.product_interfaces import confirm_interface

        try:
            return confirm_interface(store, sid, owner, interface_id, body.version)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/interfaces/validate")
    def validate(
        sid: str, body: InterfaceValidation, owner: str = Depends(owner_dependency)
    ) -> Any:
        from forge_mind.workspace.product_interfaces import validate_interfaces

        try:
            session = store.get(sid, owner)
            if session["state"] in {"running", "queued", "cancelling"}:
                raise WorkspaceError("Wait for the active operation before checking interfaces")
            return validate_interfaces(service, sid, owner, body.revision_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return routes
