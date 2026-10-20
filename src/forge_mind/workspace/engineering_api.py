"""Shared document, annotated edits, imports, reports and explicit proposal acceptance."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.authoring_policy import AuthoringMode
from forge_mind.workspace.document import DesignDocument, Feature, GeometryRef, Sketch
from forge_mind.workspace.document_service import read_topology
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.engineering_service import release_package, report_path
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.mechanism_models import MechanismTemplate
from forge_mind.workspace.models import Contract, DesignPlan, Record
from forge_mind.workspace.motion_check import MotionCheckSettings
from forge_mind.workspace.processing import copy_artifact
from forge_mind.workspace.store import Store, WorkspaceError


class DocumentEdit(Record):
    document: DesignDocument
    contract: Contract | None = None
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    explanation: str = Field(default="Manual feature edit", max_length=1000)
    expected_current_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    expected_input_version: int | None = Field(default=None, ge=0)


class InputEdit(Record):
    version: int = Field(ge=0)
    inputs: EngineeringInputs


class ManualProject(Record):
    name: str = Field(default="Manual engineering design", min_length=1, max_length=160)
    intent: (
        Annotated[str, Field(min_length=1, max_length=10000, strict=True, pattern=r"\S")] | None
    ) = None
    authoring_mode: AuthoringMode | None = None


class PlanEdit(Record):
    plan: DesignPlan
    expected_plan_version: int = Field(ge=0, strict=True)
    contract: Contract | None = None
    expected_current_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    expected_input_version: int | None = Field(default=None, ge=0, strict=True)


class Accept(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)


class ReportRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class PoseRequest(ReportRequest):
    report_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class MotionRequest(ReportRequest):
    settings: MotionCheckSettings = Field(default_factory=MotionCheckSettings)


def engineering_router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    router = APIRouter(prefix="/workspace")

    @router.get("/engineering-schema")
    def schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return {
            "inputs": EngineeringInputs.model_json_schema(),
            "defaults": EngineeringInputs().model_dump(),
        }

    @router.get("/document-schema")
    def document_schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        """Return the exact typed contracts used by deterministic CAD authoring."""
        return {
            "document": DesignDocument.model_json_schema(),
            "contract": Contract.model_json_schema(),
            "empty_document": DesignDocument().model_dump(),
        }

    def owned(sid: str, owner: str) -> dict[str, Any]:
        try:
            return store.get(sid, owner)
        except WorkspaceError as exc:
            raise HTTPException(404, "Workspace not found") from exc

    def idle(session: dict[str, Any]) -> None:
        if session["state"] in {"queued", "running", "cancelling"}:
            raise HTTPException(409, "Wait for the active operation")

    def enqueue(sid: str, owner: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            job = store.enqueue(sid, owner, payload)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        service.start()
        return {"id": sid, "job_id": job, "state": "queued"}

    def validate_ref(sid: str, ref: GeometryRef) -> None:
        topology = read_topology(service, sid, ref.revision_id)
        if not any(
            e["id"] == ref.entity_id
            and e["signature"] == ref.signature
            and e["kind"] == ref.kind
            and e["body_signature"] == ref.body_signature
            for e in topology["entities"]
        ):
            raise WorkspaceError("Reference is not part of the claimed revision")

    def add_asset(
        sid: str,
        owner: str,
        path: Any,
        asset_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asset = {
            "id": asset_id,
            "name": name[:160],
            "sha256": file_sha256(path),
            **(metadata or {}),
        }
        try:
            store.append_asset(sid, owner, asset)
        except WorkspaceError:
            path.unlink(missing_ok=True)
            raise
        return asset

    @router.post("/manual-sessions")
    def manual(
        body: ManualProject | None = None, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return store.create(
            owner,
            (body.intent if body.intent is not None else body.name)
            if body
            else "Manual engineering design",
            {"requirements": [], "coverage_notes": ["No acceptance targets declared yet"]},
            authoring_mode=body.authoring_mode if body else None,
        )

    @router.post("/sessions/{sid}/design-plan")
    def design_plan(
        sid: str, body: PlanEdit, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        try:
            return store.set_design_plan(
                sid,
                owner,
                body.plan,
                expected_plan_version=body.expected_plan_version,
                contract=body.contract,
                expected_current_revision=body.expected_current_revision,
                expected_input_version=body.expected_input_version,
            )
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/sessions/{sid}/document", status_code=202)
    def document(
        sid: str, body: DocumentEdit, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        owned(sid, owner)
        if body.revision_id:
            service._load_revision(sid, body.revision_id)
        payload = body.model_dump(exclude={"expected_current_revision", "expected_input_version"})
        if "expected_current_revision" in body.model_fields_set:
            payload["expected_current_revision"] = body.expected_current_revision
        if body.expected_input_version is not None:
            payload["expected_input_version"] = body.expected_input_version
        return enqueue(sid, owner, {**payload, "preview_only": True})

    @router.post("/sessions/{sid}/motion-check", status_code=202)
    def motion_check(
        sid: str, body: MotionRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        owned(sid, owner)
        service._load_revision(sid, body.revision_id)
        return enqueue(
            sid,
            owner,
            {**body.model_dump(), "engineering_operation": "motion_check"},
        )

    @router.get("/sessions/{sid}/motion-checks/{check_id}")
    def motion_result(
        sid: str, check_id: str, owner: str = Depends(owner_dependency)
    ) -> FileResponse:
        from forge_mind.workspace.motion_service import check_path

        try:
            return FileResponse(
                check_path(service, sid, owner, check_id), media_type="application/json"
            )
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/sketch/solve")
    def sketch(body: Sketch, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        # No native parser or kernel; bounded 80-point, 120-constraint numerical system.
        from forge_mind.workspace.sketch import solve_sketch

        try:
            return solve_sketch(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/steering-template")
    def steering(body: dict[str, Any], owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        from forge_mind.workspace.steering_template import SteeringTemplate, steering_document

        try:
            return steering_document(SteeringTemplate.model_validate(body)).model_dump()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/sessions/{sid}/topology/{rid}")
    def topology(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        owned(sid, owner)
        try:
            return read_topology(service, sid, rid)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/sessions/{sid}/engineering-inputs")
    def inputs(sid: str, body: InputEdit, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        session = owned(sid, owner)
        idle(session)
        try:
            for annotation in body.inputs.annotations:
                validate_ref(sid, annotation.reference)
            for protection in body.inputs.protections:
                if protection.reference:
                    if protection.reference.revision_id != protection.revision_id:
                        raise WorkspaceError(
                            "Protection and geometry reference must use the same baseline revision"
                        )
                    validate_ref(sid, protection.reference)
                if protection.feature_id:
                    revision = service._load_revision(sid, protection.revision_id)
                    if not any(
                        f["id"] == protection.feature_id
                        for f in ((revision.get("document") or {}).get("assembly") or {}).get(
                            "parts", (revision.get("document") or {}).get("features", [])
                        )
                    ):
                        raise WorkspaceError("Protected feature does not exist in the baseline")
            for dimension in body.inputs.dimensions:
                validate_ref(sid, dimension.a)
                validate_ref(sid, dimension.b)
            for feedback in body.inputs.feedback:
                service._load_revision(sid, feedback.revision_id)
            if body.inputs.assembly_revision:
                service._load_revision(sid, body.inputs.assembly_revision)
            store.save_engineering(sid, owner, body.version, body.inputs.model_dump())
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return store.engineering(sid)

    @router.post("/sessions/{sid}/accept")
    def accept(
        sid: str, body: Accept, compact: bool = False, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        session = store.session_record(sid, owner) if compact else owned(sid, owner)
        revision = service._load_revision(sid, body.revision_id)
        if (
            revision["measurement"]["engine_fingerprint"] != engine_fingerprint()
            or (session.get("proposal") or {}).get("engine_fingerprint") != engine_fingerprint()
        ):
            raise HTTPException(409, "Candidate instruments changed; regenerate before accepting")
        try:
            store.accept_proposal(
                sid, owner, body.revision_id, body.expected_revision, body.input_version
            )
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        if compact:
            from forge_mind.workspace.project_access import project_overview

            return project_overview(store, sid, owner)
        return owned(sid, owner)

    @router.post("/sessions/{sid}/assets")
    async def upload(
        sid: str, request: Request, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        session = owned(sid, owner)
        idle(session)
        asset_id = uuid.uuid4().hex
        directory = store.root / "assets" / sid
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (asset_id + ".step")
        total = 0
        try:
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > 32 * 1024 * 1024:
                        raise HTTPException(413, "STEP import exceeds 32 MB")
                    stream.write(chunk)
            with path.open("rb") as stream:
                header = stream.read(512)
            if b"ISO-10303-21;" not in header:
                raise HTTPException(422, "Upload an ISO 10303-21 STEP file")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return add_asset(
            sid, owner, path, asset_id, request.headers.get("X-Filename", "Imported STEP")
        )

    @router.post("/sessions/{sid}/base/{rid}")
    def capture_base(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        session = owned(sid, owner)
        idle(session)
        revision = service._load_revision(sid, rid)
        if not revision["measurement"]["geometry_valid"]:
            raise HTTPException(409, "A base requires valid measured geometry")
        asset_id = uuid.uuid4().hex
        path = store.root / "assets" / sid / (asset_id + ".step")
        path.parent.mkdir(parents=True, exist_ok=True)
        copy_artifact(store.directory(sid, rid) / "part.step", path)
        from forge_mind.workspace.hardware import bill_of_materials

        document = (
            DesignDocument.model_validate(revision["document"])
            if revision.get("document")
            else None
        )
        return add_asset(
            sid,
            owner,
            path,
            asset_id,
            "Immutable revision " + rid[:8],
            {
                "source_revision": rid,
                "solid_count": revision["measurement"]["facts"]["solid_count"],
                "inventory": bill_of_materials(document, revision.get("assets", [])),
            },
        )

    @router.get("/catalog")
    def catalog(
        q: str | None = None,
        family: str | None = Query(default=None, max_length=100),
        subtype: str | None = Query(default=None, max_length=100),
        thread_d: float | None = Query(default=None, gt=0, le=100000, allow_inf_nan=False),
        length: float | None = Query(default=None, gt=0, le=100000, allow_inf_nan=False),
        limit: int = Query(default=30, ge=1, le=500),
        owner: str = Depends(owner_dependency),
    ) -> list[dict[str, Any]]:
        from forge_mind.workspace.hardware import search

        query = (
            q
            if q is not None
            else ""
            if any(value is not None for value in (family, subtype, thread_d, length))
            else "bearing"
        )
        return search(
            query[:100],
            family=family,
            subtype=subtype,
            thread_d=thread_d,
            length=length,
            limit=limit,
        )

    @router.get("/catalog-counts")
    def catalog_counts(owner: str = Depends(owner_dependency)) -> dict[str, int]:
        from forge_mind.agentic.catalog.registry import catalog_counts as counts

        return counts()

    @router.post("/mechanism-template")
    def mechanism_template(
        body: MechanismTemplate, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        from forge_mind.workspace.mechanism_templates import template_document

        try:
            return template_document(body).model_dump()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/sessions/{sid}/pose", status_code=202)
    def pose(sid: str, body: PoseRequest, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        session = owned(sid, owner)
        idle(session)
        report = json.loads(
            report_path(service, sid, owner, body.report_id, "report.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            report["revision_id"] != body.revision_id
            or report["input_version"] != session["engineering"]["version"]
            or report["engine_fingerprint"] != engine_fingerprint()
        ):
            raise WorkspaceError("Pose report is stale; run engineering review again")
        transforms = report["assembly"].get("feature_transforms")
        if not transforms:
            raise WorkspaceError("Assembly pose is unresolved")
        asset = capture_base(sid, body.revision_id, owner)
        features = []
        outputs = []
        for i, params in enumerate(transforms):
            base_id, pose_id = f"body_{i}", f"pose_{i}"
            features.extend(
                [
                    Feature(
                        id=base_id,
                        name=f"Immutable body {i}",
                        kind="import",
                        asset_id=asset["id"],
                        params={"solid_index": i},
                    ),
                    Feature(
                        id=pose_id,
                        name=f"Pose body {i}",
                        kind="transform",
                        inputs=[base_id],
                        params=params,
                    ),
                ]
            )
            outputs.append(pose_id)
        document = DesignDocument(
            name="Assembly pose alternative", features=features, outputs=outputs
        )
        return enqueue(
            sid,
            owner,
            {
                "document": document.model_dump(),
                "revision_id": body.revision_id,
                "preview_only": True,
                "expected_input_version": report["input_version"],
                "explanation": "Preview assembly pose from immutable body bases; source history remains in parent revision",
            },
        )

    @router.post("/sessions/{sid}/engineering", status_code=202)
    def engineering(
        sid: str, body: ReportRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        owned(sid, owner)
        service._load_revision(sid, body.revision_id)
        return enqueue(
            sid,
            owner,
            {
                "engineering_operation": "review",
                "revision_id": body.revision_id,
            },
        )

    @router.get("/sessions/{sid}/reports/{report_id}/{name}")
    def report(
        sid: str, report_id: str, name: str, owner: str = Depends(owner_dependency)
    ) -> FileResponse:
        owned(sid, owner)
        try:
            path = report_path(service, sid, owner, report_id, name)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(
            path,
            filename=name,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/sessions/{sid}/studies/{study_id}")
    def study(sid: str, study_id: str, owner: str = Depends(owner_dependency)) -> FileResponse:
        from forge_mind.workspace.parameter_study import study_path

        owned(sid, owner)
        try:
            path = study_path(service, sid, owner, study_id)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(
            path,
            filename="parameter-study.json",
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/sessions/{sid}/release/{rid}")
    def release(sid: str, rid: str, owner: str = Depends(owner_dependency)) -> FileResponse:
        owned(sid, owner)
        try:
            path = release_package(service, sid, owner, rid)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(
            path,
            filename="engineering-" + rid[:8] + ".zip",
            headers={"Cache-Control": "private, no-store"},
        )

    return router
