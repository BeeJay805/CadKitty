"""HTTP workspace boundary: every session and artifact lookup is owner-scoped."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import Field

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_approval import (
    ApprovalRequest,
    ProposalRevisionRequest,
    approve_proposal,
    read_proposal,
    revise_proposal,
)
from forge_mind.workspace.customer_projects import (
    CustomerProjectRequest,
    customer_capabilities,
    customer_project,
    hosted_customer_limits,
)
from forge_mind.workspace.feature_delta import FeatureDelta
from forge_mind.workspace.generation_policy import generation_configuration
from forge_mind.workspace.geometry_access_contract import AccessQuery, access_tool_result
from forge_mind.workspace.geometry_distance_contract import DistanceQuery, distance_tool_result
from forge_mind.workspace.geometry_neighbors_contract import NeighborQuery, neighbors_tool_result
from forge_mind.workspace.input_application_api import AbandonInputRequest
from forge_mind.workspace.intent_attribution import IntentResolution
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import EditRequest, Record, WorkspaceRequest
from forge_mind.workspace.occurrence_focus import OccurrenceFocus
from forge_mind.workspace.photo_profile_api import PhotoProfileAction, execute_photo_profile
from forge_mind.workspace.plan_draft import (
    Collection,
    PlanDraftCommit,
    PlanDraftPatch,
    PlanDraftRead,
)
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.store import Store, WorkspaceError


class ProjectName(Record):
    title: str = Field(min_length=1, max_length=80)


class ResumeRequest(Record):
    max_actions: int | None = Field(default=None, ge=1)
    max_seconds: int | None = Field(default=None, ge=1)


class RetryOperationRequest(Record):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class FeatureEditRequest(Record):
    delta: FeatureDelta
    max_seconds: int | None = Field(default=None, ge=1, strict=True)


class IntentResolutionRequest(Record):
    resolution: IntentResolution
    max_seconds: int | None = Field(default=None, ge=1, strict=True)
    max_actions: int | None = Field(default=None, ge=1, strict=True)


class OccurrenceInspectionRequest(Record):
    occurrence_path: str = Field(min_length=1, max_length=2048)
    surface_offset: int = Field(default=0, ge=0)
    surface_limit: int = Field(default=20, ge=1, le=100)
    include_image: bool = True
    focus: OccurrenceFocus | None = None


class SectionInspectionRequest(Record):
    axis: Literal["x", "y", "z"]
    offset_mm: float = Field(allow_inf_nan=False, strict=True)


class ProductPhotoRequest(Record):
    preset: Literal["metal", "plastic"]


def router(
    store: Store, service: WorkspaceService, owner_dependency: Callable[..., Any]
) -> APIRouter:
    routes = APIRouter(prefix="/workspace")

    @routes.post("/sessions/{sid}/photo-profile")
    def photo_profile(
        sid: str, body: PhotoProfileAction, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        try:
            return execute_photo_profile(service, sid, owner, body)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def get(sid: str, owner: str) -> dict[str, Any]:
        try:
            return store.get(sid, owner)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace not found") from exc

    def enqueue(sid: str, owner: str, payload: dict[str, Any]) -> str:
        try:
            doc = store.get(sid, owner)
            if customer_project(doc) and service.executor.mode == "docker":
                if not customer_capabilities(service.executor)["enabled"]:
                    raise HTTPException(503, "Hosted customer delivery is not qualified")
                for key, maximum in (hosted_customer_limits() or {}).items():
                    supplied = payload.get(key)
                    payload[key] = maximum if supplied is None else min(supplied, maximum)
            jid = store.enqueue(sid, owner, payload)
            service.start()
            return jid
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/schema")
    def customer_schema() -> dict[str, Any]:
        """Public-safe availability only; no model, owner, paths or project state."""
        return {"customer_projects": customer_capabilities(service.executor)}

    @routes.get("/capabilities")
    def capabilities(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return {
            "execution_mode": service.executor.mode,
            **generation_configuration(service.client_factory),
            "engineering_workspace": service.executor.mode == "trusted_local",
            "customer_projects": customer_capabilities(service.executor),
            "engine_fingerprint": engine_fingerprint(),
            "claims": "Evidence for displayed requirements; physical manufacture and complete prompt coverage are not certified.",
        }

    @routes.post("/customer-projects", status_code=202)
    def create_customer(
        body: CustomerProjectRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        if not customer_capabilities(service.executor)["enabled"]:
            raise HTTPException(503, "Customer delivery is not qualified on this executor")
        try:
            result = store.create_customer_project(
                owner,
                body.model_dump(
                    mode="json",
                    exclude={"printable"} if "printable" not in body.model_fields_set else set(),
                ),
                hosted_limits=hosted_customer_limits()
                if service.executor.mode == "docker"
                else None,
            )
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        service.start()
        return result

    @routes.get("/customer-requests/{request_id}")
    def recover_customer(request_id: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        try:
            return store.customer_request(owner, request_id)
        except WorkspaceError as exc:
            raise HTTPException(404, "Customer request not found") from exc

    @routes.get("/sessions")
    def sessions(owner: str = Depends(owner_dependency)) -> list[dict[str, Any]]:
        return store.list_sessions(owner)

    @routes.post("/sessions", status_code=202)
    def create(body: WorkspaceRequest, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        if not body.prompt.strip():
            raise HTTPException(422, "prompt must not be blank")
        contract = body.contract.model_dump() if body.contract else None
        doc = store.create(owner, body.prompt, contract, authoring_mode=body.authoring_mode)
        enqueue(
            doc["id"],
            owner,
            {
                **body.model_dump(exclude={"prompt"}),
                "preview_only": service.executor.mode == "trusted_local",
            },
        )
        return get(doc["id"], owner)

    @routes.get("/sessions/{sid}")
    def session(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return get(sid, owner)

    @routes.get("/sessions/{sid}/customer-reference")
    def customer_reference(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        get(sid, owner)
        from forge_mind.workspace.customer_references import reference_image

        try:
            _, metadata = reference_image(store, sid, owner)
            return metadata
        except (WorkspaceError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/customer-reference/image")
    def customer_reference_image(sid: str, owner: str = Depends(owner_dependency)) -> Response:
        get(sid, owner)
        from forge_mind.workspace.customer_references import reference_image

        try:
            content, _ = reference_image(store, sid, owner)
            return Response(
                content,
                media_type="image/png",
                headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
            )
        except (WorkspaceError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/customer-proposal")
    def proposal(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        get(sid, owner)
        return read_proposal(store, sid, owner)

    @routes.post("/sessions/{sid}/customer-proposal/approve", status_code=202)
    def approve(
        sid: str, body: ApprovalRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        get(sid, owner)
        if not customer_capabilities(service.executor)["enabled"]:
            raise HTTPException(503, "Customer delivery is not qualified on this executor")
        try:
            jid = approve_proposal(store, sid, owner, body)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        service.start()
        return {"id": sid, "job_id": jid, "state": get(sid, owner)["state"]}

    @routes.post("/sessions/{sid}/customer-proposal/revise", status_code=202)
    def revise(
        sid: str, body: ProposalRevisionRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        get(sid, owner)
        if not customer_capabilities(service.executor)["enabled"]:
            raise HTTPException(503, "Customer delivery is not qualified on this executor")
        try:
            jid = revise_proposal(store, sid, owner, body)
        except WorkspaceError as exc:
            raise HTTPException(409, str(exc)) from exc
        service.start()
        return {"id": sid, "job_id": jid, "state": get(sid, owner)["state"]}

    @routes.get("/sessions/{sid}/contract-verification")
    def contract_verification(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        """Read the exact current postcapture proof without generation or acceptance."""
        get(sid, owner)
        from forge_mind.workspace.requirement_verification import read_contract_verification

        try:
            return read_contract_verification(service, sid, owner)
        except (WorkspaceError, ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/edit-basis")
    def get_basis(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        """Read the owned current base before any focused edit."""
        get(sid, owner)
        from forge_mind.workspace.edit_service import read_edit_basis

        try:
            return read_edit_basis(service, sid, owner)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.get("/sessions/{sid}/intent-attribution")
    def get_intent_attribution(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        """Read the owned interpretation and questions without starting work."""
        get(sid, owner)
        from forge_mind.workspace.intent_attribution_service import read_attribution

        try:
            return read_attribution(store, sid, owner)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/intent-attribution/resolve", status_code=202)
    def resolve_intent(
        sid: str, body: IntentResolutionRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Queue current owner answers; interpretation is not approval or evidence."""
        get(sid, owner)
        jid = enqueue(
            sid,
            owner,
            {
                "intent_resolution": body.resolution.model_dump(),
                "max_seconds": body.max_seconds,
                "max_actions": body.max_actions,
            },
        )
        return {"id": sid, "job_id": jid, "state": "queued"}

    @routes.post("/sessions/{sid}/input-application/abandon")
    def abandon_input(
        sid: str, body: AbandonInputRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Abandon exact pending owner input; preserve its history and accepted CAD."""
        get(sid, owner)
        try:
            return store.abandon_input(sid, owner, body.input_id, body.basis, body.source_hash)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/feature-edit", status_code=202)
    def feature_edit(
        sid: str, body: FeatureEditRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Queue a guarded Feature delta; queued does not mean applied."""
        get(sid, owner)
        from forge_mind.workspace.edit_service import enqueue_feature_edit

        try:
            return enqueue_feature_edit(service, sid, owner, body.delta, body.max_seconds)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(
                409, {"applied": False, "code": "rejected", "message": str(exc)}
            ) from exc

    @routes.post("/sessions/{sid}/plan-patch")
    def plan_patch(
        sid: str, body: PlanPatch, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Save focused intent atomically; no CAD or rejected edit is applied."""
        get(sid, owner)
        from forge_mind.workspace.edit_service import patch_plan

        try:
            return patch_plan(service, sid, owner, body)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(
                409, {"applied": False, "code": "rejected", "message": str(exc)}
            ) from exc

    @routes.get("/sessions/{sid}/plan-draft")
    def read_plan_draft(
        sid: str,
        collection: Collection = "metadata",
        cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=20, ge=1, le=50),
        record_id: str | None = Query(default=None, min_length=1, max_length=256),
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Read an owned draft page; draft intent is not admitted CAD or evidence."""
        get(sid, owner)
        from forge_mind.workspace.plan_draft import read_draft

        try:
            query = PlanDraftRead(
                collection=collection, cursor=cursor, limit=limit, record_id=record_id
            )
            return read_draft(store, sid, owner, query)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(
                409, {"applied": False, "code": "rejected", "message": str(exc)}
            ) from exc

    @routes.post("/sessions/{sid}/plan-draft")
    def patch_plan_draft(
        sid: str, body: PlanDraftPatch, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Save draft corrections only; no admitted plan, CAD or evidence changes."""
        get(sid, owner)
        from forge_mind.workspace.plan_draft import patch_draft

        try:
            return patch_draft(store, sid, owner, body)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(
                409, {"applied": False, "code": "rejected", "message": str(exc)}
            ) from exc

    @routes.post("/sessions/{sid}/plan-draft/commit")
    def commit_plan_draft(
        sid: str, body: PlanDraftCommit, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        """Validate and publish exact draft intent; this never constructs CAD."""
        get(sid, owner)
        from forge_mind.workspace.plan_draft import commit_draft

        try:
            return commit_draft(store, sid, owner, body)
        except (WorkspaceError, ValueError) as exc:
            raise HTTPException(
                409, {"applied": False, "code": "rejected", "message": str(exc)}
            ) from exc

    @routes.post("/sessions/{sid}/design/{rid}/inspect-occurrence")
    def inspect_occurrence(
        sid: str,
        rid: str,
        body: OccurrenceInspectionRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Derived read-only observation; never enqueue or modify the candidate."""
        get(sid, owner)
        from forge_mind.workspace.occurrence_inspection import inspect_occurrence as inspect

        try:
            if body.focus is not None and not body.include_image:
                raise ValueError("Occurrence focus requires include_image=True")
            arguments = body.model_dump(exclude={"focus"})
            if body.focus is not None:
                arguments["focus"] = body.focus
            return inspect(service, sid, owner, rid, **arguments, timeout=30)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/design/{rid}/measure-distance")
    def measure_distance(
        sid: str,
        rid: str,
        body: DistanceQuery,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Measure saved solids or trimmed faces without changing the candidate."""
        get(sid, owner)
        from forge_mind.workspace.geometry_distance import measure_distance as measure

        try:
            value = measure(service, sid, owner, rid, body.a, body.b, timeout=30)
            return distance_tool_result(value, sid, rid, body.a, body.b)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except OSError as exc:
            raise HTTPException(409, "Distance source or output files are unavailable") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/design/{rid}/cylindrical-access")
    def inspect_cylindrical_access(
        sid: str,
        rid: str,
        body: AccessQuery,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Observe a declared cylinder path without changing saved geometry or acceptance."""
        get(sid, owner)
        from forge_mind.workspace.geometry_access import inspect_cylindrical_access as inspect

        try:
            value = inspect(service, sid, owner, rid, body, timeout=30)
            return access_tool_result(value, sid, rid, body)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except OSError as exc:
            raise HTTPException(
                409, "Cylindrical access source or output files are unavailable"
            ) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/design/{rid}/topology-neighbors")
    def inspect_topology_neighbors(
        sid: str,
        rid: str,
        body: NeighborQuery,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Inspect one saved topology neighborhood without changing the candidate."""
        get(sid, owner)
        from forge_mind.workspace.geometry_neighbors import inspect_topology_neighbors as inspect

        try:
            value = inspect(
                service,
                sid,
                owner,
                rid,
                body.selector,
                offset=body.offset,
                limit=body.limit,
                timeout=30,
            )
            return neighbors_tool_result(value, sid, rid, body.selector, body.offset, body.limit)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except OSError as exc:
            raise HTTPException(
                409, "Topology neighbor source or output files are unavailable"
            ) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/design/{rid}/inspect-section")
    def inspect_section(
        sid: str,
        rid: str,
        body: SectionInspectionRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Observe a whole-revision mesh plane without changing the candidate."""
        get(sid, owner)
        from forge_mind.workspace.section_inspection import inspect_section as inspect

        try:
            return inspect(service, sid, owner, rid, body.axis, body.offset_mm, timeout=30)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except OSError as exc:
            raise HTTPException(409, "Section source or output files are unavailable") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/design/{rid}/product-photo")
    def product_photo(
        sid: str,
        rid: str,
        body: ProductPhotoRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        """Read a derived appearance image without changing production or geometry."""
        get(sid, owner)
        from forge_mind.workspace.product_photo import get_product_photo

        try:
            return get_product_photo(service, sid, owner, rid, body.preset, timeout=30)
        except WorkspaceError as exc:
            raise HTTPException(404, "workspace or revision not found") from exc
        except OSError as exc:
            raise HTTPException(
                409, "Product photo source or output files are unavailable"
            ) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @routes.post("/sessions/{sid}/name")
    def name(sid: str, body: ProjectName, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        get(sid, owner)
        title = body.title.strip()
        if not title:
            raise HTTPException(422, "project name must not be blank")
        store.update(sid, title=title)
        return get(sid, owner)

    @routes.post("/sessions/{sid}/edit", status_code=202)
    def edit(sid: str, body: EditRequest, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        doc = get(sid, owner)
        if customer_project(doc) and doc.get("state") == "needs_approval":
            raise HTTPException(
                409, "Use the bound proposal approval or revision action before continuing"
            )
        if (
            not body.instruction.strip()
            and not body.parameters
            and not body.revision_id
            and not body.contract
        ):
            raise HTTPException(
                422, "provide an edit, parameters, a contract, or a revision to restore"
            )
        if body.revision_id:
            try:
                store.revision(sid, body.revision_id)
            except WorkspaceError as exc:
                raise HTTPException(404, "revision not found") from exc
        payload = body.model_dump(exclude_none=True)
        payload["preview_only"] = service.executor.mode == "trusted_local" and not customer_project(
            doc
        )
        if customer_project(doc):
            if not customer_capabilities(service.executor)["enabled"]:
                raise HTTPException(503, "Customer delivery is not qualified on this executor")
            if body.parameters:
                raise HTTPException(422, "Use a written customer edit to complete fresh delivery")
            payload["delivery_requested"] = True
        payload["restore_only"] = bool(
            body.revision_id and not body.instruction and not body.parameters and not body.contract
        )
        enqueue(sid, owner, payload)
        return {"id": doc["id"], "state": "queued"}

    @routes.post("/sessions/{sid}/cancel")
    def cancel(sid: str, owner: str = Depends(owner_dependency)) -> dict[str, str]:
        get(sid, owner)
        store.cancel(sid, owner)
        return {"state": str(store.get(sid, owner)["state"])}

    @routes.post("/sessions/{sid}/resume", status_code=202)
    def resume(
        sid: str, body: ResumeRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        doc = get(sid, owner)
        complete = customer_project(doc)
        if complete and doc.get("state") == "needs_approval":
            raise HTTPException(409, "Approve the current saved plan before resuming construction")
        if complete and not customer_capabilities(service.executor)["enabled"]:
            raise HTTPException(503, "Customer delivery is not qualified on this executor")
        enqueue(
            sid,
            owner,
            {
                **body.model_dump(),
                "resume": True,
                "preview_only": not complete,
                **({"delivery_requested": True} if complete else {}),
            },
        )
        return {"id": sid, "state": "queued"}

    @routes.post("/sessions/{sid}/retry-operation", status_code=202)
    def retry_operation(
        sid: str, body: RetryOperationRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        get(sid, owner)
        job_id = enqueue(sid, owner, {"retry_operation": body.job_id})
        # A repeated request can resolve to an already finished linked attempt.
        # Read live status separately rather than claiming it was queued again.
        return {"id": sid, "job_id": job_id}

    @routes.get("/sessions/{sid}/revisions/{rid}/{name}")
    def artifact(
        sid: str, rid: str, name: str, owner: str = Depends(owner_dependency)
    ) -> FileResponse:
        get(sid, owner)
        allowed = {
            "part.stl",
            "part.step",
            "part.glb",
            "render.png",
            "render-inspection.png",
            "render-inspection.json",
            "program.py",
            "evidence.json",
            "contract.json",
            "topology.json",
            "document.json",
            "motion.json",
            "motion-rig.json",
            "assembly.json",
            "parts.json",
            "definition-meshes.json",
        }
        if name not in allowed and not re.fullmatch(
            r"definition_[a-zA-Z][a-zA-Z0-9_]{0,63}\.(?:step|stl)", name
        ):
            raise HTTPException(404, "artifact not found")
        try:
            revision = store.revision(sid, rid)
            directory = store.directory(sid, rid)
        except WorkspaceError as exc:
            raise HTTPException(404, "artifact not found") from exc
        path = directory / name
        if (
            name == "definition-meshes.json"
            or (name.endswith(".stl") and name.startswith("definition_"))
        ) and not regular_file(path):
            raise HTTPException(404, "artifact not found")
        if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise HTTPException(404, "artifact not found")
        if name in {"evidence.json", "contract.json"}:
            expected = revision["measurement" if name == "evidence.json" else "contract"]
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise HTTPException(409, "Artifact no longer matches its revision record")
        elif file_sha256(path) != revision["measurement"]["artifact_hashes"].get(name):
            raise HTTPException(409, "Artifact content changed; evidence is stale")
        return FileResponse(
            path,
            filename=name,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        )

    @routes.get("/sessions/{sid}/sections/{name}")
    def section(sid: str, name: str, owner: str = Depends(owner_dependency)) -> FileResponse:
        get(sid, owner)
        if (
            len(name) != 36
            or not name.endswith(".png")
            or not all(c in "0123456789abcdef" for c in name[:-4])
        ):
            raise HTTPException(404, "section not found")
        path = store.root / "sections" / sid / name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "section not found")
        return FileResponse(path, headers={"Cache-Control": "private, no-store"})

    return routes
