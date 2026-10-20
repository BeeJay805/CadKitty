"""One private engineering investigation service for people and assistant tools."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.routing import APIRoute
from pydantic import Field

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_reasoning import (
    ChallengeRequest,
    RationaleRequest,
    add_rationale,
    apply_challenge,
    explain_design,
    preview_challenge,
)
from forge_mind.workspace.experiment_models import (
    CalibrationRequest,
    ExperimentCondition,
    ExperimentRequest,
    Hypothesis,
    OutcomeRequest,
    PredictionRequest,
)
from forge_mind.workspace.experiment_service import (
    calibrate_experiment,
    commit_prediction,
    create_experiment,
    record_outcome,
    synthetic_tutorial,
)
from forge_mind.workspace.lab_records import list_records, read_record, record_summary
from forge_mind.workspace.models import Record
from forge_mind.workspace.synthesis_models import SynthesisRequest


class ComparisonRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    request: SynthesisRequest


class RealizationRequest(Record):
    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    name: str = Field(default="Mechanism investigation", min_length=1, max_length=120)


class CalibratedDesignRequest(Record):
    variable_name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    hypothesis: Hypothesis
    condition: ExperimentCondition
    target_displacement_mm: float = Field(gt=0, le=100)
    min_thickness_mm: float = Field(gt=0, le=1000)
    max_thickness_mm: float = Field(gt=0, le=1000)
    mapping_basis: str = Field(min_length=1, max_length=2000)
    feature_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    part_id: str | None = Field(default=None, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    thickness_parameter: Literal["x", "y", "z"]


class LabRoute(APIRoute):
    """Unsupported calculations are useful errors, never successful empty results or HTTP 500."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await original(request)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

        return handle


def lab_schema() -> dict[str, Any]:
    models: dict[str, type[Record]] = {
        "experiment": ExperimentRequest,
        "calibration": CalibrationRequest,
        "prediction": PredictionRequest,
        "outcome": OutcomeRequest,
        "comparison": ComparisonRequest,
        "rationale": RationaleRequest,
        "challenge": ChallengeRequest,
        "calibrated_design": CalibratedDesignRequest,
    }
    return {key: model.model_json_schema() for key, model in models.items()}


def lab_router(service: WorkspaceService, owner_dependency: Callable[..., Any]) -> APIRouter:
    from forge_mind.workspace.lab_service import (
        calibrated_design,
        create_comparison,
        realize_comparison,
    )
    from forge_mind.workspace.synthesis_models import example_synthesis_request

    routes = APIRouter(prefix="/workspace", route_class=LabRoute)

    @routes.get("/lab/schema")
    def schema(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return lab_schema()

    @routes.get("/lab/experiment-example")
    def experiment_example(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return synthetic_tutorial()

    @routes.get("/lab/synthesis-example")
    def synthesis_example(owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return example_synthesis_request().model_dump()

    @routes.get("/sessions/{sid}/lab/records")
    def records(
        sid: str, kind: str | None = None, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return {"records": [record_summary(r) for r in list_records(service, sid, owner, kind)]}

    @routes.get("/sessions/{sid}/lab/records/{identifier}")
    def record(sid: str, identifier: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return read_record(service, sid, owner, identifier)

    @routes.post("/sessions/{sid}/lab/experiments")
    def experiment(
        sid: str, body: ExperimentRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return create_experiment(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/calibrations")
    def calibration(
        sid: str, body: CalibrationRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return calibrate_experiment(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/predictions")
    def prediction(
        sid: str, body: PredictionRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return commit_prediction(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/outcomes")
    def outcome(
        sid: str, body: OutcomeRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return record_outcome(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/comparisons")
    def comparison(
        sid: str, body: ComparisonRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return create_comparison(
            service, sid, owner, body.revision_id, body.input_version, body.request
        )

    @routes.post("/sessions/{sid}/lab/comparisons/{identifier}/realize", status_code=202)
    def realize(
        sid: str, identifier: str, body: RealizationRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return realize_comparison(service, sid, owner, identifier, body.candidate_id, body.name)

    @routes.get("/sessions/{sid}/lab/explain")
    def explain(
        sid: str,
        revision_id: str = Query(pattern=r"^[a-f0-9]{32}$"),
        body_index: int | None = Query(default=None, ge=0),
        part_id: str | None = None,
        feature_id: str | None = None,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        return explain_design(
            service,
            sid,
            owner,
            revision_id,
            body_index=body_index,
            part_id=part_id,
            feature_id=feature_id,
        )

    @routes.post("/sessions/{sid}/lab/rationales")
    def rationale(
        sid: str, body: RationaleRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return add_rationale(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/challenges")
    def challenge(
        sid: str, body: ChallengeRequest, owner: str = Depends(owner_dependency)
    ) -> dict[str, Any]:
        return preview_challenge(service, sid, owner, body)

    @routes.post("/sessions/{sid}/lab/challenges/{identifier}/apply", status_code=202)
    def apply(sid: str, identifier: str, owner: str = Depends(owner_dependency)) -> dict[str, Any]:
        return apply_challenge(service, sid, owner, identifier)

    @routes.post("/sessions/{sid}/lab/calibrations/{identifier}/design-proposal")
    def improve(
        sid: str,
        identifier: str,
        body: CalibratedDesignRequest,
        owner: str = Depends(owner_dependency),
    ) -> dict[str, Any]:
        return calibrated_design(service, sid, owner, identifier, body)

    return routes
