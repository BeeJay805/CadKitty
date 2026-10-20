"""Typed assistant access to the same experiment, synthesis and challenge services."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from forge_mind.workspace.design_reasoning import ChallengeRequest, RationaleRequest
from forge_mind.workspace.experiment_models import (
    CalibrationRequest,
    ExperimentRequest,
    OutcomeRequest,
    PredictionRequest,
)
from forge_mind.workspace.lab_api import CalibratedDesignRequest, ComparisonRequest

if TYPE_CHECKING:
    from forge_mind.workspace.mcp_server import WorkspaceAdapter

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """CAD documents and original CSV remain fetchable; routine reasoning uses receipts."""
    result = {k: v for k, v in record.items() if k not in {"csv_text", "candidate_document"}}
    if "candidates" in result:
        result["candidates"] = [
            {k: v for k, v in c.items() if k != "document"} for c in result["candidates"]
        ]
    return result


def register_lab_tools(server: MCPServer, adapter: WorkspaceAdapter) -> None:
    async def post(project_id: str, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        return compact_record(
            json.loads(
                await adapter.request(
                    "POST",
                    f"/workspace/sessions/{project_id}/lab/{operation}",
                    body=body,
                )
            )
        )

    @server.tool(annotations=READ)
    async def get_lab_schema() -> dict[str, Any]:
        """Read exact experiment, synthesis, rationale and challenge schemas. Physics scopes are bounded, not product certification."""
        return dict(json.loads(await adapter.request("GET", "/workspace/lab/schema")))

    @server.tool(annotations=READ)
    async def list_lab_records(project_id: Identifier) -> dict[str, Any]:
        """Read compact immutable investigation metadata and freshness before choosing an experiment or comparison."""
        return dict(
            json.loads(
                await adapter.request("GET", f"/workspace/sessions/{project_id}/lab/records")
            )
        )

    @server.tool(annotations=READ)
    async def get_lab_record(project_id: Identifier, record_id: Identifier) -> dict[str, Any]:
        """Read a hashed experiment, calibration, prediction, outcome or mechanism comparison with ancestry and stale status."""
        return compact_record(
            json.loads(
                await adapter.request(
                    "GET",
                    f"/workspace/sessions/{project_id}/lab/records/{record_id}",
                )
            )
        )

    @server.tool(annotations=WRITE)
    async def import_experiment(
        project_id: Identifier, experiment: ExperimentRequest
    ) -> dict[str, Any]:
        """Import supplied measurements or explicit synthetic controls. physical_reported is an operator claim, never authenticated physical proof. Never invent observations."""
        return await post(project_id, "experiments", experiment.model_dump())

    @server.tool(annotations=WRITE)
    async def calibrate_experiment(
        project_id: Identifier, calibration: CalibrationRequest
    ) -> dict[str, Any]:
        """Compare material and fixture-compliance hypotheses against specified immutable training rows, with uncertainty and identifiability."""
        return await post(project_id, "calibrations", calibration.model_dump())

    @server.tool(annotations=WRITE)
    async def register_prediction(
        project_id: Identifier, prediction: PredictionRequest
    ) -> dict[str, Any]:
        """Commit a prediction and tolerance before submitting an outcome. Previously observed conditions are refused; same-geometry force changes only test linear scaling."""
        return await post(project_id, "predictions", prediction.model_dump())

    @server.tool(annotations=WRITE)
    async def record_experiment_outcome(
        project_id: Identifier, outcome: OutcomeRequest
    ) -> dict[str, Any]:
        """Attach supplied measurements once to a frozen prospective prediction. Never manufacture data or change acceptance to match an observation."""
        return await post(project_id, "outcomes", outcome.model_dump())

    @server.tool(annotations=WRITE)
    async def compare_mechanism_architectures(
        project_id: Identifier, comparison: ComparisonRequest
    ) -> dict[str, Any]:
        """Screen editable screw, rack and symmetric linkage candidates against common functional targets and explicit drive/efficiency assumptions."""
        return await post(project_id, "comparisons", comparison.model_dump())

    @server.tool(annotations=WRITE)
    async def realize_mechanism_candidate(
        project_id: Identifier,
        comparison_id: Identifier,
        candidate_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")],
        name: Annotated[str, Field(min_length=1, max_length=120)],
    ) -> dict[str, Any]:
        """Build the screened feature document in a new candidate project; original project and frozen task remain preserved. Poll returned project_id for real CAD results."""
        return await post(
            project_id,
            f"comparisons/{comparison_id}/realize",
            {"candidate_id": candidate_id, "name": name},
        )

    @server.tool(annotations=READ)
    async def explain_design_part(
        project_id: Identifier,
        revision_id: Identifier,
        body_index: Annotated[int | None, Field(ge=0)] = None,
    ) -> dict[str, Any]:
        """Read selected-part variables, authored purpose, source assumptions and linked engineering evidence. Unknown purpose is not invented."""
        params: dict[str, Any] = {"revision_id": revision_id}
        if body_index is not None:
            params["body_index"] = body_index
        return dict(
            json.loads(
                await adapter.request(
                    "GET", f"/workspace/sessions/{project_id}/lab/explain", params=params
                )
            )
        )

    @server.tool(annotations=WRITE)
    async def record_design_rationale(
        project_id: Identifier, rationale: RationaleRequest
    ) -> dict[str, Any]:
        """Record an authored engineering explanation linked to real variables, requirements, sources and receipts; links do not independently validate causality."""
        return await post(project_id, "rationales", rationale.model_dump())

    @server.tool(annotations=WRITE)
    async def preview_design_challenge(
        project_id: Identifier, challenge: ChallengeRequest
    ) -> dict[str, Any]:
        """Preview an executable parameter change and affected parts/evidence. Original engineering requirements cannot be rewritten by this operation."""
        return await post(project_id, "challenges", challenge.model_dump())

    @server.tool(annotations=WRITE)
    async def apply_design_challenge(
        project_id: Identifier, challenge_id: Identifier
    ) -> dict[str, Any]:
        """Queue the exact current challenge document as a guarded candidate, retaining protections and frozen requirements. Does not accept it."""
        return await post(project_id, f"challenges/{challenge_id}/apply", {})

    @server.tool(annotations=WRITE)
    async def propose_calibrated_design(
        project_id: Identifier,
        calibration_id: Identifier,
        request: CalibratedDesignRequest,
    ) -> dict[str, Any]:
        """Propose a thickness change under a calibrated beam/fixture hypothesis and explicit CAD mapping. Rebuild and a new physical experiment remain required."""
        return await post(
            project_id, f"calibrations/{calibration_id}/design-proposal", request.model_dump()
        )
