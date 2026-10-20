"""Discover, propose, evaluate and compare designs through the public tool contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from forge_mind.workspace.action_payloads import DesignCollection, read_design_arguments
from forge_mind.workspace.design_study_models import (
    AcceptStudyCandidate,
    AddStudyCandidate,
    CreateDesignStudy,
    RunDesignStudy,
    SearchDesignStudy,
)

if TYPE_CHECKING:
    from forge_mind.workspace.mcp_server import WorkspaceAdapter

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


def register_design_study_tools(server: MCPServer, adapter: WorkspaceAdapter) -> None:
    @server.tool(annotations=READ)
    async def get_design_study_schema() -> dict[str, Any]:
        """Discover measured objectives, exact evaluator settings, capabilities and installation state."""
        return dict(await adapter.json("GET", "/workspace/design-study-schema"))

    @server.tool(annotations=READ)
    async def list_design_studies(project_id: Identifier) -> dict[str, Any]:
        """List preserved general design studies, without losing older experiments."""
        return dict(await adapter.json("GET", f"/workspace/sessions/{project_id}/design-studies"))

    @server.tool(annotations=WRITE)
    async def create_design_study(
        project_id: Identifier, request: CreateDesignStudy
    ) -> dict[str, Any]:
        """Freeze baseline, objectives, constraints, parameter bounds and evaluator assumptions before comparing candidates. This does not change product acceptance requirements."""
        return dict(
            await adapter.json(
                "POST",
                f"/workspace/sessions/{project_id}/design-studies",
                body=request.model_dump(),
            )
        )

    @server.tool(annotations=READ)
    async def get_design_study(project_id: Identifier, study_id: Identifier) -> dict[str, Any]:
        """Read actual candidate measurements, constraint failures, progress and admissible comparisons. Failed or unconverged evaluations never win."""
        return dict(
            await adapter.json("GET", f"/workspace/sessions/{project_id}/design-studies/{study_id}")
        )

    @server.tool(annotations=WRITE)
    async def add_design_candidate(
        project_id: Identifier, study_id: Identifier, request: AddStudyCandidate
    ) -> dict[str, Any]:
        """Add parameter changes or an existing editable revision, including topology-changing designs, to one frozen protocol."""
        return dict(
            await adapter.json(
                "POST",
                f"/workspace/sessions/{project_id}/design-studies/{study_id}/candidates",
                body=request.model_dump(),
            )
        )

    @server.tool(annotations=READ)
    async def get_design_study_progress(
        project_id: Identifier, study_id: Identifier
    ) -> dict[str, Any]:
        """Poll diagnostic solver stage, iteration, refinement and saved-stage counts cheaply. Progress does not verify evidence; read the full study before publishing measurements."""
        return dict(
            await adapter.json(
                "GET", f"/workspace/sessions/{project_id}/design-studies/{study_id}/progress"
            )
        )

    @server.tool(annotations=WRITE)
    async def run_design_study(
        project_id: Identifier, study_id: Identifier, retry_failed: bool = False
    ) -> dict[str, Any]:
        """Queue pending or interrupted candidates; retain verified finished evaluations. Use existing project cancellation to stop workers."""
        return dict(
            await adapter.json(
                "POST",
                f"/workspace/sessions/{project_id}/design-studies/{study_id}/run",
                body=RunDesignStudy(retry_failed=retry_failed).model_dump(),
            )
        )

    @server.tool(annotations=WRITE)
    async def search_design_study(
        project_id: Identifier, study_id: Identifier, request: SearchDesignStudy
    ) -> dict[str, Any]:
        """Iterate a declared number of coordinate-search rounds within frozen bounds, rebuilding and measuring every candidate. Best evaluated is not a global optimum."""
        return dict(
            await adapter.json(
                "POST",
                f"/workspace/sessions/{project_id}/design-studies/{study_id}/search",
                body=request.model_dump(),
            )
        )

    @server.tool(annotations=WRITE)
    async def propose_study_candidate(
        project_id: Identifier, study_id: Identifier, request: AcceptStudyCandidate
    ) -> dict[str, Any]:
        """Create a guarded proposal from a feasible evaluated candidate, preserving original requirements and study baseline."""
        return dict(
            await adapter.json(
                "POST",
                f"/workspace/sessions/{project_id}/design-studies/{study_id}/accept",
                body=request.model_dump(),
            )
        )

    @server.tool(annotations=READ)
    async def read_design_page(
        project_id: Identifier,
        revision_id: Identifier,
        collection: DesignCollection,
        scope: Annotated[str, Field(strict=True)] | None = None,
        cursor: Annotated[str, Field(strict=True)] | None = None,
        feature_id: Annotated[str, Field(strict=True)] | None = None,
        limit: Annotated[int, Field(strict=True, ge=1, le=1000)] = 100,
    ) -> dict[str, Any]:
        """Read complete large designs through revision-bound pages. Select variables, relations, parts, features, occurrences, nodes or edges; follow the returned cursor."""
        arguments = read_design_arguments(
            collection, {"scope": scope, "cursor": cursor, "feature_id": feature_id, "limit": limit}
        )
        query = {key: str(value) for key, value in arguments.model_dump(exclude_none=True).items()}
        return dict(
            await adapter.json(
                "GET",
                f"/workspace/sessions/{project_id}/design/{revision_id}/page",
                params=query,
            )
        )
