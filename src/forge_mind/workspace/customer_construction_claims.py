"""Typed decisions from the single command approval claim boundary."""

from typing import Any, Literal, TypedDict


class ExecuteConstructionClaim(TypedDict):
    mode: Literal["execute_saved_command"]
    action: dict[str, Any]
    feature_basis: dict[str, Any] | None
    command_hash: str


class PublishedConstructionContinuation(TypedDict):
    mode: Literal["continue_published"]
    revision_id: str
    execution_job_id: str
    command_hash: str


ConstructionClaim = ExecuteConstructionClaim | PublishedConstructionContinuation
