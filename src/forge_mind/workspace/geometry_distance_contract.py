"""Shared, native-free request and result contracts for saved geometry distance."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{32}$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{64}$")]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = tuple[Number, Number, Number]
FRAME = "cad_z_up_mm_world"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024
SOURCE_NAMES = frozenset({"part.step", "topology.json", "document.json"})
SCOPE = (
    "Minimum Euclidean distance between the selected saved solids or trimmed faces, "
    "with one representative native point pair. Multiple minima may exist. Zero does "
    "not distinguish contact, overlap or containment. native_inner_solution is not "
    "an overlap classifier. Solid witnesses may lie inside a solid. No signed "
    "penetration, wall-thickness, motion, physical-fit or acceptance claim."
)


class DistanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DistanceSelector(DistanceRecord):
    body_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    occurrence_path: Annotated[str, Field(strict=True, min_length=1, max_length=2048)] | None = None
    face_signature: Digest | None = None

    @model_validator(mode="after")
    def one_body(self) -> DistanceSelector:
        if (self.body_index is None) == (self.occurrence_path is None):
            raise ValueError("Select exactly one saved body index or occurrence path")
        return self


class DistanceQuery(DistanceRecord):
    a: DistanceSelector
    b: DistanceSelector


class ResolvedDistanceSelector(DistanceRecord):
    requested: DistanceSelector
    body_index: Annotated[int, Field(strict=True, ge=0)]
    body_signature: Digest
    face_signature: Digest | None = None

    @model_validator(mode="after")
    def matches_request(self) -> ResolvedDistanceSelector:
        if self.requested.body_index is not None and self.body_index != self.requested.body_index:
            raise ValueError("Resolved body does not match requested index")
        if self.face_signature != self.requested.face_signature:
            raise ValueError("Resolved face does not match requested face")
        return self


class DistanceContext(DistanceRecord):
    project_id: Identifier
    revision_id: Identifier
    operation: Literal["minimum_distance"] = "minimum_distance"
    frame: Literal["cad_z_up_mm_world"] = "cad_z_up_mm_world"
    units: Literal["mm"] = "mm"
    a: ResolvedDistanceSelector
    b: ResolvedDistanceSelector
    source_hashes: dict[str, Digest]
    saved_engine_fingerprint: Digest
    query_engine_fingerprint: Digest

    @model_validator(mode="after")
    def sources(self) -> DistanceContext:
        if set(self.source_hashes) != SOURCE_NAMES:
            raise ValueError("Distance requires exact STEP, topology and document hashes")
        if (self.a.body_index, self.a.face_signature) == (
            self.b.body_index,
            self.b.face_signature,
        ):
            raise ValueError("Select two distinct saved entities")
        return self


class DistanceResult(DistanceContext):
    minimum_distance_mm: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
    witness_a_mm: Vector
    witness_b_mm: Vector
    solution_count: Annotated[int, Field(strict=True, ge=1)]
    native_completed: Annotated[bool, Field(strict=True)]
    native_inner_solution: Annotated[bool, Field(strict=True)]
    scope: str = SCOPE

    @model_validator(mode="after")
    def completed_measurement(self) -> DistanceResult:
        if not self.native_completed or self.scope != SCOPE:
            raise ValueError("Distance requires completed native measurement and exact scope")
        witness_distance = math.dist(self.witness_a_mm, self.witness_b_mm)
        if not math.isfinite(witness_distance) or not math.isclose(
            witness_distance, self.minimum_distance_mm, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError("Native point witnesses do not match the reported distance")
        return self


def distance_tool_result(
    value: Any,
    project_id: str,
    revision_id: str,
    a: DistanceSelector,
    b: DistanceSelector,
) -> dict[str, Any]:
    """Validate the response and retain the exact requested selection on both transports."""
    result = DistanceResult.model_validate(value)
    if (
        result.project_id != project_id
        or result.revision_id != revision_id
        or result.a.requested != a
        or result.b.requested != b
    ):
        raise ValueError("Distance result does not match the requested revision and entities")
    encoded = result.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Distance result exceeds its response byte limit")
    return result.model_dump(mode="json")
