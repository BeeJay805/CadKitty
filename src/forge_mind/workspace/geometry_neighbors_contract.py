"""Native-free contracts for one selected saved topology neighborhood."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{32}$")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{64}$")]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Vector = tuple[Number, Number, Number]
Index = Annotated[int, Field(strict=True, ge=0)]
Count = Annotated[int, Field(strict=True, ge=1)]
Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
FRAME = "cad_z_up_mm_world"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
SOURCE_NAMES = frozenset({"part.step", "topology.json", "document.json"})
SCOPE = (
    "One-ring native boundary relationships of a selected saved face or edge. "
    "Neighbors are unique entities; incidence counts retain repeated edge uses "
    "within a face, including seams. Edge center is the native curve midpoint, "
    "not a circle center. Identities are revision-local. No tangent "
    "continuity, feature function, across-edit identity, intermediate feature-stage "
    "selection validity, physical-fit or acceptance claim."
)


class NeighborRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NeighborSelector(NeighborRecord):
    body_index: Index | None = None
    occurrence_path: Annotated[str, Field(strict=True, min_length=1, max_length=2048)] | None = None
    kind: Literal["face", "edge"]
    signature: Digest

    @model_validator(mode="after")
    def one_body(self) -> NeighborSelector:
        if (self.body_index is None) == (self.occurrence_path is None):
            raise ValueError("Select exactly one saved body index or occurrence path")
        return self


class ResolvedNeighborSelector(NeighborRecord):
    requested: NeighborSelector
    body_index: Index
    body_signature: Digest

    @model_validator(mode="after")
    def matches_request(self) -> ResolvedNeighborSelector:
        if self.requested.body_index is not None and self.body_index != self.requested.body_index:
            raise ValueError("Resolved body does not match requested index")
        return self


class NeighborQuery(NeighborRecord):
    selector: NeighborSelector
    offset: Index = 0
    limit: Limit = 40


class NeighborContext(NeighborRecord):
    project_id: Identifier
    revision_id: Identifier
    selector: ResolvedNeighborSelector
    offset: Index = 0
    limit: Limit = 40
    operation: Literal["topology_neighbors"] = "topology_neighbors"
    frame: Literal["cad_z_up_mm_world"] = "cad_z_up_mm_world"
    units: Literal["mm"] = "mm"
    source_hashes: dict[str, Digest]
    saved_engine_fingerprint: Digest
    query_engine_fingerprint: Digest

    @model_validator(mode="after")
    def exact_sources(self) -> NeighborContext:
        if set(self.source_hashes) != SOURCE_NAMES:
            raise ValueError("Neighbors require exact STEP, topology and document hashes")
        return self


class EntityFact(NeighborRecord):
    id: Annotated[str, Field(strict=True, min_length=1, max_length=160)]
    kind: Literal["face", "edge"]
    signature: Digest
    body_index: Index
    body_signature: Digest
    center: Vector
    min: Vector
    max: Vector
    measure: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
    unit: Literal["mm", "mm^2"]
    surface: Annotated[str, Field(strict=True, min_length=1, max_length=64)] | None = None
    normal: Vector | None = None
    curve: Annotated[str, Field(strict=True, min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def matching_geometry(self) -> EntityFact:
        if any(lower > upper for lower, upper in zip(self.min, self.max, strict=True)):
            raise ValueError("Native entity bounds are not ordered")
        if self.unit != ("mm^2" if self.kind == "face" else "mm"):
            raise ValueError("Native entity measure unit disagrees with kind")
        if self.kind == "face" and self.curve is not None:
            raise ValueError("A face cannot carry an edge curve type")
        if self.kind == "edge" and (self.surface is not None or self.normal is not None):
            raise ValueError("An edge cannot carry face surface/normal facts")
        return self


class NeighborItem(NeighborRecord):
    entity: EntityFact
    incidence_count: Count


class NeighborResult(NeighborContext):
    focal: EntityFact
    relation: Literal["face_boundary_edges", "edge_incident_faces"]
    neighbors: Annotated[list[NeighborItem], Field(max_length=100)]
    total_neighbors: Index
    total_incidences: Index
    next_offset: Index | None
    native_completed: Annotated[bool, Field(strict=True)]
    coverage: Literal["complete_selected_neighborhood"] = "complete_selected_neighborhood"
    scope: str = SCOPE

    @model_validator(mode="after")
    def consistent_neighborhood(self) -> NeighborResult:
        requested = self.selector.requested
        if not self.native_completed or self.scope != SCOPE:
            raise ValueError("Neighbors require completed native traversal and exact scope")
        if (
            self.focal.kind != requested.kind
            or self.focal.signature != requested.signature
            or self.focal.body_index != self.selector.body_index
            or self.focal.body_signature != self.selector.body_signature
        ):
            raise ValueError("Focal entity does not match resolved selection")
        expected_kind = "edge" if requested.kind == "face" else "face"
        expected_relation = (
            "face_boundary_edges" if requested.kind == "face" else "edge_incident_faces"
        )
        if self.relation != expected_relation:
            raise ValueError("Neighborhood relation disagrees with selection kind")
        ids, signatures = set(), set()
        for item in self.neighbors:
            entity = item.entity
            if (
                entity.kind != expected_kind
                or entity.body_index != self.selector.body_index
                or entity.body_signature != self.selector.body_signature
                or entity.id == self.focal.id
                or entity.id in ids
                or entity.signature in signatures
            ):
                raise ValueError("Neighbors have duplicate or mismatched saved identities")
            ids.add(entity.id)
            signatures.add(entity.signature)
        expected_size = max(0, min(self.limit, self.total_neighbors - self.offset))
        if len(self.neighbors) != expected_size:
            raise ValueError("Neighborhood page does not match total/offset/limit")
        end = self.offset + len(self.neighbors)
        expected_next = end if end < self.total_neighbors else None
        if self.next_offset != expected_next:
            raise ValueError("Neighborhood next offset is inconsistent")
        page_incidences = sum(item.incidence_count for item in self.neighbors)
        if self.total_neighbors == 0 and self.total_incidences != 0:
            raise ValueError("An empty native neighborhood cannot contain incidences")
        if self.total_incidences < page_incidences + self.total_neighbors - len(self.neighbors):
            raise ValueError("Neighborhood incidence total cannot cover its unique neighbors")
        if (
            self.offset == 0
            and len(self.neighbors) == self.total_neighbors
            and self.total_incidences != page_incidences
        ):
            raise ValueError("Complete page incidence total disagrees with neighbor counts")
        return self


def neighbors_tool_result(
    value: Any,
    project_id: str,
    revision_id: str,
    selector: NeighborSelector,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Bind a validated result to the exact caller request on either transport."""
    query = NeighborQuery(selector=selector, offset=offset, limit=limit)
    result = NeighborResult.model_validate(value)
    if (
        result.project_id != project_id
        or result.revision_id != revision_id
        or result.selector.requested != query.selector
        or result.offset != query.offset
        or result.limit != query.limit
    ):
        raise ValueError("Neighbors result does not match requested revision/selection/page")
    if len(result.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Neighbors result exceeds its response byte limit")
    return result.model_dump(mode="json")
