"""Typed measured print poses; no printer settings or physical approval."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
PartName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
XYZ = Annotated[list[float], Field(min_length=3, max_length=3)]
Row4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Matrix4 = Annotated[list[Row4], Field(min_length=4, max_length=4)]

ALGORITHM: Literal["measured-planar-mesh-placement-v1"] = "measured-planar-mesh-placement-v1"
SCOPE = (
    "Measured planar STEP faces corroborated against every delivered mesh triangle. "
    "A rigid plate pose and serialized mesh checks do not establish exact curved STEP "
    "support, adhesion, support strategy, print success, strength or physical fit. "
    "The geometric ranking is not a physically optimal print orientation."
)
RANKING = "centroid_inside_contact_hull_then_contact_area_desc_then_native_face_index"


class PrintRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PrintPlane(PrintRecord):
    """A current imported planar face, bound through its source STEP hash."""

    face_index: StrictInt = Field(ge=0)
    point_mm: XYZ
    outward_normal: XYZ
    area_mm2: float = Field(gt=0)
    kernel_tolerance_mm: float = Field(ge=0)


class PrintSetupRequest(PrintRecord):
    schema_version: Literal[1] = 1
    operation: Literal["inspect", "prepare"]
    revision_id: Identifier
    part_ids: list[PartName] = Field(min_length=1)
    artifact_hashes: dict[str, Hash]

    @model_validator(mode="after")
    def unique_parts(self) -> PrintSetupRequest:
        if len(set(self.part_ids)) != len(self.part_ids):
            raise ValueError("Print setup definitions must be unique")
        return self


class PrintSource(PrintRecord):
    revision_id: Identifier
    part_id: PartName
    quantity: StrictInt = Field(gt=0)
    document_sha256: Hash
    manifest_sha256: Hash
    step_file: str
    step_sha256: Hash
    stl_file: str
    stl_sha256: Hash
    stl_bytes: StrictInt = Field(gt=0)
    body_count: StrictInt = Field(gt=0)
    frame: Literal["definition_local"] = "definition_local"
    units: Literal["mm"] = "mm"


class PrintCandidate(PrintRecord):
    candidate_hash: Hash
    plane: PrintPlane
    transform: Matrix4
    source_frame: Literal["definition_local"] = "definition_local"
    target_frame: Literal["plate_z_up"] = "plate_z_up"
    units: Literal["mm"] = "mm"
    status: Literal["measured", "unresolved"]
    reason: str
    observations: dict[str, Any]


class PrintPartInspection(PrintRecord):
    source: PrintSource
    status: Literal["measured", "unresolved"]
    candidates: list[PrintCandidate]
    recommended_candidate_hash: Hash | None
    mesh: dict[str, Any]
    reason: str


class PrintSetupResult(PrintRecord):
    schema_version: Literal[1] = 1
    algorithm: Literal["measured-planar-mesh-placement-v1"] = ALGORITHM
    revision_id: Identifier
    operation: Literal["inspect", "prepare"]
    status: Literal["measured", "prepared", "unresolved"]
    parts: dict[PartName, PrintPartInspection]
    files: dict[str, dict[str, Any]]
    ranking: str = RANKING
    scope: str = SCOPE
    physical_readiness: Literal["unverified"] = "unverified"
    manufacturing_ready: Literal[False] = False
