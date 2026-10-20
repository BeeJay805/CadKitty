"""Closed frozen nominal predicates; authored bindings never contain verdicts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

Name = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]
Quote = Annotated[str, Field(min_length=1, max_length=2000, pattern=r"\S")]
FiniteNumber = Annotated[float, Field(strict=True)]


class VerificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class NominalPatternHole(VerificationRecord):
    center_mm: tuple[FiniteNumber, FiniteNumber]
    diameter_mm: float = Field(gt=0, strict=True)
    diameter_tolerance_mm: float = Field(ge=0, strict=True)
    position_tolerance_mm: float = Field(ge=0, strict=True)


class VerificationVersion(VerificationRecord):
    version: Literal[1] = 1

    @field_validator("version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Verification version must be integer 1")
        return value


class NominalPattern(VerificationVersion):
    scope: Literal["nominal_cad"]
    subject_id: Name
    subject_role: Literal["designed"]
    frame: Literal["definition_local"]
    axis: Literal["x", "y", "z"]
    count_policy: Literal["exact"]
    holes: list[NominalPatternHole] = Field(min_length=1)
    source_quote: Quote

    @model_validator(mode="after")
    def distinct_centers(self) -> NominalPattern:
        if len({hole.center_mm for hole in self.holes}) != len(self.holes):
            raise ValueError("Nominal pattern centers must be distinct")
        return self


class NominalThroughHolePattern(NominalPattern):
    kind: Literal["nominal_through_hole_pattern"]


class NominalPerforatedBox(NominalPattern):
    """Outer prism center, not the centroid after material is removed."""

    kind: Literal["nominal_perforated_box"]
    size_mm: tuple[
        Annotated[float, Field(gt=0, strict=True)],
        Annotated[float, Field(gt=0, strict=True)],
        Annotated[float, Field(gt=0, strict=True)],
    ]
    center_mm: tuple[FiniteNumber, FiniteNumber, FiniteNumber]


class DigitalDelivery(VerificationVersion):
    kind: Literal["digital_delivery"]
    scope: Literal["digital_artifacts"]
    deliverable: Literal[
        "project_stl",
        "project_step",
        "editable_native",
        "prepared_print_package",
        "bound_instruction_plan",
    ]
    coverage: Literal["all_made"]
    source_quote: Quote


class NominalInventoryItem(VerificationRecord):
    subject_id: Name
    subject_role: Literal["designed"]
    required_name: Annotated[str, Field(strict=True, min_length=1, max_length=2000, pattern=r"\S")]
    quantity: int = Field(strict=True, gt=0)


class NominalSubjectInventory(VerificationVersion):
    """Exact PartDefinition.name and installed quantities, never solid count."""

    kind: Literal["nominal_subject_inventory"]
    scope: Literal["nominal_cad"]
    coverage: Literal["all_made"]
    items: list[NominalInventoryItem] = Field(min_length=1)
    source_quote: Quote

    @model_validator(mode="after")
    def unique_inventory(self) -> NominalSubjectInventory:
        if len({item.subject_id for item in self.items}) != len(self.items):
            raise ValueError("Inventory subjects must be unique")
        if len({item.required_name for item in self.items}) != len(self.items):
            raise ValueError("Inventory names must be unique")
        return self


class NominalDeliveryScope(VerificationVersion):
    """Enforced software scope only; no physical or arbitrary prose proof."""

    kind: Literal["nominal_delivery_scope"]
    scope: Literal["software_policy"]
    policy: Literal["nominal_cad_and_digital_only"]
    source_quote: Quote


PhotoClaim = Literal[
    "visible_profile",
    "rectangular_outline",
    "extent_u",
    "extent_v",
    "thickness",
    "opening_count",
    "straight_through",
]


class NominalPhotoSolid(VerificationVersion):
    """Prospective recorded interpretation; image coordinates are not owner facts."""

    kind: Literal["nominal_photo_solid"] = "nominal_photo_solid"
    scope: Literal["recorded_photo_nominal"] = "recorded_photo_nominal"
    subject_id: Name
    subject_role: Literal["designed"] = "designed"
    frame: Literal["definition_local"] = "definition_local"
    axis: Literal["z"] = "z"
    claims: list[PhotoClaim] = Field(min_length=1)
    size_mm: tuple[
        Annotated[float, Field(strict=True, gt=0)],
        Annotated[float, Field(strict=True, gt=0)],
        Annotated[float, Field(strict=True, gt=0)],
    ]
    hole_count: int | None = Field(default=None, strict=True, ge=0)
    size_fact_ids: tuple[Name, Name, Name]
    hole_count_fact_id: Name | None = None
    source_quote: Quote
    source_text_policy: Literal["source_only_v2"] | None = None

    @model_serializer(mode="wrap")
    def legacy_serialization(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value: dict[str, Any] = handler(self)
        if self.source_text_policy is None:
            value.pop("source_text_policy", None)
        return value

    @model_validator(mode="after")
    def unique_claims(self) -> NominalPhotoSolid:
        if len(set(self.claims)) != len(self.claims):
            raise ValueError("Photo nominal claims must be unique")
        if len(set(self.size_fact_ids)) != 3:
            raise ValueError("Photo dimensions need three distinct source facts")
        if (self.hole_count is None) != (self.hole_count_fact_id is None):
            raise ValueError(
                "A source count needs its exact source fact; image counts have neither"
            )
        return self


# V3 is a retained wire protocol. New authority families cannot silently add
# required response keys or acquire a generic repair route through this registry.
VerificationRepairCriterionV3 = (
    NominalThroughHolePattern
    | NominalPerforatedBox
    | DigitalDelivery
    | NominalSubjectInventory
    | NominalDeliveryScope
)
VerificationCriterion = VerificationRepairCriterionV3 | NominalPhotoSolid


class VerificationSubjectBinding(VerificationRecord):
    subject_id: Name
    part_id: Name
    frame: Literal["definition_local"]
