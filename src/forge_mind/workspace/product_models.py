"""Declared production intent, never a manufacturing or supplier certificate."""

from typing import Literal

from pydantic import Field, StrictBool, StrictStr

from forge_mind.workspace.models import Record


class DeliveryGuidance(Record):
    """Saved customer instructions, not physical or manufacturing approval."""

    print: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Declared print orientation/settings and their limits; material remains in production.",
    )
    installation: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Installation steps and prerequisites, including known unmodeled hardware needs. Refer to modeled bought parts without duplicating their quantity authority.",
    )
    use_and_removal: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Customer use and removal instructions, with unverified access or fit stated.",
    )
    unresolved: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description="Unknown choices and checks, such as anchors, load or physical fit. Do not invent specifications.",
    )


class ProductionMaterial(Record):
    name: str = Field(min_length=1, max_length=160)
    grade: str | None = Field(default=None, max_length=160)
    density_kg_m3: float | None = Field(default=None, gt=0, le=100000)
    source_claim_ids: list[str] = Field(default_factory=list, max_length=64)


class Procurement(Record):
    catalog_id: str | None = Field(default=None, max_length=160)
    supplier: str | None = Field(default=None, max_length=300)
    part_number: str | None = Field(default=None, max_length=160)
    source_claim_ids: list[str] = Field(default_factory=list, max_length=64)


class PartAppearance(Record):
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class PartProduction(Record):
    role: Literal["make", "buy", "reference", "unspecified"] = "unspecified"
    deliver_stl: StrictBool = False
    material: ProductionMaterial | None = None
    process: Literal[
        "additive",
        "cnc",
        "sheet_metal",
        "casting",
        "injection_molding",
        "laser_cutting",
        "manual",
        "other",
        "unspecified",
    ] = "unspecified"
    finish: str | None = Field(default=None, max_length=300)
    appearance: PartAppearance | None = None
    procurement: Procurement | None = None
    source_claim_ids: list[str] = Field(default_factory=list, max_length=64)
