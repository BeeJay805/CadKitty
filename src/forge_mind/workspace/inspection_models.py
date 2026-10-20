"""Authored inspection intent, never supplied measurements or acceptance."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceText = Annotated[str, Field(min_length=1, max_length=2000)]
OccurrencePath = Annotated[str, Field(min_length=1, max_length=256)]
RequirementId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class InspectionProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    check_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    result: Literal["probe", "ordered_separation_mm", "boundary_midpoint_mm"] = "probe"


class InspectionCharacteristic(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    label: str = Field(min_length=1, max_length=160)
    requirement_ids: list[RequirementId] = Field(default_factory=list)
    occurrence_paths: list[OccurrencePath] = Field(default_factory=list)
    characteristic: str = Field(min_length=1, max_length=2000)
    datum: str = Field(min_length=1, max_length=2000)
    units: Literal["mm"] = "mm"
    nominal_mm: float | None = Field(default=None, strict=True)
    minus_mm: float | None = Field(default=None, ge=0, strict=True)
    plus_mm: float | None = Field(default=None, ge=0, strict=True)
    basis: str = Field(min_length=1, max_length=2000)
    physical_method: str = Field(min_length=1, max_length=2000)
    missing_evidence: list[EvidenceText] = Field(default_factory=list)
    probe: InspectionProbe | None = None

    @model_validator(mode="after")
    def limits(self) -> InspectionCharacteristic:
        if (self.minus_mm is None) != (self.plus_mm is None):
            raise ValueError("Supply both asymmetric allowances or leave both unknown")
        if self.minus_mm is not None and self.nominal_mm is None:
            raise ValueError("Allowances require a proposed nominal")
        if (
            self.nominal_mm is not None
            and self.minus_mm is not None
            and self.plus_mm is not None
            and not all(
                math.isfinite(v)
                for v in (self.nominal_mm - self.minus_mm, self.nominal_mm + self.plus_mm)
            )
        ):
            raise ValueError("Proposed limits must be finite")
        return self
