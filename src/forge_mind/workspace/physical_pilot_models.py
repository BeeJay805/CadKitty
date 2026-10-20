"""Small owner-reported bench observations, never physical certification."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator

from forge_mind.workspace.models import Record

Text = Annotated[StrictStr, Field(min_length=1, max_length=2000, pattern=r".*\S.*")]
Identifier = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")]
Digest = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
CheckId = Literal[
    "print_completion",
    "part_inspection",
    "hardware_fit",
    "assembly_fit",
    "removal",
    "wall_installation",
]


class PilotPrinter(Record):
    name: Text
    profile: Text | None = None
    nozzle_mm: StrictFloat | None = Field(default=None, gt=0, allow_inf_nan=False)


class PilotMaterial(Record):
    name: Text
    supplier_or_batch: Text | None = None


class PilotHardware(Record):
    name: Text
    quantity: StrictInt = Field(gt=0)
    part_id: StrictStr | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    note: Text


class PilotMeasurement(Record):
    name: Text
    value: StrictFloat = Field(allow_inf_nan=False)
    unit: Text
    method: Text
    uncertainty: StrictFloat | None = Field(default=None, ge=0, allow_inf_nan=False)


class PilotCheckOutcome(Record):
    check_id: CheckId
    status: Literal["observed_pass", "observed_fail", "unknown", "not_run"]
    observation: Text
    measurements: list[PilotMeasurement] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def measurements_require_observation(self) -> PilotCheckOutcome:
        if self.status == "not_run" and self.measurements:
            raise ValueError("A check that was not run cannot contain measured results")
        return self


class PhysicalPilotOutcomeRequest(Record):
    revision_id: Identifier
    protocol_hash: Digest
    specimen_id: Text
    provenance: Literal["physical_reported"]
    printer: PilotPrinter
    material: PilotMaterial
    hardware: list[PilotHardware] = Field(max_length=128)
    checks: list[PilotCheckOutcome] = Field(min_length=6, max_length=6)
    acquisition_note: Text

    @model_validator(mode="after")
    def complete_unique_checks(self) -> PhysicalPilotOutcomeRequest:
        identifiers = [check.check_id for check in self.checks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Supply each of the six protocol checks once")
        parts = [item.part_id for item in self.hardware if item.part_id is not None]
        if len(set(parts)) != len(parts):
            raise ValueError("Declare each mapped hardware definition once")
        return self
