"""Owner-declared, bounded cantilever experiments; imports are not physical certification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record

Hypothesis = Literal["material_stiffness", "fixture_translation", "fixture_rotation"]
Provenance = Literal["physical_reported", "synthetic"]


class ExperimentProtocol(Record):
    specimen_id: str = Field(min_length=1, max_length=120)
    material_id: str = Field(min_length=1, max_length=120)
    fixture_id: str = Field(min_length=1, max_length=120)
    apparatus: str = Field(min_length=1, max_length=2000)
    reference_modulus_mpa: float = Field(gt=0, le=2000000)
    width_mm: float = Field(gt=0, le=1000)
    thickness_mm: float = Field(gt=0, le=1000)
    min_span_mm: float = Field(gt=0, le=10000)
    max_span_mm: float = Field(gt=0, le=10000)
    max_force_n: float = Field(gt=0, le=100000)
    max_elastic_stress_mpa: float = Field(gt=0, le=100000)
    geometry_basis: str = Field(min_length=1, max_length=2000)
    assumptions: str = Field(min_length=1, max_length=2000)
    uncertainty_basis: str = Field(min_length=1, max_length=2000)
    # Uncertainties are independent one-standard-deviation values. Geometry is conditional.
    model: Literal["rectangular_cantilever_tip_load_v1"] = "rectangular_cantilever_tip_load_v1"

    @model_validator(mode="after")
    def applicable(self) -> ExperimentProtocol:
        if self.max_span_mm < self.min_span_mm or self.min_span_mm < 10 * self.thickness_mm:
            raise ValueError("Span interval must be ordered and at least ten thicknesses")
        return self


class ExperimentCondition(Record):
    span_mm: float = Field(gt=0, le=10000)
    force_n: float = Field(gt=0, le=100000)
    force_uncertainty_n: float = Field(gt=0, le=10000)


class Observation(ExperimentCondition):
    observation_id: str = Field(min_length=1, max_length=120)
    displacement_mm: float = Field(ge=0, le=1000)
    displacement_uncertainty_mm: float = Field(gt=0, le=100)


class ExperimentRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_input_version: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=200)
    provenance: Provenance
    protocol: ExperimentProtocol
    observations: list[Observation] = Field(default_factory=list, max_length=200)
    csv_text: str | None = Field(default=None, min_length=1, max_length=200000)

    @model_validator(mode="after")
    def one_source(self) -> ExperimentRequest:
        if bool(self.observations) == bool(self.csv_text):
            raise ValueError("Provide either observations or CSV, exclusively")
        return self


class CalibrationRequest(Record):
    experiment_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    observation_ids: list[str] = Field(min_length=3, max_length=200)

    @model_validator(mode="after")
    def unique_rows(self) -> CalibrationRequest:
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("Calibration observation IDs must be unique")
        return self


class PredictionRequest(Record):
    calibration_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    hypothesis: Hypothesis
    condition: ExperimentCondition
    comparison_tolerance_mm: float = Field(gt=0, le=100)
    planned_displacement_uncertainty_mm: float = Field(gt=0, le=100)
    rationale: str = Field(min_length=1, max_length=2000)


class OutcomeRequest(Record):
    prediction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    provenance: Provenance
    observation: Observation
    acquisition_note: str = Field(min_length=1, max_length=2000)
