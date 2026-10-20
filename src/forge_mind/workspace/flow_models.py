"""Declared physics and numerical settings for external viscous flow experiments."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record


class FlowMeshSettings(Record):
    reference_length_m: float | None = Field(default=None, gt=0)
    grading_rate: float = Field(default=0, ge=0)
    surface_size_factor: float = Field(default=0.12, gt=0, le=0.5)
    farfield_size_factor: float = Field(default=1.5, gt=0)
    refinement_distance_factor: float = Field(default=3, gt=0)
    domain_padding_factor: float = Field(default=12, ge=3)
    refinement: Literal["coarse", "fine"] = "coarse"
    fine_scale: float = Field(default=0.65, gt=0, lt=1)
    wall_layers: int = Field(default=0, ge=0)
    first_layer_size_factor: float = Field(default=1e-5, gt=0, lt=0.1)
    layer_growth: float = Field(default=1.3, gt=1, le=2)


class FlowNumerics(Record):
    iterations: int = Field(default=1000, ge=20)
    cfl: float = Field(default=10, gt=0)
    beta_factor: float = Field(default=4.1, gt=0)
    linear_solver_iterations: int = Field(default=60, ge=1)
    residual_log10_max: float = Field(default=-6, le=-2)
    force_window: int = Field(default=30, ge=5)
    force_relative_tolerance: float = Field(default=0.005, gt=0, le=0.1)


class FlowResources(Record):
    threads: int = Field(default=2, ge=1)
    max_seconds: float | None = Field(default=None, gt=0)
    max_memory_mb: float | None = Field(default=None, gt=0)
    max_mesh_nodes: int | None = Field(default=None, ge=4)
    max_total_seconds: float | None = Field(default=None, gt=0)


class FlowAdaptation(Record):
    protocol: Literal["controlled_cross_v1"] = "controlled_cross_v1"
    domain_scale: float = Field(default=1.5, ge=1.1)
    max_refinements: int | None = Field(default=None, ge=1)


class FlowSettings(Record):
    density_kg_m3: float = Field(gt=0)
    dynamic_viscosity_pa_s: float = Field(gt=0)
    velocity_m_s: tuple[float, float, float]
    reference_area_m2: float = Field(gt=0)
    reference_length_m: float = Field(gt=0)
    regime: Literal["laminar", "sa"] = "laminar"
    wall_policy: Literal["resolved_no_slip"] = "resolved_no_slip"
    sa_freestream_nu_factor: float = Field(default=3, gt=0)
    solid_index: int | None = Field(default=None, ge=0)
    mesh: FlowMeshSettings = Field(default_factory=FlowMeshSettings)
    numerics: FlowNumerics = Field(default_factory=FlowNumerics)
    resources: FlowResources = Field(default_factory=FlowResources)
    adaptation: FlowAdaptation = Field(default_factory=FlowAdaptation)

    @model_validator(mode="after")
    def physical_setup(self) -> FlowSettings:
        if math.hypot(*self.velocity_m_s) <= 0:
            raise ValueError("External flow needs a nonzero finite freestream velocity")
        if self.numerics.force_window > self.numerics.iterations:
            raise ValueError("Force convergence window exceeds requested iterations")
        if self.mesh.farfield_size_factor < self.mesh.surface_size_factor:
            raise ValueError("Farfield mesh size must be at least the surface mesh size")
        if self.regime == "sa" and self.mesh.wall_layers < 3:
            raise ValueError(
                "SA resolved-wall studies require at least three explicit prismatic wall layers"
            )
        return self


FLOW_SCOPE = (
    "Steady, constant-density incompressible viscous external flow around one exported solid. "
    "Laminar low-Reynolds-number or explicitly selected Spalart-Allmaras RANS screening; "
    "no transitional, compressible, "
    "moving-body or fluid-structure prediction. No-slip wall on the actual CAD boundary. "
    "RANS requires prismatic wall layers and measured y-plus <= 1, which alone does not "
    "certify boundary-layer resolution or turbulence-model accuracy. Numerical "
    "convergence is not mesh independence or physical validation. Compare baseline and winner "
    "on finer meshes and enlarged domains under identical conditions before claiming improvement."
)
