"""Explicit owner-authored inputs. Their presence is never a validation result."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from forge_mind.workspace.document import Annotation, GeometryRef, Name, Protection, Vector3, digest
from forge_mind.workspace.mechanism_models import MechanismLoad
from forge_mind.workspace.models import Record


class Assumption(Record):
    id: Name
    text: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=1000)
    confirmed: bool = False


class MaterialInput(Record):
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=1000)
    density_kg_m3: float = Field(gt=0, le=30000)
    elastic_modulus_mpa: float = Field(gt=0, le=2000000)
    poisson_ratio: float = Field(gt=-1, lt=0.5)
    yield_strength_mpa: float | None = Field(default=None, gt=0)


class ProcessInput(Record):
    process: Literal["fdm", "cnc", "resin", "other"]
    source: str = Field(min_length=1, max_length=1000)
    build_volume_mm: Vector3
    min_wall_mm: float | None = Field(default=None, gt=0)
    max_unsupported_angle_deg: float | None = Field(default=None, ge=0, le=90)
    tool_diameter_mm: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def positive_volume(self) -> ProcessInput:
        if any(v <= 0 for v in self.build_volume_mm):
            raise ValueError("Build volume dimensions must be positive")
        return self


class ToleranceContributor(Record):
    name: str = Field(min_length=1, max_length=120)
    nominal_mm: float
    minus_mm: float = Field(ge=0)
    plus_mm: float = Field(ge=0)
    source: str = Field(min_length=1, max_length=1000)
    coefficient: float | None = None


class FitInput(Record):
    id: Name
    hole: ToleranceContributor
    shaft: ToleranceContributor
    extras: list[ToleranceContributor] = Field(default_factory=list)
    min_clearance_mm: float
    max_clearance_mm: float
    fit_class: Literal["clearance", "interference", "transition"]
    source: str = Field(min_length=1, max_length=1000)


class JointInput(Record):
    id: Name
    kind: Literal["fixed", "revolute", "slider"]
    parent: int = Field(ge=0)
    child: int = Field(ge=0)
    parent_origin: Vector3 = (0, 0, 0)
    child_origin: Vector3 = (0, 0, 0)
    # Coordinate frames use a Z axis and an X reference direction.
    parent_axis: Vector3 = (0, 0, 1)
    child_axis: Vector3 = (0, 0, 1)
    parent_x: Vector3 = (1, 0, 0)
    child_x: Vector3 = (1, 0, 0)
    value: float = 0
    lower: float = 0
    upper: float = 0

    @model_validator(mode="after")
    def limits(self) -> JointInput:
        if self.parent == self.child or not self.lower <= self.value <= self.upper:
            raise ValueError("Joint needs different bodies and a value inside ordered limits")
        if self.kind == "fixed" and (self.value or self.lower or self.upper):
            raise ValueError("Fixed joint has no motion")
        return self


class LoadInput(Record):
    id: Name
    method: Literal["linear_fea", "rectangular_cantilever"]
    force_n: float = Field(gt=0, le=100000000)
    direction: Vector3 = (1, 0, 0)
    # Actual supported FEA boundary: lowest Z face fixed, traction over highest Z face.
    restraint: Literal["lowest_z_fixed"] = "lowest_z_fixed"
    load_region: Literal["highest_z_face"] = "highest_z_face"
    max_deflection_mm: float = Field(gt=0)
    min_safety_factor: float = Field(ge=1)
    mesh_factors: tuple[float, float] = (0.12, 0.08)
    convergence_tolerance: float = Field(default=0.1, gt=0, le=0.25)
    source: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def mesh(self) -> LoadInput:
        if not 0.03 <= self.mesh_factors[1] < self.mesh_factors[0] <= 0.3:
            raise ValueError("Provide coarse/fine mesh factors from 0.03 to 0.3")
        if sum(v * v for v in self.direction) < 1e-12:
            raise ValueError("Load direction cannot be zero")
        return self


class ServicePath(Record):
    id: Name
    body: int = Field(ge=0)
    travel: Vector3
    samples: int = Field(default=15, ge=2, le=50)
    tool_radius_mm: float = Field(default=0, ge=0)
    task: str = Field(min_length=1, max_length=1000)


class PhysicalFeedback(Record):
    id: Name
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    feature_id: Name | None = None
    parameter: Name | None = None
    parameter_gain: float = Field(
        default=1,
        ge=-100,
        le=100,
        description="Parameter change per millimeter of measured error; diameter-to-radius uses 0.5. This mapping is a user assumption.",
    )
    nominal_mm: float
    measured_mm: float
    uncertainty_mm: float = Field(ge=0)
    method: str = Field(min_length=1, max_length=1000)
    process: str = Field(min_length=1, max_length=1000)
    observation: str = Field(min_length=1, max_length=2000)


class DrawingDimension(Record):
    id: Name
    a: GeometryRef
    b: GeometryRef
    label: str = Field(min_length=1, max_length=120)


class EngineeringInputs(Record):
    annotations: list[Annotation] = Field(default_factory=list)
    protections: list[Protection] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    material: MaterialInput | None = None
    process: ProcessInput | None = None
    fits: list[FitInput] = Field(default_factory=list)
    joints: list[JointInput] = Field(default_factory=list)
    assembly_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    loads: list[LoadInput] = Field(default_factory=list)
    service_paths: list[ServicePath] = Field(default_factory=list)
    feedback: list[PhysicalFeedback] = Field(default_factory=list)
    dimensions: list[DrawingDimension] = Field(default_factory=list)
    mechanism_loads: list[MechanismLoad] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> EngineeringInputs:
        for field in (
            "annotations",
            "protections",
            "assumptions",
            "fits",
            "joints",
            "loads",
            "service_paths",
            "feedback",
            "dimensions",
            "mechanism_loads",
        ):
            values = getattr(self, field)
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"Duplicate IDs in {field}")
        if (self.joints or self.service_paths) and not self.assembly_revision:
            raise ValueError("Assembly bodies and paths must identify their source revision")
        return self

    def fingerprint(self) -> str:
        return digest(self.model_dump())
