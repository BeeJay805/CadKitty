"""Explicit component-study setup; calculations never invent engineering acceptance."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from forge_mind.workspace.document import Vector3
from forge_mind.workspace.engineering_models import MaterialInput
from forge_mind.workspace.models import Record
from forge_mind.workspace.product_tools import Selector


class StructuralStudy(Record):
    name: str = Field(min_length=1, max_length=160)
    target: Selector
    material: MaterialInput
    axis: Vector3 = (0, 0, 1)
    fixed_end: Literal["minimum", "maximum"] = "minimum"
    force_n: float = Field(gt=0, le=1e8)
    direction: Vector3
    mesher: Literal["surface", "cad"] = "surface"
    mesh_factors: tuple[float, float] = (0.16, 0.1)
    convergence_tolerance: float = Field(default=0.1, gt=0, le=0.25)
    max_deflection_mm: float | None = Field(default=None, gt=0, le=100000)
    min_safety_factor: float | None = Field(default=None, ge=1, le=100000)
    basis: str = Field(min_length=1, max_length=3000)
    source_claim_ids: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def setup(self) -> StructuralStudy:
        if any(
            not text.strip()
            for text in (self.name, self.basis, self.material.name, self.material.source)
        ):
            raise ValueError(
                "Study name, material identity/source and boundary rationale cannot be blank"
            )
        if any(sum(v * v for v in vector) < 1e-12 for vector in (self.axis, self.direction)):
            raise ValueError("Analysis axis and force direction must be nonzero")
        if not 0.03 <= self.mesh_factors[1] < self.mesh_factors[0] <= 0.3:
            raise ValueError("Provide ordered coarse/fine mesh factors from 0.03 to 0.3")
        if len(set(self.source_claim_ids)) != len(self.source_claim_ids):
            raise ValueError("Source claim IDs must be unique")
        return self


class StructuralRequest(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    input_version: int = Field(ge=0)
    study: StructuralStudy


STRUCTURAL_SCOPE = (
    "One exported connected solid under declared homogeneous, isotropic, small-displacement "
    "linear elasticity. The complete planar cap at the selected axis end is fixed; the opposite "
    "planar cap receives uniform traction with the declared total force. Neighbouring parts are "
    "context only. No inferred assembly load transfer, contact, preload, plasticity, fatigue, "
    "buckling, thermal response or process anisotropy. Mesh comparison is not an error bound."
)
