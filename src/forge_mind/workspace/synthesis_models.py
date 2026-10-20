"""Explicit functional targets and bounded opposing-jaw architecture proposals."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")]


class Efficiency(Record):
    low: float = Field(gt=0, le=1)
    high: float = Field(gt=0, le=1)
    basis: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def ordered(self) -> Efficiency:
        if self.low > self.high or not self.basis.strip():
            raise ValueError("Efficiency needs ordered bounds and an explicit basis")
        return self


class GripperTask(Record):
    opening_min_mm: float = Field(ge=2, le=300)
    opening_max_mm: float = Field(gt=2, le=300)
    required_force_per_jaw_n: float | None = Field(default=None, gt=0, le=100000)
    max_close_time_s: float | None = Field(default=None, gt=0, le=100000)
    max_width_mm: float | None = Field(default=None, gt=0, le=1000)

    @model_validator(mode="after")
    def interval(self) -> GripperTask:
        if self.opening_min_mm >= self.opening_max_mm:
            raise ValueError("Opening range must be increasing")
        return self


class DriveAssumption(Record):
    torque_nm: float = Field(gt=0, le=1000)
    rpm: float = Field(gt=0, le=10000)
    basis: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def stated(self) -> DriveAssumption:
        if not self.basis.strip():
            raise ValueError("State the basis for simultaneous torque and speed")
        return self


class CandidateBase(Record):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    efficiency: Efficiency | None = None
    assumption_basis: str = Field(min_length=1, max_length=2000)
    source_claim_ids: list[Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def stated(self) -> CandidateBase:
        if not self.name.strip() or not self.assumption_basis.strip():
            raise ValueError("Candidate name and assumption basis cannot be blank")
        if len(set(self.source_claim_ids)) != len(self.source_claim_ids):
            raise ValueError("Source claim references must be unique")
        return self


class ScrewCandidate(CandidateBase):
    family: Literal["screw"] = "screw"
    lead_mm: float = Field(ge=0.5, le=30)
    shaft_diameter_mm: float = Field(ge=4, le=16)


class RackCandidate(CandidateBase):
    family: Literal["rack"] = "rack"
    module_mm: float = Field(ge=0.5, le=3)
    pinion_teeth: int = Field(ge=17, le=36, strict=True)


class LinkageCandidate(CandidateBase):
    family: Literal["linkage"] = "linkage"
    crank_mm: float = Field(ge=12, le=90)
    rod_mm: float = Field(ge=20, le=180)
    jaw_offset_mm: float = Field(ge=0, le=200)
    angle_min_deg: float = Field(ge=0, le=175)
    angle_max_deg: float = Field(gt=0, le=180)

    @model_validator(mode="after")
    def branch(self) -> LinkageCandidate:
        if self.rod_mm <= self.crank_mm or self.angle_min_deg >= self.angle_max_deg:
            raise ValueError(
                "Use rod > crank and an increasing angle interval within one half-turn"
            )
        return self


Candidate = Annotated[
    ScrewCandidate | RackCandidate | LinkageCandidate, Field(discriminator="family")
]


class SynthesisRequest(Record):
    brief: str = Field(min_length=1, max_length=6000)
    task: GripperTask
    drive: DriveAssumption
    candidates: list[Candidate] = Field(min_length=1, max_length=9)
    samples: int = Field(default=21, ge=3, le=49, strict=True)

    @model_validator(mode="before")
    @classmethod
    def numeric_booleans(cls, value: Any) -> Any:
        def visit(item: Any) -> None:
            if isinstance(item, bool):
                raise ValueError("Boolean values are not engineering quantities")
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)

        visit(value)
        return value

    @model_validator(mode="after")
    def unique(self) -> SynthesisRequest:
        if not self.brief.strip() or len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Provide a brief and unique candidate IDs")
        return self


def example_synthesis_request() -> SynthesisRequest:
    """Explicit illustrative setup, never inferred material or motor ratings."""
    common = {
        "efficiency": {
            "low": 0.65,
            "high": 0.85,
            "basis": "Illustrative assumption; not sourced or measured",
        },
        "assumption_basis": "Equal opposing jaw loads, rigid guidance, constant input torque/speed; transmission and purchased parts need validation",
    }
    return SynthesisRequest.model_validate(
        {
            "brief": "Compare three opposing-jaw mechanisms over 20 to 80 mm opening. Screen 100 N per jaw and a 4 second closing time; these are illustrative targets.",
            "task": {
                "opening_min_mm": 20,
                "opening_max_mm": 80,
                "required_force_per_jaw_n": 100,
                "max_close_time_s": 4,
                "max_width_mm": 300,
            },
            "drive": {
                "torque_nm": 1,
                "rpm": 60,
                "basis": "Illustrative simultaneous operating point; no motor has been selected",
            },
            "candidates": [
                {
                    **common,
                    "id": "opposed_screw",
                    "name": "Opposed lead screw",
                    "family": "screw",
                    "lead_mm": 4,
                    "shaft_diameter_mm": 8,
                },
                {
                    **common,
                    "id": "opposed_rack",
                    "name": "Opposed rack and pinion",
                    "family": "rack",
                    "module_mm": 1.5,
                    "pinion_teeth": 20,
                },
                {
                    **common,
                    "id": "symmetric_linkage",
                    "name": "Geared symmetric slider-cranks",
                    "family": "linkage",
                    "crank_mm": 22,
                    "rod_mm": 70,
                    "jaw_offset_mm": 50,
                    "angle_min_deg": 5,
                    "angle_max_deg": 175,
                },
            ],
        }
    )
