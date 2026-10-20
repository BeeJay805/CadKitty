"""Planar rigid-body intent in local coordinates, placed in a principal CAD plane."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]
Scalar = float | Identifier
Point = tuple[Scalar, Scalar]


class Component(Record):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    feature: Identifier = Field(
        description="Simple document: top-level output feature ID. Assembly: unique assembly.roots occurrence ID, not a part-definition ID or part-local feature ID. Every output/root needs exactly one component; all component feature values must be unique."
    )
    group: str = Field(default="Assembly", max_length=80)
    grounded: bool = False
    x: Scalar = 0.0
    y: Scalar = 0.0
    z: Scalar = 0.0
    angle: Scalar = 0.0


class PlanarJoint(Record):
    id: Identifier
    kind: Literal["pin", "fixed", "slider"]
    a: Identifier
    b: Identifier
    pa: Point = (0.0, 0.0)
    pb: Point = (0.0, 0.0)
    angle: Scalar = 0.0
    axis: Scalar = 0.0


class GearCoupling(Record):
    id: Identifier
    a: Identifier
    b: Identifier
    # theta_a + ratio * theta_b = phase, in degrees. Positive is external mesh.
    ratio: Scalar
    phase: Scalar = 0.0


class Driver(Record):
    component: Identifier
    angle: Scalar = 0.0
    start: float = Field(default=0, ge=-720, le=720)
    end: float = Field(default=360, ge=-720, le=720)
    samples: int = Field(default=37, ge=3, le=73)


class Mechanism(Record):
    frame: Literal["planar_xy_mm_deg", "planar_yz_mm_deg", "planar_zx_mm_deg"] = "planar_xy_mm_deg"
    components: list[Component] = Field(min_length=2, max_length=24)
    joints: list[PlanarJoint] = Field(default_factory=list, max_length=48)
    gears: list[GearCoupling] = Field(default_factory=list, max_length=8)
    driver: Driver

    @model_validator(mode="after")
    def references(self) -> Mechanism:
        ids = {c.id for c in self.components}
        if len(ids) != len(self.components) or len({c.feature for c in self.components}) != len(
            ids
        ):
            component_ids = [c.id for c in self.components]
            features = [c.feature for c in self.components]
            duplicate_ids = sorted(
                {value for value in component_ids if component_ids.count(value) > 1}
            )
            duplicate_features = sorted({value for value in features if features.count(value) > 1})
            raise ValueError(
                "Components need unique IDs and unique output features; "
                f"duplicate component IDs: {duplicate_ids}; duplicate feature IDs: {duplicate_features}"
            )
        if not any(c.grounded for c in self.components):
            raise ValueError("Ground at least one component")
        if self.driver.component not in {c.id for c in self.components if not c.grounded}:
            raise ValueError("Driver must refer to a moving component")
        if abs(self.driver.end - self.driver.start) < 1e-8:
            raise ValueError("Motion range must have nonzero travel")
        constraints: list[PlanarJoint | GearCoupling] = [*self.joints, *self.gears]
        if len({c.id for c in constraints}) != len(constraints):
            raise ValueError("Constraint IDs must be unique")
        for joint in constraints:
            if joint.a not in ids or joint.b not in ids or joint.a == joint.b:
                raise ValueError("Constraints need two different existing components")
        return self


class MechanismLoad(Record):
    id: Identifier
    component: Identifier
    torque_n_mm: float = Field(gt=0, le=1e9)
    allowable_shear_mpa: float = Field(gt=0, le=10000)
    safety_factor: float = Field(ge=1, le=100)
    source: str = Field(min_length=1, max_length=1000)


class MechanismTemplate(Record):
    kind: Literal["gearbox", "slider_crank", "four_bar"] = "gearbox"
    module: float = Field(default=1.5, ge=0.5, le=5)
    pinion_teeth: int = Field(default=18, ge=17, le=32)
    wheel_teeth: int = Field(default=36, ge=17, le=36)
    thickness: float = Field(default=8, ge=3, le=20)
    shaft_diameter: float = Field(default=6, ge=3, le=20)
    clearance: float = Field(default=0.2, gt=0, le=1)
    crank: float = Field(default=20, ge=5, le=100)
    rod: float = Field(default=65, ge=10, le=200)
    ground: float = Field(default=70, ge=10, le=200)
    rocker: float = Field(default=55, ge=10, le=200)
