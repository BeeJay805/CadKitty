"""Shared geometric types used across DSL, Compiler, Validator.

Frozen Pydantic models so a PartProgram is fully hashable / JSON-serializable
and operations are safely shared between stages without aliasing surprises.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Vec2(BaseModel):
    """2D point or vector. Units: mm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class Vec3(BaseModel):
    """3D point or vector. Units: mm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Rotation(BaseModel):
    """Euler-angle rotation about X, Y, Z axes in that order. Units: degrees.

    Matches build123d's ``Rot(X=..., Y=..., Z=...)`` constructor semantics.
    Default = no rotation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class BBox(BaseModel):
    """Axis-aligned bounding box. Units: mm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: Vec3
    max: Vec3

    @property
    def size(self) -> Vec3:
        return Vec3(
            x=self.max.x - self.min.x,
            y=self.max.y - self.min.y,
            z=self.max.z - self.min.z,
        )

    @property
    def center(self) -> Vec3:
        return Vec3(
            x=(self.min.x + self.max.x) / 2,
            y=(self.min.y + self.max.y) / 2,
            z=(self.min.z + self.max.z) / 2,
        )


# Default-instance helpers for use in Pydantic defaults.
ZERO_VEC3 = Vec3()
ZERO_ROTATION = Rotation()


__all__ = [
    "ZERO_ROTATION",
    "ZERO_VEC3",
    "BBox",
    "Rotation",
    "Vec2",
    "Vec3",
]
