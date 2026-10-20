"""Bounded photo observations, never owner facts or manufacturing acceptance."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 2048 * 2048
MAX_COMPONENTS = 1024
MAX_CONTOUR_POINTS = 100_000
MAX_PROPOSAL_POINTS = 4096
CALIBRATION_RESIDUAL_PX = 1.0
OBSERVER_ERROR_MM = 0.01
SCOPE = "Derived planar image observation and design proposal; not owner dimensions, physical fit, or acceptance."
Point = tuple[float, float]


class ProfileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProfileAnchor(ProfileRecord):
    pixel: Point
    plane_mm: Point

    @model_validator(mode="after")
    def bounded_coordinates(self) -> ProfileAnchor:
        if any(abs(v) > 1e6 for v in (*self.pixel, *self.plane_mm)):
            raise ValueError("Calibration coordinates exceed the finite profile bound")
        return self


class ProfileAssumptions(ProfileRecord):
    planar: bool = False
    distortion: Literal["uncorrected", "negligible_declared"] = "uncorrected"
    source_origin: str = Field(min_length=1, max_length=500)


class PhotoProfileRequest(ProfileRecord):
    vision_image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_anchors: list[ProfileAnchor] = Field(default_factory=list, max_length=32)
    check_points: list[ProfileAnchor] = Field(default_factory=list, max_length=32)
    foreground: Literal["dark", "light"]
    threshold: Annotated[int, Field(strict=True, ge=0, le=255)]
    region: tuple[int, int, int, int] | None = None
    seed_pixel: Point | None = None
    fit_tolerance_mm: float = Field(gt=0, le=10)
    assumptions: ProfileAssumptions
    thickness_mm: float | None = Field(default=None, gt=0, le=10000)
    through_cuts: bool | None = None
    allow_circle_fits: bool = False
    outer_rectangle_mm: tuple[float, float] | None = None

    @model_validator(mode="after")
    def separate_checks(self) -> PhotoProfileRequest:
        if self.outer_rectangle_mm is not None and any(
            not 0 < value <= 10000 for value in self.outer_rectangle_mm
        ):
            raise ValueError("Proposed rectangle dimensions must be positive and bounded")
        for rows in (self.calibration_anchors, self.check_points):
            if len({row.pixel for row in rows}) != len(rows):
                raise ValueError("Calibration points must have distinct pixel positions")
        if {a.pixel for a in self.calibration_anchors} & {a.pixel for a in self.check_points}:
            raise ValueError("Check points must be separate from calibration anchors")
        if self.region is not None:
            x0, y0, x1, y1 = self.region
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError("Region must have ordered nonnegative pixel bounds")
        return self


class PhotoProfileSelector(ProfileRecord):
    step_artifact: Literal["part.step"] = "part.step"
    body_index: int = Field(ge=0, le=4095, strict=True)
    z_mm: float
