"""Two-sided saved-material comparison to an authored nominal shell specification."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_mind.workspace.nominal_boolean import (
    RESIDUAL_UNCERTAINTY_MM3,
    checked_boolean,
    material_result,
)

PROTOCOL = "nominal_open_rectangular_shell_v1"


Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class OpenRectangularShellSpec(BaseModel):
    """Authored expected dimensions, never an original-source verification criterion."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    size_mm: tuple[Positive, Positive, Positive]
    opening_axis: Literal["x", "y", "z"]
    opening_direction: Literal["positive", "negative"]
    wall_thickness_mm: Positive
    floor_thickness_mm: Positive

    @model_validator(mode="after")
    def nonempty_cavity(self) -> OpenRectangularShellSpec:
        depth = "xyz".index(self.opening_axis)
        if self.floor_thickness_mm >= self.size_mm[depth] or any(
            self.wall_thickness_mm * 2 >= size
            for index, size in enumerate(self.size_mm)
            if index != depth
        ):
            raise ValueError("Shell thicknesses must leave a positive open cavity")
        return self


def shell_boxes(
    criterion: OpenRectangularShellSpec, center_mm: tuple[float, float, float]
) -> dict[str, list[float]]:
    """Use authored dimensions and an explicit translation registration only."""
    outer = [center - size / 2 for center, size in zip(center_mm, criterion.size_mm, strict=True)]
    cavity = [value + criterion.wall_thickness_mm for value in outer]
    size = [value - 2 * criterion.wall_thickness_mm for value in criterion.size_mm]
    depth = "xyz".index(criterion.opening_axis)
    # Extend only beyond the open boundary, avoiding coincident cutter end faces.
    # This numerical construction aid does not change the retained material target.
    overrun = 1.0
    cavity[depth] = (
        outer[depth] + criterion.floor_thickness_mm
        if criterion.opening_direction == "positive"
        else outer[depth] - overrun
    )
    size[depth] = criterion.size_mm[depth] - criterion.floor_thickness_mm + overrun
    values = [*outer, *cavity, *size]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Shell bounds must remain finite")
    boxes = {
        "outer_origin_mm": outer,
        "outer_size_mm": list(criterion.size_mm),
        "cavity_origin_mm": cavity,
        "cavity_size_mm": size,
    }
    check_boundary_resolution(criterion, boxes, None)
    return boxes


def _boundary_intervals(
    req: OpenRectangularShellSpec, boxes: dict[str, list[float]]
) -> list[tuple[int, float, float, float]]:
    intervals = []
    depth = "xyz".index(req.opening_axis)
    for axis, nominal in enumerate(req.size_mm):
        low = boxes["outer_origin_mm"][axis]
        high = low + nominal
        inner_low = boxes["cavity_origin_mm"][axis]
        inner_high = inner_low + boxes["cavity_size_mm"][axis]
        intervals.append((axis, low, high, nominal))
        if axis != depth:
            intervals.extend(
                [
                    (axis, low, inner_low, req.wall_thickness_mm),
                    (axis, inner_low, inner_high, nominal - 2 * req.wall_thickness_mm),
                    (axis, inner_high, high, req.wall_thickness_mm),
                ]
            )
        elif req.opening_direction == "positive":
            intervals.extend(
                [
                    (axis, low, inner_low, req.floor_thickness_mm),
                    (axis, inner_low, high, nominal - req.floor_thickness_mm),
                ]
            )
        else:
            intervals.extend(
                [
                    (axis, low, inner_high, nominal - req.floor_thickness_mm),
                    (axis, inner_high, high, req.floor_thickness_mm),
                ]
            )
    return intervals


def check_boundary_resolution(
    req: OpenRectangularShellSpec,
    boxes: dict[str, list[float]],
    resolution_mm: float | None,
) -> None:
    """Refuse lost fixed offsets; None checks floating-point separation only."""
    if resolution_mm is not None and (not math.isfinite(resolution_mm) or resolution_mm <= 0):
        raise ValueError("Boundary resolution must be positive and finite")
    for _, low, high, nominal in _boundary_intervals(req, boxes):
        if not all(math.isfinite(value) for value in (low, high, nominal)):
            raise ValueError("Shell boundaries must remain finite")
        uncertainty = max(math.ulp(low), math.ulp(high), resolution_mm or 0)
        if high - low <= 2 * uncertainty or nominal <= 2 * uncertainty:
            raise ValueError("Requested shell boundaries are not resolvable")
        if resolution_mm is not None and abs((high - low) - nominal) > resolution_mm:
            raise ValueError("Shell boundary arithmetic changed a fixed nominal dimension")


def _expected_boundaries(
    expected: Any, req: OpenRectangularShellSpec, boxes: dict[str, list[float]]
) -> float:
    """Require the independently constructed target to retain every requested plane."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.Precision import Precision

    resolution = float(Precision.Confusion_s())
    planes = []
    for entity in [*expected.faces(), *expected.edges(), *expected.vertices()]:
        tolerance = float(BRep_Tool.Tolerance_s(entity.wrapped))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Expected boundary has invalid kernel resolution")
        resolution = max(resolution, tolerance)
    check_boundary_resolution(req, boxes, resolution)
    for face in expected.faces():
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.GetType() != GeomAbs_Plane or not math.isfinite(face.area) or face.area <= 0:
            raise ValueError("Expected shell lost a positive planar boundary")
        plane = surface.Plane()
        normal = tuple(plane.Axis().Direction().Coord())
        point = tuple(plane.Location().Coord())
        axis = max(range(3), key=lambda index: abs(normal[index]))
        if abs(normal[axis]) < 1 - 1e-12 or not math.isfinite(point[axis]):
            raise ValueError("Expected shell boundary is not an axis-aligned plane")
        planes.append((axis, point[axis]))
    for axis, low, high, _ in _boundary_intervals(req, boxes):
        for boundary in (low, high):
            if not any(
                plane_axis == axis and abs(coordinate - boundary) <= resolution
                for plane_axis, coordinate in planes
            ):
                raise ValueError("Expected shell lost a requested boundary plane")
    return resolution


def inspect_open_rectangular_shell(shape: Any, req: OpenRectangularShellSpec) -> dict[str, Any]:
    import build123d as b
    from OCP.Precision import Precision

    observed: dict[str, Any] = {
        "protocol": PROTOCOL,
        "frame": "cad_z_up_mm_world",
        "result_scope": "matches_nominal_plan",
        "scope": "Request-fixed authored geometry only; no original-source completeness, absolute placement, manufacturing, physical proof or acceptance",
        "expected": req.model_dump(mode="json", include=set(OpenRectangularShellSpec.model_fields)),
        "residual_uncertainty_mm3": RESIDUAL_UNCERTAINTY_MM3,
    }
    if (
        not shape.is_valid
        or len(shape.solids()) != 1
        or not math.isfinite(shape.volume)
        or shape.volume <= 0
    ):
        return {
            **observed,
            "status": "unverified",
            "reason": "Saved shell must be one valid positive finite solid",
        }
    bounds = shape.bounding_box(optimal=True)
    lower, upper = list(bounds.min), list(bounds.max)
    if not all(math.isfinite(value) for value in [*lower, *upper]):
        raise ValueError("Saved shell bounds must be finite")
    center = (
        (lower[0] + upper[0]) / 2,
        (lower[1] + upper[1]) / 2,
        (lower[2] + upper[2]) / 2,
    )
    boxes = shell_boxes(req, center)
    check_boundary_resolution(req, boxes, float(Precision.Confusion_s()))
    outer_size, cavity_size = boxes["outer_size_mm"], boxes["cavity_size_mm"]
    outer = b.Solid.make_box(
        outer_size[0],
        outer_size[1],
        outer_size[2],
        b.Plane(origin=tuple(boxes["outer_origin_mm"])),
    )
    cavity = b.Solid.make_box(
        cavity_size[0],
        cavity_size[1],
        cavity_size[2],
        b.Plane(origin=tuple(boxes["cavity_origin_mm"])),
    )
    expected = checked_boolean(outer, cavity, "cut")
    if (
        not expected.is_valid
        or len(expected.solids()) != 1
        or not math.isfinite(expected.volume)
        or expected.volume <= 0
    ):
        raise ValueError("Frozen shell did not construct one valid positive finite solid")
    boundary_resolution = _expected_boundaries(expected, req, boxes)
    excess = material_result(checked_boolean(shape, expected, "cut"))
    missing = material_result(checked_boolean(expected, shape, "cut"))
    states = (excess["status"], missing["status"])
    return {
        **observed,
        "status": "fail"
        if "fail" in states
        else "unverified"
        if "unverified" in states
        else "pass",
        "reason": "Two-sided saved material comparison with the authored nominal shell",
        "registration": {
            "method": "bbox_translation_only",
            "center_mm": list(center),
            "rotation_fitted": False,
        },
        "observed_bbox": {"min_mm": lower, "max_mm": upper},
        "expected_boxes": boxes,
        "expected_boundary_resolution_mm": boundary_resolution,
        "excess_material": excess,
        "missing_material": missing,
    }
