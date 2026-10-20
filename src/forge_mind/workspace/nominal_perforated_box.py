"""Bounded nominal rectangular material-set proof from one saved STEP read.

The outer prism center is not the perforated solid's mass centroid. Hole
allowances never authorize outer stock changes or physical qualification.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.nominal_through_pattern import (
    COORDINATE_EPSILON_MM,
    OBSTRUCTION_EPSILON_MM3,
    Finite,
    _Hole,
    _inspect_shape,
    _Request,
)
from forge_mind.workspace.nominal_through_pattern import (
    PROTOCOL as PATTERN_PROTOCOL,
)

PROTOCOL = "nominal_perforated_box_v1"
Positive = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]


class _BoxRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    protocol: Literal["nominal_perforated_box_v1"]
    criterion_hash: str = Field(strict=True, pattern=r"^[a-f0-9]{64}$")
    subject_id: str = Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    part_id: str = Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    frame: Literal["definition_local"]
    axis: Literal["x", "y", "z"]
    count_policy: Literal["exact"]
    holes: list[_Hole] = Field(min_length=1)
    size_mm: tuple[Positive, Positive, Positive]
    center_mm: tuple[Finite, Finite, Finite]


def _native_cut(left: Any, right: Any) -> Any:
    from forge_mind.workspace.nominal_boolean import checked_boolean

    return checked_boolean(left, right, "cut")


def _residual(shape: Any) -> dict[str, Any]:
    from forge_mind.workspace.nominal_boolean import material_result

    return material_result(shape)


def _box_bounds(shape: Any, req: _BoxRequest) -> dict[str, Any]:
    lower = [center - size / 2 for center, size in zip(req.center_mm, req.size_mm, strict=True)]
    upper = [center + size / 2 for center, size in zip(req.center_mm, req.size_mm, strict=True)]
    bounds = shape.bounding_box(optimal=True)
    actual_lower, actual_upper = list(bounds.min), list(bounds.max)
    errors = [
        abs(actual - target)
        for actual, target in zip([*actual_lower, *actual_upper], [*lower, *upper], strict=True)
    ]
    if not all(math.isfinite(value) for value in [*actual_lower, *actual_upper, *errors]):
        raise ValueError("Nonfinite outer box observation")
    facts: dict[str, Any] = {
        "expected_bbox_min_mm": lower,
        "expected_bbox_max_mm": upper,
        "actual_bbox_min_mm": actual_lower,
        "actual_bbox_max_mm": actual_upper,
        "bound_errors_mm": errors,
        "center_definition": "Center of the outer rectangular prism, not mass centroid",
    }
    if any(error > COORDINATE_EPSILON_MM for error in errors):
        return {
            **facts,
            "status": "fail",
            "reason": "Saved outer stock bounds differ from frozen box",
        }
    return {**facts, "status": "pass", "reason": "Outer stock bounds match frozen box"}


def _inspect_body(shape: Any, req: _BoxRequest, pattern: dict[str, Any]) -> dict[str, Any]:
    import build123d as b

    facts = _box_bounds(shape, req)
    if facts["status"] != "pass":
        return facts
    lower = facts["expected_bbox_min_mm"]
    expected = b.Solid.make_box(*req.size_mm, b.Plane(origin=tuple(lower)))
    depth = "xyz".index(req.axis)
    transverse = [index for index in range(3) if index != depth]
    z_dir = tuple(1 if index == depth else 0 for index in range(3))
    x_dir = (0, 1, 0) if depth == 0 else (1, 0, 0)
    cutters = []
    for observed in pattern["observed_holes"]:
        origin = [0.0, 0.0, 0.0]
        origin[depth] = lower[depth] - 1
        for index, coordinate in zip(transverse, observed["center_uv_mm"], strict=True):
            origin[index] = coordinate
        radius = observed["diameter_interval_mm"][0] / 2
        cutter = b.Solid.make_cylinder(
            radius,
            req.size_mm[depth] + 2,
            b.Plane(origin=tuple(origin), x_dir=x_dir, z_dir=z_dir),
        )
        expected = _native_cut(expected, cutter)
        cutters.append({"origin_mm": origin, "axis": req.axis, "radius_mm": radius})
    if not expected.is_valid or len(expected.solids()) != 1 or expected.volume <= 0:
        raise ValueError("Frozen box minus verified cutters is not one valid positive solid")
    excess = _residual(_native_cut(shape, expected))
    missing = _residual(_native_cut(expected, shape))
    states = [excess["status"], missing["status"]]
    status = "fail" if "fail" in states else "unverified" if "unverified" in states else "pass"
    return {
        **facts,
        "actual_cutters": cutters,
        "excess_material": excess,
        "missing_material": missing,
        "status": status,
        "reason": "Two-sided saved material comparison with frozen box and verified actual bores",
    }


def evaluate_nominal_perforated_box(step_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Prove one frozen box-minus-holes family, never manufacturing accuracy."""
    import build123d as b

    from forge_mind.workspace.processing import regular_file

    req = _BoxRequest.model_validate(request)
    pattern_request = _Request.model_validate(
        {**req.model_dump(exclude={"size_mm", "center_mm"}), "protocol": PATTERN_PROTOCOL}
    )
    result: dict[str, Any] = {
        "status": "unverified",
        "reason": "Saved body geometry has not been evaluated",
        "method": PROTOCOL,
        "criterion_hash": req.criterion_hash,
        "facts": {
            "subject_id": req.subject_id,
            "part_id": req.part_id,
            "frame": req.frame,
            "scope": "One nominal rectangular prism minus exactly verified uniform through bores",
            "qualification_status": "unverified",
            "numerical_limits": {
                "coordinate_epsilon_mm": COORDINATE_EPSILON_MM,
                "residual_uncertainty_mm3": OBSTRUCTION_EPSILON_MM3,
                "policy": (
                    "Pass requires no positive residual in either direction. Positive residual "
                    "above the fixed guard fails; any smaller positive residual or debris is "
                    "unverified. No relative volume cutoff or physical manufacturing allowance."
                ),
            },
        },
        "evidence_refs": [],
    }
    if not regular_file(step_path):
        result["reason"] = "Saved STEP is missing, linked, nonregular or outside resource policy"
        return result
    before = file_sha256(step_path)
    try:
        shape = b.import_step(step_path)
        pattern = _inspect_shape(shape, pattern_request)
        result["facts"]["pattern"] = pattern
        status, reason = pattern["evaluation_status"], pattern["evaluation_reason"]
        bounds = _box_bounds(shape, req)
        result["facts"]["body"] = bounds
        if bounds["status"] == "fail":
            status, reason = bounds["status"], bounds["reason"]
        if status == "pass":
            body = _inspect_body(shape, req, pattern)
            result["facts"]["body"] = body
            status, reason = body["status"], body["reason"]
        if not regular_file(step_path) or file_sha256(step_path) != before:
            raise ValueError("Saved STEP changed during body inspection")
        result.update(status=status, reason=reason)
        result["facts"]["step_sha256"] = before
        result["evidence_refs"] = [{"artifact": step_path.name, "sha256": before}]
    except Exception as exc:
        result["status"] = "unverified"
        result["reason"] = f"Saved body inspection unavailable: {type(exc).__name__}"
    return result
