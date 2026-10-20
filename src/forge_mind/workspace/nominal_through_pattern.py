"""Frozen uniform cylindrical through-pattern proof on one saved definition.

Run native inspection only inside the existing bounded processing worker. The
caller owns subject/role binding and frozen criterion admission. This evaluator
cannot establish threads, manufactured fit, physical accuracy or strength.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_mind.validation_provenance import file_sha256

PROTOCOL = "nominal_through_hole_pattern_v1"
COORDINATE_EPSILON_MM = 1e-5
ANGULAR_EPSILON_RAD = 1e-7
OBSTRUCTION_EPSILON_MM3 = 1e-6
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]


class _Hole(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    center_mm: tuple[Finite, Finite]
    diameter_mm: Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]
    diameter_tolerance_mm: Nonnegative
    position_tolerance_mm: Nonnegative


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    protocol: Literal["nominal_through_hole_pattern_v1"]
    criterion_hash: str = Field(strict=True, pattern=r"^[a-f0-9]{64}$")
    subject_id: str = Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    part_id: str = Field(strict=True, pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    frame: Literal["definition_local"]
    axis: Literal["x", "y", "z"]
    count_policy: Literal["exact"]
    holes: list[_Hole] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_centers(self) -> _Request:
        if len({hole.center_mm for hole in self.holes}) != len(self.holes):
            raise ValueError("Frozen pattern centers must be distinct")
        return self


def _difference_status(error: float, allowance: float) -> str:
    """Zero means computational nominal equality, never a physical allowance."""
    if allowance == 0:
        return "pass" if error <= COORDINATE_EPSILON_MM else "fail"
    if error > allowance + COORDINATE_EPSILON_MM:
        return "fail"
    if error < allowance - COORDINATE_EPSILON_MM:
        return "pass"
    return "unverified"


def _aggregate(states: list[str]) -> str:
    return "fail" if "fail" in states else "unverified" if "unverified" in states else "pass"


def evaluate_nominal_through_pattern(step_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Read the actual STEP; source mutation or kernel failure never supplies proof."""
    from forge_mind.workspace.processing import regular_file

    req = _Request.model_validate(request)
    result: dict[str, Any] = {
        "status": "unverified",
        "reason": "Saved geometry has not been evaluated",
        "method": PROTOCOL,
        "criterion_hash": req.criterion_hash,
        "facts": {
            "scope": (
                "Uniform analytic cylindrical through bores on one saved made definition in "
                "its frozen local frame. No threads, manufactured accuracy, fit or strength proof."
            ),
            "subject_id": req.subject_id,
            "part_id": req.part_id,
            "frame": req.frame,
            "axis": req.axis,
            "numerical_limits": {
                "coordinate_epsilon_mm": COORDINATE_EPSILON_MM,
                "angular_epsilon_rad": ANGULAR_EPSILON_RAD,
                "obstruction_epsilon_mm3": OBSTRUCTION_EPSILON_MM3,
                "comparison_policy": (
                    "Zero allowance uses fixed computational nominal equality epsilon. "
                    "Nonzero source allowance is not widened: safely inside passes, clearly "
                    "outside fails, numerical boundary remains unverified. No manufacturing tolerance."
                ),
            },
            "qualification_status": "unverified",
        },
        "evidence_refs": [],
    }
    if not regular_file(step_path):
        result["reason"] = "Saved STEP is missing, linked, nonregular or outside resource policy"
        return result
    expected = file_sha256(step_path)
    try:
        result["facts"].update(_inspect(step_path, req))
        if not regular_file(step_path) or file_sha256(step_path) != expected:
            raise ValueError("Saved STEP changed during inspection")
        result["facts"]["step_sha256"] = expected
        result["evidence_refs"] = [{"artifact": step_path.name, "sha256": expected}]
        result["status"] = result["facts"].pop("evaluation_status")
        result["reason"] = result["facts"].pop("evaluation_reason")
    except Exception as exc:
        result["status"] = "unverified"
        result["reason"] = f"Saved geometry inspection unavailable: {type(exc).__name__}"
        result["facts"].pop("evaluation_status", None)
        result["facts"].pop("evaluation_reason", None)
    return result


def _inspect(step_path: Path, req: _Request) -> dict[str, Any]:
    import build123d as b

    return _inspect_shape(b.import_step(step_path), req)


def _inspect_shape(shape: Any, req: _Request) -> dict[str, Any]:
    """Shared private observation on the same already-read immutable solid."""
    import build123d as b
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    from forge_mind.validator.geometry import extract_cylinders
    from forge_mind.workspace.nominal_boolean import checked_boolean, material_result
    from forge_mind.workspace.topology import signature

    solids = list(shape.solids())
    if (
        not shape.is_valid
        or len(solids) != 1
        or not math.isfinite(float(solids[0].volume))
        or float(solids[0].volume) <= 0
    ):
        raise ValueError("Pattern binding requires one positive valid solid")
    # Through axes are unoriented. For Y, reverse computational depth to retain
    # increasing transverse components X,Z in a rigid right-handed frame.
    x_dir, z_dir = {
        "x": ((0, 1, 0), (1, 0, 0)),
        "y": ((1, 0, 0), (0, -1, 0)),
        "z": ((1, 0, 0), (0, 0, 1)),
    }[req.axis]
    local = b.Plane(origin=(0, 0, 0), x_dir=x_dir, z_dir=z_dir).to_local_coords(solids[0])
    unsupported: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    walls: list[dict[str, Any]] = []
    for face in local.faces():
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.GetType() == GeomAbs_Plane:
            normal = np.asarray(tuple(face.normal_at()), dtype=float)
            if np.linalg.norm(np.cross(normal, (0, 0, 1))) > ANGULAR_EPSILON_RAD:
                continue
            for wire in face.inner_wires():
                edges = list(wire.edges())
                if not wire.is_closed or len(edges) != 1 or edges[0].geom_type != b.GeomType.CIRCLE:
                    unsupported.append(
                        {"reason": "Unsupported closed aperture loop", "face": signature(face)}
                    )
                    continue
                edge = edges[0]
                circle = BRepAdaptor_Curve(edge.wrapped).Circle()
                radius = float(circle.Radius())
                if abs(float(edge.length) - 2 * math.pi * radius) > COORDINATE_EPSILON_MM:
                    unsupported.append(
                        {"reason": "Incomplete circular aperture", "face": signature(face)}
                    )
                    continue
                center = circle.Location()
                openings.append(
                    {
                        "point": [center.X(), center.Y(), center.Z()],
                        "radius": radius,
                        "normal_z": float(normal[2]),
                    }
                )
            continue
        if face.geom_type != b.GeomType.CYLINDER:
            unsupported.append(
                {"reason": "Unsupported nonplanar noncylindrical surface", "face": signature(face)}
            )
            continue
        cylinders = extract_cylinders(face)
        if len(cylinders) != 1 or cylinders[0].concave is None:
            unsupported.append(
                {"reason": "Unclassified cylindrical surface", "face": signature(face)}
            )
            continue
        cylinder = cylinders[0]
        if cylinder.concave is False:
            continue
        direction = np.asarray(cylinder.axis_dir, dtype=float)
        if np.linalg.norm(np.cross(direction, (0, 0, 1))) > ANGULAR_EPSILON_RAD:
            unsupported.append(
                {"reason": "Additional nonparallel concave cylinder", "face": signature(face)}
            )
            continue
        span_u = abs(surface.LastUParameter() - surface.FirstUParameter())
        span_v = abs(surface.LastVParameter() - surface.FirstVParameter())
        if abs(span_u - 2 * math.pi) > ANGULAR_EPSILON_RAD or abs(
            float(face.area) - 2 * math.pi * cylinder.radius * span_v
        ) > COORDINATE_EPSILON_MM * max(1.0, float(face.area)):
            unsupported.append({"reason": "Partial cylindrical wall", "face": signature(face)})
            continue
        vertices = [tuple(vertex.center()) for vertex in face.vertices()]
        if not vertices:
            unsupported.append({"reason": "Unmeasurable wall endpoints", "face": signature(face)})
            continue
        point = np.asarray(cylinder.axis_point, dtype=float)
        intercept = point - direction * point[2] / direction[2]
        observed = {
            "center_uv_mm": intercept[:2].tolist(),
            "radius_mm": float(cylinder.radius),
            "span_mm": [min(v[2] for v in vertices), max(v[2] for v in vertices)],
            "face_signature": signature(face),
        }
        if not all(
            math.isfinite(v)
            for v in [*observed["center_uv_mm"], observed["radius_mm"], *observed["span_mm"]]
        ):
            raise ValueError("Nonfinite cylindrical observation")
        walls.append(observed)

    groups: list[list[dict[str, Any]]] = []
    for wall in sorted(walls, key=lambda w: (*w["center_uv_mm"], w["radius_mm"], *w["span_mm"])):
        matches = [
            group
            for group in groups
            if all(
                math.dist(wall["center_uv_mm"], other["center_uv_mm"]) <= COORDINATE_EPSILON_MM
                for other in group
            )
        ]
        if len(matches) > 1:
            unsupported.append({"reason": "Ambiguous axis grouping"})
        elif matches:
            matches[0].append(wall)
        else:
            groups.append([wall])
    bounds = local.bounding_box(optimal=True)
    body_span = [float(bounds.min.Z), float(bounds.max.Z)]
    if not all(math.isfinite(v) for v in body_span) or body_span[0] >= body_span[1]:
        raise ValueError("Invalid body axis support")
    holes: list[dict[str, Any]] = []
    for group in groups:
        center = group[0]["center_uv_mm"]
        radii = [wall["radius_mm"] for wall in group]
        spans = sorted(wall["span_mm"] for wall in group)
        merged = list(spans[0])
        separated = False
        for low, high in spans[1:]:
            separated |= low > merged[1] + COORDINATE_EPSILON_MM
            merged[1] = max(merged[1], high)
        reason = None
        if max(radii) - min(radii) > COORDINATE_EPSILON_MM:
            reason = "Counterbored or nonuniform diameter is unsupported"
        elif separated:
            reason = "Separated coaxial spans are unsupported"
        radius = min(radii)
        witnesses = []
        for z, sign in ((merged[0], -1), (merged[1], 1)):
            witnesses.append(
                sum(
                    math.dist(opening["point"], [*center, z]) <= COORDINATE_EPSILON_MM
                    and abs(opening["radius"] - radius) <= COORDINATE_EPSILON_MM
                    and opening["normal_z"] * sign > 1 - ANGULAR_EPSILON_RAD
                    for opening in openings
                )
            )
        probe = b.Solid.make_cylinder(
            radius, body_span[1] - body_span[0] + 2, b.Plane(origin=(*center, body_span[0] - 1))
        )
        intersection = checked_boolean(local, probe, "common")
        material = material_result(intersection)
        overlap = material["volume_mm3"]
        components = material["components"]
        for component in components:
            component["frame"] = "pattern_axis_frame"
        through = (
            "fail"
            if material["status"] == "fail"
            else "unverified"
            if material["status"] != "pass" or reason or witnesses != [1, 1]
            else "pass"
        )
        holes.append(
            {
                "center_uv_mm": center,
                "diameter_interval_mm": [2 * min(radii), 2 * max(radii)],
                "wall_span_mm": merged,
                "mouth_witness_counts": witnesses,
                "obstruction_mm3": overlap,
                "obstruction_components": components,
                "obstruction_classification": material,
                "through_status": through,
                "reason": reason,
                "walls": group,
            }
        )
    facts: dict[str, Any] = {
        "body_signature": signature(solids[0]),
        "body_axis_span_mm": body_span,
        "pattern_axis_frame": {
            "x": {"u": "y", "v": "z", "depth": "x"},
            "y": {"u": "x", "v": "z", "depth": "-y"},
            "z": {"u": "x", "v": "y", "depth": "z"},
        }[req.axis],
        "observed_holes": holes,
        "unsupported_surfaces": unsupported,
        "inventory_scope": "All observed concave cylindrical axes; unsupported apertures prevent complete positive inventory",
    }
    if unsupported:
        return {
            **facts,
            "evaluation_status": "unverified",
            "evaluation_reason": "Complete supported aperture inventory is unavailable",
        }
    if len(holes) != len(req.holes):
        return {
            **facts,
            "evaluation_status": "fail",
            "evaluation_reason": "Observed exact aperture count differs from the frozen pattern",
        }
    position_matches: list[list[int]] = [
        [
            index
            for index, hole in enumerate(holes)
            if math.dist(target.center_mm, hole["center_uv_mm"])
            <= target.position_tolerance_mm + COORDINATE_EPSILON_MM
        ]
        for target in req.holes
    ]
    if any(not indices for indices in position_matches):
        return {
            **facts,
            "evaluation_status": "fail",
            "evaluation_reason": "No observed hole axis matches a required position",
        }
    if any(len(indices) != 1 for indices in position_matches) or len(
        {indices[0] for indices in position_matches}
    ) != len(req.holes):
        return {
            **facts,
            "evaluation_status": "unverified",
            "evaluation_reason": "Position-only hole association is ambiguous",
        }
    comparisons: list[dict[str, Any]] = []
    for target, indices in zip(req.holes, position_matches, strict=True):
        hole = holes[indices[0]]
        position_error = math.dist(target.center_mm, hole["center_uv_mm"])
        position_status = _difference_status(position_error, target.position_tolerance_mm)
        diameter_status = _aggregate(
            [
                _difference_status(abs(value - target.diameter_mm), target.diameter_tolerance_mm)
                for value in hole["diameter_interval_mm"]
            ]
        )
        comparisons.append(
            {
                "observation_index": indices[0],
                "target_center_mm": list(target.center_mm),
                "target_diameter_mm": target.diameter_mm,
                "diameter_allowance_mm": target.diameter_tolerance_mm,
                "position_allowance_mm": target.position_tolerance_mm,
                "position_error_mm": position_error,
                "position_status": position_status,
                "diameter_status": diameter_status,
                "status": _aggregate([position_status, diameter_status, hole["through_status"]]),
            }
        )
    status = _aggregate([comparison["status"] for comparison in comparisons])
    return {
        **facts,
        "comparisons": comparisons,
        "evaluation_status": status,
        "evaluation_reason": "Frozen uniform through-pattern comparison on saved definition geometry",
    }
