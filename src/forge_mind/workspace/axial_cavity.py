"""Finite straight cavity envelopes from saved BRep, never thread or fit proof.

Only a complete circular inner wire on an analytic planar face is a supported
mouth. A straight exterior approach and the entire inward corridor are checked.
Native callers run this inside the existing bounded product-tool worker.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.workspace.product_tools import AxialCavity

COORDINATE_TOLERANCE_MM = 1e-5
ANGULAR_TOLERANCE_RAD = 1e-7
OBSTRUCTION_TOLERANCE_MM3 = 1e-6


def inspect_cavity(req: AxialCavity, body: Any) -> tuple[str, dict[str, Any]]:
    import build123d as b
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    from forge_mind.validator.geometry import extract_cylinders
    from forge_mind.workspace.geometry_results import boolean_overlap_components, boolean_volume
    from forge_mind.workspace.topology import signature

    entry = np.asarray(req.entry_point, dtype=float)
    axis = np.asarray(req.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    radius = req.diameter / 2
    results: dict[str, Any] = {
        "entry_point_mm": entry.tolist(),
        "axis": axis.tolist(),
        "requested_envelope_diameter_mm": req.diameter,
        "requested_depth_interval_mm": [req.depth_interval.low, req.depth_interval.high],
        "continuous_corridor_interval_mm": [0.0, req.depth_interval.high],
        "coordinate_tolerance_mm": COORDINATE_TOLERANCE_MM,
        "angular_tolerance_rad": ANGULAR_TOLERANCE_RAD,
        "obstruction_tolerance_mm3": OBSTRUCTION_TOLERANCE_MM3,
        "body_signature": signature(body),
        "entry_status": "unverified",
        "terminal_depth_status": "not_measured",
        "through_status": "unverified",
        "measurement_scope": (
            "Finite nominal cylindrical-envelope accessibility on this selected solid only. "
            "The explicit depth interval is a requested target, not a measured cavity endpoint. "
            "The approach and continuous corridor include all material before the requested end. "
            "Other bodies, exact bore-diameter fit, threads and physical properties are unverified."
        ),
    }
    mouths: list[dict[str, Any]] = []
    for face in body.faces():
        surface = BRepAdaptor_Surface(face.wrapped, True)
        if surface.GetType() != GeomAbs_Plane:
            continue
        normal = np.asarray(tuple(face.normal_at()), dtype=float)
        normal /= np.linalg.norm(normal)
        origin = surface.Plane().Location()
        point = np.asarray([origin.X(), origin.Y(), origin.Z()])
        angle = math.atan2(float(np.linalg.norm(np.cross(normal, axis))), float(-normal @ axis))
        if not math.isfinite(angle) or angle > ANGULAR_TOLERANCE_RAD:
            continue
        if abs(float((entry - point) @ normal)) > COORDINATE_TOLERANCE_MM:
            continue
        for wire in face.inner_wires():
            edges = list(wire.edges())
            # Split or noncircular loops are unsupported, never guessed.
            if not wire.is_closed or len(edges) != 1 or edges[0].geom_type != b.GeomType.CIRCLE:
                continue
            edge = edges[0]
            circle = BRepAdaptor_Curve(edge.wrapped).Circle()
            center = circle.Location()
            center_xyz = np.asarray([center.X(), center.Y(), center.Z()])
            measured_radius = float(circle.Radius())
            if not math.isfinite(measured_radius) or measured_radius <= 0:
                raise ValueError("Invalid cavity mouth radius")
            if (
                np.linalg.norm(center_xyz - entry) > COORDINATE_TOLERANCE_MM
                or abs(float(edge.length) - 2 * math.pi * measured_radius) > COORDINATE_TOLERANCE_MM
            ):
                continue
            mouths.append(
                {
                    "face_signature": signature(face),
                    "edge_signature": signature(edge),
                    "center_mm": center_xyz.tolist(),
                    "diameter_mm": 2 * measured_radius,
                    "outward_normal": normal.tolist(),
                    "opposition_error_rad": angle,
                }
            )
    results["mouth_candidates"] = mouths
    if len(mouths) != 1:
        results["reason"] = "No unique supported planar circular mouth at the explicit entry"
        return "unverified", results

    walls: list[dict[str, Any]] = []
    for cylinder in extract_cylinders(body):
        wall_axis = np.asarray(cylinder.axis_dir)
        wall_point = np.asarray(cylinder.axis_point)
        if (
            cylinder.concave is not True
            or np.linalg.norm(np.cross(axis, wall_axis)) > ANGULAR_TOLERANCE_RAD
            or np.linalg.norm(np.cross(entry - wall_point, axis)) > COORDINATE_TOLERANCE_MM
        ):
            continue
        interval: list[float] | None = None
        if cylinder.axial_extent is not None:
            direction = float(axis @ wall_axis)
            offset = float(axis @ wall_point - direction * (wall_axis @ wall_point) - axis @ entry)
            interval = sorted(float(value * direction + offset) for value in cylinder.axial_extent)
            if not all(math.isfinite(value) for value in interval):
                raise ValueError("Invalid cavity wall interval")
        walls.append(
            {
                "diameter_mm": 2 * cylinder.radius,
                "axis_point_mm": list(cylinder.axis_point),
                "axis_direction": list(cylinder.axis_dir),
                "axial_interval_from_entry_mm": interval,
            }
        )
    results["cylindrical_walls"] = walls
    mouth_diameter = mouths[0]["diameter_mm"]
    # Require a measured inward cylindrical wall attached to the opening. A
    # circular rim by itself could surround an unsupported, curved cavity.
    attached = [
        wall
        for wall in walls
        if abs(wall["diameter_mm"] - mouth_diameter) <= COORDINATE_TOLERANCE_MM
        and wall["axial_interval_from_entry_mm"] is not None
        and abs(wall["axial_interval_from_entry_mm"][0]) <= COORDINATE_TOLERANCE_MM
        and wall["axial_interval_from_entry_mm"][1] > COORDINATE_TOLERANCE_MM
    ]
    if not attached or any(wall["axial_interval_from_entry_mm"] is None for wall in walls):
        results["reason"] = "Mouth has no supported attached inward wall or wall extent is unknown"
        return "unverified", results
    # An internal counterbore shoulder is not the outer mouth. Starting there
    # would silently shift the supplied usable-depth datum into an existing bore.
    if any(wall["axial_interval_from_entry_mm"][0] < -COORDINATE_TOLERANCE_MM for wall in walls):
        results["reason"] = "Entry lies inside an existing coaxial cavity rather than at its mouth"
        return "unverified", results

    frame = b.Plane(origin=tuple(entry), z_dir=tuple(axis))
    local = frame.to_local_coords(body)
    bounds = local.bounding_box(optimal=True)
    lower, upper = float(bounds.min.Z), float(bounds.max.Z)
    if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
        raise ValueError("Invalid selected body support interval")
    approach_start = min(-1.0, lower - 1.0)

    def obstruction(start: float, end: float) -> dict[str, Any]:
        probe = b.Solid.make_cylinder(
            radius,
            end - start,
            b.Plane(origin=tuple(entry + axis * start), z_dir=tuple(axis)),
        )
        if not probe.is_valid:
            raise ValueError("Invalid cavity probe")
        intersection = body.intersect(probe)
        volume = boolean_volume(intersection)
        components = boolean_overlap_components(intersection)
        return {"interval_mm": [start, end], "obstruction_mm3": volume, "components": components}

    approach = obstruction(approach_start, 0.0)
    corridor = obstruction(0.0, req.depth_interval.high)
    results.update(
        measurement_status="measured",
        body_axis_support_mm=[lower, upper],
        exterior_approach=approach,
        continuous_corridor=corridor,
    )
    if approach["obstruction_mm3"] > OBSTRUCTION_TOLERANCE_MM3:
        results["reason"] = "Exterior axial approach is obstructed"
        return "fail", results
    results["entry_status"] = "supported_planar_mouth"
    if mouth_diameter + COORDINATE_TOLERANCE_MM < req.diameter:
        results["reason"] = "Requested cylindrical envelope exceeds the measured mouth"
        return "fail", results
    if corridor["obstruction_mm3"] > OBSTRUCTION_TOLERANCE_MM3:
        results["reason"] = "Material obstructs the continuous requested cylindrical corridor"
        return "fail", results
    results["reason"] = (
        "Finite requested envelope and exterior approach are clear within stated tolerances"
    )
    return "pass", results
