"""Bounded engineering calculations and independent exported product geometry probes.

Pass means only the named algorithm and supplied limits, never product qualification.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.nominal_open_rectangular_shell import OpenRectangularShellSpec

Number = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-100000, le=100000)]
Positive = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0, le=100000)]
Vector = tuple[Number, Number, Number]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: Literal["mm"]


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: Number
    high: Number

    @model_validator(mode="after")
    def ordered(self) -> Interval:
        if self.low > self.high:
            raise ValueError("Interval low must not exceed high")
        return self


class Fastener(Input):
    tool: Literal["fastener_stack"]
    screw_length: Interval
    grip_layers: list[Interval] = Field(min_length=1, max_length=32)
    required_engagement: Positive
    available_depth: Interval
    minimum_tip_clearance: Number = Field(default=0, ge=0)

    @model_validator(mode="after")
    def positive_stack(self) -> Fastener:
        if any(i.low < 0 for i in [self.screw_length, self.available_depth, *self.grip_layers]):
            raise ValueError("Lengths cannot be negative")
        return self


class Tolerance(Input):
    tool: Literal["tolerance_stack"]
    terms: list[Interval] = Field(min_length=1, max_length=64)
    allowable: Interval


class Cable(Input):
    tool: Literal["cable_bend"]
    centerline_radii: list[Positive] = Field(min_length=1, max_length=64)
    bundle_radius: Positive
    minimum_inside_radius: Positive


class Selector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    occurrence_path: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def exactly_one(self) -> Selector:
        if (self.body_index is None) == (self.occurrence_path is None):
            raise ValueError("Select exactly one body_index or occurrence_path")
        return self


class OpenRectangularShell(Input, OpenRectangularShellSpec):
    tool: Literal["open_rectangular_shell"]
    body: Selector


class Hole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    center: Vector
    diameter: Positive


class Holes(Input):
    tool: Literal["hole_pattern"]
    body: Selector
    axis: Vector
    holes: list[Hole] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def axis_valid(self) -> Holes:
        if math.sqrt(sum(x * x for x in self.axis)) < 1e-8:
            raise ValueError("Axis must be nonzero")
        return self


class Vent(Holes):
    tool: Literal["vent_open_fraction"]  # type: ignore[assignment]
    region_origin: Vector
    region_x_axis: Vector
    region_width: Positive
    region_height: Positive
    minimum_open_fraction: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]


class MountGrid(Input):
    tool: Literal["mount_grid"]
    body: Selector
    origin: Vector
    axis: Vector
    x_axis: Vector
    pitch_x: Positive
    pitch_y: Positive
    columns: Annotated[int, Field(strict=True, ge=1, le=8)]
    rows: Annotated[int, Field(strict=True, ge=1, le=8)]
    diameter: Positive

    @model_validator(mode="after")
    def orthogonal_frame(self) -> MountGrid:
        a = math.sqrt(sum(v * v for v in self.axis))
        x = math.sqrt(sum(v * v for v in self.x_axis))
        if (
            min(a, x) < 1e-8
            or abs(sum(u * v for u, v in zip(self.axis, self.x_axis, strict=True))) / (a * x) > 1e-7
        ):
            raise ValueError("Mount grid axes must be nonzero and perpendicular")
        return self


class Clearance(Input):
    tool: Literal["clearance_volume"]
    reservation: Selector
    obstacles: list[Selector] = Field(min_length=1, max_length=64)


class FaceSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body: Selector
    face_signature: str = Field(strict=True, pattern=r"^[a-f0-9]{64}$")


class AxialCavity(Input):
    tool: Literal["axial_cavity"]
    body: Selector
    entry_point: Vector
    axis: Vector
    diameter: Positive
    depth_interval: Interval

    @model_validator(mode="after")
    def finite_corridor(self) -> AxialCavity:
        if math.sqrt(sum(value * value for value in self.axis)) < 1e-8:
            raise ValueError("Cavity axis must be nonzero and point inward from the mouth")
        if self.depth_interval.low < 0 or self.depth_interval.high <= self.depth_interval.low:
            raise ValueError("Cavity depth interval requires 0 <= low < high")
        return self


class PlanarFacePair(Input):
    tool: Literal["planar_face_pair"]
    a: FaceSelector
    b: FaceSelector
    datum: Vector
    expected_separation_mm: Number | None = None
    expected_midpoint_mm: Number | None = None
    tolerance_mm: (
        Annotated[float, Field(strict=True, allow_inf_nan=False, ge=1e-5, le=100000)] | None
    ) = None

    @model_validator(mode="after")
    def explicit_precision(self) -> PlanarFacePair:
        if (
            self.expected_separation_mm is not None or self.expected_midpoint_mm is not None
        ) and self.tolerance_mm is None:
            raise ValueError("Supplied targets require explicit tolerance_mm of at least 1e-5 mm")
        return self


class ProjectedPair(Input):
    tool: Literal["projected_pair"]
    a: Selector
    b: Selector
    axis: Vector
    datum: Vector
    expected_separation_mm: Number | None = None
    expected_midpoint_mm: Number | None = None
    tolerance_mm: (
        Annotated[float, Field(strict=True, allow_inf_nan=False, ge=1e-5, le=100000)] | None
    ) = None

    @model_validator(mode="after")
    def measurement_frame(self) -> ProjectedPair:
        if math.sqrt(sum(x * x for x in self.axis)) < 1e-8:
            raise ValueError("Projection axis must be nonzero")
        if (
            self.expected_separation_mm is not None or self.expected_midpoint_mm is not None
        ) and self.tolerance_mm is None:
            raise ValueError("Supplied targets require explicit tolerance_mm of at least 1e-5 mm")
        return self


Request = Annotated[
    Fastener
    | Tolerance
    | Cable
    | Holes
    | Vent
    | Clearance
    | MountGrid
    | AxialCavity
    | ProjectedPair
    | PlanarFacePair
    | OpenRectangularShell,
    Field(discriminator="tool"),
]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(Request)
_MODELS: dict[str, type[BaseModel]] = {
    "fastener_stack": Fastener,
    "tolerance_stack": Tolerance,
    "cable_bend": Cable,
    "hole_pattern": Holes,
    "vent_open_fraction": Vent,
    "clearance_volume": Clearance,
    "mount_grid": MountGrid,
    "axial_cavity": AxialCavity,
    "projected_pair": ProjectedPair,
    "planar_face_pair": PlanarFacePair,
    "open_rectangular_shell": OpenRectangularShell,
}
_SCOPES = {
    "open_rectangular_shell": "Exact two-sided material comparison to request-fixed authored open rectangular shell dimensions in saved STEP world XYZ. Candidate bounding-box center registers translation only; no rotation fitting or target adaptation. Reports matches_nominal_plan only; no original-source completeness, absolute placement, physical, manufacturing or acceptance claim. Positive residuals above the existing fixed guard fail; smaller residuals or debris remain unverified.",
    "fastener_stack": "Worst-case axial intervals from supplied lengths. Include washers in grip_layers; available_depth is usable receiver depth (nut thickness for a nut). Threads, preload, strength and tool access unverified.",
    "tolerance_stack": "Worst-case sum of signed intervals; no statistical independence assumed.",
    "cable_bend": "Supplied circular-arc centerline radii minus bundle radius. Exported route, transitions, slack, strain relief and physical cable properties unverified.",
    "hole_pattern": "Exported concave cylindrical wall diameter/axis plus full-body-span cylindrical Boolean clearance. Fixed 1e-5 mm coordinate tolerance and 1e-6 mm3 intersection threshold. No threads or general cutouts.",
    "vent_open_fraction": "Conservative geometric area of disjoint verified round through apertures inside supplied planar rectangle. Additional apertures omitted. No airflow, ligament or panel-strength claim.",
    "mount_grid": "Centered orthogonal rectangular mounting grid derived from supplied pitches, origin and axes, then exported full-depth hole checks. Datum/configuration applicability and source truth need independent confirmation.",
    "axial_cavity": "Specified cylindrical envelope on one saved solid, from a unique supported planar circular mouth along an explicit inward axis. Checks exterior axial approach from beyond the body's support and the continuous corridor from the mouth to depth_interval.high, including any portion before depth_interval.low. Reports actual coaxial wall observations separately. Fixed 1e-5 mm coordinate, 1e-7 angular and 1e-6 mm3 Boolean tolerances. Finite interval only: no terminal-depth discovery, through-hole, exact bore-diameter fit, thread, strength or physical-readiness claim. Unsupported or ambiguous mouths remain unverified.",
    "clearance_volume": "Static exported reservation solid intersection against selected obstacles only. No continuous service path or ergonomic certificate.",
    "projected_pair": "Ordered whole-body BRep support intervals along the normalized supplied axis relative to datum. Separation is B.min - A.max; boundary midpoint is (A.max + B.min)/2. Negative separation means overlapping projections, not collision. Protrusions affect support. No opposing-face, contact, Euclidean distance, transverse alignment or physical-readiness claim. Probe targets do not replace frozen requirements. Minimum comparison tolerance is 1e-5 mm; no target leaves status unverified.",
    "planar_face_pair": "Two uniquely identified analytic trimmed faces on distinct exported bodies. A's outward normal defines the ordered axis; B must oppose A within 1e-7 radians. Measure signed plane separation and midpoint on the axis line through the explicit datum. Dimension endpoints lie on infinite planes, not necessarily inside trimmed faces. Face areas retain holes; no overlap/contact area, minimum Euclidean distance, clearance, continuous motion, strength or physical-readiness claim. Targets never replace frozen requirements. Minimum comparison tolerance is 1e-5 mm; no target leaves status unverified.",
}


def available_product_tools() -> list[dict[str, str]]:
    return [{"tool": name, "scope": scope} for name, scope in _SCOPES.items()]


def product_tool_schema(tool: str | None = None) -> dict[str, Any]:
    if tool is None:
        return _ADAPTER.json_schema()
    if tool not in _MODELS:
        raise ValueError("Unsupported product tool")
    return _MODELS[tool].model_json_schema()


def _receipt(
    tool: str, status: str, results: Any, hashes: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": status,
        "algorithm_version": 2
        if tool in {"hole_pattern", "mount_grid", "vent_open_fraction"}
        else 1,
        "algorithm_scope": _SCOPES.get(tool, "Unsupported tool"),
        "artifact_hashes": hashes or {},
        "results": results,
        "qualification_status": "unverified",
    }


def run_product_tool(request: dict[str, Any], revision_directory: Path) -> dict[str, Any]:
    """Validate input and inspect STEP inside the service's bounded CAD worker.

    Native geometry callers must use processing.process_revision for time/memory bounds.
    """
    req = _ADAPTER.validate_python(request)
    if isinstance(req, Fastener):
        low = req.screw_length.low - sum(i.high for i in req.grip_layers)
        high = req.screw_length.high - sum(i.low for i in req.grip_layers)
        clearance = req.available_depth.low - high
        return _receipt(
            req.tool,
            "pass"
            if low >= req.required_engagement and clearance >= req.minimum_tip_clearance
            else "fail",
            {"engagement_mm": [low, high], "worst_tip_clearance_mm": clearance},
        )
    if isinstance(req, Tolerance):
        low, high = sum(i.low for i in req.terms), sum(i.high for i in req.terms)
        return _receipt(
            req.tool,
            "pass" if low >= req.allowable.low and high <= req.allowable.high else "fail",
            {"sum_mm": [low, high]},
        )
    if isinstance(req, Cable):
        radii = [r - req.bundle_radius for r in req.centerline_radii]
        return _receipt(
            req.tool,
            "pass" if min(radii) >= req.minimum_inside_radius else "fail",
            {"inside_radii_mm": radii, "evidence": "supplied dimensions only"},
        )
    step = revision_directory / "part.step"
    if not step.is_file() or step.is_symlink() or step.stat().st_size > 50_000_000:
        return _receipt(
            req.tool, "unverified", {"reason": "STEP unavailable or exceeds 50 MB bound"}
        )
    hashes = {"part.step": file_sha256(step)}
    try:
        result = _geometry(req, step, hashes["part.step"])
        if file_sha256(step) != hashes["part.step"]:
            raise ValueError("STEP changed during inspection")
        return result
    except Exception as exc:
        return _receipt(
            req.tool,
            "unverified",
            {"reason": f"Geometry inspection unavailable: {type(exc).__name__}"},
            hashes,
        )


def _volume(shape: Any) -> float:
    from forge_mind.workspace.geometry_results import boolean_volume

    return boolean_volume(shape)


def _geometry(req: Any, step: Path, expected_hash: str) -> dict[str, Any]:
    import build123d as b
    import numpy as np

    from forge_mind.validator.geometry import extract_cylinders
    from forge_mind.workspace.assembly import named_solids, occurrence_label

    if isinstance(req, MountGrid):
        axis = np.array(req.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x_axis = np.array(req.x_axis, dtype=float)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(axis, x_axis)
        centers = [
            np.array(req.origin)
            + x_axis * ((col - (req.columns - 1) / 2) * req.pitch_x)
            + y_axis * ((row - (req.rows - 1) / 2) * req.pitch_y)
            for row in range(req.rows)
            for col in range(req.columns)
        ]
        holes = Holes(
            tool="hole_pattern",
            units="mm",
            body=req.body,
            axis=req.axis,
            holes=[Hole(center=tuple(c), diameter=req.diameter) for c in centers],
        )
        receipt = _geometry(holes, step, expected_hash)
        receipt["tool"] = req.tool
        receipt["algorithm_scope"] = _SCOPES[req.tool]
        receipt["results"]["derived_grid"] = {
            "pitch_x_mm": req.pitch_x,
            "pitch_y_mm": req.pitch_y,
            "columns": req.columns,
            "rows": req.rows,
        }
        return receipt
    hashes = {"part.step": file_sha256(step)}
    if hashes["part.step"] != expected_hash:
        raise ValueError("Artifact hash changed")
    shape = b.import_step(step)
    solids = list(shape.solids())
    if not shape.is_valid:
        raise ValueError("Export geometry is invalid")

    def select(s: Selector) -> Any:
        if s.body_index is not None:
            if s.body_index >= len(solids):
                raise ValueError("Selected body index does not exist in this export")
            return solids[s.body_index]
        return named_solids(shape)[occurrence_label(str(s.occurrence_path))]

    if isinstance(req, OpenRectangularShell):
        from forge_mind.workspace.nominal_open_rectangular_shell import (
            inspect_open_rectangular_shell,
        )
        from forge_mind.workspace.topology import same_exported_solid

        body = select(req.body)
        indices = [i for i, solid in enumerate(solids) if same_exported_solid(body, solid)]
        if len(indices) != 1:
            return _receipt(
                req.tool,
                "unverified",
                {"reason": "Shell selector does not resolve to one exported body"},
                hashes,
            )
        observed = inspect_open_rectangular_shell(body, req)
        observed.update(body_index=indices[0], selector=req.body.model_dump(exclude_none=True))
        return _receipt(req.tool, observed["status"], observed, hashes)

    if isinstance(req, AxialCavity):
        from forge_mind.workspace.axial_cavity import inspect_cavity
        from forge_mind.workspace.topology import same_exported_solid

        body = select(req.body)
        indices = [i for i, solid in enumerate(solids) if same_exported_solid(body, solid)]
        if len(indices) != 1:
            return _receipt(
                req.tool,
                "unverified",
                {"reason": "Cavity selector does not resolve to one exported body"},
                hashes,
            )
        status, cavity_results = inspect_cavity(req, body)
        cavity_results.update(
            body_index=indices[0], selector=req.body.model_dump(exclude_none=True)
        )
        return _receipt(req.tool, status, cavity_results, hashes)

    if isinstance(req, PlanarFacePair):
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane

        from forge_mind.workspace.topology import same_exported_solid, signature

        faces = []
        for face_selector in (req.a, req.b):
            body = select(face_selector.body)
            indices = [i for i, solid in enumerate(solids) if same_exported_solid(body, solid)]
            matches = [
                face for face in body.faces() if signature(face) == face_selector.face_signature
            ]
            if len(indices) != 1 or len(matches) != 1:
                return _receipt(
                    req.tool,
                    "unverified",
                    {"reason": "Face or body identity is missing or ambiguous"},
                    hashes,
                )
            face = matches[0]
            surface = BRepAdaptor_Surface(face.wrapped, True)
            if surface.GetType() != GeomAbs_Plane:
                return _receipt(
                    req.tool,
                    "unverified",
                    {"reason": "Selected face is not an analytic plane"},
                    hashes,
                )
            normal = np.asarray(tuple(face.normal_at()), dtype=float)
            if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
                raise ValueError("Planar face has no finite normal")
            normal /= np.linalg.norm(normal)
            origin = surface.Plane().Location()
            faces.append(
                {
                    "selector": face_selector.model_dump(exclude_none=True),
                    "body_index": indices[0],
                    "body_signature": signature(body),
                    "face_signature": signature(face),
                    "center_mm": list(face.center()),
                    "normal": normal.tolist(),
                    "plane_origin_mm": [origin.X(), origin.Y(), origin.Z()],
                    "area_mm2": float(face.area),
                }
            )
            if (
                not all(
                    math.isfinite(value)
                    for value in [
                        *faces[-1]["center_mm"],
                        *faces[-1]["plane_origin_mm"],
                        faces[-1]["area_mm2"],
                    ]
                )
                or faces[-1]["area_mm2"] <= 0
            ):
                raise ValueError("Planar face geometry is non-finite or empty")
        if faces[0]["body_index"] == faces[1]["body_index"]:
            return _receipt(
                req.tool,
                "unverified",
                {"reason": "Faces must belong to distinct exported bodies"},
                hashes,
            )
        axis = np.asarray(faces[0]["normal"])
        other_normal = np.asarray(faces[1]["normal"])
        angular_error = math.atan2(
            float(np.linalg.norm(np.cross(axis, other_normal))), float(-np.dot(axis, other_normal))
        )
        if angular_error > 1e-7:
            return _receipt(
                req.tool,
                "unverified",
                {"reason": "Face normals are not opposed and parallel within 1e-7 radians"},
                hashes,
            )
        datum = np.asarray(req.datum)
        coordinates = []
        for face_record in faces:
            normal = np.asarray(face_record["normal"])
            coordinate = float(
                np.dot(normal, np.asarray(face_record["plane_origin_mm"]) - datum)
                / np.dot(normal, axis)
            )
            coordinates.append(coordinate)
            face_record["plane_coordinate_mm"] = coordinate
            face_record["plane_dimension_point_mm"] = (datum + axis * coordinate).tolist()
        separation = coordinates[1] - coordinates[0]
        midpoint = (coordinates[0] + coordinates[1]) / 2
        if not all(math.isfinite(v) for v in [*coordinates, separation, midpoint, angular_error]):
            raise ValueError("Plane measurement is non-finite")
        comparisons = {}
        for name, measured, target in (
            ("ordered_separation_mm", separation, req.expected_separation_mm),
            ("boundary_midpoint_mm", midpoint, req.expected_midpoint_mm),
        ):
            if target is not None:
                assert req.tolerance_mm is not None
                comparisons[name] = {
                    "measured_mm": measured,
                    "expected_mm": target,
                    "tolerance_mm": req.tolerance_mm,
                    "status": "pass" if abs(measured - target) <= req.tolerance_mm else "fail",
                }
        return _receipt(
            req.tool,
            ("pass" if all(row["status"] == "pass" for row in comparisons.values()) else "fail")
            if comparisons
            else "unverified",
            {
                "measurement_status": "measured",
                "a": faces[0],
                "b": faces[1],
                "axis": axis.tolist(),
                "datum": list(req.datum),
                "units": "mm",
                "ordered_separation_mm": separation,
                "boundary_midpoint_mm": midpoint,
                "normal_opposition_error_rad": angular_error,
                "parallel_angle_tolerance_rad": 1e-7,
                "minimum_comparison_tolerance_mm": 1e-5,
                "comparisons": comparisons,
            },
            hashes,
        )

    if isinstance(req, ProjectedPair):
        from forge_mind.workspace.topology import same_exported_solid

        axis = np.array(req.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        frame = b.Plane(origin=req.datum, z_dir=tuple(axis))
        pair: list[dict[str, Any]] = []
        for selector in (req.a, req.b):
            selected = select(selector)
            indices = [i for i, solid in enumerate(solids) if same_exported_solid(selected, solid)]
            if len(indices) != 1:
                raise ValueError("Projection selector does not resolve to one exported body")
            # Transform analytic BRep first, never the corners of its world-axis bounding box.
            local = frame.to_local_coords(selected)
            bounds = local.bounding_box(optimal=True)
            interval = [float(bounds.min.Z), float(bounds.max.Z)]
            if not all(math.isfinite(value) for value in interval) or interval[0] > interval[1]:
                raise ValueError("Projection interval is invalid")
            pair.append(
                {
                    "selector": selector.model_dump(exclude_none=True),
                    "body_index": indices[0],
                    "interval_mm": interval,
                }
            )
        if pair[0]["body_index"] == pair[1]["body_index"]:
            return _receipt(
                req.tool,
                "unverified",
                {
                    "reason": "Projection requires two distinct exported bodies; selectors are aliases"
                },
                hashes,
            )
        end_a, start_b = pair[0]["interval_mm"][1], pair[1]["interval_mm"][0]
        separation, midpoint = start_b - end_a, (end_a + start_b) / 2
        comparisons = {}
        for name, value, expected in (
            ("ordered_separation_mm", separation, req.expected_separation_mm),
            ("boundary_midpoint_mm", midpoint, req.expected_midpoint_mm),
        ):
            if expected is not None:
                assert req.tolerance_mm is not None
                comparisons[name] = {
                    "expected_mm": expected,
                    "measured_mm": value,
                    "tolerance_mm": req.tolerance_mm,
                    "status": "pass" if abs(value - expected) <= req.tolerance_mm else "fail",
                }
        return _receipt(
            req.tool,
            ("pass" if all(c["status"] == "pass" for c in comparisons.values()) else "fail")
            if comparisons
            else "unverified",
            {
                "measurement_status": "measured",
                "a": pair[0],
                "b": pair[1],
                "axis": axis.tolist(),
                "datum": list(req.datum),
                "units": "mm",
                "ordered_separation_mm": separation,
                "boundary_midpoint_mm": midpoint,
                "minimum_comparison_tolerance_mm": 1e-5,
                "comparisons": comparisons,
            },
            hashes,
        )

    if isinstance(req, Clearance):
        reservation = select(req.reservation)
        rows: list[dict[str, Any]] = [
            {
                "obstacle": s.model_dump(exclude_none=True),
                "overlap_mm3": _volume(reservation.intersect(select(s))),
            }
            for s in req.obstacles
        ]
        return _receipt(
            req.tool,
            "pass" if all(r["overlap_mm3"] <= 1e-6 for r in rows) else "fail",
            rows,
            hashes,
        )
    body = select(req.body)
    axis = np.array(req.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    box = body.bounding_box()
    corners = [
        np.array((x, y, z))
        for x in (box.min.X, box.max.X)
        for y in (box.min.Y, box.max.Y)
        for z in (box.min.Z, box.max.Z)
    ]
    lower, upper = min(float(p @ axis) for p in corners), max(float(p @ axis) for p in corners)
    cylinders = extract_cylinders(body)
    rows = []
    for hole in req.holes:
        center = np.array(hole.center)
        # Expose actual analytic geometry separately from the requested target.
        # Coaxial counterbores and split faces remain separate observations;
        # selecting a diameter must never silently discard an ambiguous wall.
        walls = []
        for cylinder in cylinders:
            offset = float(np.linalg.norm(np.cross(center - np.array(cylinder.axis_point), axis)))
            if (
                cylinder.concave is True
                and np.linalg.norm(np.cross(axis, cylinder.axis_dir)) <= 1e-7
                and offset <= 1e-5
            ):
                diameter = 2 * cylinder.radius
                walls.append(
                    {
                        "diameter_mm": diameter,
                        "axis_point_mm": list(cylinder.axis_point),
                        "axis_direction": list(cylinder.axis_dir),
                        "axis_offset_mm": offset,
                        "diameter_error_mm": diameter - hole.diameter,
                        "matches_requested_diameter": abs(diameter - hole.diameter) <= 1e-5,
                    }
                )
        matched = any(wall["matches_requested_diameter"] for wall in walls)
        origin = center + axis * (lower - float(center @ axis) - 1)
        probe = b.Solid.make_cylinder(
            hole.diameter / 2, upper - lower + 2, b.Plane(origin=tuple(origin), z_dir=tuple(axis))
        )
        overlap = _volume(body.intersect(probe))
        rows.append(
            {
                "center_mm": hole.center,
                "diameter_mm": hole.diameter,
                "cylindrical_walls": walls,
                "wall_matched": bool(matched),
                "obstruction_mm3": overlap,
                "status": "pass" if matched and overlap <= 1e-6 else "fail",
            }
        )
    passed = all(r["status"] == "pass" for r in rows)
    results: dict[str, Any] = {
        "holes": rows,
        "body_axis_span_mm": [lower, upper],
        "measurement_scope": "center_mm and diameter_mm are requested targets. cylindrical_walls are independently extracted concave analytic faces whose axes match the requested line within 1e-5 mm and 1e-7 cross-product tolerance. Axis points are arbitrary points on the measured infinite axes, not hole endpoints. Split faces and counterbore diameters are retained separately; absent observations do not supply a measured value. No manufacturing or physical measurements.",
    }
    if isinstance(req, Vent):
        x = np.array(req.region_x_axis, dtype=float)
        if np.linalg.norm(x) < 1e-8 or abs(float(x @ axis)) / np.linalg.norm(x) > 1e-7:
            raise ValueError("Region x axis must be perpendicular to aperture axis")
        x /= np.linalg.norm(x)
        y = np.cross(axis, x)
        for i, h in enumerate(req.holes):
            delta = np.array(h.center) - np.array(req.region_origin)
            u, v, radius = float(delta @ x), float(delta @ y), h.diameter / 2
            if not (
                radius <= u <= req.region_width - radius
                and radius <= v <= req.region_height - radius
            ):
                raise ValueError("Aperture is outside region")
            for other in req.holes[:i]:
                if (
                    np.linalg.norm(np.cross(np.array(h.center) - np.array(other.center), axis))
                    < (h.diameter + other.diameter) / 2
                ):
                    raise ValueError("Aperture inventory overlaps")
        fraction = sum(
            math.pi * (h.diameter / 2) ** 2
            for h, r in zip(req.holes, rows, strict=True)
            if r["status"] == "pass"
        ) / (req.region_width * req.region_height)
        results["verified_open_fraction_lower_bound"] = fraction
        passed = passed and fraction >= req.minimum_open_fraction
    return _receipt(req.tool, "pass" if passed else "fail", results, hashes)
