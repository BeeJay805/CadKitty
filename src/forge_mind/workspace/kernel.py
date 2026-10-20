"""Trusted bounded-worker compiler for the shared feature document."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import build123d as b

from forge_mind.workspace.capacity import enforce_limit
from forge_mind.workspace.document import DesignDocument, Sketch
from forge_mind.workspace.sketch import solve_sketch
from forge_mind.workspace.topology import (
    FEATURE_EDGE_LIMIT,
    FEATURE_FACE_LIMIT,
    PATTERN_COUNT_LIMIT,
    PATTERN_INSTANCE_LIMIT,
    selected_shapes,
    signature,
)


def _normalize_boolean_result(shape: Any) -> Any:
    """Keep multi-island results as geometry, never ShapeList's list arithmetic."""
    if shape is None or (isinstance(shape, b.ShapeList) and not shape):
        raise ValueError("Operation did not produce valid positive-volume geometry")
    return b.Compound(children=shape) if isinstance(shape, b.ShapeList) else shape


def compile_document(
    document: DesignDocument,
    assets: dict[str, Path],
    directory: Path | None = None,
    cache_source: Path | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if document.assembly:
        from forge_mind.workspace.assembly import compile_assembly

        return compile_assembly(document, assets, directory, cache_source)
    shapes: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    instance_counts: dict[str, int] = {}
    values = document.values()
    for feature in document.features:
        if feature.suppressed:
            history.append({"id": feature.id, "state": "suppressed"})
            continue
        try:
            sources = [shapes[i] for i in feature.inputs]
            p = {
                **feature.params,
                **{k: values[v] for k, v in feature.bindings.items()},
            }

            def number(
                key: str,
                default: float | None = None,
                positive: bool = False,
                _params: dict[str, float] = p,
            ) -> float:
                value = _params.get(key, default)
                if value is None or (positive and value <= 0):
                    raise ValueError(
                        f"{key} requires {'a positive' if positive else 'an explicit'} value"
                    )
                return float(value)

            def source(_sources: list[Any] = sources) -> Any:
                if len(_sources) != 1:
                    raise ValueError("Operation requires one input feature")
                return _sources[0]

            kind = feature.kind
            shape: Any
            solver = None
            pattern_estimate = None
            instance_count = max(1, sum(instance_counts[i] for i in feature.inputs))
            if kind == "box":
                shape = b.Box(
                    number("x", positive=True),
                    number("y", positive=True),
                    number("z", positive=True),
                )
            elif kind == "cylinder":
                shape = b.Cylinder(number("radius", positive=True), number("height", positive=True))
            elif kind == "cone":
                bottom, top = number("bottom_radius"), number("top_radius")
                if bottom < 0 or top < 0 or max(bottom, top) <= 0:
                    raise ValueError("Cone radii must be nonnegative, with at least one positive")
                height = number("height", positive=True)
                shape = b.Cylinder(bottom, height) if bottom == top else b.Cone(bottom, top, height)
            elif kind == "torus":
                major, minor = (
                    number("major_radius", positive=True),
                    number("minor_radius", positive=True),
                )
                if minor >= major:
                    raise ValueError("Torus requires minor_radius < major_radius")
                shape = b.Torus(major, minor)
            elif kind == "sphere":
                shape = b.Sphere(number("radius", positive=True))
            elif kind == "profile":
                shape = (
                    b.Pos(number("x", 0), number("y", 0), number("z", 0))
                    * b.Rot(number("rx", 0), number("ry", 0), number("rz", 0))
                    * b.Ellipse(
                        number("radius_x", positive=True), number("radius_y", positive=True)
                    )
                )
            elif kind == "rounded_profile":
                width = number("width", positive=True)
                height = number("height", positive=True)
                radius = number("radius", positive=True)
                if radius >= min(width, height) / 2:
                    raise ValueError("Rounded profile requires radius < min(width, height) / 2")
                shape = (
                    b.Pos(number("x", 0), number("y", 0), number("z", 0))
                    * b.Rot(number("rx", 0), number("ry", 0), number("rz", 0))
                    * b.RectangleRounded(width, height, radius)
                )
            elif kind == "curve":
                curve_points = [
                    tuple(values[v] if isinstance(v, str) else v for v in point)
                    for point in feature.points
                ]
                if len(curve_points) < 2:
                    raise ValueError("Curve requires at least two distinct 3D points")
                shape = b.Wire([b.Spline(*curve_points)])
            elif kind == "loft":
                if len(sources) < 2 or any(len(s.faces()) != 1 for s in sources):
                    raise ValueError("Loft requires two or more single-face profiles in order")
                shape = b.loft([s.faces()[0] for s in sources], ruled=number("ruled", 0) == 1)
            elif kind == "sweep":
                if (
                    len(sources) != 2
                    or len(sources[0].faces()) != 1
                    or len(sources[1].wires()) != 1
                ):
                    raise ValueError("Sweep requires one profile and one continuous path")
                shape = b.sweep(
                    sources[0].faces()[0],
                    path=sources[1].wires()[0],
                    is_frenet=number("frenet", 0) == 1,
                )
            elif kind == "python":
                if not feature.code.strip():
                    raise ValueError("Python feature requires retained source with result assigned")
                from forge_mind.workspace.program_feature import compile_program_feature

                shape = compile_program_feature(feature.code, p, sources)
            elif kind == "sketch":
                if feature.sketch is None:
                    raise ValueError("Sketch feature requires sketch geometry")
                solver = solve_sketch(feature.sketch)
                if solver["status"] == "conflicting":
                    raise ValueError(
                        "Conflicting sketch constraints: " + ", ".join(solver["conflicts"])
                    )
                sketch = Sketch.model_validate(solver["sketch"])
                points = {pt.id: (pt.x, pt.y, 0) for pt in sketch.points}
                lines, faces = [], []
                for entity in sketch.entities:
                    if entity.construction:
                        continue
                    if entity.kind == "line":
                        lines.append(
                            b.Edge.make_line(points[entity.points[0]], points[entity.points[1]])
                        )
                    else:
                        faces.append(b.Pos(*points[entity.points[0]]) * b.Circle(entity.radius))
                if lines:
                    wires = b.Wire.combine(lines)
                    if not all(w.is_closed for w in wires):
                        raise ValueError("Extrudable sketch profiles must be closed")
                    # Explicit separate profiles. Nested loops require a cut feature, not guessed holes.
                    faces.extend(b.Face(w) for w in wires)
                if not faces:
                    raise ValueError("Sketch has no closed non-construction profiles")
                plane = getattr(b.Plane, sketch.plane).offset(sketch.offset)
                shape = plane * b.Compound(children=faces)
            elif kind == "extrude":
                faces = list(source().faces())
                taper = number("taper", 0)
                if not -90 < taper < 90:
                    raise ValueError("Extrude taper must be between -90 and 90 degrees")
                shape = b.Compound(
                    children=[
                        b.extrude(face, amount=number("amount"), taper=taper) for face in faces
                    ]
                )
            elif kind == "revolve":
                axis = b.Axis(
                    (number("ox", 0), number("oy", 0), number("oz", 0)),
                    (number("ax", 0), number("ay", 1), number("az", 0)),
                )
                angle = number("angle", 360, positive=True)
                if angle > 360:
                    raise ValueError("Revolution cannot exceed 360 degrees")
                shape = b.revolve(source().faces(), axis=axis, revolution_arc=angle)
            elif kind in {"union", "cut", "intersect"}:
                if len(sources) < 2:
                    raise ValueError("Boolean requires at least two input features")
                shape = sources[0]
                for tool in sources[1:]:
                    shape = _normalize_boolean_result(
                        shape + tool
                        if kind == "union"
                        else shape - tool
                        if kind == "cut"
                        else shape & tool
                    )
            elif kind in {"fillet", "chamfer"}:
                edges = selected_shapes(source(), "edge", feature.selection)
                shape = (
                    b.fillet(edges, number("radius", positive=True))
                    if kind == "fillet"
                    else b.chamfer(edges, number("length", positive=True))
                )
            elif kind == "shell":
                solids = source().solids()
                if len(solids) != 1:
                    raise ValueError("Shell requires one solid")
                openings = selected_shapes(solids[0], "face", feature.selection)
                shape = solids[0].offset_3d(openings, -number("thickness", positive=True))
            elif kind == "transform":
                shape = (
                    b.Pos(number("x", 0), number("y", 0), number("z", 0))
                    * b.Rot(number("rx", 0), number("ry", 0), number("rz", 0))
                    * source()
                )
            elif kind == "mirror":
                axis_index = number("plane", 0)
                if axis_index not in (0, 1, 2):
                    raise ValueError("Mirror plane: 0=YZ, 1=XZ, 2=XY")
                shape = b.mirror(
                    source(), about=(b.Plane.YZ, b.Plane.XZ, b.Plane.XY)[int(axis_index)]
                )
            elif kind == "pattern":
                count = number("count", positive=True)
                if not count.is_integer():
                    raise ValueError("Pattern count must be a positive integer")
                enforce_limit(int(count), PATTERN_COUNT_LIMIT, "pattern instances")
                base = source()
                instance_count *= int(count)
                pattern_estimate = {
                    "instances": instance_count,
                    "faces": len(base.faces()) * int(count),
                    "edges": len(base.edges()) * int(count),
                }
                enforce_limit(instance_count, PATTERN_INSTANCE_LIMIT, "expanded pattern instances")
                enforce_limit(
                    pattern_estimate["faces"], FEATURE_FACE_LIMIT, "estimated pattern faces"
                )
                enforce_limit(
                    pattern_estimate["edges"], FEATURE_EDGE_LIMIT, "estimated pattern edges"
                )
                shape = b.Compound(
                    children=[
                        b.Pos(i * number("x", 0), i * number("y", 0), i * number("z", 0))
                        * b.Rot(0, 0, i * number("angle", 0))
                        * source()
                        for i in range(int(count))
                    ]
                )
            elif kind == "import":
                if feature.asset_id not in assets:
                    raise ValueError("Imported base is unavailable or not owned by this workspace")
                shape = b.import_step(str(assets[feature.asset_id]))
                if "solid_index" in p:
                    index = number("solid_index")
                    if not index.is_integer() or index < 0 or index >= len(shape.solids()):
                        raise ValueError("Imported solid index is outside its immutable base")
                    shape = shape.solids()[int(index)]
            elif kind == "hardware":
                from forge_mind.workspace.hardware import hardware_shape

                if number("envelope", 0) not in {0, 1}:
                    raise ValueError("Hardware envelope must be 0 or 1")
                shape = hardware_shape(
                    feature.catalog_id or "", envelope=number("envelope", 0) == 1
                )
            elif kind == "gear":
                from forge_mind.agentic.gears import mesh_clock_deg, spur_gear

                teeth = number("teeth", positive=True)
                if not teeth.is_integer() or teeth < 17:
                    raise ValueError(
                        "Gear teeth must be an integer of at least 17 for this involute profile"
                    )
                pressure = number("pressure_angle", 20)
                if not 20 <= pressure <= 25:
                    raise ValueError("Supported pressure angle is 20 to 25 degrees")
                shape = spur_gear(
                    number("module", positive=True),
                    int(teeth),
                    number("thickness", positive=True),
                    bore=number("bore", 0),
                    pressure_angle=pressure,
                    clock_deg=mesh_clock_deg(int(teeth))
                    if number("auto_clock", 0) == 1
                    else number("clock", 0),
                )
            elif kind in {"shaft", "bearing"}:
                diameter, bore = number("diameter", positive=True), number("bore", 0)
                height = number("length" if kind == "shaft" else "width", positive=True)
                if not 0 <= bore < diameter:
                    raise ValueError("Bore must be nonnegative and smaller than outside diameter")
                shape = b.Cylinder(diameter / 2, height)
                if bore:
                    shape = shape - b.Cylinder(bore / 2, height + 2)
            elif kind == "link":
                length, width = number("length", positive=True), number("width", positive=True)
                height, bore = number("thickness", positive=True), number("bore", positive=True)
                if not bore < width < length:
                    raise ValueError("Link requires bore < width < pivot distance")
                shape = b.Pos(length / 2, 0, 0) * b.Box(length, width, height)
                for x in (0, length):
                    shape = shape + b.Pos(x, 0, 0) * b.Cylinder(width / 2, height)
                for x in (0, length):
                    shape = shape - b.Pos(x, 0, 0) * b.Cylinder(bore / 2, height + 2)
            else:
                raise ValueError("Unsupported feature")
            if kind in {"gear", "shaft"} and "key_width" in p:
                kw, kh = number("key_width", positive=True), number("key_height", positive=True)
                diameter = number(
                    "diameter" if kind == "shaft" else "shaft_diameter", positive=True
                )
                height = number("length" if kind == "shaft" else "thickness", positive=True)
                if kw >= diameter or kh >= diameter / 2:
                    raise ValueError("Key dimensions exceed the supported shaft section")
                shape = shape - b.Pos(0, diameter / 2, 0) * b.Box(kw, kh, height + 2)
            shape = _normalize_boolean_result(shape)
            if kind not in {"sketch", "profile", "rounded_profile", "curve"} and shape.solids():
                solids = list(shape.solids())
                shape = solids[0] if len(solids) == 1 else b.Compound(children=solids)
            if not shape.is_valid or (
                kind not in {"sketch", "profile", "rounded_profile", "curve"}
                and (not shape.solids() or shape.volume <= 0)
            ):
                raise ValueError("Operation did not produce valid positive-volume geometry")
            face_count, edge_count = len(shape.faces()), len(shape.edges())
            enforce_limit(face_count, FEATURE_FACE_LIMIT, "feature faces")
            enforce_limit(edge_count, FEATURE_EDGE_LIMIT, "feature edges")
            instance_counts[feature.id] = instance_count
            shapes[feature.id] = shape
            history.append(
                {
                    "id": feature.id,
                    "state": "built",
                    "signature": signature(shape),
                    "volume_mm3": float(shape.volume),
                    "solid_count": len(shape.solids()),
                    "sketch_solution": solver,
                    "source_sha256": hashlib.sha256(feature.code.encode()).hexdigest()
                    if kind == "python"
                    else None,
                    "history_scope": "Retained parametric source; deterministic regeneration unverified; internal operations are source-editable"
                    if kind == "python"
                    else "Native feature",
                    "resources": {
                        "faces": face_count,
                        "edges": edge_count,
                        "expanded_instances": instance_count,
                        "pattern_preallocation_estimate": pattern_estimate,
                        "face_limit": FEATURE_FACE_LIMIT,
                        "edge_limit": FEATURE_EDGE_LIMIT,
                    },
                }
            )
        except Exception as exc:
            raise ValueError(f"Feature {feature.id} ({feature.name}): {exc}") from exc
    outputs = document.outputs or ([document.features[-1].id] if document.features else [])
    if not outputs:
        raise ValueError("Document has no output")
    if document.mechanism:
        from forge_mind.workspace.mechanism_geometry import place_components

        shapes, mechanism_record = place_components(document, shapes)
        history.append(mechanism_record)
    try:
        result = b.Compound(
            children=[shapes[output] for output in outputs]
            if document.mechanism
            else [solid for output in outputs for solid in shapes[output].solids()]
        )
    except KeyError as exc:
        raise ValueError(
            "Output or dependency is suppressed; choose a valid rollback output"
        ) from exc
    if not result.solids():
        raise ValueError("Document output requires at least one solid")
    enforce_limit(len(result.faces()), FEATURE_FACE_LIMIT, "document output faces")
    enforce_limit(len(result.edges()), FEATURE_EDGE_LIMIT, "document output edges")
    return result, history
