"""Shared manual/AI feature document. No evaluation of expressions or invented history."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from forge_mind.workspace.mechanism_models import Mechanism, Scalar
from forge_mind.workspace.models import Record
from forge_mind.workspace.motion_rig import MotionRig
from forge_mind.workspace.product_models import DeliveryGuidance, PartProduction

Name = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")]
Vector3 = tuple[float, float, float]
MAX_DOCUMENT_VARIABLES: int | None = None
MAX_DOCUMENT_RELATIONS: int | None = None


class Point2(Record):
    id: Name
    x: float = Field()
    y: float = Field()


class SketchEntity(Record):
    id: Name
    kind: Literal["line", "circle"]
    points: list[Name] = Field(min_length=1, max_length=2)
    radius: float = Field(default=1, gt=0)
    construction: bool = False


class SketchConstraint(Record):
    id: Name
    kind: Literal[
        "fix",
        "coincident",
        "horizontal",
        "vertical",
        "distance",
        "distance_x",
        "distance_y",
        "parallel",
        "perpendicular",
        "equal",
        "radius",
        "tangent",
        "symmetry",
    ]
    points: list[Name] = Field(default_factory=list, max_length=3)
    entities: list[Name] = Field(default_factory=list, max_length=2)
    value: float = 0


class Sketch(Record):
    points: list[Point2] = Field(default_factory=list)
    entities: list[SketchEntity] = Field(default_factory=list)
    constraints: list[SketchConstraint] = Field(default_factory=list)
    plane: Literal["XY", "XZ", "YZ"] = "XY"
    offset: float = Field(default=0)

    @model_validator(mode="after")
    def references(self) -> Sketch:
        p = {point.id for point in self.points}
        e = {entity.id for entity in self.entities}
        c = {constraint.id for constraint in self.constraints}
        if (
            len(p) != len(self.points)
            or len(e) != len(self.entities)
            or len(c) != len(self.constraints)
        ):
            raise ValueError("Sketch IDs must be unique")
        for entity in self.entities:
            if not set(entity.points) <= p or len(entity.points) != (
                2 if entity.kind == "line" else 1
            ):
                raise ValueError("Sketch entity has invalid point references")
        for constraint in self.constraints:
            if not set(constraint.points) <= p or not set(constraint.entities) <= e:
                raise ValueError("Constraint references missing geometry")
            arity = {
                "fix": (1, 0),
                "coincident": (2, 0),
                "distance": (2, 0),
                "distance_x": (2, 0),
                "distance_y": (2, 0),
                "horizontal": (0, 1),
                "vertical": (0, 1),
                "parallel": (0, 2),
                "perpendicular": (0, 2),
                "equal": (0, 2),
                "radius": (0, 1),
                "tangent": (0, 2),
                "symmetry": (3, 0),
            }[constraint.kind]
            if (len(constraint.points), len(constraint.entities)) != arity:
                raise ValueError(f"{constraint.kind} needs {arity[0]} points and {arity[1]} curves")
            if constraint.kind in {"distance", "radius"} and constraint.value <= 0:
                raise ValueError("Distance/radius constraint must be positive")
        return self


class GeometryRef(Record):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    entity_id: str = Field(pattern=r"^(face|edge|body)_[a-f0-9]{20}_[0-9]+$")
    signature: str = Field(pattern=r"^[a-f0-9]{64}$")
    body_signature: str = Field(pattern=r"^[a-f0-9]{64}$")
    kind: Literal["face", "edge", "body"]
    point: Vector3


class Feature(Record):
    id: Name
    name: str = Field(min_length=1, max_length=160)
    kind: Literal[
        "profile",
        "rounded_profile",
        "curve",
        "loft",
        "sweep",
        "python",
        "box",
        "cylinder",
        "cone",
        "torus",
        "sphere",
        "sketch",
        "extrude",
        "revolve",
        "union",
        "cut",
        "intersect",
        "fillet",
        "chamfer",
        "shell",
        "pattern",
        "mirror",
        "transform",
        "import",
        "hardware",
        "gear",
        "link",
        "shaft",
        "bearing",
    ]
    inputs: list[Name] = Field(default_factory=list)
    params: dict[Name, float] = Field(default_factory=dict)
    bindings: dict[Name, Name] = Field(default_factory=dict)
    points: list[tuple[Scalar, Scalar, Scalar]] = Field(default_factory=list)
    code: str = ""
    sketch: Sketch | None = None
    asset_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    catalog_id: str | None = Field(default=None, max_length=160)
    selection: list[str] = Field(default_factory=list)
    suppressed: bool = False


class VariableRelation(Record):
    name: Name
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "max",
        "min",
        "sin_deg",
        "cos_deg",
        "tan_deg",
        "atan2_deg",
        "sqrt",
    ]
    a: Scalar
    b: Scalar = 0.0

    @model_validator(mode="after")
    def operands(self) -> VariableRelation:
        self.validate_operands()
        return self

    def validate_operands(self) -> None:
        if self.operation in {"sin_deg", "cos_deg", "tan_deg", "sqrt"}:
            if isinstance(self.b, str) or self.b != 0:
                raise ValueError("Unary variable relations require b to be omitted or literal zero")
        elif "b" not in self.model_fields_set:
            raise ValueError("Binary variable relations require an explicit b operand")


class PartPort(Record):
    name: Name
    kind: Literal["bore", "shaft"]
    origin: tuple[Scalar, Scalar, Scalar] = (0.0, 0.0, 0.0)
    axis: Vector3 = (0.0, 0.0, 1.0)
    diameter: Scalar


class PartDefinition(Record):
    production: PartProduction = Field(default_factory=PartProduction)
    id: Name
    name: str = Field(min_length=1, max_length=120)
    features: list[Feature] = Field(min_length=1)
    output: Name
    ports: list[PartPort] = Field(default_factory=list)
    source: str = Field(
        default="Custom part; material and manufacturing unverified", max_length=1000
    )

    @model_validator(mode="after")
    def stl_delivery(self) -> PartDefinition:
        if self.production.deliver_stl and self.production.role != "make":
            raise ValueError(
                f"Part {self.id}: deliver_stl requires an explicitly declared make role"
            )
        return self


class Occurrence(Record):
    id: Name
    definition: Name
    position: tuple[Scalar, Scalar, Scalar] = (0.0, 0.0, 0.0)
    rotation: tuple[Scalar, Scalar, Scalar] = (0.0, 0.0, 0.0)


class Subassembly(Record):
    id: Name
    name: str = Field(min_length=1, max_length=120)
    children: list[Occurrence] = Field(min_length=1)


class InterfaceMate(Record):
    id: Name
    a: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_/]*\.[a-zA-Z][a-zA-Z0-9_]*$")
    b: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_/]*\.[a-zA-Z][a-zA-Z0-9_]*$")
    min_clearance: float = Field(default=0, ge=-10, le=10)
    max_clearance: float = Field(default=0.5, ge=-10, le=10)
    axial_tolerance: float = Field(default=0.01, ge=0, le=100)


class AssemblySpec(Record):
    parts: list[PartDefinition] = Field(min_length=1)
    subassemblies: list[Subassembly] = Field(default_factory=list)
    roots: list[Occurrence] = Field(min_length=1)
    mates: list[InterfaceMate] = Field(default_factory=list)


class DesignDocument(Record):
    delivery_guidance: DeliveryGuidance | None = None
    production: PartProduction | None = None
    version: Literal[1] = 1
    name: str = Field(default="Untitled design", min_length=1, max_length=160)
    units: Literal["mm"] = "mm"
    variables: dict[Name, float] = Field(default_factory=dict)
    relations: list[VariableRelation] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    outputs: list[Name] = Field(default_factory=list)
    mechanism: Mechanism | None = None
    motion_rig: MotionRig | None = None
    assembly: AssemblySpec | None = None

    @model_serializer(mode="wrap")
    def legacy_guidance_absence(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep historical default-expanded document hashes when guidance is absent."""
        result: dict[str, Any] = handler(self)
        if self.delivery_guidance is None:
            result.pop("delivery_guidance", None)
        return result

    def values(self) -> dict[str, float]:
        import math

        result = dict(self.variables)
        for relation in self.relations:
            relation.validate_operands()
            if relation.name in result:
                raise ValueError("Derived variable cannot overwrite a variable")
            try:
                a, b = (result[v] if isinstance(v, str) else v for v in (relation.a, relation.b))
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError("Variable relation operands must be finite")
                operation = relation.operation
                if operation == "add":
                    value = a + b
                elif operation == "subtract":
                    value = a - b
                elif operation == "multiply":
                    value = a * b
                elif operation == "divide":
                    value = a / b
                elif operation == "max":
                    value = max(a, b)
                elif operation == "min":
                    value = min(a, b)
                elif operation in {"sin_deg", "cos_deg", "tan_deg"}:
                    angle = math.remainder(a, 180 if operation == "tan_deg" else 360)
                    if operation == "tan_deg" and abs(abs(angle) - 90) <= 1e-10:
                        raise ValueError("tan_deg is undefined at odd multiples of 90 degrees")
                    radians = math.radians(angle)
                    value = (
                        math.sin(radians)
                        if operation == "sin_deg"
                        else math.cos(radians)
                        if operation == "cos_deg"
                        else math.tan(radians)
                    )
                elif operation == "atan2_deg":
                    if a == 0 and b == 0:
                        raise ValueError("atan2_deg requires a nonzero vector; a is y and b is x")
                    value = math.degrees(math.atan2(a, b))
                elif operation == "sqrt":
                    if a < 0:
                        raise ValueError("sqrt requires a nonnegative operand")
                    value = math.sqrt(a)
                else:
                    raise ValueError("Unknown named variable relation")
            except (KeyError, ZeroDivisionError) as exc:
                raise ValueError(
                    "Variable relations must be ordered and reference existing values; division by zero is unsupported"
                ) from exc
            if not math.isfinite(value):
                raise ValueError("Derived variable must be finite")
            result[relation.name] = value
        return result

    @model_validator(mode="after")
    def dependencies(self) -> DesignDocument:
        values = self.values()
        if self.production is not None and self.production.deliver_stl:
            raise ValueError("deliver_stl is supported only on assembly make part definitions")
        if self.assembly:
            if self.production is not None:
                raise ValueError("Assembly production must be declared on each part definition")
            if self.features or self.outputs:
                raise ValueError("Assembly definitions replace top-level features and outputs")
            from forge_mind.workspace.assembly import validate_assembly

            validate_assembly(self)
        if any(not math.isfinite(v) for v in self.variables.values()):
            raise ValueError("Variable must be finite")
        seen: set[str] = set()
        kinds: dict[str, str] = {}
        for feature in self.features:
            if len(set(feature.inputs)) != len(feature.inputs):
                raise ValueError(
                    "Feature inputs must be unique; duplicate operands do not represent extra physical parts"
                )
            if feature.id in seen or not set(feature.inputs) <= seen:
                raise ValueError("Feature IDs must be unique; inputs must precede their dependent")
            if not set(feature.bindings.values()) <= values.keys():
                raise ValueError("Feature binding references an unknown variable")
            if any(not math.isfinite(v) for v in feature.params.values()):
                raise ValueError("Feature parameter must be finite")
            point_refs = {v for point in feature.points for v in point if isinstance(v, str)}
            if not point_refs <= values.keys():
                raise ValueError("Feature points reference an unknown variable")
            if any(
                not math.isfinite(values[v] if isinstance(v, str) else v)
                for point in feature.points
                for v in point
            ):
                raise ValueError("Feature points must resolve to finite coordinates")
            if feature.kind == "curve" and len(feature.points) < 2:
                raise ValueError("Curve requires at least two points")
            if feature.kind == "python" and not feature.code.strip():
                raise ValueError("Python feature requires explicit source code")
            if feature.kind != "curve" and feature.points:
                raise ValueError("Points are only supported by curve features")
            if feature.kind != "python" and feature.code:
                raise ValueError("Code is only supported by Python features")
            allowed = {
                "profile": {"radius_x", "radius_y", "x", "y", "z", "rx", "ry", "rz"},
                "rounded_profile": {
                    "width",
                    "height",
                    "radius",
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                },
                "curve": set(),
                "loft": {"ruled"},
                "sweep": {"frenet"},
                "python": feature.params.keys() | feature.bindings.keys(),
                "box": {"x", "y", "z"},
                "cylinder": {"radius", "height"},
                "cone": {"bottom_radius", "top_radius", "height"},
                "torus": {"major_radius", "minor_radius"},
                "sphere": {"radius"},
                "sketch": set(),
                "extrude": {"amount", "taper"},
                "revolve": {"angle", "ox", "oy", "oz", "ax", "ay", "az"},
                "union": set(),
                "cut": set(),
                "intersect": set(),
                "fillet": {"radius"},
                "chamfer": {"length"},
                "shell": {"thickness"},
                "pattern": {"count", "x", "y", "z", "angle"},
                "mirror": {"plane"},
                "transform": {"x", "y", "z", "rx", "ry", "rz"},
                "import": {"solid_index"},
                "hardware": {"envelope"},
                "gear": {
                    "module",
                    "teeth",
                    "thickness",
                    "bore",
                    "pressure_angle",
                    "auto_clock",
                    "clock",
                    "key_width",
                    "key_height",
                    "shaft_diameter",
                },
                "link": {"length", "width", "thickness", "bore"},
                "shaft": {"diameter", "length", "bore", "key_width", "key_height"},
                "bearing": {"diameter", "bore", "width"},
            }[feature.kind]
            if not (feature.params.keys() | feature.bindings.keys()) <= allowed:
                raise ValueError(f"Unknown parameter for {feature.kind}")
            arity = (
                0
                if feature.kind
                in {
                    "profile",
                    "rounded_profile",
                    "curve",
                    "box",
                    "cylinder",
                    "cone",
                    "torus",
                    "sphere",
                    "sketch",
                    "import",
                    "hardware",
                    "gear",
                    "link",
                    "shaft",
                    "bearing",
                }
                else 2
                if feature.kind in {"union", "cut", "intersect", "loft", "sweep"}
                else 1
            )
            if feature.kind != "python" and (
                len(feature.inputs) < arity or (arity < 2 and len(feature.inputs) != arity)
            ):
                raise ValueError(
                    f"{feature.kind} requires {'at least ' if arity == 2 else ''}{arity} inputs"
                )
            flag = (
                "ruled" if feature.kind == "loft" else "frenet" if feature.kind == "sweep" else None
            )
            if flag is not None:
                flag_value = (
                    values[feature.bindings[flag]]
                    if flag in feature.bindings
                    else feature.params.get(flag, 0)
                )
                if flag_value not in (0, 1):
                    raise ValueError(f"{flag} must be zero or one")
            if feature.kind == "loft" and any(
                kinds.get(i) not in {"profile", "rounded_profile"} for i in feature.inputs
            ):
                raise ValueError("Loft requires profile inputs")
            if feature.kind == "sweep" and (
                len(feature.inputs) != 2
                or kinds.get(feature.inputs[0]) not in {"profile", "rounded_profile"}
                or kinds.get(feature.inputs[1]) != "curve"
            ):
                raise ValueError("Sweep requires a profile followed by a curve")
            if feature.kind in {"extrude", "revolve"} and any(
                kinds.get(i) != "sketch" for i in feature.inputs
            ):
                raise ValueError(
                    "Extrude/revolve requires an explicit sketch profile, not every face of a solid"
                )
            seen.add(feature.id)
            kinds[feature.id] = feature.kind
        if not set(self.outputs) <= seen or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Outputs must be unique existing feature IDs")
        if self.mechanism:
            from forge_mind.workspace.mechanism import resolve_mechanism

            resolve_mechanism(self.mechanism, values)
            outputs = {r.id for r in self.assembly.roots} if self.assembly else set(self.outputs)
            owned = {c.feature for c in self.mechanism.components}
            if owned != outputs:
                raise ValueError(
                    "Every mechanism output needs exactly one named component; "
                    f"missing output/root IDs: {sorted(outputs - owned)}; "
                    f"unexpected component feature IDs: {sorted(owned - outputs)}"
                )
            if any(f.suppressed for f in self.features if f.id in self.outputs):
                raise ValueError("Mechanism component outputs cannot be suppressed")
        if self.motion_rig:
            from forge_mind.workspace.motion_rig import validate_document_rig

            validate_document_rig(self)
        return self

    def fingerprint(self) -> str:
        return digest(self.model_dump())


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Annotation(Record):
    id: Name
    text: str = Field(min_length=1, max_length=3000)
    reference: GeometryRef
    # World-space polyline. It remains on its recorded revision when an anchor is unresolved.
    marks: list[Vector3] = Field(default_factory=list)


class Protection(Record):
    id: Name
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    feature_id: Name | None = None
    reference: GeometryRef | None = None
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def target(self) -> Protection:
        if (self.feature_id is None) == (self.reference is None):
            raise ValueError("Protection requires exactly one feature or geometry reference")
        if self.reference and self.reference.revision_id != self.revision_id:
            raise ValueError("Protection reference and baseline revision disagree")
        return self
