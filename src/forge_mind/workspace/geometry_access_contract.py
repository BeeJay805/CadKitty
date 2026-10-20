"""Native-free contracts for exact finite cylindrical access observations."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge_mind.workspace.geometry_distance_contract import Digest, Identifier, Number, Vector

FRAME: Literal["cad_z_up_mm_world"] = "cad_z_up_mm_world"
SOURCE_NAMES = frozenset({"part.step", "topology.json", "document.json"})
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 3 * MAX_INPUT_BYTES
MAX_REQUEST_BYTES = MAX_RESPONSE_BYTES = 32 * 1024
MAX_OBSTACLES = 16
MAX_OVERLAP_COMPONENTS = 32
MAX_TIMEOUT = 60
DISTANCE_GUARD = 1e-6
VOLUME_GUARD = 1e-6
FRAME_GUARD = 1e-12
SCOPE = (
    "Complete static, axial or transverse translation of the declared finite capped cylinder "
    "against only the selected saved bodies. Zero volume does not prove noncontact. "
    "Near-zero distance remains contact or unresolved. No arbitrary-body motion, rotation, "
    "mixed-direction travel, curved path, physical fit, strength or acceptance claim."
)
Index = Annotated[int, Field(strict=True, ge=0)]
Positive = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0)]
Nonnegative = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
Relation = Literal["positive_volume_interference", "separated", "contact_or_unresolved"]


class AccessRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Access geometry must have finite derived quantities")
    return value


def _unit(vector: Vector) -> Vector:
    scale = max(abs(value) for value in vector)
    if not scale:
        raise ValueError("Access frame requires nonzero vectors")
    scaled = tuple(value / scale for value in vector)
    norm = math.hypot(*scaled)
    return tuple(_finite(value / norm) for value in scaled)  # type: ignore[return-value]


def _dot(a: Vector, b: Vector) -> float:
    return _finite(math.fsum(left * right for left, right in zip(a, b, strict=True)))


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        _finite(a[1] * b[2] - a[2] * b[1]),
        _finite(a[2] * b[0] - a[0] * b[2]),
        _finite(a[0] * b[1] - a[1] * b[0]),
    )


class AccessBodySelector(AccessRecord):
    body_index: Index | None = None
    occurrence_path: Annotated[str, Field(strict=True, min_length=1, max_length=2048)] | None = None

    @model_validator(mode="after")
    def one_body(self) -> AccessBodySelector:
        if (self.body_index is None) == (self.occurrence_path is None):
            raise ValueError("Select exactly one saved body index or occurrence path")
        return self


class AccessCylinder(AccessRecord):
    center_mm: Vector
    axis: Vector
    radial_reference: Vector
    radius_mm: Positive
    length_mm: Positive


class StaticMotion(AccessRecord):
    mode: Literal["static"]


class AxialMotion(AccessRecord):
    mode: Literal["axial"]
    distance_mm: Number

    @model_validator(mode="after")
    def nonzero(self) -> AxialMotion:
        if self.distance_mm == 0:
            raise ValueError("Use static mode for zero travel")
        return self


class TransverseMotion(AccessRecord):
    mode: Literal["transverse"]
    distance_mm: Number

    @model_validator(mode="after")
    def nonzero(self) -> TransverseMotion:
        if self.distance_mm == 0:
            raise ValueError("Use static mode for zero travel")
        return self


AccessMotion = Annotated[StaticMotion | AxialMotion | TransverseMotion, Field(discriminator="mode")]


class AccessQuery(AccessRecord):
    cylinder: AccessCylinder
    motion: AccessMotion
    obstacles: Annotated[list[AccessBodySelector], Field(min_length=1, max_length=MAX_OBSTACLES)]

    @model_validator(mode="after")
    def unique_requests(self) -> AccessQuery:
        keys = [(row.body_index, row.occurrence_path) for row in self.obstacles]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate requested access obstacles")
        if len(self.model_dump_json().encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("Access query exceeds its request byte budget")
        return self


class FrameAxes(AccessRecord):
    x: Vector
    y: Vector
    z: Vector

    @model_validator(mode="after")
    def proper_frame(self) -> FrameAxes:
        if any(abs(math.hypot(*v) - 1) > FRAME_GUARD for v in (self.x, self.y, self.z)):
            raise ValueError("Access frame axes must be unit vectors")
        if any(
            abs(_dot(a, b)) > FRAME_GUARD
            for a, b in ((self.x, self.y), (self.x, self.z), (self.y, self.z))
        ):
            raise ValueError("Access frame axes must be orthogonal")
        if abs(_dot(_cross(self.x, self.y), self.z) - 1) > FRAME_GUARD:
            raise ValueError("Access frame must have positive orientation")
        return self


class EvaluatedCylinder(AccessRecord):
    center_mm: Vector
    axis: Vector
    frame_axes: FrameAxes
    radius_mm: Positive
    length_mm: Positive

    @model_validator(mode="after")
    def aligned_frame(self) -> EvaluatedCylinder:
        if self.axis != self.frame_axes.z:
            raise ValueError("Access cylinder axis differs from its frame")
        return self


class EvaluatedMotion(AccessRecord):
    mode: Literal["static", "axial", "transverse"]
    translation_mm: Vector
    start_center_mm: Vector
    end_center_mm: Vector


def evaluate_query(query: AccessQuery) -> tuple[EvaluatedCylinder, EvaluatedMotion]:
    """Resolve the declared radial frame and signed local motion deterministically."""
    query = AccessQuery.model_validate(query)
    source = query.cylinder
    z = _unit(source.axis)
    q = _unit(source.radial_reference)
    dot = _dot(q, z)
    projection = tuple(q[i] - z[i] * dot for i in range(3))
    if math.hypot(*projection) <= FRAME_GUARD:
        raise ValueError("Access radial reference is parallel or ill-conditioned")
    x = _unit(projection)  # type: ignore[arg-type]
    y = _cross(z, x)
    frame = FrameAxes(x=x, y=y, z=z)
    cylinder = EvaluatedCylinder(
        center_mm=source.center_mm,
        axis=z,
        frame_axes=frame,
        radius_mm=source.radius_mm,
        length_mm=source.length_mm,
    )
    distance = 0.0 if query.motion.mode == "static" else query.motion.distance_mm
    direction = frame.x if query.motion.mode == "transverse" else frame.z
    translation = tuple(_finite(value * distance) for value in direction)
    end = tuple(_finite(a + b) for a, b in zip(source.center_mm, translation, strict=True))
    motion = EvaluatedMotion(
        mode=query.motion.mode,
        translation_mm=(translation[0], translation[1], translation[2]),
        start_center_mm=source.center_mm,
        end_center_mm=(end[0], end[1], end[2]),
    )
    _finite(source.length_mm + (abs(distance) if query.motion.mode == "axial" else 0))
    _finite(2 * source.radius_mm + (abs(distance) if query.motion.mode == "transverse" else 0))
    area = _finite(math.pi * source.radius_mm * source.radius_mm)
    if query.motion.mode == "transverse":
        area = _finite(area + 2 * source.radius_mm * abs(distance))
    _finite(area * (source.length_mm + (abs(distance) if query.motion.mode == "axial" else 0)))
    # Conservative analytic world bounds must also be finite before constructing CAD.
    for start, stop in zip(source.center_mm, end, strict=True):
        extent = _finite(source.length_mm / 2 + source.radius_mm)
        _finite(min(start, stop) - extent)
        _finite(max(start, stop) + extent)
    return cylinder, motion


class ResolvedAccessBody(AccessRecord):
    requested: AccessBodySelector
    body_index: Index
    body_signature: Digest

    @model_validator(mode="after")
    def requested_index(self) -> ResolvedAccessBody:
        if self.requested.body_index is not None and self.requested.body_index != self.body_index:
            raise ValueError("Resolved access body differs from requested index")
        return self


class AccessContext(AccessRecord):
    project_id: Identifier
    revision_id: Identifier
    operation: Literal["cylindrical_access"] = "cylindrical_access"
    frame: Literal["cad_z_up_mm_world"] = FRAME
    units: Literal["mm"] = "mm"
    query: AccessQuery
    evaluated_cylinder: EvaluatedCylinder
    evaluated_motion: EvaluatedMotion
    obstacles: Annotated[list[ResolvedAccessBody], Field(min_length=1, max_length=MAX_OBSTACLES)]
    inventory_body_count: Annotated[int, Field(strict=True, ge=1)]
    omitted_body_indices: list[Index]
    source_hashes: dict[str, Digest]
    saved_engine_fingerprint: Digest
    query_engine_fingerprint: Digest

    @model_validator(mode="after")
    def bound_context(self) -> AccessContext:
        if set(self.source_hashes) != SOURCE_NAMES:
            raise ValueError("Access requires exact STEP, topology and document hashes")
        cylinder, motion = evaluate_query(self.query)
        if self.evaluated_cylinder != cylinder or self.evaluated_motion != motion:
            raise ValueError("Access evaluated geometry differs from requested frame and motion")
        if [row.requested for row in self.obstacles] != self.query.obstacles:
            raise ValueError("Access obstacle order differs from requested selections")
        selected = [row.body_index for row in self.obstacles]
        omitted = self.omitted_body_indices
        if len(set(selected)) != len(selected):
            raise ValueError("Access obstacle aliases resolve to a duplicate body")
        if omitted != sorted(set(omitted)) or set(selected) & set(omitted):
            raise ValueError("Access omitted inventory is duplicate, unordered or selected")
        if len(selected) + len(omitted) != self.inventory_body_count:
            raise ValueError("Access selected and omitted inventory is incomplete")
        # No allocation based on an untrusted possibly huge body count.
        if any(index != value for index, value in enumerate(sorted(selected + omitted))):
            raise ValueError("Access inventory must cover every contiguous saved body index")
        return self


class BoundsRecord(AccessRecord):
    bbox_min_mm: Vector
    bbox_max_mm: Vector

    @model_validator(mode="after")
    def ordered_bounds(self) -> BoundsRecord:
        if any(a > b for a, b in zip(self.bbox_min_mm, self.bbox_max_mm, strict=True)):
            raise ValueError("Access bounds must be ordered")
        return self


class SweepFact(BoundsRecord):
    method: Literal["static_cylinder", "axial_extended_cylinder", "transverse_capsule_prism"]
    volume_mm3: Positive


class OverlapComponent(BoundsRecord):
    volume_mm3: Positive


class NumericalGuards(AccessRecord):
    distance_mm: Nonnegative = DISTANCE_GUARD
    volume_mm3: Nonnegative = VOLUME_GUARD

    @model_validator(mode="after")
    def fixed_guards(self) -> NumericalGuards:
        if self.distance_mm != DISTANCE_GUARD or self.volume_mm3 != VOLUME_GUARD:
            raise ValueError("Access numerical guards cannot be caller selected")
        return self


def classify_observation(volume: float, distance: float) -> Relation:
    if not math.isfinite(volume) or not math.isfinite(distance) or volume < 0 or distance < 0:
        raise ValueError("Access observation requires finite nonnegative measurements")
    if volume > 0 and distance > DISTANCE_GUARD:
        raise ValueError("Access overlap and separation measurements disagree")
    if volume > VOLUME_GUARD:
        return "positive_volume_interference"
    if volume == 0 and distance > DISTANCE_GUARD:
        return "separated"
    return "contact_or_unresolved"


class AccessObservation(AccessRecord):
    obstacle: ResolvedAccessBody
    overlap_components: Annotated[list[OverlapComponent], Field(max_length=MAX_OVERLAP_COMPONENTS)]
    overlap_volume_mm3: Nonnegative
    minimum_distance_mm: Nonnegative
    witness_sweep_mm: Vector
    witness_obstacle_mm: Vector
    solution_count: Annotated[int, Field(strict=True, ge=1)]
    native_inner_solution: Annotated[bool, Field(strict=True)]
    native_completed: Annotated[bool, Field(strict=True)]
    relation: Relation

    @model_validator(mode="after")
    def measured_facts(self) -> AccessObservation:
        if not self.native_completed:
            raise ValueError("Access requires completed native observations")
        try:
            amount = math.fsum(row.volume_mm3 for row in self.overlap_components)
        except OverflowError as exc:
            raise ValueError("Access overlap component sum is nonfinite") from exc
        if not math.isfinite(amount) or not math.isclose(
            amount, self.overlap_volume_mm3, rel_tol=1e-9, abs_tol=0
        ):
            raise ValueError("Access overlap components do not account for measured volume")
        distance = math.dist(self.witness_sweep_mm, self.witness_obstacle_mm)
        if not math.isfinite(distance) or not math.isclose(
            distance, self.minimum_distance_mm, rel_tol=1e-9, abs_tol=DISTANCE_GUARD
        ):
            raise ValueError("Access distance witnesses disagree with measured distance")
        if self.relation != classify_observation(self.overlap_volume_mm3, self.minimum_distance_mm):
            raise ValueError("Access relation differs from measured facts")
        return self


class AccessResult(AccessContext):
    sweep: SweepFact
    observations: Annotated[list[AccessObservation], Field(min_length=1, max_length=MAX_OBSTACLES)]
    numerical_guards: NumericalGuards
    native_completed: Annotated[bool, Field(strict=True)]
    scope: str = SCOPE

    @model_validator(mode="after")
    def complete_result(self) -> AccessResult:
        if not self.native_completed or self.scope != SCOPE:
            raise ValueError("Access requires complete native result and exact scope")
        if [row.obstacle for row in self.observations] != self.obstacles:
            raise ValueError("Access result must retain every selected obstacle in order")
        expected = {
            "static": "static_cylinder",
            "axial": "axial_extended_cylinder",
            "transverse": "transverse_capsule_prism",
        }[self.query.motion.mode]
        if self.sweep.method != expected:
            raise ValueError("Access swept-solid method differs from requested motion")
        for row in self.observations:
            for component in row.overlap_components:
                if any(
                    a < lower - DISTANCE_GUARD or b > upper + DISTANCE_GUARD
                    for a, b, lower, upper in zip(
                        component.bbox_min_mm,
                        component.bbox_max_mm,
                        self.sweep.bbox_min_mm,
                        self.sweep.bbox_max_mm,
                        strict=True,
                    )
                ):
                    raise ValueError("Access overlap component lies outside measured sweep bounds")
        return self


def access_tool_result(
    value: Any, project_id: str, revision_id: str, query: AccessQuery
) -> dict[str, Any]:
    """Revalidate exact query, provenance context and complete bounded observations."""
    query = AccessQuery.model_validate(query)
    result = AccessResult.model_validate(value)
    if (
        result.project_id != project_id
        or result.revision_id != revision_id
        or result.query != query
    ):
        raise ValueError("Access result differs from requested revision or query")
    if len(result.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Access result exceeds its response byte budget")
    return result.model_dump(mode="json")
