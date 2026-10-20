"""Prescribed rigid motion in exported CAD reference space, not a joint solver."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from forge_mind.workspace.mechanism_models import Identifier, Scalar
from forge_mind.workspace.models import Record

if TYPE_CHECKING:
    from forge_mind.workspace.document import DesignDocument


class MotionDriver(Record):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    unit: Literal["mm", "deg"]
    minimum: float = Field()
    maximum: float = Field()
    reference: Scalar = 0.0


class MotionGroup(Record):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    parent: Identifier | None = None
    driver: Identifier
    kind: Literal["slide", "rotate"]
    # All axes/pivots are in the exported design's rest/world CAD XYZ frame.
    axis: tuple[float, float, float] = (0, 0, 1)
    pivot: tuple[Scalar, Scalar, Scalar] = (0.0, 0.0, 0.0)
    ratio: float = Field(default=1)
    occurrences: list[str] = Field(default_factory=list)


class MotionRig(Record):
    kind: Literal["prescribed_rigid_v1"] = "prescribed_rigid_v1"
    frame: Literal["cad_z_up_mm_rest_world"] = "cad_z_up_mm_rest_world"
    drivers: list[MotionDriver] = Field(min_length=1)
    groups: list[MotionGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def references(self) -> MotionRig:
        drivers = {d.id for d in self.drivers}
        groups = {g.id: g for g in self.groups}
        if len(drivers) != len(self.drivers) or len(groups) != len(self.groups):
            raise ValueError("Motion IDs must be unique within drivers and groups")
        owned: set[str] = set()
        acyclic: set[str] = set()
        for d in self.drivers:
            if d.minimum >= d.maximum:
                raise ValueError("Motion driver needs increasing limits")
        for g in self.groups:
            if g.driver not in drivers or (g.parent and g.parent not in groups):
                raise ValueError("Motion group refers to a missing driver or parent")
            if (
                any(not math.isfinite(a) for a in g.axis)
                or not math.isfinite(math.hypot(*g.axis))
                or math.hypot(*g.axis) < 1e-6
                or g.ratio == 0
            ):
                raise ValueError("Motion needs a nonzero axis and ratio")
            if len(set(g.occurrences)) != len(g.occurrences) or owned.intersection(g.occurrences):
                raise ValueError("Each occurrence can belong directly to only one motion group")
            owned.update(g.occurrences)
            seen = {g.id}
            parent = g.parent
            while parent and parent not in acyclic:
                if parent not in groups:
                    raise ValueError("Motion group refers to a missing ancestor parent")
                if parent in seen:
                    raise ValueError("Motion parent hierarchy is cyclic")
                seen.add(parent)
                parent = groups[parent].parent
            acyclic.update(seen)
        if {g.driver for g in self.groups} != drivers:
            raise ValueError("Every motion driver must move a group")
        if not owned:
            raise ValueError("Motion must include at least one physical occurrence")
        return self


def resolve_rig(rig: MotionRig, values: dict[str, float]) -> dict[str, Any]:
    def scalar(v: Scalar) -> float:
        if isinstance(v, str) and v not in values:
            raise ValueError("Motion references an unknown variable: " + v)
        n = values[v] if isinstance(v, str) else v
        if not math.isfinite(n):
            raise ValueError("Motion coordinate must be finite")
        return float(n)

    result = rig.model_dump()
    for d in result["drivers"]:
        d["reference"] = scalar(d["reference"])
        if not d["minimum"] <= d["reference"] <= d["maximum"]:
            raise ValueError("Motion reference must lie within driver limits")
    for g in result["groups"]:
        g["pivot"] = [scalar(v) for v in g["pivot"]]
        length = math.hypot(*g["axis"])
        g["axis"] = [v / length for v in g["axis"]]
    return result


def validate_document_rig(document: DesignDocument) -> None:
    from forge_mind.workspace.assembly import expand

    assert document.motion_rig
    if not document.assembly or document.mechanism:
        raise ValueError("Prescribed motion requires an assembly without a planar mechanism")
    resolve_rig(document.motion_rig, document.values())
    paths = {r["path"] for r in expand(document)}
    if any(set(g.occurrences) - paths for g in document.motion_rig.groups):
        raise ValueError("Motion group references a missing leaf occurrence")


def bind_rig(document: DesignDocument, assembly: dict[str, Any], **identity: Any) -> dict[str, Any]:
    assert document.motion_rig
    rig = resolve_rig(document.motion_rig, document.values())
    groups = {path: g["id"] for g in rig["groups"] for path in g["occurrences"]}
    rows = assembly["occurrences"]
    if len({r["path"] for r in rows}) != len(rows) or len({r["body_index"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Motion requires unique exported occurrence bindings")
    if set(groups) - {r["path"] for r in rows}:
        raise ValueError("Motion occurrence was not bound in exported STEP")
    return {
        **identity,
        "rig": rig,
        "bindings": [
            {"path": r["path"], "body_index": r["body_index"], "group": groups.get(r["path"])}
            for r in rows
        ],
        "status": "preview_ready",
        "scope": "Prescribed rigid motion. Unassigned occurrences remain fixed. No joint closure, contact, load or continuous clearance certification.",
    }


def group_matrices(rig: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Parent * local delta; identity at driver references. No accumulated integration."""
    from scipy.spatial.transform import Rotation

    drivers = {d["id"]: d for d in rig["drivers"]}
    if set(values) - drivers.keys():
        raise ValueError("Unknown motion driver")
    for key, value in values.items():
        d = drivers[key]
        if not math.isfinite(value) or not d["minimum"] <= value <= d["maximum"]:
            raise ValueError("Motion value outside finite driver limits")
    groups = {g["id"]: g for g in rig["groups"]}
    result: dict[str, Any] = {}

    def local_matrix(key: str) -> Any:
        g = groups[key]
        d = drivers[g["driver"]]
        delta = (values.get(d["id"], d["reference"]) - d["reference"]) * g["ratio"]
        if not math.isfinite(delta):
            raise ValueError("Motion displacement must remain finite")
        matrix = np.eye(4)
        axis = np.asarray(g["axis"], dtype=float)
        if g["kind"] == "slide":
            matrix[:3, 3] = axis * delta
        else:
            rotation = Rotation.from_rotvec(axis * math.radians(delta)).as_matrix()
            pivot = np.asarray(g["pivot"], dtype=float)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = pivot - rotation @ pivot
        return matrix

    for key in groups:
        chain: list[str] = []
        active: set[str] = set()
        current: str | None = key
        while current and current not in result:
            if current in active or current not in groups:
                raise ValueError("Motion parent hierarchy is cyclic or missing")
            active.add(current)
            chain.append(current)
            current = groups[current]["parent"]
        for child in reversed(chain):
            parent = groups[child]["parent"]
            matrix = local_matrix(child)
            result[child] = result[parent] @ matrix if parent else matrix
            if not np.isfinite(result[child]).all():
                raise ValueError("Motion transform must remain finite")
    return result
