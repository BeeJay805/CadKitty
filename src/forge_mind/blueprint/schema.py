"""Blueprint — the structured spec between intent and DSL.

More concrete than IntentSpec (it commits to an envelope, a set of named
features, and which interfaces sit on which feature) but more abstract than a
PartProgram (no exact operations or coordinates yet). The DSLGenerator (Phase 5)
turns a Blueprint into operations.

Fields are required (no defaults) so structured-output strict mode can't leave
critical values like the envelope at zero.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RequestedInterface(StrEnum):
    """The vocabulary the Blueprint uses to NAME a required interface.

    A SUPERSET of the validator's face/board catalog (``dsl.interfaces.declaration.InterfaceType``,
    legacy spellings kept verbatim) PLUS the standards-backed HARDWARE features. Without the hardware
    tier a requested heat-set insert / bearing seat / magnet pocket / dowel had no name and got
    mis-mapped to the closest bolt (``m5_tnut``); now each maps to its OWN ref. The intent-fidelity
    gate (``validator.fidelity``) compares these against what the part actually built, so the
    vocabulary here IS the structured contract the gate holds the generator to.
    """

    # Face / board interfaces (validator face checkers; legacy spellings, do not rename).
    NEMA17_FACE = "nema17_face"
    EXTRUSION_2020_SLOT = "extrusion_2020_slot"
    M5_TNUT = "m5_tnut"
    PI4_MOUNT = "pi4_mount"
    PI4_PORTS = "pi4_ports"
    FAN_FACE = "fan_face"
    # Standards-backed hardware features (conformance-checked; component_id carries the key).
    CLEARANCE_HOLE = "clearance_hole"
    COUNTERBORE = "counterbore"
    HEAT_SET_INSERT = "heat_set_insert"
    CAPTIVE_NUT_POCKET = "captive_nut_pocket"
    BEARING_SEAT = "bearing_seat"
    DOWEL_HOLE = "dowel_hole"
    MAGNET_POCKET = "magnet_pocket"


class BlueprintVec3(BaseModel):
    """A 3-vector with all components required (mm or unit direction)."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class FeatureSpec(BaseModel):
    """A named, high-level feature the part needs."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique feature name, e.g. 'motor_flange'.")
    kind: str = Field(..., description="Shape family: plate, boss, rib, wall, bracket, etc.")
    purpose: str = Field(..., description="Why this feature exists.")
    approx_size_mm: BlueprintVec3 = Field(..., description="Rough [x,y,z] extent in mm.")


class InterfaceSpec(BaseModel):
    """An interface the part must present, bound to a feature + facing."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Declaration name, e.g. 'motor_face'.")
    interface_type: RequestedInterface
    on_feature: str = Field(..., description="Name of the feature this interface sits on.")
    facing: str = Field(..., description="Outward direction, e.g. '+Z', '-Z', '+Y'.")
    component_id: str | None = Field(
        default=None,
        description="Component key for a parameterised interface: 'M3' for a clearance hole / "
        "heat-set insert / counterbore, '608' for a bearing seat, '6x3' for a magnet pocket, a fan "
        "id for a fan_face. Null when the interface takes no component key.",
    )


class Blueprint(BaseModel):
    """A complete structured spec ready for DSL generation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    summary: str
    envelope_mm: BlueprintVec3 = Field(
        ...,
        description="Working bounding-box estimate the part should "
        "fit; the deterministic interpreter GROWS it to the real footprint if it is too small.",
    )
    max_envelope_mm: BlueprintVec3 | None = Field(
        default=None,
        description="User-stated HARD max bounding box if the intent gave one (else null). Set "
        "deterministically from intent, NOT chosen here; the interpreter never grows past it.",
    )
    print_orientation: BlueprintVec3 = Field(..., description="Which way is up when printed.")
    material: str
    min_wall_mm: float = Field(..., gt=0)
    max_overhang_deg: float = Field(..., gt=0, le=90)
    features: list[FeatureSpec] = Field(..., min_length=1)
    required_interfaces: list[InterfaceSpec] = Field(
        ...,
        description="Every interface the part must present. EMPTY ([]) for a BARE structural member "
        "(a strut / beam / brace / lattice with no mounts, holes, or hardware) -- do NOT invent an "
        "interface it does not need. Required field: always present, but may be the empty list.",
    )


__all__ = [
    "Blueprint",
    "BlueprintVec3",
    "FeatureSpec",
    "InterfaceSpec",
    "RequestedInterface",
]
