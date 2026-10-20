"""Deterministic catalog enumeration of composable plate family SHAPES (ADR-0016 Q1).

The composable spec is a small tuple of bounded categoricals, so the family space is
finite and enumerable in full -- no LLM proposal, no out-of-vocabulary risk. Motors +
extrusions are enumerated by component id (ADR-0017), the registry multiplier. Each shape
is pruned PRE-TRIAL (``$0``, no LLM) by the interpreter's own feasibility; with the
extrusion checker now localized (ADR-0017) there is no oracle-level exclusion -- dual-
extrusion and same-M5 motor+extrusion are certifiable. (The L-bracket stays a single legacy
NEMA 17 + 2020 combo: component hardware on it is deferred -- the gusset derive is coupled.)

A ``FamilyShape`` is the categorical menu choice (base + interface placements); the
geometry (dims/offsets) is derived deterministically here for the feasibility probe and,
later, by the engine for each trial -- the LLM only ever picks the menu.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from forge_mind.layout.composable_interpreter import (
    _LEGACY_EXTRUSION_ID,
    _LEGACY_MOTOR_ID,
    _derive_flat_geometry,
    interpret_composable_plate,
)
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    BaseDimsSpec,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.interpreter import LayoutError
from forge_mind.types import Vec3

# Representative sizing for the deterministic feasibility probe + engine-derived geometry.
_PROBE_MIN_WALL_MM = 4.0
_PROBE_MATERIAL = "PLA"


@dataclass(frozen=True)
class Placement:
    """A categorical interface placement (no geometry)."""

    ref: InterfaceRef
    face: Face
    component_id: str | None = None

    def label(self) -> str:
        c = f":{self.component_id}" if self.component_id else ""
        return f"{self.ref.value}@{self.face.value}{c}"


@dataclass(frozen=True)
class FamilyShape:
    """A base archetype + the categorical interface placements -- one candidate family."""

    base: BaseArchetype
    placements: tuple[Placement, ...]

    @property
    def key(self) -> str:
        return (
            f"{self.base.value}["
            + ",".join(p.label() for p in sorted(self.placements, key=lambda p: p.label()))
            + "]"
        )

    def to_placements(self) -> list[InterfacePlacement]:
        return [
            InterfacePlacement(ref=p.ref, face=p.face, component_id=p.component_id)
            for p in self.placements
        ]


# Flat-plate placement options (the catalog the enumerator draws from). Motors + extrusions
# are now enumerated BY COMPONENT ID (ADR-0017), exactly like fans -- the registry multiplier.
# The legacy default ids (nema17 / ext_2020) map back to component_id=None so their catalog
# keys (and the engine's 5 controls) stay byte-stable; only the NEW hardware carries an id.
def _flat_options(motor_ids: list[str], ext_ids: list[str], fan_ids: list[str]) -> list[Placement]:
    opts: list[Placement] = []
    for mid in motor_ids:
        cid = None if mid == _LEGACY_MOTOR_ID else mid
        opts.append(Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP, cid))
        opts.append(Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_BOTTOM, cid))
    for eid in ext_ids:
        cid = None if eid == _LEGACY_EXTRUSION_ID else eid
        opts.append(Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM, cid))
        opts.append(Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_TOP, cid))
    for fid in fan_ids:
        opts.append(Placement(InterfaceRef.FAN_FACE, Face.PLATE_TOP, fid))
        opts.append(Placement(InterfaceRef.FAN_FACE, Face.PLATE_BOTTOM, fid))
    return opts


def enumerate_shapes(
    *,
    motor_ids: list[str] | None = None,
    ext_ids: list[str] | None = None,
    fan_ids: list[str] | None = None,
) -> list[FamilyShape]:
    """Every catalog family shape: flat_plate with 1-2 interfaces + the l_bracket combo.
    Deterministic, order-stable. NOT yet pruned for geometric feasibility (see
    ``feasible_shapes``). Since the extrusion checker is now localized (ADR-0017), dual-
    extrusion and same-M5 motor+extrusion ARE certifiable and therefore enumerated; only
    geometric infeasibility (the $0 prune) drops shapes now -- no oracle-level exclusion."""
    from forge_mind.dsl.components.registry import (
        known_extrusion_component_ids,
        known_fan_component_ids,
        known_motor_component_ids,
    )

    motor_ids = motor_ids if motor_ids is not None else sorted(known_motor_component_ids())
    ext_ids = ext_ids if ext_ids is not None else sorted(known_extrusion_component_ids())
    fan_ids = fan_ids if fan_ids is not None else sorted(known_fan_component_ids())
    shapes: list[FamilyShape] = []
    opts = _flat_options(motor_ids, ext_ids, fan_ids)

    # flat_plate: every single placement, and every unordered pair of distinct placements.
    for p in opts:
        shapes.append(FamilyShape(BaseArchetype.FLAT_PLATE, (p,)))
    for a, b in itertools.combinations(opts, 2):
        shapes.append(FamilyShape(BaseArchetype.FLAT_PLATE, (a, b)))

    # l_bracket_bend: the one supported combo, LEGACY NEMA 17 + 2020 only. Component hardware
    # on the L-bracket is DEFERRED (its gusset derive is NEMA17/bore-coupled -- ADR-0017), so
    # the L-bracket is NOT multiplied by the registries.
    shapes.append(
        FamilyShape(
            BaseArchetype.L_BRACKET_BEND,
            (
                Placement(InterfaceRef.NEMA17_FACE, Face.BASE_TOP),
                Placement(InterfaceRef.EXTRUSION_2020, Face.UPRIGHT_FRONT),
            ),
        )
    )
    return shapes


def derive_flat_geometry(
    shape: FamilyShape, min_wall_mm: float = _PROBE_MIN_WALL_MM
) -> tuple[BaseDimsSpec, list[float]]:
    """Engine-derived explicit dims + per-placement X offsets for a flat_plate shape -- the
    'code derives geometry' half of the thesis (the LLM only picks the menu). Delegates to
    the interpreter's shared derive (``_derive_flat_geometry``) so the engine probes the SAME
    geometry the interpreter builds. Returns (base_dims, offsets aligned to shape.placements)."""
    width, depth, offsets = _derive_flat_geometry(shape.to_placements(), min_wall_mm)
    return BaseDimsSpec(width_mm=width, depth_mm=depth), offsets


def build_spec(
    shape: FamilyShape,
    *,
    name: str,
    envelope_mm: Vec3,
    min_wall_mm: float = _PROBE_MIN_WALL_MM,
    material: str = _PROBE_MATERIAL,
) -> ComposablePlateSpec:
    """A concrete ComposablePlateSpec for a shape with engine-derived geometry (explicit
    dims/offsets for flat_plate; the l_bracket combo derives its own dims)."""
    if shape.base is BaseArchetype.FLAT_PLATE:
        base_dims, offsets = derive_flat_geometry(shape, min_wall_mm)
        interfaces = [
            InterfacePlacement(ref=p.ref, face=p.face, component_id=p.component_id, offset_mm=off)
            for p, off in zip(shape.placements, offsets, strict=True)
        ]
        return ComposablePlateSpec(
            name=name,
            envelope_mm=envelope_mm,
            material=material,
            min_wall_mm=min_wall_mm,
            base=shape.base,
            base_dims=base_dims,
            interfaces=interfaces,
        )
    return ComposablePlateSpec(
        name=name,
        envelope_mm=envelope_mm,
        material=material,
        min_wall_mm=min_wall_mm,
        base=shape.base,
        interfaces=shape.to_placements(),
    )


def is_feasible(shape: FamilyShape) -> bool:
    """Deterministic ($0) pre-trial feasibility: build a representative spec at a generous
    envelope and try to interpret it. An infeasible shape (e.g. a 40 mm fan whose bore
    can't keep its bolt land at 4 mm wall) raises LayoutError and is pruned."""
    spec = build_spec(shape, name="probe", envelope_mm=Vec3(x=400.0, y=400.0, z=40.0))
    try:
        interpret_composable_plate(spec)
    except LayoutError:
        return False
    return True


def feasible_shapes(
    *,
    motor_ids: list[str] | None = None,
    ext_ids: list[str] | None = None,
    fan_ids: list[str] | None = None,
) -> list[FamilyShape]:
    """The enumerated shapes that pass the deterministic feasibility prune ($0)."""
    return [
        s
        for s in enumerate_shapes(motor_ids=motor_ids, ext_ids=ext_ids, fan_ids=fan_ids)
        if is_feasible(s)
    ]


__all__ = [
    "FamilyShape",
    "Placement",
    "build_spec",
    "derive_flat_geometry",
    "enumerate_shapes",
    "feasible_shapes",
    "is_feasible",
]
