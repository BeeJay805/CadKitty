"""Computed-body interpreter (ADR-0019): a ``ComputedBodySpec`` (categorical menu pick +
coarse intent) -> a fully-RESOLVED ``PartProgram`` with a single ``computed_solid`` op.

This is the "code derives geometry" half: the LLM emits only the categorical
``ComputedOp``; every dimension is derived HERE from the coarse envelope + min_wall
(parallel to the composable plate interpreter's dim derivation). An infeasible request
(cross-section can't meet min_wall within the envelope) raises ``LayoutError`` up front —
the spec-completeness backstop (ADR-0019 Q2).

Convention: the member's long axis is local X, so the axis-wise bbox check stays consistent
(the builder extrudes the prism along X).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.blueprint.computed_sizing import derive_computed_envelope
from forge_mind.dsl.computed_ops import ComputedOp, FacetedStrutOp, LatticeOp
from forge_mind.dsl.primitives import Box, ComputedSolid, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.envelope import grow_envelope
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.spec import Pattern
from forge_mind.types import Vec3

# Mirror the builder's rectangle aspect (compiler/build_ops/computed.py).
_RECT_DEPTH_RATIO = 1.4
# Cross-section fills ~this fraction of the envelope's minor dimension (structural default).
_STRUT_FILL = 0.45

# Lattice density -> target cell interior as a multiple of the wall thickness (the mass lever).
_DENSITY_CELL_FACTOR = {"light": 6.0, "medium": 4.0, "dense": 2.5}
# A real opening must be at least this * wall_thickness wide, else the envelope is infeasible.
_MIN_CELL_FACTOR = 1.0


class ComputedBodySpec(BaseModel):
    """A standalone computed-geometry body: coarse intent (from the Blueprint) + the
    categorical op (the LLM's menu pick). Parallel to ``ComposablePlateSpec``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pattern: Literal[Pattern.COMPUTED_BODY] = Pattern.COMPUTED_BODY
    name: str = Field(..., min_length=1, max_length=128)
    envelope_mm: Vec3 | None = (
        None  # null -> DERIVE from size_class; explicit -> used as the estimate
    )
    max_envelope_mm: Vec3 | None = None  # user-stated HARD max (from intent); null = grow freely
    size_class: Literal["compact", "standard", "large"] = (
        "standard"  # coarse size when envelope is derived
    )
    material: str = "PLA"
    min_wall_mm: float = Field(..., gt=0)
    max_overhang_deg: float = Field(45.0, gt=0, le=90)
    op: ComputedOp


def _derive_faceted_strut(
    cross_section: Literal["rectangular", "hexagonal"], envelope: Vec3, min_wall: float
) -> ComputedSolid:
    """Derive the resolved prism (length + cross-section size) from coarse intent.

    length = envelope.x (member axis). The cross-section fills ~``_STRUT_FILL`` of the minor
    envelope dimension, floored so the THIN dimension still clears min_wall, capped so the
    LARGE dimension fits the envelope. Infeasible -> LayoutError.
    """
    length = envelope.x
    minor = min(envelope.y, envelope.z)
    # Thinnest cross-section dimension as a fraction of `size`: hex across-flats = size*sqrt(3),
    # rect depth = size*RATIO. Floor `size` so that thin dimension >= min_wall.
    thin_ratio = math.sqrt(3.0) if cross_section == "hexagonal" else _RECT_DEPTH_RATIO
    floor = min_wall / thin_ratio
    max_size = minor / 2.0  # largest cross-section extent (2*size) must fit the minor envelope
    if floor > max_size + 1e-9:
        raise LayoutError(
            f"faceted_strut {cross_section}: cross-section cannot meet min_wall {min_wall}mm "
            f"within envelope minor {minor}mm (need size>={floor:.2f}, max {max_size:.2f})"
        )
    size = min(max_size, max(floor, _STRUT_FILL * minor))
    return ComputedSolid(
        op_id="strut",
        mode=OpMode.ADD,
        kind="faceted_strut",
        section=cross_section,
        size=size,
        length=length,
    )


def _lattice_counts(span: float, t: float, density: str) -> int:
    """Cells along a lateral span for wall thickness ``t`` and coarse ``density``. Shrinks the
    count until the resulting cell interior is a real opening (>= _MIN_CELL_FACTOR * t)."""
    target_cell = _DENSITY_CELL_FACTOR[density] * t
    n = max(1, round((span - t) / (t + target_cell)))
    while n > 1 and (span - t * (n + 1)) / n < _MIN_CELL_FACTOR * t:
        n -= 1
    return n


def _derive_lattice(cell: str, density: str, envelope: Vec3, min_wall: float) -> list[Operation]:
    """Derive the lattice as a grid of straight vertical wall-members (the compiler unions them).

    Walls run the FULL height along the print Z (so every face is vertical -> no overhang) and
    are ``min_wall`` thick (axis-aligned, 90 deg -> the ray-cast reads true thickness; safe by
    construction). (nx+1) walls perpendicular to X + (ny+1) perpendicular to Y, outer walls flush
    with the envelope -> a watertight waffle slab with nx*ny vertical through-cells, lighter than
    the solid envelope. Infeasible (no real opening fits at min_wall) -> LayoutError.
    """
    t = min_wall
    nx = _lattice_counts(envelope.x, t, density)
    ny = _lattice_counts(envelope.y, t, density)
    cell_x = (envelope.x - t * (nx + 1)) / nx
    cell_y = (envelope.y - t * (ny + 1)) / ny
    if min(cell_x, cell_y) < _MIN_CELL_FACTOR * t:
        raise LayoutError(
            f"lattice {cell}: envelope {envelope.x:.0f}x{envelope.y:.0f}mm too small for a cell "
            f"at min_wall {min_wall}mm (cell interior {min(cell_x, cell_y):.1f}mm "
            f"< {_MIN_CELL_FACTOR * t:.1f}mm); use a larger envelope or a solid member."
        )
    ops: list[Operation] = []
    for i in range(nx + 1):
        cx = -envelope.x / 2 + t / 2 + i * (envelope.x - t) / nx
        ops.append(
            Box(
                op_id=f"lat_x_{i}",
                mode=OpMode.ADD,
                position=Vec3(x=cx, y=0.0, z=0.0),
                width=t,
                depth=envelope.y,
                height=envelope.z,
            )
        )
    for j in range(ny + 1):
        cy = -envelope.y / 2 + t / 2 + j * (envelope.y - t) / ny
        ops.append(
            Box(
                op_id=f"lat_y_{j}",
                mode=OpMode.ADD,
                position=Vec3(x=0.0, y=cy, z=0.0),
                width=envelope.x,
                depth=t,
                height=envelope.z,
            )
        )
    return ops


def _min_computed_envelope(op: ComputedOp, env: Vec3, min_wall: float) -> Vec3:
    """The smallest envelope that lets this op meet min_wall -- the grow target. Keeps the
    functional span (local X for a strut) and only floors the constrained dims, so a too-small LLM
    estimate grows to the minimum buildable body instead of being rejected."""
    if isinstance(op, FacetedStrutOp):
        thin_ratio = math.sqrt(3.0) if op.cross_section == "hexagonal" else _RECT_DEPTH_RATIO
        need = 2.0 * min_wall / thin_ratio  # the minor must host a 2*floor cross-section
        return Vec3(x=env.x, y=max(env.y, need), z=max(env.z, need))
    if isinstance(op, LatticeOp):
        need = 3.0 * min_wall  # one real cell at min_wall on the lateral footprint (X,Y); Z is full
        return Vec3(x=max(env.x, need), y=max(env.y, need), z=env.z)
    return env  # pragma: no cover - closed union


def interpret_computed_body(spec: ComputedBodySpec) -> PartProgram:
    op_spec = spec.op
    # DERIVE the envelope from the coarse size_class when the LLM did not size it (null); an explicit
    # envelope is honoured as the estimate. Either way GROW to meet min_wall + honour a user hard max
    # (instead of rejecting -> a silent fallback). A feasible estimate is unchanged -> byte-identical.
    base_env = (
        spec.envelope_mm
        if spec.envelope_mm is not None
        else derive_computed_envelope(spec.size_class)
    )
    eff_env = grow_envelope(
        base_env, _min_computed_envelope(op_spec, base_env, spec.min_wall_mm), spec.max_envelope_mm
    )
    ops: list[Operation]
    if isinstance(op_spec, FacetedStrutOp):
        ops = [_derive_faceted_strut(op_spec.cross_section, eff_env, spec.min_wall_mm)]
    elif isinstance(op_spec, LatticeOp):
        ops = _derive_lattice(op_spec.cell, op_spec.density, eff_env, spec.min_wall_mm)
    else:  # pragma: no cover - exhaustiveness; ComputedOp is a closed union
        raise LayoutError(f"unsupported computed op kind: {op_spec.kind!r}")
    return PartProgram(
        metadata=Metadata(name=spec.name, units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=ops,
        constraints=Constraints(
            min_wall_thickness=spec.min_wall_mm,
            max_overhang_deg=spec.max_overhang_deg,
            max_bbox=eff_env,
        ),
    )


__all__ = ["ComputedBodySpec", "interpret_computed_body"]
