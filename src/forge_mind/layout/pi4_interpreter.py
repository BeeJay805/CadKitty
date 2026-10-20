"""ElectronicsTrayInterpreter — ElectronicsTraySpec -> PartProgram (ADR-0005, ADR-0007).

The second LayoutSpec interpreter, alongside the l_bracket one. It reproduces the
exact geometry decisions the hand-authored open-tray reference
(examples/handwritten_pi4_case.py) proved against the pi4_mount + pi4_ports
checkers, parameterised by the spec, with every relational invariant enforced
BEFORE geometry exists.

Open tray only (no lid, no snap fits). Box + Cylinder primitives only.

Key invariants guaranteed:
- floor + perimeter walls >= min_wall; cavity = board + 2*clearance,
- 4 standoffs on the (centred) hole rectangle, each with a screw clearance hole
  and a material land >= min_wall (the ADR-0005 standoff-land resolution: an OD
  too small to give that land is rejected, not silently weakened),
- standoff-to-wall gap >= min_wall (rejects the tight-corner case ADR-0005 flagged
  rather than producing a fragile near-collision; merging is a future refinement),
- top-open port notches so their top edge never overhangs (prints floor-down),
- everything inside the envelope.

Centred-rectangle simplification: the 4 mounting holes are placed on a hole_dx x
hole_dy rectangle centred at the origin (the checkers verify rectangle size, not
absolute offset). The real Pi 4 rectangle is offset toward the SD end; matching
that exactly is deferred (it does not change validation).
"""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.dsl.components.board import BoardPortFact, PortEdge
from forge_mind.dsl.components.registry import get_board_component
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.envelope import grow_envelope
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.pi4_spec import ElectronicsTraySpec, PortPolicy, VentilationPolicy
from forge_mind.layout.rules import centered_rect_corners, edge_land
from forge_mind.types import Vec3

# Wall height above the standoff top that retains the board (open tray).
WALL_ABOVE_STANDOFF_MM = 10.0
# Where the port opening starts above the floor.
PORT_BOTTOM_ABOVE_FLOOR_MM = 2.0


@dataclass(frozen=True)
class _ResolvedBoard:
    """The board facts the geometry needs, from EITHER a component or legacy fields."""

    length_mm: float
    width_mm: float
    hole_dx_mm: float
    hole_dy_mm: float
    screw_radius_mm: float
    edge_inset_mm: float
    standoff_od_mm: float
    standoff_height_mm: float
    ports: tuple[BoardPortFact, ...]


def _derived_standoff_od(screw_radius_mm: float, min_wall_mm: float) -> float:
    """Tightest legal standoff OD given the screw clearance + min_wall.

    The post must keep a >= min_wall ring of material between the screw hole
    and the standoff's outer edge: ``OD/2 - screw_r >= min_wall``. A 1e-6 FP
    epsilon keeps the derived value strictly above the strict `< w` check
    (mirrors the bench-001/bench-003 derive helpers; ADR-0008).
    """
    return 2.0 * (screw_radius_mm + min_wall_mm) + 2e-6


def _resolve_board(spec: ElectronicsTraySpec) -> _ResolvedBoard:
    """Normalise the board source. ``board_component_id`` (canonical) wins; otherwise
    the legacy explicit ``board`` + ``standoffs`` + ``ports`` policy is used. Legacy
    STANDARD ports resolve to the Raspberry Pi 4 component's ports, so old Pi 4 specs
    produce byte-identical geometry.

    The standoff OD is resolved as ``max(picked_value, derived_minimum)`` -- the
    registry's per-board default (or the LLM's explicit legacy value) is treated
    as a FLOOR, and the interpreter grows it when ``spec.min_wall_mm`` demands a
    larger land. This is the bench-002 perturbation fix: at min_wall = 2 the
    derived minimum (~6.7) is below the registry's 7.0 so geometry is unchanged;
    at min_wall = 4 the derived minimum (10.7) wins and the standoff grows to
    keep the land rule structurally satisfied (ADR-0006: structure does what's
    structural).
    """
    w = spec.min_wall_mm
    if spec.board_component_id is not None:
        c = get_board_component(spec.board_component_id)
        screw_r = c.mount.screw_clearance_radius_mm
        return _ResolvedBoard(
            length_mm=c.length_mm,
            width_mm=c.width_mm,
            hole_dx_mm=c.mount.hole_dx_mm,
            hole_dy_mm=c.mount.hole_dy_mm,
            screw_radius_mm=screw_r,
            edge_inset_mm=c.mount.edge_inset_mm,
            standoff_od_mm=max(
                c.defaults.standoff_outer_diameter_mm, _derived_standoff_od(screw_r, w)
            ),
            standoff_height_mm=c.defaults.standoff_height_mm,
            ports=c.ports,
        )
    b, so = spec.board, spec.standoffs
    screw_r = so.screw_clearance_mm / 2.0
    legacy_ports = (
        get_board_component("raspberry_pi_4").ports if spec.ports is PortPolicy.STANDARD else ()
    )
    return _ResolvedBoard(
        length_mm=b.length_mm,
        width_mm=b.width_mm,
        hole_dx_mm=so.hole_dx_mm,
        hole_dy_mm=so.hole_dy_mm,
        screw_radius_mm=screw_r,
        edge_inset_mm=so.edge_inset_mm,
        standoff_od_mm=max(so.outer_diameter_mm, _derived_standoff_od(screw_r, w)),
        standoff_height_mm=so.height_mm,
        ports=legacy_ports,
    )


def _effective_clearance(spec: ElectronicsTraySpec, rb: _ResolvedBoard) -> float:
    """The clearance the interpreter actually uses: ``max(spec.clearance_mm,
    derived_minimum)``. The derived minimum is the smallest clearance that
    keeps the standoff-to-wall gap >= min_wall given the resolved standoff OD.

    The spec's value is treated as a FLOOR (a user's stated tightness preference);
    the structure grows it when the resolved standoff OD demands more room. Same
    bench-002 perturbation fix as the OD resolution above: when the registry's
    standoff OD grows under thicker walls, clearance must grow with it.
    """
    w = spec.min_wall_mm
    standoff_r = rb.standoff_od_mm / 2.0
    # gap_y = (board_width/2 + clearance) - (hole_dy/2 + standoff_r) must be >= w
    # The tight axis is whichever (length, width) makes the hole-to-edge offset
    # smaller; take the max needed across both axes so neither gap fails.
    needed_y = w + (rb.hole_dy_mm / 2.0 + standoff_r) - rb.width_mm / 2.0
    needed_x = w + (rb.hole_dx_mm / 2.0 + standoff_r) - rb.length_mm / 2.0
    derived_min = max(needed_x, needed_y, w) + 2e-6
    return max(spec.clearance_mm, derived_min)


def _check_feasible(spec: ElectronicsTraySpec, rb: _ResolvedBoard) -> None:
    """Reject specs that cannot be realised with legal lands / gaps / envelope."""
    w = spec.min_wall_mm
    screw_r = rb.screw_radius_mm
    standoff_r = rb.standoff_od_mm / 2.0
    clearance = _effective_clearance(spec, rb)

    # print_orientation is no longer a spec field: electronics_tray always
    # prints floor-down (+Z up) and the interpreter assumes it.

    problems: list[str] = []

    # Standoff land: post must clear the screw hole by >= min_wall.
    land = standoff_r - screw_r
    if land < w - 1e-9:
        problems.append(
            f"standoff OD {rb.standoff_od_mm}mm too small: land {land:.2f}mm < min_wall {w}mm "
            f"(need OD >= {2 * (screw_r + w):.1f}mm for the {2 * screw_r:.1f}mm screw hole)."
        )

    # Standoff-to-wall gap (the tight axis is the board width / hole_dy side).
    # Uses the EFFECTIVE clearance: if the LLM picked a too-small clearance for
    # the resolved standoff, _effective_clearance grows it; if it is already big
    # enough, this check is a no-op.
    gap_y = edge_land(rb.width_mm / 2.0 + clearance, rb.hole_dy_mm / 2.0, standoff_r)
    gap_x = edge_land(rb.length_mm / 2.0 + clearance, rb.hole_dx_mm / 2.0, standoff_r)
    if min(gap_x, gap_y) < w - 1e-9:
        problems.append(
            f"effective clearance {clearance:.2f}mm too small: a corner standoff sits "
            f"{min(gap_x, gap_y):.2f}mm from the wall (< min_wall {w}mm). Increase clearance "
            "or reduce the standoff OD."
        )

    # The hole rectangle must fit inside the board footprint with the declared inset.
    if rb.hole_dy_mm + 2 * rb.edge_inset_mm > rb.width_mm + 1e-6:
        problems.append("mounting-hole rectangle (with inset) does not fit the board width.")
    if rb.hole_dx_mm + 2 * rb.edge_inset_mm > rb.length_mm + 1e-6:
        problems.append("mounting-hole rectangle (with inset) does not fit the board length.")

    outer_l = rb.length_mm + 2 * (clearance + w)
    outer_w = rb.width_mm + 2 * (clearance + w)

    # Each connector-bank port must fit within its wall with a min_wall land at each end.
    for port in rb.ports:
        wall = outer_l if port.edge is PortEdge.LONG_PLUS_Y else outer_w
        if port.width_mm + 2 * w > wall + 1e-6:
            problems.append(
                f"port {port.name} ({port.width_mm:.0f}mm) does not fit the {wall:.0f}mm wall."
            )

    # NOTE (ADR-0035 A2): the envelope is deliberately NOT checked here. The case's outer
    # footprint DERIVES from board registry facts + clearance + wall, so ``spec.envelope_mm``
    # (an LLM estimate) is grown to fit in ``interpret_electronics_tray`` via the shared
    # ``grow_envelope`` -- the same grow-don't-reject policy as the composable and computed
    # interpreters. A user-stated ``max_envelope_mm`` is the one real limit; growth past it
    # raises an honest InfeasibleRequestError there. This interpreter was the one family that
    # missed that migration, which made every Pi-4 prompt infeasible at the production
    # blueprint default min_wall=4 (derived standoff OD -> ~105x76mm vs the ~100x70 estimate).

    if problems:
        raise LayoutError("infeasible electronics_tray spec: " + "; ".join(problems))


def _vent_ops(spec: ElectronicsTraySpec) -> list[Operation]:
    """Simple, deterministic floor slots. Empty for VentilationPolicy.NONE."""
    if spec.ventilation is VentilationPolicy.NONE:
        return []
    w = spec.min_wall_mm
    # Three central through-floor slots, clear of the standoffs, >= min_wall apart.
    ops: list[Operation] = []
    for i, y in enumerate((-8.0, 0.0, 8.0)):
        ops.append(
            Box(
                op_id=f"vent_{i}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=0.0, y=y, z=w / 2.0),
                width=30.0,
                depth=3.0,
                height=2.0 * w,  # through the floor
            )
        )
    return ops


def _port_op(
    port: BoardPortFact, outer_l: float, outer_w: float, w: float, port_z: float, port_h: float
) -> Box:
    """A top-open rectangular notch through the wall the port's edge names."""
    if port.edge is PortEdge.LONG_PLUS_Y:
        return Box(
            op_id=f"port_{port.name}",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=outer_w / 2.0 - w / 2.0, z=port_z),
            width=port.width_mm,
            depth=2.0 * w,
            height=port_h,
        )
    # PortEdge.SHORT_MINUS_X
    return Box(
        op_id=f"port_{port.name}",
        mode=OpMode.SUBTRACT,
        position=Vec3(x=-(outer_l / 2.0 - w / 2.0), y=0.0, z=port_z),
        width=2.0 * w,
        depth=port.width_mm,
        height=port_h,
    )


def interpret_electronics_tray(spec: ElectronicsTraySpec) -> PartProgram:
    """Deterministically realise an ElectronicsTraySpec as a validated PartProgram.

    Board-generic: the board (size, mount, screw, standoff defaults, ports) is
    resolved from ``board_component_id`` when set, else from the legacy explicit
    fields (ADR-0007). Raises LayoutError if the spec is infeasible (checked before
    any geometry).
    """
    rb = _resolve_board(spec)
    _check_feasible(spec, rb)

    w = spec.min_wall_mm
    screw_r = rb.screw_radius_mm
    standoff_r = rb.standoff_od_mm / 2.0
    clearance = _effective_clearance(spec, rb)

    floor_top = w
    standoff_top = floor_top + rb.standoff_height_mm
    case_h = standoff_top + WALL_ABOVE_STANDOFF_MM
    inner_l = rb.length_mm + 2 * clearance
    inner_w = rb.width_mm + 2 * clearance
    outer_l = inner_l + 2 * w
    outer_w = inner_w + 2 * w

    # Grow-don't-reject (ADR-0035 A2): the outer footprint is a COMPUTED CONSTRAINT (board facts
    # + clearance + wall), the spec envelope an LLM estimate. Grow the envelope to fit; a
    # user-stated hard max is honoured inside grow_envelope (honest refusal, never silent
    # growth past stated intent). An envelope that already fits comes back unchanged, so every
    # previously-feasible spec stays byte-identical.
    eff_env = grow_envelope(
        spec.envelope_mm,
        Vec3(x=outer_l, y=outer_w, z=case_h),
        spec.max_envelope_mm,
    )

    ops: list[Operation] = [
        Box(
            op_id="outer",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=case_h / 2.0),
            width=outer_l,
            depth=outer_w,
            height=case_h,
        ),
        Box(
            op_id="cavity",
            mode=OpMode.SUBTRACT,
            position=Vec3(x=0.0, y=0.0, z=floor_top + (case_h - floor_top) / 2.0),
            width=inner_l,
            depth=inner_w,
            height=case_h - floor_top,
        ),
    ]

    corners = centered_rect_corners(rb.hole_dx_mm, rb.hole_dy_mm)
    for tag, cx, cy in corners:
        ops.append(
            Cylinder(
                op_id=f"standoff_{tag}",
                mode=OpMode.ADD,
                position=Vec3(x=cx, y=cy, z=floor_top + rb.standoff_height_mm / 2.0),
                radius=standoff_r,
                height=rb.standoff_height_mm,
            )
        )
    for tag, cx, cy in corners:
        ops.append(
            Cylinder(
                op_id=f"screw_{tag}",
                mode=OpMode.SUBTRACT,
                position=Vec3(x=cx, y=cy, z=floor_top + (rb.standoff_height_mm + 1.0) / 2.0),
                radius=screw_r,
                height=rb.standoff_height_mm + 1.0,  # blind at the floor, open into the cavity
            )
        )

    ops.extend(_vent_ops(spec))

    if rb.ports:
        port_bottom = floor_top + PORT_BOTTOM_ABOVE_FLOOR_MM
        port_top = case_h + w  # reaches above the wall top -> open notch, no overhang
        port_z = (port_bottom + port_top) / 2.0
        port_h = port_top - port_bottom
        ops.extend(_port_op(p, outer_l, outer_w, w, port_z, port_h) for p in rb.ports)

    interfaces = [
        InterfaceDeclaration(
            name="pi4_mount",
            interface_type=InterfaceType.PI4_MOUNT,
            location=Vec3(x=0.0, y=0.0, z=standoff_top),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
            # Declare the actual mount built so the checker validates THIS board's
            # rectangle + screw size (board-generic; defaults to the Pi 4 facts).
            params={
                "hole_dx": rb.hole_dx_mm,
                "hole_dy": rb.hole_dy_mm,
                "screw_radius": screw_r,
            },
        ),
    ]
    # Declare the ports interface ONLY when the board actually has ports: a closed
    # tray (e.g. a generic PCB) must not be forced through Pi 4 port validation.
    if rb.ports:
        interfaces.append(
            InterfaceDeclaration(
                name="pi4_ports",
                interface_type=InterfaceType.PI4_PORTS,
                location=Vec3(x=0.0, y=0.0, z=case_h / 2.0),
                normal=Vec3(x=0.0, y=1.0, z=0.0),
            )
        )

    # Constant +Z: electronics_tray always prints floor-down.
    metadata = Metadata(name=spec.name, units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0))
    constraints = Constraints(
        min_wall_thickness=w,
        max_overhang_deg=45.0,  # standard FDM no-support threshold
        max_bbox=eff_env,  # the grown envelope; identical to spec.envelope_mm when no growth
    )
    return PartProgram(
        metadata=metadata, operations=ops, interfaces=interfaces, constraints=constraints
    )


# Backward-compatible alias (ADR-0007 Stage E): old code and tests call interpret_pi4.
interpret_pi4 = interpret_electronics_tray


__all__ = ["interpret_electronics_tray", "interpret_pi4"]
