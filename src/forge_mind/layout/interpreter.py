"""LayoutInterpreter — LayoutSpec -> PartProgram, deterministically (ADR-0004).

This is the payoff of the LayoutSpec bet: the relational layout decisions the LLM
kept getting wrong (hole-to-edge land >= min_wall, holes on the exact bolt
pattern, the L printed base-down so nothing overhangs, everything inside the
envelope) are computed in code and guaranteed BEFORE any geometry exists. An
infeasible request (envelope too small to hold the interfaces with legal lands)
is rejected up front with ``LayoutError`` rather than emitting a part that fails
validation.

The emitted PartProgram is exactly the contract the compiler + validator already
consume; PartDSL stays the internal IR. The geometry mirrors the hand-authored
bench-001 ground truth (examples/handwritten_nema17.py), parameterised by the
spec.

Only the ``l_bracket_motor_mount`` pattern is implemented.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m5 import (
    M5_CLEARANCE_RADIUS_MM as M5_CLEARANCE_RADIUS,
)
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM as NEMA17_BOLT_PITCH,
)
from forge_mind.dsl.interfaces.nema17 import (
    M3_CLEARANCE_RADIUS_MM as M3_CLEARANCE_RADIUS,
)
from forge_mind.dsl.interfaces.nema17 import (
    SHAFT_BORE_RADIUS_MM as SHAFT_BORE_RADIUS,
)
from forge_mind.dsl.primitives import Box, Operation, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.interfaces import place_m5_pair, place_nema17_face
from forge_mind.layout.rules import edge_land
from forge_mind.layout.spec import BaseDims, LBracketMotorMountSpec, UprightDims
from forge_mind.types import Vec3

# NEMA17 and M5 facts now come from the shared interface definitions so placement
# and the validator checkers can't drift. The NEMA17 half-pitch is derived here.
NEMA17_HALF_PITCH = NEMA17_BOLT_PITCH / 2.0  # 15.5
# Outermost in-plane extent of the M3 cluster from the motor centre.
NEMA17_M3_HALF_EXTENT = NEMA17_HALF_PITCH + M3_CLEARANCE_RADIUS  # 17.1


class LayoutError(ValueError):
    """Raised when a LayoutSpec cannot be realised with legal lands / envelope."""


class InfeasibleRequestError(LayoutError):
    """A GENUINE physical infeasibility: the request cannot be built at the given min_wall (e.g. a
    fan's airflow bore vs its corner-bolt land) or cannot fit a user-stated HARD max envelope. A
    LayoutError subclass so existing reject paths (library / composition builders) still catch it;
    the pipeline treats it as a CLEAN REFUSAL (the reason is real), not a direct-DSL fallback that
    would build a wrong part."""


def derive_l_bracket_dims(spec: LBracketMotorMountSpec) -> tuple[BaseDims, UprightDims]:
    """Tightest legal (BaseDims, UprightDims) for an L-bracket given min_wall +
    the spec's m5_spacing + the shared NEMA17 / M5 / shaft-bore facts.

    This is the "structure derives layout" half of the bench-001 fix (the
    ADR-0008 pattern applied per-pattern). The L-bracket's base hosts the
    NEMA17 cluster + shaft bore; the upright hosts the 2 M5 holes. Each
    dim is set to "the smallest value that keeps every land >= min_wall":

    - base.width  = 2 * (max(M3 half-extent, shaft-bore radius) + min_wall)
                  = 2 * (max(17.1, 11.0) + min_wall) = 2 * (17.1 + min_wall)
    - base.depth  = same constraint (the M3 cluster also bounds Y)
    - upright.width  = 2 * (m5_spacing / 2 + M5 radius + min_wall)
    - upright.height = 2 * (M5 radius + min_wall)  (single centred M5 row)

    The bench-001 perturbation experiment showed the LLM gets these
    *almost* right under envelope pressure (off-by-0.2 mm fence-post errors
    on the thicker-wall variant). Computing them deterministically removes
    that failure class entirely.
    """
    w = spec.min_wall_mm
    m5_spacing = spec.extrusion_2020.m5_spacing_mm
    # Tiny epsilon keeps the derived value strictly above the boundary so the
    # _check_feasible strict `< w` comparison does not trip on FP rounding
    # (specifically, the upright-height boundary where 2*(2.6+4)/2 - 2.6
    # rounds to 3.999999... in IEEE 754 even though the algebra says 4.0).
    # 1e-6 mm is well below any meaningful mechanical tolerance.
    eps = 1e-6
    half = max(NEMA17_M3_HALF_EXTENT, SHAFT_BORE_RADIUS) + w + eps
    base = BaseDims(width_mm=2.0 * half, depth_mm=2.0 * half)
    up_half_w = m5_spacing / 2.0 + M5_CLEARANCE_RADIUS + w + eps
    up_half_h = M5_CLEARANCE_RADIUS + w + eps
    upright = UprightDims(width_mm=2.0 * up_half_w, height_mm=2.0 * up_half_h)
    return (base, upright)


def _resolve_l_bracket_dims(spec: LBracketMotorMountSpec) -> tuple[BaseDims, UprightDims]:
    """Pick (base, upright) either from explicit spec values or by deriving them.
    Explicit values win when given (backward compat with hand-authored fixtures).
    """
    derived_base, derived_up = derive_l_bracket_dims(spec)
    base = spec.base if spec.base is not None else derived_base
    upright = spec.upright if spec.upright is not None else derived_up
    return (base, upright)


def _check_feasible(spec: LBracketMotorMountSpec) -> None:
    """Reject specs that cannot be built with hole-to-edge lands >= min_wall and
    within the envelope. Every check is the relational invariant that the direct
    DSL path failed to satisfy reliably.
    """
    w = spec.min_wall_mm
    base, up = _resolve_l_bracket_dims(spec)
    m5_half = spec.extrusion_2020.m5_spacing_mm / 2.0

    # Print orientation is no longer a spec field: this pattern prints base-down
    # (+Z up) by construction, so the upright is a single vertical wall with no
    # overhang. No runtime check required since the LLM can't override it.

    problems: list[str] = []

    # NEMA17 M3 holes sit at +/-15.5 in both base axes; their outer edge must keep
    # a >= min_wall land to the base edge (the relational invariant, via edge_land).
    if edge_land(base.width_mm / 2.0, NEMA17_HALF_PITCH, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"base width {base.width_mm}mm too small: M3 hole-to-edge land < min_wall {w}mm "
            f"(half-pitch {NEMA17_HALF_PITCH} + M3 r {M3_CLEARANCE_RADIUS} + wall {w})."
        )
    if edge_land(base.depth_mm / 2.0, NEMA17_HALF_PITCH, M3_CLEARANCE_RADIUS) < w:
        problems.append(
            f"base depth {base.depth_mm}mm too small: M3 hole-to-edge land < min_wall {w}mm."
        )

    # Central shaft bore must keep a min_wall land to the base edges.
    if edge_land(min(base.width_mm, base.depth_mm) / 2.0, 0.0, SHAFT_BORE_RADIUS) < w:
        problems.append(
            f"base too small for the {2 * SHAFT_BORE_RADIUS}mm shaft bore + {w}mm land "
            f"(smaller base axis)."
        )

    # 2x M5 holes on the upright: half-spacing + M5 edge + land must fit the width;
    # a single centred row needs the M5 edge + land within the half-height.
    if edge_land(up.width_mm / 2.0, m5_half, M5_CLEARANCE_RADIUS) < w:
        problems.append(f"upright width {up.width_mm}mm too small for the M5 holes + {w}mm land.")
    if edge_land(up.height_mm / 2.0, 0.0, M5_CLEARANCE_RADIUS) < w:
        problems.append(f"upright height {up.height_mm}mm too small for the M5 row + {w}mm land.")

    # Envelope: assembled bbox must fit. X = widest plate, Y = base depth,
    # Z = upright height (the base sits within 0..wall, below the upright top).
    env = spec.envelope_mm
    bbox_x = max(base.width_mm, up.width_mm)
    bbox_y = base.depth_mm
    bbox_z = max(up.height_mm, w)
    if bbox_x > env.x + 1e-6:
        problems.append(f"part X {bbox_x}mm exceeds envelope X {env.x}mm.")
    if bbox_y > env.y + 1e-6:
        problems.append(f"part Y {bbox_y}mm exceeds envelope Y {env.y}mm.")
    if bbox_z > env.z + 1e-6:
        problems.append(f"part Z {bbox_z}mm exceeds envelope Z {env.z}mm.")

    if problems:
        raise LayoutError("infeasible l_bracket_motor_mount spec: " + "; ".join(problems))


def _gusset_op(spec: LBracketMotorMountSpec, ext_front_y: float) -> Box | None:
    """A central, overhang-safe box rib in the inner corner, clearing all holes.

    Always returns either a fitting Box (auto-include) or None (silently
    skip when there isn't room). The previous ``GussetPolicy`` field on
    ``LBracketMotorMountSpec`` was removed in ADR-0009 Task 2: the LLM
    picked ``required`` 23/30 times on the decontaminated bench-001
    prompt for "NEMA 17 stepper motor", trapping the interpreter against
    the derived (tightest legal) base depth. The interpreter now owns
    the decision; if a rib fits cleanly it gets added (stronger part for
    free), if it doesn't the part is built without one (still valid
    L-bracket geometry, same as ``gusset: none`` used to produce).
    """
    w = spec.min_wall_mm
    rib_width = 3.0 * w
    rib_height = 2.0 * w

    # The rib runs from the upright front face forward, but must stop a min_wall
    # land short of the shaft bore. Size its depth to the room available rather
    # than a fixed value, so a slightly shallower base still gets a (thinner) rib
    # instead of being skipped outright.
    max_depth = ext_front_y - (SHAFT_BORE_RADIUS + w)
    rib_depth = min(2.0 * w, max_depth)

    clears_bolts_x = rib_width / 2.0 < (NEMA17_HALF_PITCH - M3_CLEARANCE_RADIUS)
    fits_height = (w + rib_height) <= spec.envelope_mm.z + 1e-6
    if rib_depth < w or not clears_bolts_x or not fits_height:
        return None

    front_y = ext_front_y - rib_depth
    return Box(
        op_id="corner_gusset",
        mode=OpMode.ADD,
        position=Vec3(x=0.0, y=front_y + rib_depth / 2.0, z=w + rib_height / 2.0),
        width=rib_width,
        depth=rib_depth,
        height=rib_height,
    )


def interpret(spec: LBracketMotorMountSpec) -> PartProgram:
    """Deterministically realise a LayoutSpec as a validated PartProgram.

    Raises LayoutError if the spec is infeasible (checked before any geometry).
    """
    _check_feasible(spec)

    w = spec.min_wall_mm
    base, up = _resolve_l_bracket_dims(spec)

    rear_y = base.depth_mm / 2.0
    ext_y_center = rear_y - w / 2.0  # upright spans [rear_y - w, rear_y]
    ext_front_y = rear_y - w
    hole_z = up.height_mm / 2.0

    ops: list[Operation] = [
        Box(
            op_id="motor_flange",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=0.0, z=w / 2.0),
            width=base.width_mm,
            depth=base.depth_mm,
            height=w,
        ),
        Box(
            op_id="ext_flange",
            mode=OpMode.ADD,
            position=Vec3(x=0.0, y=ext_y_center, z=up.height_mm / 2.0),
            width=up.width_mm,
            depth=w,
            height=up.height_mm,
        ),
    ]

    gusset = _gusset_op(spec, ext_front_y)
    if gusset is not None:
        ops.append(gusset)

    # NEMA17 motor face: 4 M3 holes on the 31mm square + central shaft bore,
    # drilled +Z through the base. The interface helper owns the placement math
    # and reads the same facts the validator checker validates against.
    ops.extend(place_nema17_face(Vec3(x=0.0, y=0.0, z=w / 2.0), w))

    # 2x M5 T-nut clearance holes through the upright (drilled +Y). The interface
    # helper owns the placement and reads the same fact the extrusion/m5 checkers
    # validate against.
    ops.extend(
        place_m5_pair(
            Vec3(x=0.0, y=ext_y_center, z=hole_z),
            spec.extrusion_2020.m5_spacing_mm,
            w,
        )
    )

    interfaces = [
        InterfaceDeclaration(
            name=spec.nema17.name,
            interface_type=InterfaceType.NEMA17_FACE,
            location=Vec3(x=0.0, y=0.0, z=w / 2.0),
            normal=Vec3(x=0.0, y=0.0, z=1.0),
        ),
        InterfaceDeclaration(
            name=spec.extrusion_2020.name,
            interface_type=InterfaceType.EXTRUSION_2020_SLOT,
            location=Vec3(x=0.0, y=ext_y_center, z=hole_z),
            normal=Vec3(x=0.0, y=1.0, z=0.0),
            params={"spacing": spec.extrusion_2020.m5_spacing_mm},
        ),
    ]

    metadata = Metadata(
        name=spec.name,
        units="mm",
        # Constant +Z: the l_bracket pattern always prints base-down.
        print_orientation=Vec3(x=0.0, y=0.0, z=1.0),
    )
    constraints = Constraints(
        min_wall_thickness=w,
        max_overhang_deg=spec.max_overhang_deg,
        max_bbox=spec.envelope_mm,
    )

    return PartProgram(
        metadata=metadata,
        operations=ops,
        interfaces=interfaces,
        constraints=constraints,
    )


__all__ = ["InfeasibleRequestError", "LayoutError", "derive_l_bracket_dims", "interpret"]
