"""Shared deterministic layout rules (ADR-0006 Stage 2).

Small, pure geometry helpers extracted from the two proven LayoutSpec
interpreters -- ``interpreter.py`` (``l_bracket_motor_mount``) and
``pi4_interpreter.py`` (``pi4_case``) -- at the points where they were computing
the SAME relational invariant with duplicated arithmetic.

Extraction criterion (ADR-0006 Stage 2): a helper lands here only if BOTH
interpreters already use it. Single-pattern logic (port notches, vent slots, the
standoff post+screw assembly) stays in its interpreter until a second caller
appears -- abstraction is earned by a real second use, not anticipated.

Nothing here changes geometry or validator thresholds; each function is a
behaviour-preserving factoring of arithmetic the interpreters already ran. The
module imports only ``types`` (no interpreter/spec imports), so either
interpreter can depend on it without an import cycle and ``LayoutError`` stays
owned by the interpreter that raises it.
"""

from __future__ import annotations

from forge_mind.types import Vec3


def is_floor_down(orientation: Vec3) -> bool:
    """True if a print orientation points predominantly +Z (floor/base-down).

    Both patterns print +Z up so their tall feature -- the L's upright wall, the
    case's perimeter walls -- is a vertical wall with no overhang. The test is
    "+Z dominant": z is positive and at least as large in magnitude as x and y.
    Callers raise their own pattern-specific ``LayoutError`` on a False result.
    """
    return (
        orientation.z > 0
        and abs(orientation.z) >= abs(orientation.x)
        and abs(orientation.z) >= abs(orientation.y)
    )


def edge_land(half_extent: float, center_offset: float, feature_radius: float) -> float:
    """Signed distance from a centred circular feature's outer edge to a planar
    boundary at +/- ``half_extent`` from the same centre::

        edge_land = half_extent - (center_offset + feature_radius)

    This is the relational invariant the direct DSL path kept violating: a hole,
    bore, or post sitting near an edge must keep >= ``min_wall`` to it. Read the
    result as a material land (hole edge -> solid edge, ``l_bracket``) or an air
    gap (standoff post -> inner wall, ``pi4_case``) depending on what the boundary
    is -- the geometry is identical. Negative means the feature crosses the
    boundary.

    Returns a pure geometric quantity; the caller compares it to its own
    threshold (typically ``min_wall``) with its own tolerance and diagnostic
    message.
    """
    return half_extent - (center_offset + feature_radius)


def centered_rect_corners(dx: float, dy: float) -> list[tuple[str, float, float]]:
    """The four corners of a ``dx`` x ``dy`` rectangle centred at the origin, as
    ``(tag, x, y)`` in NW, NE, SW, SE order.

    Both interpreters place a 4-feature rectangular pattern at exactly these
    corners in exactly this order: the NEMA 17 31 mm bolt square (dx == dy) and
    the Pi 4 58 x 49 mm standoff/screw rectangle. The ``tag`` is the corner name;
    callers prefix it to build op ids (``motor_bolt_{tag}``, ``standoff_{tag}``,
    ``screw_{tag}``). The order is part of the contract -- it fixes the operation
    order the compiler consumes, so the emitted geometry is unchanged.
    """
    hx = dx / 2.0
    hy = dy / 2.0
    return [
        ("nw", -hx, hy),
        ("ne", hx, hy),
        ("sw", -hx, -hy),
        ("se", hx, -hy),
    ]


__all__ = ["centered_rect_corners", "edge_land", "is_floor_down"]
