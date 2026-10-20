"""Raspberry Pi 4 mounting interface.

A valid pi4_mount presents 4 M2.5 clearance holes on a 58 x 49 mm rectangle (the
Pi 4 Model B hole pattern), each concentric inside a standoff post with at least
``min_wall`` of material land around the hole. Cylinder-based, like nema17_face,
but the pattern is a rectangle (two distinct side lengths) and we additionally
verify the standoff land that ADR-0005 flagged as the feasibility risk.
"""

from __future__ import annotations

import math

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.pi4 import HOLE_RECT_DX_MM, HOLE_RECT_DY_MM, M25_CLEARANCE_RADIUS_MM
from forge_mind.validator.geometry import CylinderFeature, planar_distance
from forge_mind.validator.interfaces.base import BaseInterface, InterfaceContext, holes_matching
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

# Nominal Pi 4 mount facts come from the shared interface definition -- the SAME
# constants the ElectronicsTraySpec (Pi 4 case) defaults resolve to, so validation
# can't drift from what the spec defaults build. Tolerances below are validator policy.
M25_CLEARANCE_RADIUS = M25_CLEARANCE_RADIUS_MM
RECT_DX = HOLE_RECT_DX_MM
RECT_DY = HOLE_RECT_DY_MM
M25_RADIUS_TOL = 0.35  # accepts 2.0-3.4mm dia
RECT_TOL = 0.5
_CONCENTRIC_TOL = 0.6  # in-plane distance for a post to count as "around" a hole
_DEFAULT_MIN_WALL = 2.0


class Pi4MountInterface(BaseInterface):
    type_name = InterfaceType.PI4_MOUNT.value

    def check(
        self,
        declaration: InterfaceDeclaration,
        cylinders: list[CylinderFeature],
        ctx: InterfaceContext | None = None,
    ) -> tuple[list[Check], list[Fix]]:
        checks: list[Check] = []
        fixes: list[Fix] = []
        normal = declaration.normal
        axis = (normal.x, normal.y, normal.z)
        prefix = f"interfaces.{declaration.name}"
        min_wall = ctx.min_wall_mm if (ctx and ctx.min_wall_mm) else _DEFAULT_MIN_WALL

        # Board-specific nominals come from the interface declaration (the interpreter
        # passes the actual mount it built); fall back to the Pi 4 facts. The tolerances
        # below stay validator policy. This lets a second rectangular board validate
        # through the same checker without a registry (ADR-0007 Stage C; same move as
        # extrusion_2020_slot reading its spacing from declaration.params).
        rect_dx = float(declaration.params.get("hole_dx", RECT_DX))
        rect_dy = float(declaration.params.get("hole_dy", RECT_DY))
        screw_radius = float(declaration.params.get("screw_radius", M25_CLEARANCE_RADIUS))

        holes = holes_matching(cylinders, normal, screw_radius, M25_RADIUS_TOL)

        count_ok = len(holes) == 4
        checks.append(
            Check(
                id=f"{prefix}.mount_holes_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if count_ok else CheckStatus.FAIL,
                message=(
                    "Found the 4 M2.5 mounting holes."
                    if count_ok
                    else f"Expected 4 M2.5 mounting holes along the normal, found {len(holes)}."
                ),
                expected=f"4 holes, r~{screw_radius:.2f}mm",
                actual=f"{len(holes)} holes",
            )
        )
        if not count_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=f"Pi 4 mount needs exactly 4 M2.5 clearance holes; found {len(holes)}.",
                    suggested_change={"hole_count": 4, "hole_radius": screw_radius},
                )
            )
            return checks, fixes

        # The 4 holes must form a 58 x 49 rectangle: sorted pairwise distances are
        # [49, 49, 58, 58, diag, diag].
        pts = [h.axis_point for h in holes]
        dists = sorted(
            planar_distance(pts[i], pts[j], axis) for i in range(4) for j in range(i + 1, 4)
        )
        short_side = min(rect_dx, rect_dy)
        long_side = max(rect_dx, rect_dy)
        diag = math.hypot(rect_dx, rect_dy)
        short_ok = abs(dists[0] - short_side) <= RECT_TOL and abs(dists[1] - short_side) <= RECT_TOL
        long_ok = abs(dists[2] - long_side) <= RECT_TOL and abs(dists[3] - long_side) <= RECT_TOL
        diag_ok = all(abs(d - diag) <= RECT_TOL * math.sqrt(2) for d in dists[4:])
        rect_ok = short_ok and long_ok and diag_ok
        checks.append(
            Check(
                id=f"{prefix}.hole_rectangle_58x49",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if rect_ok else CheckStatus.FAIL,
                message=(
                    f"Mounting holes form the {rect_dx:.0f}x{rect_dy:.0f} mm rectangle."
                    if rect_ok
                    else f"Holes do not form a {rect_dx:.0f}x{rect_dy:.0f} rectangle "
                    f"(sides {[round(d, 1) for d in dists[:4]]})."
                ),
                expected=f"{rect_dx:.0f}x{rect_dy:.0f}mm",
                actual=f"sides {[round(d, 1) for d in dists[:4]]}",
                tolerance=f"+/-{RECT_TOL}mm",
            )
        )
        if not rect_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=f"Mounting holes are not a {rect_dx:.0f}x{rect_dy:.0f} rectangle.",
                    suggested_change={"rect_dx": rect_dx, "rect_dy": rect_dy},
                )
            )

        # Each hole must sit inside a standoff post with >= min_wall land. The post
        # is a larger concentric cylinder along the same axis.
        worst_land = math.inf
        missing = 0
        for h in holes:
            posts = [
                c
                for c in cylinders
                if c.axis_is_parallel_to(axis)
                and c.radius > h.radius + 1e-6
                and planar_distance(c.axis_point, h.axis_point, axis) <= _CONCENTRIC_TOL
            ]
            if not posts:
                missing += 1
                continue
            land = min(posts, key=lambda c: c.radius).radius - h.radius
            worst_land = min(worst_land, land)

        land_ok = missing == 0 and worst_land >= min_wall - 1e-6
        measured = "no standoff" if missing else f"{worst_land:.2f} mm"
        checks.append(
            Check(
                id=f"{prefix}.standoff_land",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if land_ok else CheckStatus.FAIL,
                message=(
                    "Each screw hole has a standoff with adequate material land."
                    if land_ok
                    else f"Standoff land {measured} is below the required {min_wall:.2f} mm "
                    "(or a standoff is missing)."
                ),
                expected=f">= {min_wall:.2f} mm land",
                actual=measured,
            )
        )
        if not land_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"Standoff material land is below {min_wall:.2f}mm. Increase the standoff "
                        "outer diameter so it clears the screw hole by at least min_wall."
                    ),
                    suggested_change={"min_land": min_wall},
                )
            )

        return checks, fixes


__all__ = ["Pi4MountInterface"]
