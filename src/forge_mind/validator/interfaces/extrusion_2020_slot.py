"""T-slot extrusion slot interface (registry-driven; ADR-0017).

Verifies the part can bolt to a T-slot extrusion via two T-nuts: two clearance holes
along the mating normal, spaced to land in the slot (default 20mm apart). The bolt
clearance radius is read from the declaration's ``params`` (the ExtrusionComponent feeds
M6 for 3030), falling back to the shared M5 fact when absent -- so a bare (legacy) 2020
declaration is BYTE-IDENTICAL, and a 3030 certifies its M6 pair. Spacing tolerance is
generous because T-nuts slide in the channel.

(The class keeps its ``Extrusion2020SlotInterface`` name + the ``extrusion_2020_slot``
InterfaceType value for byte-equivalence; the cosmetic rename to ``extrusion_slot`` is
deferred, ADR-0017 Q7.)
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.validator.geometry import CylinderFeature, planar_distance
from forge_mind.validator.interfaces.base import (
    BaseInterface,
    InterfaceContext,
    holes_in_region,
    holes_matching,
    region_radius_for_pair,
)
from forge_mind.validator.interfaces.m5_tnut import M5_RADIUS_TOL
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

# M5 radius from the shared fact (single source with placement); the M5 radius
# tolerance is reused from m5_tnut, and the slot spacing policy below is local.
M5_CLEARANCE_RADIUS = M5_CLEARANCE_RADIUS_MM
DEFAULT_SPACING = 20.0
SPACING_TOL = 2.0  # T-nuts slide in the slot, so spacing is forgiving


class Extrusion2020SlotInterface(BaseInterface):
    type_name = InterfaceType.EXTRUSION_2020_SLOT.value

    def check(
        self,
        declaration: InterfaceDeclaration,
        cylinders: list[CylinderFeature],
        ctx: InterfaceContext | None = None,
    ) -> tuple[list[Check], list[Fix]]:
        checks: list[Check] = []
        fixes: list[Fix] = []
        prefix = f"interfaces.{declaration.name}"
        normal = declaration.normal
        axis = (normal.x, normal.y, normal.z)

        # Per-extrusion bolt clearance: read from params (the ExtrusionComponent feeds M6
        # for 3030); fall back to the M5 fact so a bare 2020 declaration is byte-identical.
        bolt_r = float(declaration.params.get("bolt_clearance_radius", M5_CLEARANCE_RADIUS))
        target = float(declaration.params.get("spacing", DEFAULT_SPACING))

        # Localize the hole count to THIS interface's region (ADR-0017): scope to holes within
        # the T-nut pair's half-spacing + slot tolerance (+margin) of the declared centre, so a
        # SECOND same-radius interface on the plate (a NEMA 23's M5 bolts, or another extrusion)
        # does not inflate the count -- while an in-tolerance loose-fit pair stays in-region so
        # the spacing check still fires. Single-interface / radius-distinct parts are unchanged
        # (the own 2 holes are in-region; no foreign same-radius holes exist). Mirrors M3 local.
        region_r = region_radius_for_pair(target, SPACING_TOL)
        holes = holes_in_region(
            holes_matching(cylinders, normal, bolt_r, M5_RADIUS_TOL),
            declaration.location,
            normal,
            region_r,
        )
        count_ok = len(holes) >= 2
        checks.append(
            Check(
                id=f"{prefix}.two_m5_holes",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if count_ok else CheckStatus.FAIL,
                message=(
                    "Two T-nut clearance holes present."
                    if count_ok
                    else f"Expected 2 T-nut clearance holes, found {len(holes)}."
                ),
                expected=f"2 holes, r~{bolt_r:.1f}mm",
                actual=f"{len(holes)} holes",
            )
        )
        if not count_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(f"Extrusion mount needs 2 T-nut clearance holes; found {len(holes)}."),
                    suggested_change={"hole_count": 2, "hole_radius": bolt_r},
                )
            )
            return checks, fixes

        # Spacing between the two closest-matching holes (both now scoped to this region).
        spacing = planar_distance(holes[0].axis_point, holes[1].axis_point, axis)
        spacing_ok = abs(spacing - target) <= SPACING_TOL
        checks.append(
            Check(
                id=f"{prefix}.m5_spacing",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if spacing_ok else CheckStatus.FAIL,
                message=(
                    "M5 holes are spaced for the extrusion slot."
                    if spacing_ok
                    else f"M5 hole spacing {spacing:.1f}mm is off the {target:.0f}mm target."
                ),
                expected=f"{target:.0f}mm",
                actual=f"{spacing:.1f}mm",
                tolerance=f"+/-{SPACING_TOL}mm",
            )
        )
        if not spacing_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=f"M5 hole spacing {spacing:.1f}mm should be ~{target:.0f}mm.",
                    suggested_change={"spacing": target},
                )
            )
        return checks, fixes


__all__ = ["Extrusion2020SlotInterface"]
