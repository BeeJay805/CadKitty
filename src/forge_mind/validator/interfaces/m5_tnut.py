"""M5 T-nut interface — a single M5 clearance hole sized for a T-nut bolt.

Reusable building block; ``extrusion_2020_slot`` checks a pair of these. As a
standalone declared interface it verifies one M5 clearance hole along the normal.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.validator.geometry import CylinderFeature
from forge_mind.validator.interfaces.base import BaseInterface, InterfaceContext, holes_matching
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

# Nominal M5 clearance comes from the shared interface fact -- the SAME constant
# the placement helper builds from, so what we validate can't drift from what gets
# built. The radius tolerance below is validator policy.
M5_CLEARANCE_RADIUS = M5_CLEARANCE_RADIUS_MM
M5_RADIUS_TOL = 0.4  # accepts 4.4-6.0mm dia


class M5TnutInterface(BaseInterface):
    type_name = InterfaceType.M5_TNUT.value

    def check(
        self,
        declaration: InterfaceDeclaration,
        cylinders: list[CylinderFeature],
        ctx: InterfaceContext | None = None,
    ) -> tuple[list[Check], list[Fix]]:
        prefix = f"interfaces.{declaration.name}"
        holes = holes_matching(cylinders, declaration.normal, M5_CLEARANCE_RADIUS, M5_RADIUS_TOL)
        ok = len(holes) >= 1
        check = Check(
            id=f"{prefix}.m5_clearance_present",
            tier=Tier.INTERFACES,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message=(
                "M5 T-nut clearance hole present."
                if ok
                else "No M5 clearance hole found along the interface normal."
            ),
            expected=f"1+ hole, r~{M5_CLEARANCE_RADIUS:.1f}mm",
            actual=f"{len(holes)} holes",
        )
        fixes: list[Fix] = []
        if not ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason="Add an M5 clearance hole (~5.2mm dia) along the interface normal.",
                    suggested_change={"hole_radius": M5_CLEARANCE_RADIUS},
                )
            )
        return [check], fixes


__all__ = ["M5TnutInterface"]
