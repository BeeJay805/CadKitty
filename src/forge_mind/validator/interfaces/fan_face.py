"""Axial-fan mounting-face interface (bench-006).

A valid fan face presents:
- 4x M3 clearance holes at the corners of a square whose side is the fan's
  size-specific bolt spacing (32 mm for a 40 mm fan, 50 mm for a 60 mm fan),
  axis along the mating normal,
- a central airflow bore, axis along the normal, of at least the fan's spec
  opening radius (bigger is fine -- the bore must not occlude the fan).

This is the structural sibling of ``nema17_face`` -- "4 bolts on a square + a
central bore" -- parameterised instead of copied: the bolt square side and the
minimum airflow radius are read from the declaration's ``params`` (fed per fan
size by the future FanComponent), the screw clearance reuses the shared M3 fact,
and the square geometry is checked by the shared ``is_square_pattern`` helper.
Acceptance tolerances are validator policy and stay here.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.fan import (
    FAN_40MM_AIRFLOW_BORE_RADIUS_MM,
    FAN_40MM_BOLT_SQUARE_MM,
    FAN_BOLT_CLEARANCE_RADIUS_MM,
)
from forge_mind.validator.geometry import CylinderFeature
from forge_mind.validator.interfaces.base import (
    BaseInterface,
    InterfaceContext,
    holes_in_region,
    holes_matching,
    is_square_pattern,
    region_radius_for_square,
)
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

# Nominal fan-bolt clearance comes from the shared fact (same constant the
# placement layer builds from). Acceptance tolerances below are validator policy.
FAN_BOLT_CLEARANCE_RADIUS = FAN_BOLT_CLEARANCE_RADIUS_MM
M3_RADIUS_TOL = 0.35  # accepts ~2.5-3.9 mm dia clearance (a fan's M3 hole is ~3.4 mm dia)
SQUARE_TOL = 0.5  # bolt-square side tolerance (mm)
AIRFLOW_RADIUS_TOL = 0.5  # a bore this much under the spec opening still counts as "present"


class FanFaceInterface(BaseInterface):
    """Checks a fan mounting face: 4-bolt square (size from params) + airflow bore."""

    type_name = InterfaceType.FAN_FACE.value

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

        # Per-fan-size nominals: read from the declaration params (the FanComponent
        # feeds these); fall back to the 40 mm facts so a bare declaration still
        # checks against a real fan.
        bolt_square = float(declaration.params.get("bolt_square", FAN_40MM_BOLT_SQUARE_MM))
        min_airflow_r = float(
            declaration.params.get("airflow_bore_radius", FAN_40MM_AIRFLOW_BORE_RADIUS_MM)
        )

        # Localize to THIS fan's region (ADR-0015): scope to holes within the bolt
        # square's corner radial (+margin) of the declared centre, so a second M3
        # interface on the same plate does not inflate the count. Single-interface /
        # mixed-radius parts are unchanged.
        region_r = region_radius_for_square(bolt_square)
        bolts = holes_in_region(
            holes_matching(cylinders, normal, FAN_BOLT_CLEARANCE_RADIUS, M3_RADIUS_TOL),
            declaration.location,
            normal,
            region_r,
        )

        # Check 1: exactly 4 M3 corner bolt holes.
        count_ok = len(bolts) == 4
        checks.append(
            Check(
                id=f"{prefix}.bolt_holes_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if count_ok else CheckStatus.FAIL,
                message=(
                    "Found the 4 M3 fan-corner holes."
                    if count_ok
                    else f"Expected 4 M3 corner holes along the face normal, found {len(bolts)}."
                ),
                expected=f"4 holes, r~{FAN_BOLT_CLEARANCE_RADIUS:.1f}mm",
                actual=f"{len(bolts)} holes",
            )
        )
        if not count_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"Fan face needs exactly 4 M3 corner holes; found {len(bolts)}. "
                        "Add or correct the bolt-pattern holes."
                    ),
                    suggested_change={"hole_count": 4, "hole_radius": FAN_BOLT_CLEARANCE_RADIUS},
                )
            )
            return checks, fixes  # can't check the square without 4 holes

        # Check 2: the 4 holes form the fan's bolt square (size from params).
        square_ok = is_square_pattern(bolts, axis, bolt_square, SQUARE_TOL)
        checks.append(
            Check(
                id=f"{prefix}.bolt_pattern_square",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if square_ok else CheckStatus.FAIL,
                message=(
                    f"Bolt holes form the {bolt_square:.0f}mm fan square."
                    if square_ok
                    else f"Bolt holes do not form a {bolt_square:.0f}mm square."
                ),
                expected=f"{bolt_square:.0f}mm square",
                actual="4 holes" if square_ok else "non-square / wrong-size pattern",
                tolerance=f"+/-{SQUARE_TOL}mm",
            )
        )
        if not square_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"Fan bolt pattern is not a {bolt_square:.0f}mm square; move the 4 holes "
                        f"to +/-{bolt_square / 2:.1f}mm in both in-plane axes."
                    ),
                    suggested_change={"bolt_square": bolt_square},
                )
            )

        # Check 3: central airflow bore present and large enough (localized to region).
        bores = holes_in_region(
            [
                c
                for c in cylinders
                if c.axis_is_parallel_to(axis) and c.radius >= (min_airflow_r - AIRFLOW_RADIUS_TOL)
            ],
            declaration.location,
            normal,
            region_r,
        )
        bore_ok = len(bores) >= 1
        checks.append(
            Check(
                id=f"{prefix}.airflow_bore_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if bore_ok else CheckStatus.FAIL,
                message=(
                    "Central airflow bore present and unoccluded."
                    if bore_ok
                    else f"Missing or too-small airflow bore (need r >= {min_airflow_r:.1f}mm)."
                ),
                expected=f"r >= {min_airflow_r:.1f}mm",
                actual=f"{len(bores)} matching bore(s)",
            )
        )
        if not bore_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"No central airflow bore >= {min_airflow_r:.1f}mm found; the fan would be "
                        "occluded. Add a central clearance bore at the face center."
                    ),
                    suggested_change={"airflow_bore_radius": min_airflow_r},
                )
            )

        return checks, fixes


__all__ = ["FanFaceInterface"]
