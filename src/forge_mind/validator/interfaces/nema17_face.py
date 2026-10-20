"""Motor face interface (registry-driven; ADR-0017).

A valid motor face presents:
- 4x corner clearance holes on a square whose side is the motor's bolt pitch (31 mm for
  NEMA 17, 26 mm for NEMA 14, 47.14 mm for NEMA 23), axis along the mating normal,
- a central pilot-boss (register) clearance bore, axis along the normal.

This is the generalized form of the original NEMA-17-only checker: the bolt pitch, the
corner-bolt clearance radius, and the pilot-boss radius are read from the declaration's
``params`` (the MotorComponent feeds them), falling back to the shared NEMA 17 facts when
absent -- so a bare (legacy) declaration is BYTE-IDENTICAL to before (same check id +
status), and a NEMA 23 certifies its 47.14 mm M5 pattern, not 31 mm M3. The square check
id is templated from the pitch (``bolt_pattern_31mm_square`` for NEMA 17, by construction).
Acceptance tolerances stay here (validator policy).

(The class keeps its ``Nema17FaceInterface`` name + the ``nema17_face`` InterfaceType value
for byte-equivalence -- the value is serialized in the PartProgram; the cosmetic rename to
``motor_face`` is deferred, ADR-0017 Q7.)
"""

from __future__ import annotations

import math

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.nema17 import (
    BOLT_PITCH_MM,
    M3_CLEARANCE_RADIUS_MM,
    SHAFT_BORE_RADIUS_MM,
)
from forge_mind.validator.geometry import CylinderFeature, planar_distance
from forge_mind.validator.interfaces.base import (
    BaseInterface,
    InterfaceContext,
    holes_in_region,
    holes_matching,
    region_radius_for_square,
)
from forge_mind.validator.report import Check, CheckStatus, Fix, Tier

# Nominal NEMA 17 facts come from the shared interface definition -- the SAME constants the
# placement helper builds from, and the FALLBACK when a declaration carries no per-motor
# params (so a legacy NEMA 17 face stays byte-identical). The acceptance tolerances below
# are validator policy.
M3_CLEARANCE_RADIUS = M3_CLEARANCE_RADIUS_MM
BOLT_PITCH = BOLT_PITCH_MM
SHAFT_BORE_RADIUS = SHAFT_BORE_RADIUS_MM
M3_RADIUS_TOL = 0.35  # accepts 2.5-3.9mm dia clearance
PITCH_TOL = 0.5
SHAFT_BORE_MIN = 9.0
SHAFT_BORE_MAX = 14.0
# Bore acceptance window as offsets around the expected pilot radius, derived from the
# legacy absolute window so the NEMA 17 fallback reproduces [9.0, 14.0] exactly while a
# larger pilot (NEMA 23, r19.05) shifts the window with it.
_BORE_TOL_LOW = SHAFT_BORE_RADIUS - SHAFT_BORE_MIN  # 2.0
_BORE_TOL_HIGH = SHAFT_BORE_MAX - SHAFT_BORE_RADIUS  # 3.0


class Nema17FaceInterface(BaseInterface):
    type_name = InterfaceType.NEMA17_FACE.value

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

        # Per-motor nominals: read from the declaration params (the MotorComponent feeds
        # these for NEMA 14/23); fall back to the NEMA 17 facts so a bare declaration (the
        # legacy default) checks against NEMA 17 -- byte-identical (ADR-0017).
        bolt_pitch = float(declaration.params.get("bolt_pitch", BOLT_PITCH))
        bolt_r = float(declaration.params.get("bolt_clearance_radius", M3_CLEARANCE_RADIUS))
        pilot_r = float(declaration.params.get("pilot_bore_radius", SHAFT_BORE_RADIUS))
        bore_min = pilot_r - _BORE_TOL_LOW
        bore_max = pilot_r + _BORE_TOL_HIGH

        # Localize the hole count to THIS interface's region (ADR-0015): scope to holes
        # within the bolt pattern's corner radial (+margin) of the declared centre, so a
        # SECOND same-radius interface on the same plate does not inflate the count.
        # Single-interface / radius-distinct parts are unchanged.
        region_r = region_radius_for_square(bolt_pitch)
        bolts = holes_in_region(
            holes_matching(cylinders, normal, bolt_r, M3_RADIUS_TOL),
            declaration.location,
            normal,
            region_r,
        )

        # Check 1: exactly 4 corner bolt holes.
        count_ok = len(bolts) == 4
        checks.append(
            Check(
                id=f"{prefix}.bolt_holes_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if count_ok else CheckStatus.FAIL,
                message=(
                    "Found the 4 corner clearance holes."
                    if count_ok
                    else f"Expected 4 clearance holes along the face normal, found {len(bolts)}."
                ),
                expected=f"4 holes, r~{bolt_r:.1f}mm",
                actual=f"{len(bolts)} holes",
            )
        )
        if not count_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"Motor face needs exactly 4 clearance holes; found {len(bolts)}. "
                        "Add or correct the bolt-pattern holes."
                    ),
                    suggested_change={"hole_count": 4, "hole_radius": bolt_r},
                )
            )
            # Can't check the square without 4 holes.
            return checks, fixes

        # Check 2: the 4 holes form the motor's bolt square.
        pts = [b.axis_point for b in bolts]
        dists = sorted(
            planar_distance(pts[i], pts[j], axis) for i in range(4) for j in range(i + 1, 4)
        )
        sides = dists[:4]
        diagonals = dists[4:]
        expected_diag = bolt_pitch * math.sqrt(2)
        sides_ok = all(abs(s - bolt_pitch) <= PITCH_TOL for s in sides)
        diag_ok = all(abs(d - expected_diag) <= PITCH_TOL * math.sqrt(2) for d in diagonals)
        square_ok = sides_ok and diag_ok
        checks.append(
            Check(
                id=f"{prefix}.bolt_pattern_{bolt_pitch:.0f}mm_square",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if square_ok else CheckStatus.FAIL,
                message=(
                    f"Bolt holes form the {bolt_pitch:.0f}mm square pattern."
                    if square_ok
                    else f"Bolt holes do not form a {bolt_pitch:.0f}mm square (sides {[round(s, 1) for s in sides]})."
                ),
                expected=f"{bolt_pitch:.0f}mm square",
                actual=f"sides~{sum(sides) / 4:.1f}mm",
                tolerance=f"+/-{PITCH_TOL}mm",
            )
        )
        if not square_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"Bolt pattern is not a {bolt_pitch:.0f}mm square "
                        f"(measured sides {[round(s, 1) for s in sides]}). "
                        f"Move the 4 holes to +/-{bolt_pitch / 2:.1f}mm in both in-plane axes."
                    ),
                    suggested_change={"bolt_pitch": bolt_pitch},
                )
            )

        # Check 3: central pilot-boss bore present (localized to this interface's region).
        bores = holes_in_region(
            [
                c
                for c in cylinders
                # A pilot bore is a HOLE: a solid central boss (concave False) or an unmeasurable
                # face (None) must not satisfy it (audit H1, same fail-closed rule as holes_matching).
                if c.concave is True
                and c.axis_is_parallel_to(axis)
                and bore_min <= c.radius <= bore_max
            ],
            declaration.location,
            normal,
            region_r,
        )
        bore_ok = len(bores) >= 1
        checks.append(
            Check(
                id=f"{prefix}.shaft_bore_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if bore_ok else CheckStatus.FAIL,
                message=(
                    "Central pilot-boss clearance bore present."
                    if bore_ok
                    else f"Missing central pilot-boss clearance bore (~{pilot_r * 2:.0f}mm dia)."
                ),
                expected=f"r in [{bore_min:.1f}, {bore_max:.1f}]mm",
                actual=f"{len(bores)} matching bore(s)",
            )
        )
        if not bore_ok:
            fixes.append(
                Fix(
                    op_id=None,
                    reason=(
                        f"No central pilot-boss bore found; add a ~{pilot_r * 2:.0f}mm clearance "
                        "hole at the face center."
                    ),
                    suggested_change={"shaft_bore_radius": pilot_r},
                )
            )

        return checks, fixes


__all__ = ["Nema17FaceInterface"]
