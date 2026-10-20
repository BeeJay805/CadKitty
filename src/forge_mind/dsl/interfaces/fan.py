"""Axial-fan mounting-face mechanical facts — single source of truth (bench-006).

Pure data: the nominal dimensions of a standard axial cooling fan's mounting face,
parameterised by fan size. A fan face is the structural sibling of a NEMA 17 face
(``dsl/interfaces/nema17.py``): four corner bolt holes on a square plus a central
bore. The differences are exactly three nominals -- the bolt square is the fan's
size-specific spacing (not the fixed 31 mm), and the central feature is an *airflow*
opening (a required minimum), not a fixed shaft clearance.

Like ``nema17.py``, this module sits in ``dsl/interfaces`` (data, no behaviour) so
both the future placement layer and the validator checker can read the same numbers
and never drift. The screw clearance reuses the M3 fact from ``nema17.py``
(wrap-don't-delete, ADR-0007): a fan's corner holes are M3 on the 40 / 60 mm sizes.

Sizes from the axial-fan industry standard (40 mm -> 32 mm bolt spacing, 60 mm ->
50 mm spacing; both M3). The airflow bore is the designed *open* diameter the part
should leave clear so the fan is not occluded; bigger is fine, so the checker treats
it as a minimum.
"""

from __future__ import annotations

from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM

# The fan corner bolts are M3 on the 40/60 mm sizes -> reuse the canonical fact
# (one literal, no duplication). Re-exported under a fan-facing name for clarity.
FAN_BOLT_CLEARANCE_RADIUS_MM = M3_CLEARANCE_RADIUS_MM
"""Clearance-hole radius for a fan's M3 corner bolts (reuses the NEMA17 M3 fact)."""


# Per-size nominal facts: (bolt_square_mm, airflow_bore_radius_mm). The airflow
# radius is the designed open radius (a minimum the part must meet or exceed).
FAN_40MM_BOLT_SQUARE_MM = 32.0
"""40 mm axial fan: center-to-center spacing of the 4 corner bolts."""

FAN_40MM_AIRFLOW_BORE_RADIUS_MM = 19.0
"""40 mm axial fan: designed open airflow radius (~38 mm dia opening)."""

FAN_60MM_BOLT_SQUARE_MM = 50.0
"""60 mm axial fan: center-to-center spacing of the 4 corner bolts."""

FAN_60MM_AIRFLOW_BORE_RADIUS_MM = 28.0
"""60 mm axial fan: designed open airflow radius (~56 mm dia opening)."""


__all__ = [
    "FAN_40MM_AIRFLOW_BORE_RADIUS_MM",
    "FAN_40MM_BOLT_SQUARE_MM",
    "FAN_60MM_AIRFLOW_BORE_RADIUS_MM",
    "FAN_60MM_BOLT_SQUARE_MM",
    "FAN_BOLT_CLEARANCE_RADIUS_MM",
]
