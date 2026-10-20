"""Raspberry Pi 4 mount mechanical facts — the single source of truth (ADR-0007 Stage B).

Pure data: the nominal Pi 4 Model B mounting-hole geometry. The ``ElectronicsTraySpec``
``StandoffLayout`` DEFAULTS and the validator ``pi4_mount`` checker both resolve to
these constants, so the mounting rectangle and the M2.5 clearance the spec defaults
to and the checker validates against can't drift. Mirrors ``dsl/interfaces/nema17.py``
and ``m5.py``.

Pi 4 differs from NEMA 17 in one way worth noting: its mount geometry is
*spec-parameterized* (``StandoffLayout`` exposes ``hole_dx_mm``/``hole_dy_mm``/
``screw_clearance_mm`` as overridable fields), so the interpreter reads the spec
(which DEFAULTS to these facts) rather than a ``place_*()`` helper. There is no
``place_pi4_mount`` because the placement reads spec-parameterized values, not these
constants directly -- a helper would remove no duplication.

Canonical fastener form is RADIUS (ADR-0007 Q1): ``M25_CLEARANCE_RADIUS_MM`` is the
one literal; the diameter the human-facing spec field carries is derived from it.
Nominal facts only -- the checker's acceptance tolerances (``RECT_TOL``,
``M25_RADIUS_TOL``, ...) stay validator policy.
"""

from __future__ import annotations

from forge_mind.dsl.standards.fasteners import clearance_hole
from forge_mind.dsl.standards.framework import Fit

HOLE_RECT_DX_MM = 58.0
"""Mounting-hole rectangle span along X (Pi 4 Model B)."""

HOLE_RECT_DY_MM = 49.0
"""Mounting-hole rectangle span along Y (Pi 4 Model B)."""

# Routed to the standards registry (A3): the existing 2.7 mm dia IS the ISO 273 close-fit M2.5
# clearance, so this derives from it byte-identically (2.7 / 2 = 1.35).
M25_CLEARANCE_RADIUS_MM = clearance_hole("M2.5", Fit.CLOSE) / 2.0
"""Clearance-hole radius for an M2.5 screw (ISO 273 close, 2.7 mm dia) -- the canonical fastener fact."""

M25_CLEARANCE_DIAMETER_MM = 2.0 * M25_CLEARANCE_RADIUS_MM
"""M2.5 clearance diameter, derived from the canonical radius (the spec field is a diameter)."""

__all__ = [
    "HOLE_RECT_DX_MM",
    "HOLE_RECT_DY_MM",
    "M25_CLEARANCE_DIAMETER_MM",
    "M25_CLEARANCE_RADIUS_MM",
]
