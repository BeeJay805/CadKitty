"""M6 T-nut bolt-clearance fact — the single source of truth (ADR-0017).

Pure data: the M6 clearance-hole radius, for 30-series extrusions (3030) whose T-nuts
are M6 rather than the 2020/2040 M5. BOTH the placement helper and the validator's
extrusion checker read this constant, so built geometry can't drift from validated
geometry. Mirrors ``dsl/interfaces/m5.py``.

Nominal fact only (a medium/free-fit M6 clearance hole is ~6.6 mm dia). The checker's
radius tolerance and the slot-spacing policy stay in the validator: those are acceptance
policy, not the interface dimension.
"""

from __future__ import annotations

from forge_mind.dsl.standards.fasteners import clearance_hole
from forge_mind.dsl.standards.framework import Fit

# Routed to the standards registry (A3): the existing 6.6 mm dia IS the ISO 273 normal-fit M6
# clearance, so this derives from it byte-identically (6.6 / 2 = 3.3).
M6_CLEARANCE_RADIUS_MM = clearance_hole("M6", Fit.NORMAL) / 2.0
"""Clearance-hole radius for an M6 T-nut bolt (ISO 273 normal, 6.6 mm dia)."""

__all__ = ["M6_CLEARANCE_RADIUS_MM"]
