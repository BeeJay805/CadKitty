"""M5 T-nut bolt-clearance fact — the single source of truth (ADR-0006 Stage 3).

Pure data: the M5 clearance-hole radius. BOTH the placement helper
(``layout.interfaces.place_m5_pair``) and the validator checkers
(``m5_tnut``, and ``extrusion_2020_slot`` which checks a pair of them) read this
constant, so the geometry that gets built can't drift from the geometry that gets
validated. Mirrors ``dsl/interfaces/nema17.py``.

Nominal fact only. The checkers' radius tolerance and the extrusion-slot spacing
policy stay in the validator: those are acceptance policy, not the interface
dimension.
"""

from __future__ import annotations

# FLAGGED (A3): NOT routed to the standards registry. The existing 5.2 mm dia matches NO ISO 273
# class (close 5.3 / normal 5.5 / coarse 5.8) -- it is tighter than the close fit. Left UNCHANGED
# to preserve byte-equivalence of every existing M5 part (bench-001/003, the 102 catalog, NEMA 23);
# flagged for a human decision: adopt ISO 273 close (5.3) on NEW M5 parts, or keep 5.2 as an
# intentional snug T-nut clearance. The conformance check (A4) reports this divergence.
M5_CLEARANCE_RADIUS_MM = 2.6
"""Clearance-hole radius for an M5 T-nut bolt (5.2 mm dia -- non-standard, see the A3 flag above)."""

__all__ = ["M5_CLEARANCE_RADIUS_MM"]
