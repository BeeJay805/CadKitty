"""Repair blast-radius advisory (steal-list #10; filed by the L78 handoff queue).

A revision is supposed to apply the fix list to the working best -- but the measured
tail shows "revisions" that silently rebuild the whole part: across 119 consecutive
iteration pairs in the three most recent breadth arms (D1/A1/L2, 2026-07-22), the
legitimate-progression volume swing sits at p50 0.3% / p75 3.0% / p90 19.9%, while the
incident tail runs 79% to 2,684% (wall_hook A1 i3->i4: 4,311 -> 119,891 mm^3, a 27.8x
rebuild answering a fix list that asked for neither).

v1 is ADVISORY-ONLY and volume-only: a swing beyond the threshold prepends one line to
the next revision's feedback naming the measured change, conditioned on the fix list
("if the fixes didn't ask for bulk changes..."). It never gates, never affects keep.
Extents are deferred with a trigger: add them the first time an incident shows a
volume-neutral envelope blowout. Flag: FORGE_REPAIR_DIFF=1 (default OFF; flag off =
feedback byte-identical).
"""

from __future__ import annotations

import os

# Above the measured p90 of legitimate refinement (19.9%), below the incident tail
# (>=79%). From the 119-pair distribution, not feel.
_VOLUME_SWING_THRESHOLD_PCT = 40.0


def repair_diff_enabled() -> bool:
    return os.environ.get("FORGE_REPAIR_DIFF") == "1"


def repair_diff_advisory(
    anchor_volume_mm3: float | None, candidate_volume_mm3: float | None
) -> str | None:
    """One advisory line when the just-tried revision's volume swung past the measured
    threshold vs the working best; None otherwise (and always None when the flag is off
    or either volume is unmeasurable -- advisory, so silence over guessing)."""
    if not repair_diff_enabled():
        return None
    if not anchor_volume_mm3 or not candidate_volume_mm3 or anchor_volume_mm3 <= 0:
        return None
    swing_pct = abs(candidate_volume_mm3 - anchor_volume_mm3) / anchor_volume_mm3 * 100.0
    if swing_pct < _VOLUME_SWING_THRESHOLD_PCT:
        return None
    direction = "grew" if candidate_volume_mm3 > anchor_volume_mm3 else "shrank"
    return (
        f"BLAST-RADIUS NOTE: your last attempt {direction} the part's total volume by "
        f"{swing_pct:.0f}% vs the working version ({anchor_volume_mm3:,.0f} -> "
        f"{candidate_volume_mm3:,.0f} mm^3). If the fix list did not ask for bulk or "
        "envelope changes, you likely rebuilt more than the fixes required -- change "
        "ONLY what the fixes name and keep the rest of the working geometry."
    )


__all__ = ["repair_diff_advisory", "repair_diff_enabled"]
