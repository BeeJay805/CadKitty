"""Tendon-routability audit v1 (ADR-0124's R4 gate; ambition-sprint wave 3).

The tendon finger's atom-level claim: a continuous channel a 0.5 mm braided line can run
through. v1 measures the PER-PART half of that claim from the produced geometry -- the
same measured-not-trusted stance as the servo-fit audit:

  A brief feature DECLARING a tendon channel (name carries ``tendon``, or ``standard``
  carries ``TENDON``) must measure as a concave cylindrical channel with
    * diameter inside the routable band [1.0, 1.6] mm (0.5 mm Spectra + clearance;
      wider drops the tendon's fair-lead, narrower binds), and
    * an axial extent spanning >= 55% of the part's longest bounding dimension --
      a THROUGH-channel, not a dimple.

  Declared-but-unmeasured GATES (fail-closed, the desk_grommet law). Undeclared parts
  are out of scope. Assembly-level continuity across joints and the >= 3 mm bend radius
  at joint crossings are DEFERRED with a named trigger: they need mate frames, and land
  with the R4 finger's first assembly arm.

Flag: FORGE_TENDON_AUDIT=1 (default OFF; off = validate byte-identical).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from forge_mind.agentic.servo_fit_audit import Hole

_TENDON_D_BAND = (1.0, 1.6)  # 0.5 mm braid + fair-lead clearance
_THROUGH_SPAN_FRACTION = 0.55


def tendon_audit_enabled() -> bool:
    return os.environ.get("FORGE_TENDON_AUDIT") == "1"


@dataclass(frozen=True)
class TendonFinding:
    check: str  # "tendon_channel"
    message: str
    gates: bool


def _declares_tendon(feature: Any) -> bool:
    name = str(getattr(feature, "name", "")).lower()
    std = str(getattr(feature, "standard", "") or "").lower()
    return "tendon" in name or "tendon" in std


def match_tendon_channel(
    holes: list[Hole], part_longest_mm: float, extents_by_hole: list[tuple[float, float] | None]
) -> str | None:
    """None when a routable through-channel is measured; else what is missing.
    ``extents_by_hole`` pairs each hole with its measured axial (min, max) span or None."""
    in_band = [
        (h, ext)
        for h, ext in zip(holes, extents_by_hole, strict=True)
        if _TENDON_D_BAND[0] <= h.diameter <= _TENDON_D_BAND[1]
    ]
    if not in_band:
        return (
            f"no channel measured in d{_TENDON_D_BAND[0]}-{_TENDON_D_BAND[1]} "
            "(the routable band for a 0.5 mm tendon)"
        )
    if part_longest_mm <= 0:
        return "part extent unmeasurable -- cannot certify a through-channel"
    need = _THROUGH_SPAN_FRACTION * part_longest_mm
    for _h, ext in in_band:
        if ext is None:
            continue
        span = abs(ext[1] - ext[0])
        if span >= need:
            return None
    return (
        f"channel found in-band but no measured axial span reaches "
        f"{_THROUGH_SPAN_FRACTION:.0%} of the part length ({need:.1f} mm) -- "
        "a dimple is not a through-channel"
    )


def tendon_findings(brief: Any, part: Any, part_longest_mm: float) -> list[TendonFinding]:
    """Findings for every part whose brief declares a tendon channel. Empty when nothing
    is declared or the flag is off. Declared-but-unmeasured GATES."""
    if not tendon_audit_enabled():
        return []
    declared = [f for f in (getattr(brief, "features", None) or []) if _declares_tendon(f)]
    if not declared:
        return []
    from forge_mind.validator.geometry import extract_cylinders

    holes: list[Hole] = []
    extents: list[tuple[float, float] | None] = []
    for c in extract_cylinders(part):
        if c.concave is not True:
            continue
        p, d = c.axis_point, c.axis_dir
        holes.append(
            Hole(
                point=(float(p[0]), float(p[1]), float(p[2])),
                axis=(float(d[0]), float(d[1]), float(d[2])),
                diameter=float(c.radius) * 2.0,
            )
        )
        ext = c.axial_extent
        extents.append((float(ext[0]), float(ext[1])) if ext is not None else None)
    miss = match_tendon_channel(holes, part_longest_mm, extents)
    name = str(getattr(declared[0], "name", "tendon_channel"))
    if miss is None:
        return [
            TendonFinding(
                check="tendon_channel",
                message=f"tendon audit: '{name}' routable through-channel measured present",
                gates=False,
            )
        ]
    return [
        TendonFinding(
            check="tendon_channel",
            message=(
                f"tendon GATE: '{name}' declares a tendon channel but {miss} -- "
                "a declared channel must be measured, not claimed"
            ),
            gates=True,
        )
    ]


__all__ = ["TendonFinding", "match_tendon_channel", "tendon_audit_enabled", "tendon_findings"]
