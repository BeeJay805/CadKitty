"""Servo-fit audit (ADR-0124 deep-build; design: docs/research/2026-07-22-servo-fit-audit-design.md).

The PORT of the DSL-path mount-pattern idea into the agentic single-part path: a part
whose brief DECLARES an MG996R interface (a feature with ``standard`` naming MG996R)
must expose the cited hole pattern, measured from the produced geometry — never trusted
from the code or the critique. Servos are BOUGHT parts: the printed part mates to the
registry's cited (and now partially founder-verified) dimensions.

Patterns, frame-invariant (axis-perpendicular center distances, so placement/rotation
never matter):
  * HORN interface (feature name carries horn/hub): a concave center bore d8.1 +/- 0.3
    AND four concave screw holes d in [1.9, 2.6] with mutually parallel axes whose
    centers sit on a d14.0 +/- 0.8 circle. 14.0 is the FOUNDER-VERIFIED bolt circle
    (physical caliper 2026-07-22; the registry's prior 17.0 was sourced-and-wrong --
    the case study for why this audit measures instead of trusting).
  * TAB/FLANGE interface (feature name carries tab/flange/mount): four concave holes
    d in [4.0, 4.8] (the datasheet's own clearance band) with parallel axes whose
    pairwise axis-distance multiset matches the cited 49.5 x 10.0 rectangle
    ({10.0 x2, 49.5 x2, 50.47 x2} +/- 0.8).

FAIL-CLOSED (the desk_grommet law): a DECLARED interface whose pattern cannot be
measured in the geometry GATES -- an unmeasurable claim is a standing claim, not a
pass. Prose mentions of "servo" without a ``standard`` declaration are out of scope
(not_run) so structural arm links are never taxed. Calibration: pinned synthesized
specimens (known-good hub / known-bad solid disc), per ADR-0124 -- the archive holds
zero true servo-interface positives (516 accepted parts probed). Flag:
FORGE_SERVO_FIT_AUDIT=1 (default OFF; off = validate byte-identical).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from forge_mind.assembly.servo import (
    MG996R_HORN_SCREW_BC_D_MM,
    MG996R_HUB_CENTER_BORE_D_MM,
    MG996R_MOUNT_HOLE_SPAN_X_MM,
    MG996R_MOUNT_HOLE_SPAN_Y_MM,
)

_HUB_BORE_TOL_MM = 0.3  # the already-shipped +/-0.3 band (design doc: cited bands only)
_HORN_SCREW_D_BAND = (1.9, 2.6)  # self-tap clearance around the cited d2.2
_HORN_BC_TOL_MM = 0.8
_TAB_D_BAND = (4.0, 4.8)  # the datasheet's own clearance range, verbatim
_TAB_SPAN_TOL_MM = 0.8
_PARALLEL_DOT = 0.985  # ~10 degrees


def servo_fit_audit_enabled() -> bool:
    return os.environ.get("FORGE_SERVO_FIT_AUDIT") == "1"


@dataclass(frozen=True)
class ServoFitFinding:
    check: str  # "horn_interface" | "tab_interface"
    message: str
    gates: bool


@dataclass(frozen=True)
class Hole:
    """A measured concave cylinder reduced to what the patterns need."""

    point: tuple[float, float, float]  # a point on the axis
    axis: tuple[float, float, float]  # unit axis direction
    diameter: float


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _axis_distance(a: Hole, b: Hole) -> float:
    """Distance between two PARALLEL hole axes: the axis-perpendicular component of the
    center offset -- invariant to where along each axis the sample point sits."""
    dp = (b.point[0] - a.point[0], b.point[1] - a.point[1], b.point[2] - a.point[2])
    along = _dot(dp, a.axis)
    perp = (dp[0] - along * a.axis[0], dp[1] - along * a.axis[1], dp[2] - along * a.axis[2])
    return math.sqrt(_dot(perp, perp))


def _parallel_groups(holes: list[Hole]) -> list[list[Hole]]:
    groups: list[list[Hole]] = []
    for h in holes:
        for g in groups:
            if abs(_dot(h.axis, g[0].axis)) >= _PARALLEL_DOT:
                g.append(h)
                break
        else:
            groups.append([h])
    return groups


def match_horn_pattern(holes: list[Hole]) -> str | None:
    """None when the cited horn interface is measured present; else what is missing."""
    bore_lo = MG996R_HUB_CENTER_BORE_D_MM - _HUB_BORE_TOL_MM
    bore_hi = MG996R_HUB_CENTER_BORE_D_MM + _HUB_BORE_TOL_MM
    bores = [h for h in holes if bore_lo <= h.diameter <= bore_hi]
    screws = [h for h in holes if _HORN_SCREW_D_BAND[0] <= h.diameter <= _HORN_SCREW_D_BAND[1]]
    if not bores:
        return (
            f"no center bore measured in d{bore_lo:.1f}-{bore_hi:.1f} "
            f"(cited hub bore d{MG996R_HUB_CENTER_BORE_D_MM})"
        )
    for bore in bores:
        ring = [s for s in screws if abs(_dot(s.axis, bore.axis)) >= _PARALLEL_DOT]
        radius = MG996R_HORN_SCREW_BC_D_MM / 2.0
        on_circle = [
            s for s in ring if abs(_axis_distance(bore, s) - radius) <= _HORN_BC_TOL_MM / 2.0 + 0.4
        ]
        if len(on_circle) >= 4:
            return None
    return (
        f"center bore found but not 4 screw holes d{_HORN_SCREW_D_BAND[0]}-"
        f"{_HORN_SCREW_D_BAND[1]} on the verified d{MG996R_HORN_SCREW_BC_D_MM} bolt circle"
    )


def match_tab_pattern(holes: list[Hole]) -> str | None:
    """None when the cited tab rectangle is measured present; else what is missing."""
    tabs = [h for h in holes if _TAB_D_BAND[0] <= h.diameter <= _TAB_D_BAND[1]]
    sx, sy = MG996R_MOUNT_HOLE_SPAN_X_MM, MG996R_MOUNT_HOLE_SPAN_Y_MM
    diag = math.hypot(sx, sy)
    expected = sorted([sy, sy, sx, sx, diag, diag])
    for group in _parallel_groups(tabs):
        if len(group) < 4:
            continue
        # any 4-subset matching the rectangle's pairwise-distance multiset
        from itertools import combinations

        for four in combinations(group, 4):
            dists = sorted(_axis_distance(a, b) for a, b in combinations(four, 2))
            if all(abs(d - e) <= _TAB_SPAN_TOL_MM for d, e in zip(dists, expected, strict=True)):
                return None
    return (
        f"no 4-hole pattern d{_TAB_D_BAND[0]}-{_TAB_D_BAND[1]} matching the cited "
        f"{sx} x {sy} rectangle"
    )


def _declared_interfaces(brief: Any) -> list[tuple[str, str]]:
    """(kind, feature_name) for every feature whose ``standard`` names the MG996R.
    kind: 'horn' when the name carries horn/hub, else 'tab'."""
    out: list[tuple[str, str]] = []
    for feature in getattr(brief, "features", None) or []:
        std = str(getattr(feature, "standard", "") or "").lower()
        if "mg996r" not in std:
            continue
        name = str(getattr(feature, "name", "")).lower()
        kind = "horn" if ("horn" in name or "hub" in name) else "tab"
        out.append((kind, str(getattr(feature, "name", ""))))
    return out


def measured_holes(part: Any) -> list[Hole]:
    """Every concave cylindrical face of the part reduced to a measured Hole -- the
    shared extraction the interface-audit family (servo-fit, tendon) builds patterns on."""
    from forge_mind.validator.geometry import extract_cylinders

    holes: list[Hole] = []
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
    return holes


def servo_fit_findings(brief: Any, part: Any) -> list[ServoFitFinding]:
    """Findings for every DECLARED MG996R interface. Empty when nothing is declared or
    the flag is off. A declared interface that cannot be measured GATES (fail-closed)."""
    if not servo_fit_audit_enabled():
        return []
    declared = _declared_interfaces(brief)
    if not declared:
        return []
    holes = measured_holes(part)
    findings: list[ServoFitFinding] = []
    for kind, feat_name in declared:
        miss = match_horn_pattern(holes) if kind == "horn" else match_tab_pattern(holes)
        if miss is None:
            findings.append(
                ServoFitFinding(
                    check=f"{kind}_interface",
                    message=f"servo-fit: '{feat_name}' {kind} interface measured present",
                    gates=False,
                )
            )
        else:
            findings.append(
                ServoFitFinding(
                    check=f"{kind}_interface",
                    message=(
                        f"servo-fit GATE: '{feat_name}' declares an MG996R {kind} interface "
                        f"but {miss} -- a declared interface must be measured, not claimed"
                    ),
                    gates=True,
                )
            )
    return findings


__all__ = [
    "Hole",
    "ServoFitFinding",
    "match_horn_pattern",
    "match_tab_pattern",
    "measured_holes",
    "servo_fit_audit_enabled",
    "servo_fit_findings",
]
