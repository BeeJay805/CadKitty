"""Deterministic spec-feasibility pre-check (engine deep review, 2026-07-04; founder ruling).

The measured case: cam_indexing_template_plate demands 8 detent holes on a 90 mm bolt circle
THROUGH a 120 x 80 mm plate. The 90-degree stations sit at y = +/-45 mm on a +/-40 mm plate --
no rotation fixes it (best case leaves 4 of 8 outside). Both author arms burned 9 iterations
against the impossibility; Fable's brief even documented it. Ruling: an infeasible spec is
REFUSED before any authoring, with the arithmetic stated, instead of iterated against.

Precision-first: a false refusal is a product killer, so this module only fires when
  * the envelope parses cleanly, AND
  * a circular-pattern feature parses cleanly (circle diameter + count), AND
  * NO rotation of the pattern fits inside the envelope's in-plane extents (an exact sweep,
    not a heuristic -- a pattern the author could save by rotating is never refused), AND
  * the violation clears a 1 mm margin.
Anything unparseable fails soft to "no conflict" -- the geometry gates still apply downstream.
"""

from __future__ import annotations

import math
import re
from typing import Any

from forge_mind.agentic.brief import parse_envelope_mm

# "8 ... holes on a 90 mm (diameter) (bolt) circle" / "90 mm PCD" / "bolt circle of 90 mm"
_CIRCLE_RE = re.compile(
    r"(?:on\s+an?\s+)?(\d+(?:\.\d+)?)\s*mm\s*(?:diameter\s+)?(?:bolt\s+|index\s+)?circle"
    r"|(\d+(?:\.\d+)?)\s*mm\s+(?:pcd|bcd)\b"
    r"|(?:bolt\s+)?circle\s+(?:of\s+)?(?:diameter\s+)?(\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE,
)
_HOLE_D_RE = re.compile(
    r"(?:[øØ⌀]\s*|(?:\bdia(?:meter)?)\s*)(\d+(?:\.\d+)?)\s*mm"
    r"|(\d+(?:\.\d+)?)\s*mm\s*(?:dia(?:meter)?|hole)",
    re.IGNORECASE,
)
_MARGIN_MM = 1.0
_SWEEP_STEP_DEG = 0.5


def _pattern_fits(station_r: float, n: int, half_x: float, half_y: float) -> bool:
    """Is there ANY rotation at which all n stations (radius station_r) fit the rectangle?
    Exact by sweep: the pattern repeats every 360/n degrees, stepped at 0.5 degrees."""
    if station_r <= min(half_x, half_y):
        return True  # fits at every rotation
    steps = max(1, int(360.0 / max(n, 1) / _SWEEP_STEP_DEG))
    for k in range(steps):
        theta = math.radians(k * _SWEEP_STEP_DEG)
        ok = True
        for i in range(n):
            a = theta + 2.0 * math.pi * i / n
            if abs(station_r * math.cos(a)) > half_x or abs(station_r * math.sin(a)) > half_y:
                ok = False
                break
        if ok:
            return True
    return False


def spec_conflicts(brief: Any) -> list[str]:
    """Geometric contradictions INSIDE the spec, each stated with its arithmetic. [] when the
    spec is feasible or cannot be checked (fail-soft; the downstream gates still apply)."""
    try:
        env = parse_envelope_mm(str(getattr(brief, "overall_envelope_mm", "") or ""))
        if env is None:
            return []
        exts = sorted(
            env, reverse=True
        )  # in-plane = the two largest extents (a plate is L x W x t)
        half_x, half_y = exts[0] / 2.0, exts[1] / 2.0
        conflicts: list[str] = []
        for f in getattr(brief, "features", []) or []:
            count = max(1, int(getattr(f, "count", 1) or 1))
            if count < 2:
                continue  # a single feature has no pattern to fit
            text = f"{getattr(f, 'name', '')} {getattr(f, 'approx_dims_mm', '') or ''}"
            cm = _CIRCLE_RE.search(text)
            if cm is None:
                continue
            circle_d = float(cm.group(1) or cm.group(2) or cm.group(3))
            hm = _HOLE_D_RE.search(text)
            hole_d = float(hm.group(1) or hm.group(2)) if hm else 0.0
            station_r = (circle_d + hole_d) / 2.0  # hole EDGE must stay inside the part
            if _pattern_fits(station_r, count, half_x - _MARGIN_MM, half_y - _MARGIN_MM):
                continue
            conflicts.append(
                f"INFEASIBLE SPEC: '{getattr(f, 'name', 'pattern')}' puts {count} holes on a "
                f"{circle_d:g} mm circle (station reach ±{station_r:g} mm including the "
                f"Ø{hole_d:g} hole edge), but the stated envelope {env[0]:g} x {env[1]:g} x "
                f"{env[2]:g} mm allows only ±{half_x:g} x ±{half_y:g} mm in-plane — at EVERY "
                f"rotation at least one station falls outside the part. The spec contradicts "
                f"itself: either the circle must shrink to ≤ {2 * (min(half_x, half_y) - _MARGIN_MM) - hole_d:g} mm "
                f"or the part must grow to ≥ {2 * station_r + 2 * _MARGIN_MM:g} mm across. "
                f"Refusing rather than building a part that cannot satisfy both numbers."
            )
        return conflicts
    except Exception:
        return []  # never let the pre-check itself break a build


__all__ = ["spec_conflicts"]
