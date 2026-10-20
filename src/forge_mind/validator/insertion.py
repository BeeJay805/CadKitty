"""Assemblability (Wave C, row 54's 'one-piece crank cannot enter a one-piece block'):
the motion sweep proves the ASSEMBLED mechanism moves -- nothing proved a human can get
it assembled. This module checks, deterministically:

  * per part: a collision-free STRAIGHT-LINE extraction path exists (test the reverse:
    translate the part out of the assembly along +-X/+-Y/+-Z; if some direction is
    clear, the part can be inserted along it);
  * globally: a valid ASSEMBLY ORDER exists (greedy peel -- repeatedly remove any
    extractable part; the reverse peel order is an assembly order).

Honesty: absence of a straight-line path does NOT prove un-assemblability (curved /
rotate-in insertions exist -- the i4 crank drops into open saddles), so findings are
CAVEAT-tier, never a hard fail. Fail-open on any measurement error."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

_SEED = 7
# 600 requested (rejection sampling yields ~half): thin walls vs the >2-point block
# threshold need ~10 expected hits in a wall-disc overlap, not ~2 (flaky-miss zone)
_POINTS = 600
_STEPS = 8
_BLOCK_POINTS = 2  # > this many sample points inside another part = blocked

_DIRECTIONS: list[tuple[str, tuple[float, float, float]]] = [
    ("+X", (1, 0, 0)),
    ("-X", (-1, 0, 0)),
    ("+Y", (0, 1, 0)),
    ("-Y", (0, -1, 0)),
    ("+Z", (0, 0, 1)),
    ("-Z", (0, 0, -1)),
]


def _interior(mesh: trimesh.Trimesh) -> np.ndarray:
    state = np.random.get_state()
    try:
        np.random.seed(_SEED)
        pts = np.asarray(trimesh.sample.volume_mesh(mesh, _POINTS), dtype=float)
    except Exception:
        pts = np.empty((0, 3))
    finally:
        np.random.set_state(state)
    return pts


def _extraction_direction(
    pts: np.ndarray, others: list[trimesh.Trimesh], travel: float
) -> str | None:
    """The first +-axis direction along which the sampled part translates OUT of the
    assembly without ever burying >_BLOCK_POINTS points inside another part.

    Steps are sized PER OBSTACLE: travel only until the mover has fully passed that
    obstacle's extent along the direction, in steps small enough (<= 6 mm) that a thin
    wall cannot slip between samples (the fixed-global-step version aliased right through
    a 5 mm shell)."""
    if len(pts) == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    for name, d in _DIRECTIONS:
        vec = np.asarray(d, dtype=float)
        axis = int(np.argmax(np.abs(vec)))
        sign = 1.0 if vec[axis] > 0 else -1.0
        clear = True
        for other in others:
            ob_lo, ob_hi = other.bounds
            if sign > 0:
                t_max = float(ob_hi[axis] - lo[axis]) + 2.0
            else:
                t_max = float(hi[axis] - ob_lo[axis]) + 2.0
            t_max = min(max(t_max, 0.0), travel)
            if t_max <= 0.0:
                continue  # already past this obstacle along d
            n_steps = int(np.clip(np.ceil(t_max / 6.0), 6, 24))
            for t in np.linspace(t_max / n_steps, t_max, n_steps):
                moved = pts + vec * t
                try:
                    inside = int(np.count_nonzero(other.contains(moved)))
                except Exception:
                    inside = _BLOCK_POINTS + 1  # cannot certify -> blocked
                if inside > _BLOCK_POINTS:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            return name
    return None


def assembly_order_issues(named_meshes: list[tuple[str, Any]]) -> dict[str, Any] | None:
    """Greedy-peel assemblability over the home-pose assembly. Returns a dict with the
    found (reverse-peel) assembly order and any stuck parts; None when unmeasurable or
    trivially fine (< 3 parts). CAVEAT-tier by contract."""
    parts = [
        (n, m) for n, m in named_meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0
    ]
    if len(parts) < 3:
        return None
    try:
        bounds = np.vstack([m.bounds for _n, m in parts])
        travel = float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))) + 10.0
        interiors = {n: _interior(m) for n, m in parts}
        remaining = dict(parts)
        peel: list[tuple[str, str]] = []
        while len(remaining) > 1:
            extracted = None
            for n in list(remaining):
                others = [m for o, m in remaining.items() if o != n]
                d = _extraction_direction(interiors[n], others, travel)
                if d is not None:
                    extracted = (n, d)
                    break
            if extracted is None:
                return {
                    "order": [n for n, _d in reversed(peel)],
                    "stuck": sorted(remaining),
                    "note": (
                        "no straight-line extraction found for the remaining parts -- "
                        "hand-assembly may need rotation/tilt insertion (verify manually)"
                    ),
                }
            peel.append(extracted)
            remaining.pop(extracted[0])
        peel.append((next(iter(remaining)), "base"))
        return {
            "order": [f"{n} ({d})" for n, d in reversed(peel)],
            "stuck": [],
            "note": "straight-line assembly order exists",
        }
    except Exception:
        return None


__all__ = ["assembly_order_issues"]
