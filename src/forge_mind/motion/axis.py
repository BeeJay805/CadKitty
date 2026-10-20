"""T1 spin-axis discovery: a single body's REAL rotation axis, or nothing (ADR-0086).

A motion claim starts with a measured axis. Two honest sources, in priority order:

  * **B-rep bore** (when a B-rep part is available): the dominant coaxial concave cylinder
    stack — a gear/pulley/knob spins about its bore's true center line, even when the body
    is deliberately unbalanced (a crank). Dominance = the deepest large-radius hole stack.
  * **Mesh rotor signature** (mesh-only uploads): the unique principal inertia axis whose
    two perpendicular moments match within tolerance. Every k>=3 rotor (gear, handwheel,
    impeller, bearing race) has this signature; a bracket or box does not — so a bracket can
    never earn a spin claim. The axis passes through the center of mass.

``rotational_balance`` about the claimed axis is attached as EVIDENCE (quality finding); it
never fabricates an axis and never blocks one — an unbalanced rotor still physically spins.
Fail-closed: no measurable axis -> ``None`` -> the part renders static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import trimesh

from forge_mind.validator.balance import BalanceReport, rotational_balance

# The perpendicular principal moments of a k>=3 rotor are equal by symmetry; measured meshes
# carry tessellation noise, so "equal" is a relative band. A box with square cross-section
# also matches the pair test — the distinctness gate (the spin moment must differ from the
# pair by more than the band) rejects near-spherical/cubic bodies that have no meaningful
# unique axis rather than inventing one.
_PAIR_REL_TOL = 0.08
_DISTINCT_REL_MIN = 0.12
# A bore must be a real feature to define the functional axis: at least this radius and at
# least this fraction of the body's axis-aligned extent deep.
_MIN_BORE_RADIUS_MM = 1.0
_MIN_BORE_DEPTH_FRACTION = 0.3

Vec3T = tuple[float, float, float]


@dataclass(frozen=True)
class SpinAxis:
    """A measured single-body spin axis and the evidence that earned it."""

    origin: Vec3T
    direction: Vec3T
    method: Literal["brep_bore", "inertia_rotor"]
    evidence: str
    balance: BalanceReport
    bore_radius_mm: float | None = None


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def _brep_bore_axis(
    brep_part: Any, mesh: trimesh.Trimesh
) -> tuple[Vec3T, Vec3T, float, str] | None:
    """The dominant concave coaxial stack, if one is deep enough to be the functional bore."""
    try:
        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        stacks = [s for s in cluster_coaxial(extract_cylinders(brep_part)) if s.concave is True]
    except Exception:
        return None  # unmeasurable B-rep: fall through to the mesh path, never guess
    best: tuple[float, Vec3T, Vec3T, float, str] | None = None
    for s in stacks:
        if s.radius < _MIN_BORE_RADIUS_MM or s.depth_mm is None:
            continue
        d = _unit(np.asarray(s.axis_dir, dtype=float))
        if d is None:
            continue
        # depth must be a real fraction of the body's extent along the axis
        proj = mesh.vertices @ d
        body_span = float(proj.max() - proj.min())
        if body_span <= 0 or s.depth_mm < _MIN_BORE_DEPTH_FRACTION * body_span:
            continue
        score = s.depth_mm * s.radius  # deepest large bore wins
        if best is None or score > best[0]:
            best = (
                score,
                (float(s.axis_point[0]), float(s.axis_point[1]), float(s.axis_point[2])),
                (float(d[0]), float(d[1]), float(d[2])),
                float(s.radius),
                f"dominant bore Ø{2 * s.radius:.1f} mm, {s.depth_mm:.1f} mm deep "
                f"({s.depth_mm / body_span:.0%} of the body span)",
            )
    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


def _inertia_rotor_axis(mesh: trimesh.Trimesh) -> tuple[Vec3T, Vec3T, str] | None:
    """The unique principal axis of a rotor-signature body; None when the signature fails."""
    try:
        moments = np.asarray(mesh.principal_inertia_components, dtype=float)
        vectors = np.asarray(mesh.principal_inertia_vectors, dtype=float)
        com = np.asarray(mesh.center_mass, dtype=float)
    except Exception:
        return None
    if moments.shape != (3,) or not np.all(np.isfinite(moments)) or np.any(moments <= 0):
        return None
    mean = float(np.mean(moments))
    if mean <= 0:
        return None
    # find the pairing: the two closest moments are the perpendicular pair; the spin axis is
    # the remaining (unique) one.
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    best = min(pairs, key=lambda p: abs(moments[p[0]] - moments[p[1]]))
    i, j, k = best
    pair_dev = abs(moments[i] - moments[j]) / mean
    distinct = abs(moments[k] - 0.5 * (moments[i] + moments[j])) / mean
    if pair_dev > _PAIR_REL_TOL or distinct < _DISTINCT_REL_MIN:
        return None
    d = _unit(vectors[k])
    if d is None:
        return None
    return (
        (float(com[0]), float(com[1]), float(com[2])),
        (float(d[0]), float(d[1]), float(d[2])),
        f"rotor inertia signature: perpendicular moments match within {pair_dev:.1%}, "
        f"spin moment distinct by {distinct:.1%}",
    )


def spin_axis(mesh: trimesh.Trimesh, brep_part: Any = None) -> SpinAxis | None:
    """The single body's measured functional spin axis, or None (static — no claim)."""
    if not bool(getattr(mesh, "is_watertight", False)):
        return None  # mass/inertia are untrustworthy on an open shell: fail closed
    bore = _brep_bore_axis(brep_part, mesh) if brep_part is not None else None
    if bore is not None:
        origin, direction, radius, why = bore
        return SpinAxis(
            origin=origin,
            direction=direction,
            method="brep_bore",
            evidence=why,
            balance=rotational_balance(mesh, origin, direction),
            bore_radius_mm=radius,
        )
    rotor = _inertia_rotor_axis(mesh)
    if rotor is None:
        return None
    origin, direction, why = rotor
    return SpinAxis(
        origin=origin,
        direction=direction,
        method="inertia_rotor",
        evidence=why,
        balance=rotational_balance(mesh, origin, direction),
    )


__all__ = ["SpinAxis", "spin_axis"]
