"""T2 pair rotation: can body B actually rotate against body A? (ADR-0086)

The claim is earned in three measured steps, each fail-closed:

  1. **Axis + engagement** — the rotor must have a MEASURED spin axis (its own rotor
     signature / bore, via ``motion.axis``), and that rotation must ENGAGE the other body:
     the stator must sit within reach of the rotor's swept envelope about the axis. A body
     spinning in place far from its partner is a true but meaningless fact, not a pair
     claim (measured live: a small square prism has a legitimate k=4 rotor signature — its
     lone spin must not masquerade as the pair's motion). Candidates are tried smaller body
     first; no engaged axis on either body -> ``refused`` (nothing animates).
  2. **As-built interference** — exact boolean overlap (``validator.collision``). Bodies
     that interpenetrate as built are ``locked``: a press-fit or fused pair cannot rotate,
     and the claim cites the measured overlap volume.
  3. **The 360-degree swept collision** — the rotor is posed through a full turn about its
     axis and tested against the stator at every pose with the assembly gate's method
     (seeded interior samples, containment confirmed DEEPER than tessellation noise — the
     same constants as ``validator.motion_sweep``). Every pose clear -> ``free`` with the
     measured minimum clearance; a blocked pose -> ``arc_limited`` with the maximal clear
     arc around the home pose, so a scrubber can never enter the blocked arc.

Nothing here guesses: an unsampleable or non-solid mesh, a missing axis, or an oversized
input refuses with the reason instead of animating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

from forge_mind.motion.axis import spin_axis
from forge_mind.motion.sampling import (
    SWEEP_POSES,
    aabb,
    aabb_disjoint,
    deep_overlap,
    interior_points,
    posed,
    radial_distances,
    rotation_about_line,
)
from forge_mind.validator.collision import interference_report, min_surface_clearance_mm

_SWEEP_POSES = SWEEP_POSES
# Latency guard: above this combined face count the sweep refuses rather than stalling the
# upload (a timeout must never masquerade as "static").
_MAX_FACES = 200_000
# Engagement reach: the stator counts as engaged when its nearest point to the axis line is
# within the rotor's maximal radial extent plus this margin (relative + absolute) — wide
# enough that a clearance pair (bore 1 mm larger than the shaft) is engaged, tight enough
# that a body spinning far from its partner is not.
_ENGAGE_REL = 1.25
_ENGAGE_ABS_MM = 1.0

Vec3T = tuple[float, float, float]
PairStatus = Literal["free", "arc_limited", "locked", "refused"]


@dataclass(frozen=True)
class PairMotion:
    """The measured verdict on relative rotation of a two-body pair."""

    status: PairStatus
    rotor_index: int
    stator_index: int
    axis_origin: Vec3T | None
    axis_direction: Vec3T | None
    explanation: str
    clearance_mm: float | None = None
    interference_volume_mm3: float | None = None
    allowed_arc_deg: tuple[float, float] | None = None  # arc_limited: clear arc around 0
    blocked_at_deg: list[float] = field(default_factory=list)
    poses_tested: int = 0


def _refused(rotor: int, stator: int, why: str) -> PairMotion:
    return PairMotion(
        status="refused",
        rotor_index=rotor,
        stator_index=stator,
        axis_origin=None,
        axis_direction=None,
        explanation=why,
    )


def _engages(
    rotor: trimesh.Trimesh, stator: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray
) -> bool:
    """True when the stator sits within reach of the rotor's swept envelope about the axis."""
    reach = float(radial_distances(rotor, origin, direction).max())
    nearest = float(radial_distances(stator, origin, direction).min())
    return nearest <= reach * _ENGAGE_REL + _ENGAGE_ABS_MM


def _clear_arc_around_home(blocked: list[float], step: float) -> tuple[float, float]:
    """The maximal contiguous clear arc containing the home pose (0 deg), given blocked
    sample angles in [0, 360). Bounds stop one step short of the first blocked pose in each
    direction — the animation must never touch a pose the sweep measured as colliding."""
    lo = -180.0
    hi = 180.0
    for b in sorted(a if a <= 180.0 else a - 360.0 for a in blocked):
        if b > 0:
            hi = min(hi, b - step)
        elif b < 0:
            lo = max(lo, b + step)
    return (lo, hi)


def pair_rotation(bodies: list[trimesh.Trimesh]) -> PairMotion:
    """Certify relative rotation for EXACTLY two solid bodies (ADR-0086 T2)."""
    if len(bodies) != 2:
        return _refused(0, 1, f"pair certification needs exactly 2 bodies, got {len(bodies)}")
    if sum(len(b.faces) for b in bodies) > _MAX_FACES:
        return _refused(0, 1, "meshes too large for swept certification (face-count guard)")
    for i, b in enumerate(bodies):
        if not bool(getattr(b, "is_watertight", False)):
            return _refused(0, 1, f"body {i} is not a closed solid — rotation cannot be certified")

    # 1) the rotor is the body with a measured spin axis whose rotation ENGAGES the other
    #    body (smaller body tried first: the shaft in a block, the ring on a shaft). A
    #    measured-but-unengaged axis is skipped — its lone spin is not the pair's motion.
    order = sorted(range(2), key=lambda i: float(bodies[i].volume))
    rotor_ix: int | None = None
    axis = None
    saw_axis = False
    for i in order:
        cand = spin_axis(bodies[i])
        if cand is None:
            continue
        saw_axis = True
        if _engages(
            bodies[i],
            bodies[1 - i],
            np.asarray(cand.origin, dtype=float),
            np.asarray(cand.direction, dtype=float),
        ):
            rotor_ix, axis = i, cand
            break
    if rotor_ix is None or axis is None:
        why = (
            "a measured spin axis exists but its rotation never engages the other body — "
            "no pair motion to certify"
            if saw_axis
            else "no measured spin axis on either body — nothing may animate"
        )
        return _refused(order[0], order[1], why)
    stator_ix = 1 - rotor_ix
    rotor, stator = bodies[rotor_ix], bodies[stator_ix]

    # 2) as-built interference: interpenetrating bodies are locked (press-fit/fused).
    contact = interference_report(rotor, stator)
    if contact.method == "refused":
        return _refused(rotor_ix, stator_ix, f"interference check refused: {contact.summary}")
    if contact.collide:
        vol = contact.penetration_volume_mm3
        return PairMotion(
            status="locked",
            rotor_index=rotor_ix,
            stator_index=stator_ix,
            axis_origin=axis.origin,
            axis_direction=axis.direction,
            interference_volume_mm3=vol,
            explanation=(
                "bodies interpenetrate as built"
                + (f" by {vol:.2f} mm³" if vol is not None else "")
                + " — a press-fit or fused pair cannot rotate"
            ),
        )

    # 3) the full-turn swept collision, home pose first (fail closed on sampling).
    rotor_pts = interior_points(rotor)
    stator_pts = interior_points(stator)
    if rotor_pts is None or stator_pts is None:
        which = "rotor" if rotor_pts is None else "stator"
        return _refused(
            rotor_ix,
            stator_ix,
            f"{which} interior could not be sampled — a sweep that cannot see a part proves nothing",
        )
    origin = np.asarray(axis.origin, dtype=float)
    direction = np.asarray(axis.direction, dtype=float)
    stator_lo, stator_hi = stator.bounds[0], stator.bounds[1]
    step = 360.0 / _SWEEP_POSES
    blocked: list[float] = []
    for i in range(_SWEEP_POSES):
        theta = i * step
        m = rotation_about_line(origin, direction, theta)
        rotor_world = posed(rotor_pts, m)
        r_lo, r_hi = aabb(rotor_world)
        if aabb_disjoint(r_lo, r_hi, stator_lo, stator_hi):
            continue  # rotor's swept box is clear of the stator this pose
        hit = deep_overlap(rotor_world, stator)
        if not hit:
            hit = deep_overlap(posed(stator_pts, np.linalg.inv(m)), rotor)
        if hit:
            blocked.append(theta)

    if not blocked:
        return PairMotion(
            status="free",
            rotor_index=rotor_ix,
            stator_index=stator_ix,
            axis_origin=axis.origin,
            axis_direction=axis.direction,
            clearance_mm=min_surface_clearance_mm(rotor, stator),
            poses_tested=_SWEEP_POSES,
            explanation=(
                f"full turn certified clear ({_SWEEP_POSES} poses at {step:.0f}°, "
                f"axis: {axis.evidence})"
            ),
        )
    if 0.0 in blocked:
        return PairMotion(
            status="locked",
            rotor_index=rotor_ix,
            stator_index=stator_ix,
            axis_origin=axis.origin,
            axis_direction=axis.direction,
            poses_tested=_SWEEP_POSES,
            blocked_at_deg=blocked,
            explanation="the as-built pose itself reads as colliding — cannot rotate",
        )
    arc = _clear_arc_around_home(blocked, step)
    return PairMotion(
        status="arc_limited",
        rotor_index=rotor_ix,
        stator_index=stator_ix,
        axis_origin=axis.origin,
        axis_direction=axis.direction,
        allowed_arc_deg=arc,
        blocked_at_deg=blocked,
        poses_tested=_SWEEP_POSES,
        clearance_mm=min_surface_clearance_mm(rotor, stator),
        explanation=(
            f"rotation blocked at {min(blocked):.0f}° — clear arc {arc[0]:.0f}° to "
            f"{arc[1]:.0f}° around the as-built pose ({_SWEEP_POSES} poses at {step:.0f}°)"
        ),
    )


__all__ = ["SWEEP_POSES", "PairMotion", "pair_rotation"]
