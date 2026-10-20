"""T3 measured mechanisms (ADR-0086 slice 3): coupled motion, certified — never assumed.

Two mechanism patterns are recognized purely from measured geometry, and their COUPLED
trajectory is swept exactly like the assembly gate does (the follower posed at its measured
ratio for every driver pose — sweeping either body alone visits poses the mechanism never
reaches):

  * **Rolling-element bearing (3+ bodies).** Two coaxial rings + N identical rolling
    elements whose centroids share one orbit circle between the races. The orbit ratio is
    MEASURED, not assumed: for no-slip rolling with the outer race fixed, the cage/roller
    orbit turns at ``Ri / (Ri + Ro)`` of the inner race, with the contact radii taken from
    the measured orbit circle and roller radius (``Ri = r_orbit - r_roll``,
    ``Ro = r_orbit + r_roll``). Rollers are posed by ORBIT about the main axis — a sphere's
    (or coaxial cylinder's) spin about its own center changes nothing for collision, so the
    orbit pose is collision-exact for the shapes this pattern accepts.
  * **External gear pair (2 bodies).** When the uncoupled pair sweep is arc-limited (teeth
    interleave — one gear turning alone MUST hit the other), both bodies have parallel,
    non-coaxial measured axes, and both tooth counts are measurable (the ADR-0079 spectral
    radial count), the coupled trajectory (driver theta, follower -theta*z_d/z_f about its
    OWN axis) is swept; a certified full turn upgrades the claim to coupled rotation at the
    measured ratio. Anything short of certification keeps the honest uncoupled verdict.

Fail-closed everywhere: unrecognized patterns, unsampleable meshes, disagreeing axes,
unmeasurable tooth counts, or an oversized input refuse (or keep the lesser claim) — the
mechanism tier never invents a coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

from forge_mind.motion.axis import spin_axis
from forge_mind.motion.pair import PairMotion
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
from forge_mind.validator.collision import interference_report

Vec3T = tuple[float, float, float]
MechanismStatus = Literal["free", "arc_limited", "locked", "refused"]

_MAX_FACES = 300_000
# ring vs roller classification: a ring's centroid sits ON the common axis; a roller's sits
# OFF it on the orbit circle. Relative to the body's own radial size so scale cancels.
_ON_AXIS_FRAC = 0.15
_AXIS_PARALLEL_DEG = 2.0
_COAXIAL_TOL_MM = 0.75
# rolling elements must be the same part: volumes within this relative band, orbit radii too
_ROLLER_MATCH_REL = 0.12
# a sphere has no unique inertia axis: all three moments within this band of their mean
_SPHERE_MOMENT_REL = 0.06


@dataclass(frozen=True)
class Coupling:
    """One coupled follower: it turns ``ratio`` times the driver's angle about its OWN axis."""

    body_index: int
    ratio: float
    axis_origin: Vec3T
    axis_direction: Vec3T


@dataclass(frozen=True)
class MechanismMotion:
    """The measured verdict on a coupled mechanism."""

    status: MechanismStatus
    kind: Literal["rolling", "gear_pair"]
    driver_index: int
    static_indices: tuple[int, ...]
    axis_origin: Vec3T | None
    axis_direction: Vec3T | None
    couplings: tuple[Coupling, ...]
    explanation: str
    evidence: str = ""
    allowed_arc_deg: tuple[float, float] | None = None
    blocked_at_deg: list[float] = field(default_factory=list)
    poses_tested: int = 0
    interference_volume_mm3: float | None = None


def _refused(kind: Literal["rolling", "gear_pair"], why: str) -> MechanismMotion:
    return MechanismMotion(
        status="refused",
        kind=kind,
        driver_index=0,
        static_indices=(),
        axis_origin=None,
        axis_direction=None,
        couplings=(),
        explanation=why,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


def _line_distance(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    rel = p - origin
    return float(np.linalg.norm(rel - float(rel @ direction) * direction))


def _sphere_like(body: trimesh.Trimesh) -> bool:
    try:
        m = np.asarray(body.principal_inertia_components, dtype=float)
    except Exception:
        return False
    mean = float(np.mean(m))
    return mean > 0 and bool(np.all(np.abs(m - mean) / mean < _SPHERE_MOMENT_REL))


def _clear_arc_around_home(blocked: list[float], step: float) -> tuple[float, float]:
    lo, hi = -180.0, 180.0
    for b in sorted(a if a <= 180.0 else a - 360.0 for a in blocked):
        if b > 0:
            hi = min(hi, b - step)
        elif b < 0:
            lo = max(lo, b + step)
    return (lo, hi)


def _axis_symmetric(
    body: trimesh.Trimesh, pts: np.ndarray, origin: np.ndarray, direction: np.ndarray, step: float
) -> bool:
    """True when rotating the body one sweep-step about the axis leaves its own interior
    points inside it — i.e. its occupancy is INVARIANT under axis rotation (a plain ring,
    disc, or sphere). A keyed or splined ring is NOT symmetric and returns False."""
    try:
        moved = posed(pts, rotation_about_line(origin, direction, step))
        return float(np.mean(body.contains(moved))) > 0.97
    except Exception:
        return False  # can't establish symmetry -> treat as asymmetric (sweep every pose)


def _coupled_sweep(
    bodies: list[trimesh.Trimesh],
    pose_of: dict[int, tuple[np.ndarray, np.ndarray, float]],  # index -> (origin, dir, ratio)
    static_ix: list[int],
    pts: dict[int, np.ndarray],
    sym_axis: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[float], int] | str:
    """Sweep the coupled trajectory: every posed body vs every static body, and every pair of
    posed bodies whose relative pose changes. ``pts`` are the seeded interior samples (shared
    with the static check). Returns (blocked angles, poses) or a refusal.

    ``sym_axis`` (origin, direction) enables the SYMMETRY short-circuit: a pair whose relative
    configuration is INVARIANT across the sweep — because a body is rotationally symmetric
    about the axis, or two movers share one ratio (rigid) — is deep-checked ONCE, not per
    pose. For an ideal bearing (symmetric races + rigid roller orbit) every pair is invariant,
    so the full turn is certified by the home configuration in a handful of checks instead of
    hundreds. Correctness is preserved: a non-invariant pair (a keyed race, meshed teeth) is
    still swept at every pose."""
    ratio = {ix: r for ix, (_o, _d, r) in pose_of.items()}
    sym: dict[int, bool] = {}
    if sym_axis is not None:
        so, sd = sym_axis
        step0 = 360.0 / SWEEP_POSES
        sym = {ix: _axis_symmetric(bodies[ix], pts[ix], so, sd, step0) for ix in pts}

    def _invariant(a: int, b: int) -> bool:
        # a is a mover; b is a mover or a static. Their relative config is pose-invariant if
        # either body is axis-symmetric, or both orbit at the same rate (a static counts as 0).
        if sym.get(a) or sym.get(b):
            return True
        return abs(ratio[a] - ratio.get(b, 0.0)) < 1e-9

    static_aabb = {s: (bodies[s].bounds[0], bodies[s].bounds[1]) for s in static_ix}
    step = 360.0 / SWEEP_POSES
    blocked: list[float] = []
    movers = sorted(pose_of)
    checked_once: set[tuple[int, int]] = set()

    def _pair_hits(a: int, world_a: np.ndarray, mats: dict[int, np.ndarray], b: int) -> bool:
        if b in static_ix:
            s_lo, s_hi = static_aabb[b]
            a_lo, a_hi = aabb(world_a)
            if aabb_disjoint(a_lo, a_hi, s_lo, s_hi):
                return False
            return deep_overlap(world_a, bodies[b]) or deep_overlap(
                posed(pts[b], np.linalg.inv(mats[a])), bodies[a]
            )
        if np.allclose(mats[b], mats[a], atol=1e-12):
            return False
        rel = np.linalg.inv(mats[b]) @ mats[a]  # a's points in b's home frame
        return deep_overlap(posed(pts[a], rel), bodies[b])

    for i in range(SWEEP_POSES):
        theta = i * step
        mats = {ix: rotation_about_line(o, d, theta * r) for ix, (o, d, r) in pose_of.items()}
        world = {a: posed(pts[a], mats[a]) for a in movers}
        hit = False
        for a in movers:
            others = [*static_ix, *(b for b in movers if b > a)]
            for b in others:
                key = (a, b)
                if _invariant(a, b):
                    if key in checked_once:
                        continue  # relative config unchanged since it was proven clear
                    checked_once.add(key)
                if _pair_hits(a, world[a], mats, b):
                    hit = True
                    break
            if hit:
                break
        if hit:
            blocked.append(theta)
    return blocked, SWEEP_POSES


def _first_overlap(
    bodies: list[trimesh.Trimesh], pairs: list[tuple[int, int]]
) -> tuple[float, str] | None:
    """The first interpenetrating pair among ``pairs``, or None. Uses the EXACT boolean
    (sensitive to a sub-mm press-fit the point sampler's depth floor would miss) but only on
    AABB-overlapping pairs, and the caller passes a REPRESENTATIVE pair set (rollers are
    verified identical, so one stands for all) — so a bearing costs ~4 booleans, not the 36
    of the full C(n,2) product."""
    for i, j in pairs:
        if aabb_disjoint(
            bodies[i].bounds[0], bodies[i].bounds[1], bodies[j].bounds[0], bodies[j].bounds[1]
        ):
            continue
        rep = interference_report(bodies[i], bodies[j])
        if rep.collide:
            return (rep.penetration_volume_mm3 or 0.0, f"body {i} <-> body {j}")
    return None


def rolling_mechanism(bodies: list[trimesh.Trimesh]) -> MechanismMotion:
    """Certify a rolling-element bearing pattern for 3+ solid bodies (ADR-0086 T3)."""
    n = len(bodies)
    if n < 3:
        return _refused("rolling", f"rolling pattern needs 3+ bodies, got {n}")
    if sum(len(b.faces) for b in bodies) > _MAX_FACES:
        return _refused("rolling", "meshes too large for swept certification (face guard)")
    for i, b in enumerate(bodies):
        if not bool(getattr(b, "is_watertight", False)):
            return _refused("rolling", f"body {i} is not a closed solid")

    axes = {i: spin_axis(b) for i, b in enumerate(bodies)}

    # the common axis: >=2 bodies with agreeing axis LINES (parallel + mutually coaxial)
    ringish = [(i, a) for i, a in axes.items() if a is not None]
    common: tuple[np.ndarray, np.ndarray] | None = None
    for k in range(len(ringish)):
        i, ai = ringish[k]
        oi, di = np.asarray(ai.origin), _unit(np.asarray(ai.direction))
        for _j, aj in ringish[k + 1 :]:
            oj, dj = np.asarray(aj.origin), _unit(np.asarray(aj.direction))
            ang = np.degrees(np.arccos(np.clip(abs(float(di @ dj)), -1.0, 1.0)))
            if ang > _AXIS_PARALLEL_DEG:
                continue
            if (
                _line_distance(oj, oi, di) > _COAXIAL_TOL_MM
                or _line_distance(oi, oj, dj) > _COAXIAL_TOL_MM
            ):
                continue
            common = (oi, di)
            break
        if common is not None:
            break
    if common is None:
        return _refused("rolling", "no two bodies share a measured rotation axis")
    origin, direction = common

    rings: list[int] = []
    rollers: list[int] = []
    for i, b in enumerate(bodies):
        centroid = np.asarray(b.center_mass, dtype=float)
        r_c = _line_distance(centroid, origin, direction)
        body_r = float(radial_distances(b, centroid, direction).max())
        a = axes[i]
        on_axis = r_c < max(_ON_AXIS_FRAC * body_r, 0.5)
        if a is not None and on_axis:
            rings.append(i)
        elif not on_axis and (
            _sphere_like(b)
            or (
                a is not None
                and np.degrees(
                    np.arccos(
                        np.clip(abs(float(_unit(np.asarray(a.direction)) @ direction)), -1, 1)
                    )
                )
                <= _AXIS_PARALLEL_DEG
            )
        ):
            rollers.append(i)
        else:
            return _refused(
                "rolling",
                f"body {i} fits neither a coaxial ring nor a rolling element — "
                "unrecognized mechanism, nothing animates",
            )
    if len(rings) != 2 or len(rollers) < 2:
        return _refused(
            "rolling",
            f"pattern needs exactly 2 rings + 2+ rolling elements (got {len(rings)} rings, "
            f"{len(rollers)} rollers)",
        )

    # rollers must be one part on one orbit circle
    vols = np.array([abs(float(bodies[i].volume)) for i in rollers])
    orbit_rs = np.array(
        [
            _line_distance(np.asarray(bodies[i].center_mass, dtype=float), origin, direction)
            for i in rollers
        ]
    )
    if (
        float(vols.std() / vols.mean()) > _ROLLER_MATCH_REL
        or float(orbit_rs.std() / orbit_rs.mean()) > _ROLLER_MATCH_REL
    ):
        return _refused("rolling", "rolling elements are not identical on one orbit circle")

    inner, outer = sorted(
        rings, key=lambda i: float(radial_distances(bodies[i], origin, direction).max())
    )
    r_orbit = float(orbit_rs.mean())
    r_roll = float(
        np.mean(
            [
                np.linalg.norm(bodies[i].vertices - np.asarray(bodies[i].center_mass), axis=1).max()
                for i in rollers
            ]
        )
    )
    r_inner_contact = r_orbit - r_roll
    r_outer_contact = r_orbit + r_roll
    if r_inner_contact <= 0:
        return _refused("rolling", "measured roller radius exceeds the orbit radius")
    ratio = r_inner_contact / (r_inner_contact + r_outer_contact)
    evidence = (
        f"measured orbit r={r_orbit:.2f} mm, roller r={r_roll:.2f} mm -> contact radii "
        f"Ri={r_inner_contact:.2f}/Ro={r_outer_contact:.2f}; no-slip epicyclic ratio "
        f"Ri/(Ri+Ro)={ratio:.3f} (outer race fixed)"
    )

    pts: dict[int, np.ndarray] = {}
    for i in range(len(bodies)):
        p = interior_points(bodies[i])
        if p is None:
            return _refused("rolling", f"body {i} interior could not be sampled — cannot certify")
        pts[i] = p

    # As-built fit, on REPRESENTATIVE pairs (rollers verified identical -> roller[0] stands for
    # all): the two races, plus one roller against each race and against every other roller.
    r0 = rollers[0]
    fit_pairs = [(inner, outer)]
    fit_pairs += [(r0, inner), (r0, outer)]
    fit_pairs += [(r0, r) for r in rollers[1:]]
    worst = _first_overlap(bodies, fit_pairs)
    if worst is not None:
        return MechanismMotion(
            status="locked",
            kind="rolling",
            driver_index=inner,
            static_indices=(outer,),
            axis_origin=tuple(float(x) for x in origin),  # type: ignore[arg-type]
            axis_direction=tuple(float(x) for x in direction),  # type: ignore[arg-type]
            couplings=(),
            evidence=evidence,
            interference_volume_mm3=worst[0],
            explanation=(
                f"bodies interpenetrate as built ({worst[1]}, {worst[0]:.2f} mm³) — "
                "a press-fit or fused mechanism cannot move"
            ),
        )

    pose_of = {inner: (origin, direction, 1.0)}
    for i in rollers:
        pose_of[i] = (origin, direction, ratio)
    swept = _coupled_sweep(bodies, pose_of, [outer], pts, sym_axis=(origin, direction))
    if isinstance(swept, str):
        return _refused("rolling", swept)
    blocked, poses = swept
    couplings = tuple(
        Coupling(
            body_index=i,
            ratio=ratio,
            axis_origin=tuple(float(x) for x in origin),  # type: ignore[arg-type]
            axis_direction=tuple(float(x) for x in direction),  # type: ignore[arg-type]
        )
        for i in rollers
    )

    def _verdict(
        status: MechanismStatus,
        explanation: str,
        allowed_arc_deg: tuple[float, float] | None = None,
        blocked_at_deg: list[float] | None = None,
    ) -> MechanismMotion:
        return MechanismMotion(
            status=status,
            kind="rolling",
            driver_index=inner,
            static_indices=(outer,),
            axis_origin=(float(origin[0]), float(origin[1]), float(origin[2])),
            axis_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            couplings=couplings,
            evidence=evidence,
            poses_tested=poses,
            allowed_arc_deg=allowed_arc_deg,
            blocked_at_deg=blocked_at_deg or [],
            explanation=explanation,
        )

    if not blocked:
        return _verdict(
            "free",
            f"full coupled turn certified clear ({poses} poses; inner race drives, "
            f"rollers orbit at {ratio:.3f}x; {evidence})",
        )
    if 0.0 in blocked:
        return _verdict(
            "locked",
            "the as-built pose itself reads as colliding — cannot move",
            blocked_at_deg=blocked,
        )
    arc = _clear_arc_around_home(blocked, 360.0 / poses)
    return _verdict(
        "arc_limited",
        f"coupled motion blocked at {min(blocked):.0f}° — clear arc {arc[0]:.0f}° to "
        f"{arc[1]:.0f}° around the as-built pose ({poses} poses)",
        allowed_arc_deg=arc,
        blocked_at_deg=blocked,
    )


def gear_pair_coupled(bodies: list[trimesh.Trimesh], pair: PairMotion) -> MechanismMotion | None:
    """Try to upgrade an ARC-LIMITED two-body pair to certified coupled gear rotation.

    Returns None whenever the coupled claim cannot be EARNED (not a gear-mesh geometry,
    unmeasurable tooth counts, or the coupled sweep does not certify a full turn) — the
    caller keeps the honest uncoupled verdict."""
    if pair.status != "arc_limited" or len(bodies) != 2:
        return None
    a_d = spin_axis(bodies[pair.rotor_index])
    a_f = spin_axis(bodies[pair.stator_index])
    if a_d is None or a_f is None:
        return None
    od, dd = np.asarray(a_d.origin), _unit(np.asarray(a_d.direction))
    of, df = np.asarray(a_f.origin), _unit(np.asarray(a_f.direction))
    ang = np.degrees(np.arccos(np.clip(abs(float(dd @ df)), -1.0, 1.0)))
    if ang > _AXIS_PARALLEL_DEG:
        return None  # not parallel-axis gearing
    center_dist = _line_distance(of, od, dd)
    if center_dist <= _COAXIAL_TOL_MM:
        return None  # coaxial — that is the pair case, not a mesh

    from forge_mind.agentic.radial_count import measure_radial_count

    z_d = measure_radial_count(bodies[pair.rotor_index], 12).k_star
    z_f = measure_radial_count(bodies[pair.stator_index], 12).k_star
    if z_d is None or z_f is None:
        return None  # tooth counts unmeasured -> no coupling invented
    ratio = -float(z_d) / float(z_f)  # external mesh counter-rotation

    pose_of = {
        pair.rotor_index: (od, dd, 1.0),
        pair.stator_index: (of, df, ratio),
    }
    gear_pts: dict[int, np.ndarray] = {}
    for i in (pair.rotor_index, pair.stator_index):
        p = interior_points(bodies[i])
        if p is None:
            return None
        gear_pts[i] = p
    # meshed teeth are NOT axis-symmetric -> no symmetry short-circuit; the full sweep runs
    swept = _coupled_sweep(bodies, pose_of, [], gear_pts)
    if isinstance(swept, str):
        return None
    blocked, poses = swept
    if blocked:
        return None  # the coupled trajectory did not certify -> keep the honest pair claim
    evidence = (
        f"measured tooth counts z={z_d}/{z_f} (spectral radial count), parallel axes "
        f"{center_dist:.2f} mm apart -> external mesh ratio {ratio:.3f}"
    )
    return MechanismMotion(
        status="free",
        kind="gear_pair",
        driver_index=pair.rotor_index,
        static_indices=(),
        axis_origin=tuple(float(x) for x in od),  # type: ignore[arg-type]
        axis_direction=tuple(float(x) for x in dd),  # type: ignore[arg-type]
        couplings=(
            Coupling(
                body_index=pair.stator_index,
                ratio=ratio,
                axis_origin=tuple(float(x) for x in of),  # type: ignore[arg-type]
                axis_direction=tuple(float(x) for x in df),  # type: ignore[arg-type]
            ),
        ),
        evidence=evidence,
        poses_tested=poses,
        explanation=(
            f"coupled gear rotation certified over a full turn ({poses} poses; follower at "
            f"{ratio:.3f}x; {evidence}); uncoupled rotation is blocked, as meshed teeth must be"
        ),
    )


__all__ = ["Coupling", "MechanismMotion", "gear_pair_coupled", "rolling_mechanism"]
