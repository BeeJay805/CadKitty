"""Relational + kinematic validation (ADR-0010 spike) -- the new 5th tier.

What it proves, from GEOMETRY (never from the fixture):
  * each body is a watertight, positive-volume single shell (reuses the mesh tier)
  * the pin and every leaf bore are coaxial (measured B-rep cylinder axes)
  * the pin has positive running-fit clearance (bore radius - pin radius >= floor)
  * the moving leaf sweeps its full declared range without colliding with the
    fixed leaf (sampled solid interpenetration), and a FIXED joint -- a welded
    hinge -- fails because it offers no rotational DOF.

Collision = positive solid interpenetration, detected by sampling interior points
of one placed body and testing containment in the other. Two bodies merely touching
(the closed pose, or pin-in-bore contact) share zero interior volume and pass; only
real overlap fails. The sample density is reported so the check's resolution is
explicit (it can miss interpenetrations thinner than the sample spacing).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from forge_mind.agentic.gears import GearSpec, center_distance, validate_gear_mesh_kinematic
from forge_mind.assembly.joints import Joint, JointType
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody
from forge_mind.dsl.interfaces.fdm import (
    FDM_MAX_GAP_MM,
    FDM_MIN_GAP_MM,
    NOZZLE_DIAMETER_MM,
)
from forge_mind.dsl.interfaces.nema17_shaft import (
    SET_SCREW_CLEARANCE_RADIUS_MM as _SET_SCREW_RADIUS,
)
from forge_mind.dsl.interfaces.nema17_shaft import (
    SHAFT_BORE_RADIUS_MM as _SHAFT_BORE_RADIUS,
)
from forge_mind.dsl.interfaces.nema17_shaft import (
    SHAFT_RADIUS_MM as _SHAFT_RADIUS,
)
from forge_mind.validator.geometry import extract_cylinders
from forge_mind.validator.hardware_mounts import patterns_for
from forge_mind.validator.mesh_checks import run_mesh_checks
from forge_mind.validator.motion_sweep import (
    ChainSweeper,
    chained_pairs,
    collapse_pin_edges,
    downstream_ids,
    epicyclic_systems,
    four_bar_systems,
    meshless_hardware_ids,
    slider_crank_systems,
    undriven_loop_joint_ids,
)
from forge_mind.validator.report import (
    Check,
    CheckStatus,
    Evidence,
    Fix,
    Tier,
    ValidationReport,
    build_report,
)


def _sweep_evidence(
    joint: Joint, collisions: dict[tuple[str, str], tuple[float, float]]
) -> Evidence:
    """Classify a sweep result as a PROVEN violation or a fail-closed one (ADR-0127).

    A FAILing sweep gates correctly either way, but only a MEASURED one may reach the
    non-shipping strict tier, which must fail OPEN on absence of proof. Two ways a sweep
    FAILs without proving the mechanism is blocked, both excluded here:

    * **The range was never declared.** ``range_deg or (0.0, 180.0)`` substitutes a default,
      so the collision may sit at a pose the design never asked for. Failing a design on an
      engine-invented range is specification laundering in reverse.
    * **The depth is unmeasurable.** ``_pair_overlap_depth`` returns ``inf`` on a collision-
      backend failure and the sweep treats it as an overlap (fail-closed, deliberately). That
      is "could not check", not "does not fit".

    One finite depth is enough: a measured interpenetration inside the declared range proves
    the mechanism is blocked no matter how many other pairs went unmeasured.
    """
    declared = (
        joint.range_deg is not None
        if joint.type is JointType.REVOLUTE
        else (joint.range_mm is not None)
    )
    if not declared:
        return Evidence.ASSUMED
    if not any(math.isfinite(depth) for depth, _pose in collisions.values()):
        return Evidence.ASSUMED
    return Evidence.MEASURED


_CORE_MESH_CHECKS = {
    "mesh.watertight",
    "mesh.is_volume",
    "mesh.winding_consistent",
    "mesh.single_body",
    "mesh.positive_volume",
}

_AXIS_PARALLEL_TOL_DEG = 2.0
_COAXIAL_OFFSET_TOL_MM = 0.25
_MIN_CLEARANCE_MM = 0.1
_MOTION_SAMPLES = 19
_INTERIOR_SAMPLES = 800
_COLLISION_MIN_POINTS = 3  # ignore <=2 stray interior hits as numerical noise
_SHAFT_BORE_TOL = 0.35  # radius tolerance for the shaft-grip bore + the M3 set screw
# GEAR_MESH: the actual built centre distance must sit within this band of the theoretical
# (sum of pitch radii). Too tight -> teeth gross-jam; too loose -> teeth lose contact (backlash /
# no engagement). One nozzle-ish slack each way catches a real mis-spacing without flagging a
# correctly-placed pair. The PARALLEL-axis tolerance reuses the existing axis-parallel tol.
_GEAR_CENTER_DISTANCE_TOL_MM = 0.5

# trimesh.sample.volume_mesh draws from the GLOBAL numpy RNG, so an unseeded sweep flickers
# run-to-run on a marginal / near-grazing collision (the count of interior hits can cross
# _COLLISION_MIN_POINTS by chance). We seed it to a FIXED value around the sample (state saved +
# restored, so no RNG state leaks to the caller) -> the same (mesh, transform, port) yields the
# SAME verdict every run. This is the SAME save/seed/restore pattern feature_checks.py uses for
# min_wall point sampling. It does NOT weaken collision detection: the threshold and sample count
# are unchanged; a thin real interpenetration is still caught, just deterministically.
_INTERIOR_SAMPLE_SEED = 0


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def _apply_point(m: NDArray[np.float64], p: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.asarray((m @ np.array([p[0], p[1], p[2], 1.0]))[:3], dtype=np.float64)


def _apply_dir(m: NDArray[np.float64], d: tuple[float, float, float]) -> NDArray[np.float64]:
    return np.asarray(m[:3, :3] @ np.asarray(d, dtype=float), dtype=np.float64)


def _angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    d = abs(float(np.dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def _line_offset(
    p1: NDArray[np.float64], d1: NDArray[np.float64], p2: NDArray[np.float64]
) -> float:
    """Perpendicular distance from point p2 to the line (p1, d1)."""
    d = _unit(d1)
    w = p2 - p1
    return float(np.linalg.norm(w - np.dot(w, d) * d))


def _rotation_about_line(
    point: NDArray[np.float64], direction: NDArray[np.float64], angle_deg: float
) -> NDArray[np.float64]:
    d = _unit(direction)
    a = math.radians(angle_deg)
    k = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    rot = np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = point - rot @ point
    return m


def _translation(direction: NDArray[np.float64], dist: float) -> NDArray[np.float64]:
    m = np.eye(4)
    m[:3, 3] = _unit(direction) * dist
    return m


def _placed_mesh(body: PlacedBody, extra: NDArray[np.float64] | None = None) -> trimesh.Trimesh:
    m = body.mesh.copy()
    world = body.transform if extra is None else extra @ body.transform
    m.apply_transform(world)
    return m


def _bboxes_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh, pad: float = 0.0) -> bool:
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    return bool(np.all(amax + pad >= bmin) and np.all(bmax + pad >= amin))


def _interior_points(mesh: trimesh.Trimesh, n: int) -> NDArray[np.float64]:
    """Points strictly inside the solid (rejection sampling). Interior, not surface,
    so two bodies merely touching share none."""
    # Seed the global RNG to a fixed value JUST for the sample, then restore it, so the sweep /
    # interpenetration verdict is reproducible without leaking RNG state (volume_mesh is the only
    # RNG consumer here). Same save/seed/restore pattern as feature_checks._measure_min_wall.
    rng_state = np.random.get_state()
    try:
        np.random.seed(_INTERIOR_SAMPLE_SEED)
        pts = trimesh.sample.volume_mesh(mesh, n)
    except Exception:  # pragma: no cover - degenerate/no-backend
        return np.empty((0, 3))
    finally:
        np.random.set_state(rng_state)
    return np.asarray(pts, dtype=float)


def _interpenetrates(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    """True if solids a and b share interior volume (real overlap, not mere contact)."""
    if not _bboxes_overlap(a, b):
        return False
    for src, dst in ((b, a), (a, b)):
        pts = _interior_points(src, _INTERIOR_SAMPLES)
        if len(pts) == 0:
            continue
        try:
            inside = dst.contains(pts)
        except Exception:  # pragma: no cover - contains backend issue
            continue
        if int(np.count_nonzero(inside)) > _COLLISION_MIN_POINTS:
            return True
    return False


def _min_surface_gap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float | None:
    """Min surface distance between two PLACED meshes (vertex-based, both directions) -- the
    pattern functional._assembly_fit uses. Bounded + cheap; NOT a bridging/support analysis.
    None if it cannot be measured (the gap check then skips that pair rather than guess)."""
    try:
        qa: Any = trimesh.proximity.ProximityQuery(a)  # type: ignore[no-untyped-call]
        qb: Any = trimesh.proximity.ProximityQuery(b)  # type: ignore[no-untyped-call]
        d_ab = np.abs(qa.signed_distance(b.vertices))
        d_ba = np.abs(qb.signed_distance(a.vertices))
    except Exception:  # pragma: no cover - proximity backend issue
        return None
    if d_ab.size == 0 or d_ba.size == 0:
        return None
    return float(min(d_ab.min(), d_ba.min()))


def _mesh_checks_per_body(model: AssemblyModel, checks: list[Check], fixes: list[Fix]) -> None:
    for body in model.bodies:
        # An authored body whose STEP would not import has no .cad here; its mesh was
        # already validated in its own authoring loop, so skip the per-body re-check
        # rather than crash on the missing STL path. Same for program=None (an AGENTIC
        # body -- the assembly adapter's documented contract): run_mesh_checks reads the
        # program's constraints, and the per-part loop already ran the full mesh tiers on
        # that body, so skipping here drops only a duplicate check. Without this skip,
        # validate_assembly CRASHED on every agentic assembly and a declared mechanism
        # could never earn its motion sweep (caught by the articulated-GLB tests).
        if body.cad is None or body.program is None:
            continue
        for c in run_mesh_checks(body.cad.stl_path, body.program):
            if c.id not in _CORE_MESH_CHECKS:
                continue
            short = c.id.split(".", 1)[1]
            checks.append(
                Check(
                    id=f"assembly.body.{body.id}.{short}",
                    tier=Tier.ASSEMBLY,
                    status=c.status,
                    message=f"[{body.id}] {c.message}",
                    expected=c.expected,
                    actual=c.actual,
                )
            )
            if c.status == CheckStatus.FAIL:
                fixes.append(Fix(reason=f"body {body.id} failed {c.id}: {c.message}"))


def _world_cylinders(
    body: PlacedBody,
) -> list[tuple[float, NDArray[np.float64], NDArray[np.float64]]]:
    # No B-rep (an authored body whose STEP did not import) -> no measurable cylinders,
    # so the cylinder-dependent checks (pin fit, shaft coupling) gracefully find none.
    if body.cad is None:
        return []
    out = []
    for c in extract_cylinders(body.cad.part):
        out.append(
            (
                float(c.radius),
                _apply_point(body.transform, c.axis_point),
                _unit(_apply_dir(body.transform, c.axis_dir)),
            )
        )
    return out


def _is_pin(body: PlacedBody) -> bool:
    """A pin = a rod (one cylinder) OR a captured_pin (a shaft + end flanges -- all coaxial ADD
    cylinders). It rides in a bore, so _pin_fit measures its SHAFT vs that bore. Distinguishes a
    pin from a bored plate/tube (which carry a Box op or a SUBTRACT bore, so they are not
    all-cylinder).

    A DSL-compiled body is classified by its program (all-cylinder ops -- 1 for a rod, 3 for a
    captured_pin). An AGENTIC body has no program, so fall back to geometry (a pin, captured or
    not, is a solid of revolution about one axis)."""
    if body.program is None:
        return _is_rod_by_geometry(body)
    ops = body.program.operations
    return len(ops) >= 1 and all(op.op == "cylinder" for op in ops)


def _is_rod_by_geometry(body: PlacedBody) -> bool:
    """Geometry-only rod test for an authored body (no PartProgram): a rod is a solid of
    revolution, so EVERY vertex lies within ~r of one cylinder's axis. A bored plate
    fails this -- its plate corners sprawl far beyond the small bore radius -- which a
    volume ratio cannot catch (a flat plate's volume can coincidentally equal pi*r^2*L of
    its knuckle barrel). Needs the B-rep (.cad) to know the candidate radius/axis."""
    if body.cad is None:
        return False
    cyls = extract_cylinders(body.cad.part)
    if not cyls:
        return False
    biggest = max(cyls, key=lambda cy: cy.radius)  # the rod's own outer surface
    r = float(biggest.radius)
    if r <= 0.0:
        return False
    axis = _unit(np.asarray(biggest.axis_dir, dtype=float))
    axis_pt = np.asarray(biggest.axis_point, dtype=float)
    verts = np.asarray(body.mesh.vertices, dtype=float)
    if len(verts) == 0:
        return False
    rel = verts - axis_pt
    along = rel @ axis
    perp = rel - np.outer(along, axis)
    max_perp = float(np.linalg.norm(perp, axis=1).max())
    axial_extent = float(np.ptp(along))
    # A rod is a solid of revolution (every vertex within ~r of the axis -- not a wide
    # plate) AND elongated along that axis (at least as long as its radius). The second
    # gate rejects a pancake -- a flat disk / washer / spacer also sits within r of its
    # axis but spans only axially, so without it a coin would read as a pin.
    return max_perp <= r * 1.25 and axial_extent >= r


def _pin_fit(model: AssemblyModel, joint: Joint) -> tuple[float, float] | None:
    """For a rod (pin/piston) inserted at ``joint``, return (axis_offset, clearance) vs
    its bore -- or None if joint.b is not a rod sitting in a bore."""
    # A non-printed hardware endpoint (ADR-0012) has no rod geometry and no printed
    # bore, so pin-fit does not apply -- skip before any printed-body lookup.
    if (
        model.find_hardware(joint.a_part()) is not None
        or model.find_hardware(joint.b_part()) is not None
    ):
        return None
    rod_body = model.body(joint.b_part())
    if not _is_pin(rod_body):
        return None
    rod_cyls = _world_cylinders(rod_body)
    if not rod_cyls:
        return None
    receiver = model.body(joint.a_part())
    a_port = next((p for p in receiver.ports if p.name == joint.a_port()), None)
    if a_port is None:
        return None
    port_o = _apply_point(receiver.transform, a_port.origin.as_tuple())
    port_d = _unit(_apply_dir(receiver.transform, a_port.axis.as_tuple()))
    coaxial = [
        c
        for c in _world_cylinders(receiver)
        if _angle_deg(c[2], port_d) <= _AXIS_PARALLEL_TOL_DEG
        and _line_offset(port_o, port_d, c[1]) <= 1.0
    ]
    if not coaxial:
        return None
    bore_r = min(coaxial, key=lambda c: c[0])[0]
    # The pin's BEARING surface (the shaft) is the largest pin cylinder that still fits in the
    # bore -- a captured_pin's end flanges are LARGER than the bore (they trap it, they do not
    # ride in it), so they must be excluded. Fall back to the smallest cylinder defensively if
    # none fit. (For a plain single-cylinder rod this is just that cylinder -> bench byte-stable.)
    shaft_cyls = [c for c in rod_cyls if c[0] <= bore_r + _COAXIAL_OFFSET_TOL_MM]
    rod_r, rod_p, _rod_d = (
        max(shaft_cyls, key=lambda c: c[0]) if shaft_cyls else min(rod_cyls, key=lambda c: c[0])
    )
    return _line_offset(port_o, port_d, rod_p), bore_r - rod_r


def _hardware_coaxial(model: AssemblyModel, joint: Joint) -> tuple[float, float] | None:
    """For an AXIAL joint with a non-printed hardware endpoint, measure (axis_angle_deg,
    axis_offset_mm) between the two MATED PORT frames in world coords -- or None if it
    does not apply. This is the frame-level coaxial check the actuated joint needs: a
    printed part mated to the motor 'shaft' port must share its axis. Pure frames -- no
    mesh -- so it works for a non-printed node.

    ONLY axial joints (coaxial_insert / revolute / prismatic) assert coaxiality. A
    ``fixed`` joint rigidly mates two frames at ANY relative orientation (a bracket
    bolted to the motor face needn't be axis-coaxial with the shaft), so it is excluded
    -- otherwise a perfectly valid mount fails this check whenever the bracket's mating
    port is offset from the motor axis (bench-007 dispatch finding)."""
    if joint.type not in (JointType.COAXIAL_INSERT, JointType.REVOLUTE, JointType.PRISMATIC):
        return None
    a_is_hw = model.find_hardware(joint.a_part()) is not None
    b_is_hw = model.find_hardware(joint.b_part()) is not None
    if not (a_is_hw or b_is_hw):
        return None
    try:
        a_tf, a_ports = model.placed_ports(joint.a_part())
        b_tf, b_ports = model.placed_ports(joint.b_part())
        a_port = next(p for p in a_ports if p.name == joint.a_port())
        b_port = next(p for p in b_ports if p.name == joint.b_port())
    except (KeyError, StopIteration):
        return None
    a_o = _apply_point(a_tf, a_port.origin.as_tuple())
    a_d = _unit(_apply_dir(a_tf, a_port.axis.as_tuple()))
    b_o = _apply_point(b_tf, b_port.origin.as_tuple())
    b_d = _unit(_apply_dir(b_tf, b_port.axis.as_tuple()))
    return _angle_deg(a_d, b_d), _line_offset(a_o, a_d, b_o)


def _shaft_coupling_checks(model: AssemblyModel, checks: list[Check], fixes: list[Fix]) -> None:
    """For each output_arm body, verify its shaft-grip hub: a gripping bore at the
    NEMA 17 shaft running-fit radius (it GRIPS the 5 mm shaft, not a clearance bore)
    plus a radial set-screw hole. Parameterised from the shared nema17_shaft facts so
    built geometry and the check read the same numbers (bench-007, ADR-0012)."""
    arms = [
        b
        for b in model.bodies
        if b.program is not None and any(op.op_id == "hub_bore" for op in b.program.operations)
    ]
    for arm in arms:
        prefix = f"interfaces.{arm.id}"
        cyls = _world_cylinders(arm)
        # The gripping bore: a cylinder at the shaft running-fit radius (within tol).
        grip = [c for c in cyls if abs(c[0] - _SHAFT_BORE_RADIUS) <= _SHAFT_BORE_TOL]
        grip_ok = len(grip) >= 1
        checks.append(
            Check(
                id=f"{prefix}.bore_grips_shaft",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if grip_ok else CheckStatus.FAIL,
                message=(
                    f"Hub bore grips the {2 * _SHAFT_RADIUS:.0f}mm shaft (running fit)."
                    if grip_ok
                    else f"No hub bore at the {_SHAFT_BORE_RADIUS:.1f}mm shaft running-fit radius."
                ),
                expected=f"r ~ {_SHAFT_BORE_RADIUS:.1f}mm (grips the {2 * _SHAFT_RADIUS:.0f}mm shaft)",
                actual=f"{len(grip)} matching bore(s)",
            )
        )
        if not grip_ok:
            fixes.append(
                Fix(reason="Hub bore must grip the 5mm shaft with a running fit (r~2.7mm).")
            )
        # The set screw: an M3 cylinder whose axis is perpendicular to the bore axis
        # (the bore is along +Z in local; the set screw drills radially, e.g. +Y).
        bore_axis = next((c[2] for c in grip), None)
        set_screws = [
            c
            for c in cyls
            if abs(c[0] - _SET_SCREW_RADIUS) <= _SHAFT_BORE_TOL
            and (bore_axis is None or _angle_deg(c[2], bore_axis) >= 90.0 - _AXIS_PARALLEL_TOL_DEG)
        ]
        ss_ok = len(set_screws) >= 1
        checks.append(
            Check(
                id=f"{prefix}.set_screw_present",
                tier=Tier.INTERFACES,
                status=CheckStatus.PASS if ss_ok else CheckStatus.FAIL,
                message=(
                    "Radial M3 set-screw hole present in the hub."
                    if ss_ok
                    else "No radial M3 set-screw hole found in the hub."
                ),
                expected=f"1+ radial hole, r~{_SET_SCREW_RADIUS:.1f}mm",
                actual=f"{len(set_screws)} matching hole(s)",
            )
        )
        if not ss_ok:
            fixes.append(Fix(reason="Add a radial M3 set-screw hole into the hub bore wall."))


def _print_in_place_gap_check(model: AssemblyModel, checks: list[Check], fixes: list[Fix]) -> None:
    """NEW (ADR-0013 Q4b): a PRINT-IN-PLACE assembly comes off the bed already moving, so every
    moving interface must keep a printable gap -- not so tight the slicer welds the two shells into
    one solid (no joint), not so loose the joint rattles. Two BOUNDED measurements, deliberately
    NOT a support/bridging analyzer (the S4 go/no-go: keep it small):
      * FUSE (lower bound): the min surface gap between each non-fixed body pair >= FDM_MIN_GAP.
      * RATTLE (upper bound): each print-in-place pin's radial bore clearance (reused _pin_fit)
        <= FDM_MAX_GAP.
    Runs ONLY when a print_in_place joint is declared, so non-print-in-place assemblies
    (bench-004/005/007, running_fit) are byte-unaffected."""
    pip = [j for j in model.joints if j.clearance_class == "print_in_place"]
    if not pip:
        return
    # FIXED-joined bodies are MEANT to fuse (a welded mate); never flag those pairs.
    fixed_pairs = {
        frozenset((j.a_part(), j.b_part())) for j in model.joints if j.type is JointType.FIXED
    }
    placed = {b.id: _placed_mesh(b) for b in model.bodies}
    worst_pair: str | None = None
    worst_gap = math.inf
    measured = 0
    attempted = 0
    for i, a in enumerate(model.bodies):
        for b in model.bodies[i + 1 :]:
            if frozenset((a.id, b.id)) in fixed_pairs:
                continue
            attempted += 1
            gap = _min_surface_gap(placed[a.id], placed[b.id])
            if gap is None:
                continue
            measured += 1
            if gap < worst_gap:
                worst_pair, worst_gap = f"{a.id}<->{b.id}", gap

    pin_gaps = [f[1] for f in (_pin_fit(model, j) for j in pip) if f is not None]
    worst_rattle = max(pin_gaps) if pin_gaps else None

    # fuse_ok: pass only if there was nothing to measure (no non-fixed pairs) OR we actually measured
    # a gap that clears the minimum. If pairs existed but NONE could be measured, we cannot certify
    # the joint won't fuse -> fail closed (audit M4): an unmeasured interface is not a passed one.
    fuse_ok = attempted == 0 or (measured > 0 and worst_gap >= FDM_MIN_GAP_MM)
    rattle_ok = worst_rattle is None or worst_rattle <= FDM_MAX_GAP_MM
    ok = fuse_ok and rattle_ok
    if not fuse_ok and measured == 0:
        detail = (
            f"could not measure any of {attempted} moving interface(s) -- cannot certify the joint "
            "won't fuse"
        )
    elif not fuse_ok:
        detail = (
            f"{worst_pair} gap {worst_gap:.2f}mm < {FDM_MIN_GAP_MM}mm -- would fuse into one solid"
        )
    elif not rattle_ok:
        assert worst_rattle is not None
        detail = f"pin radial gap {worst_rattle:.2f}mm > {FDM_MAX_GAP_MM}mm -- joint would rattle"
    elif measured == 0:
        detail = "no moving interface to measure (no non-fixed body pairs)"
    else:
        detail = f"tightest moving gap {worst_gap:.2f}mm in [{FDM_MIN_GAP_MM}, {FDM_MAX_GAP_MM}]mm"
    checks.append(
        Check(
            id="assembly.print_in_place_gap",
            tier=Tier.ASSEMBLY,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message=(
                "Every moving interface keeps a printable gap (won't fuse, won't rattle)."
                if ok
                else f"A print-in-place interface won't print as a working joint: {detail}."
            ),
            expected=f"every moving gap in [{FDM_MIN_GAP_MM}, {FDM_MAX_GAP_MM}] mm",
            actual=detail,
            tolerance=f"nozzle {NOZZLE_DIAMETER_MM}mm",
        )
    )
    if not ok:
        fixes.append(
            Fix(
                reason=(
                    "Tighten a too-loose pin or widen a too-tight interface to ~one nozzle width: "
                    "author the axial knuckle-face gap and the pin's print_in_place radial clearance "
                    "so adjacent moving shells neither fuse nor rattle."
                )
            )
        )


def _gear_specs(joint: Joint) -> tuple[GearSpec, GearSpec] | None:
    """Build the two meshing GearSpecs from a GEAR_MESH joint's fields, or None if the joint
    is under-specified (FAIL-OPEN: a missing module / tooth count skips the check rather than
    inventing a false FAIL on an input we cannot read)."""
    if joint.module is None or joint.teeth_a is None or joint.teeth_b is None:
        return None
    try:
        m = float(joint.module)
        za = int(joint.teeth_a)
        zb = int(joint.teeth_b)
    except (TypeError, ValueError):
        return None
    if m <= 0.0 or za <= 0 or zb <= 0:
        return None
    return GearSpec(module=m, teeth=za), GearSpec(module=m, teeth=zb)


def _placed_axis(
    model: AssemblyModel, endpoint_part: str, endpoint_port: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """World (axis_origin, unit_axis_dir) for a gear's rotation axis from its joint port, or None
    if the part / port cannot be resolved (FAIL-OPEN)."""
    try:
        tf, ports = model.placed_ports(endpoint_part)
        port = next(p for p in ports if p.name == endpoint_port)
    except (KeyError, StopIteration):
        return None
    o = _apply_point(tf, port.origin.as_tuple())
    d = _unit(_apply_dir(tf, port.axis.as_tuple()))
    return o, d


def _gear_mesh_checks(model: AssemblyModel, checks: list[Check], fixes: list[Fix]) -> None:
    """For each GEAR_MESH joint (ADR-0010 K1): two independent proofs that the pair actually meshes.

    * SPACING (cheap, deterministic, no kernel): the ACTUAL built centre distance -- the perpendicular
      distance between the two gears' placed rotation axes -- must equal the theoretical
      ``center_distance(g1, g2)`` (sum of pitch radii) within tolerance, and the axes must be
      parallel. A mis-spaced or skew pair FAILS here (too close -> teeth jam; too far -> they lose
      contact). This is the no-false-PASS guard against gears placed so far apart they never touch.
    * MESHING (kinematic, builds geometry): rotate gear a and counter-rotate gear b by the gear
      RATIO over a sampled tooth cycle and assert the teeth stay in a meshing-contact band (some
      pitch overlap = contact) and never gross-jam. Promotes ``validate_gear_mesh_kinematic`` rather
      than reinventing the sweep; a mis-clocked pair BINDS and FAILS.

    Runs ONLY when a GEAR_MESH joint is declared, so every non-gear assembly is byte-unaffected.
    Every unresolvable input FAILS OPEN (skips that joint) so no new false-FAIL is introduced."""
    gear_joints = [j for j in model.joints if j.type is JointType.GEAR_MESH]
    if not gear_joints:
        return

    spacing_worst: tuple[str, float, float] | None = None  # (label, actual_cd, theoretical_cd)
    skew_worst: tuple[str, float] | None = None  # (label, axis angle deg)
    mesh_failure: str | None = None
    checked = 0
    simulated = 0
    internal_pairs = 0
    unresolved: list[str] = []

    for joint in gear_joints:
        label = f"{joint.a_part()}<->{joint.b_part()}"
        specs = _gear_specs(joint)
        if specs is None:
            unresolved.append(f"{label} (missing module/tooth counts)")
            continue
        g1, g2 = specs
        a_axis = _placed_axis(model, joint.a_part(), joint.a_port())
        b_axis = _placed_axis(model, joint.b_part(), joint.b_port())
        if a_axis is None or b_axis is None:
            unresolved.append(f"{label} (gear axis could not be resolved)")
            continue
        checked += 1

        a_o, a_d = a_axis
        b_o, b_d = b_axis
        skew = _angle_deg(a_d, b_d)
        if skew > _AXIS_PARALLEL_TOL_DEG and (skew_worst is None or skew > skew_worst[1]):
            skew_worst = (label, skew)
        actual_cd = _line_offset(a_o, a_d, b_o)
        # An INTERNAL (ring) mesh runs at the pitch-radius DIFFERENCE, an external mesh
        # at the sum -- the north-star gear-train probe's three correct ring meshes at
        # |48-18|*2/2 = 30 mm would false-FAIL against the sum (66 mm). The joint's
        # declared ``internal`` flag picks the formula; a WRONG spacing still fails
        # against its own correct theoretical either way (strictness preserved).
        if joint.internal:
            internal_pairs += 1
            theoretical_cd = abs(g1.teeth - g2.teeth) * float(g1.module) / 2.0
        else:
            theoretical_cd = center_distance(g1, g2)
        if abs(actual_cd - theoretical_cd) > _GEAR_CENTER_DISTANCE_TOL_MM and (
            spacing_worst is None
            or abs(actual_cd - theoretical_cd) > abs(spacing_worst[1] - spacing_worst[2])
        ):
            spacing_worst = (label, actual_cd, theoretical_cd)

        # The kinematic tooth sim models EXTERNAL involute pairs only; an internal ring
        # mesh is spacing/skew-gated above and its sim coverage is reported honestly in
        # the check text below (never silently counted as simulated).
        if mesh_failure is None and not joint.internal:
            simulated += 1
            result = validate_gear_mesh_kinematic(
                g1,
                g2,
                clock1_deg=joint.clock_a_deg,
                clock2_deg=joint.clock_b_deg,
            )
            if not result.meshes:
                mesh_failure = f"{label}: {result.note}"

    # A DECLARED gear mesh we could not verify must NOT pass silently (audit M3, ADR-0065). Before,
    # an under-specified joint was skipped, so if a SIBLING joint was valid the assembly still
    # reported its gears validated. Emit a FAIL instead -- fail closed on an unverifiable claim.
    if unresolved:
        checks.append(
            Check(
                id="assembly.gear_mesh_declared",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message=(
                    "A declared gear mesh could not be verified: "
                    + "; ".join(unresolved)
                    + ". A declared GEAR_MESH must specify module + both tooth counts and resolve "
                    "its gear axes."
                ),
                expected="every declared GEAR_MESH fully specified + resolvable",
                actual=f"{len(unresolved)} unverifiable of {len(gear_joints)} declared",
            )
        )
        fixes.append(
            Fix(
                reason=(
                    "Complete the under-specified GEAR_MESH joint(s): set module, teeth_a, teeth_b, "
                    "and valid ports so the mesh can be checked."
                )
            )
        )

    if checked == 0:
        return  # the spacing/meshing asserts below need at least one fully resolved pair

    spacing_ok = spacing_worst is None and skew_worst is None
    if skew_worst is not None:
        spacing_detail = f"{skew_worst[0]} gear axes skew {skew_worst[1]:.1f} deg (not parallel)"
    elif spacing_worst is not None:
        spacing_detail = (
            f"{spacing_worst[0]} centre distance {spacing_worst[1]:.2f}mm vs required "
            f"{spacing_worst[2]:.2f}mm (mis-spaced)"
        )
    else:
        spacing_detail = (
            f"all {checked} gear pair(s) spaced at the correct centre distance "
            f"({internal_pairs} internal at the pitch-radius difference)"
            if internal_pairs
            else f"all {checked} gear pair(s) spaced at the pitch-sum centre distance"
        )
    checks.append(
        Check(
            id="assembly.gear_center_distance",
            tier=Tier.ASSEMBLY,
            status=CheckStatus.PASS if spacing_ok else CheckStatus.FAIL,
            message=(
                "Every gear pair is on parallel axes at the correct centre distance."
                if spacing_ok
                else f"A gear pair is mis-spaced or skew: {spacing_detail}."
            ),
            expected="centre distance = sum of pitch radii (external) / |difference| (internal) "
            f"(+/- {_GEAR_CENTER_DISTANCE_TOL_MM}mm), axes parallel",
            actual=spacing_detail,
        )
    )
    if not spacing_ok:
        fixes.append(
            Fix(
                reason="Re-space the meshing gears on parallel axes to the correct centre "
                "distance: sum of the two pitch radii (external), |difference| (internal ring)."
            )
        )

    mesh_ok = mesh_failure is None
    sim_scope = (
        f"{simulated} external pair(s) tooth-simulated; {internal_pairs} internal ring "
        "pair(s) spacing/skew-gated only (no internal tooth sim yet)"
        if internal_pairs
        else f"all {checked} gear pair(s) tooth-simulated"
    )
    checks.append(
        Check(
            id="assembly.gear_mesh",
            tier=Tier.ASSEMBLY,
            status=CheckStatus.PASS if mesh_ok else CheckStatus.FAIL,
            message=(
                f"Gear pairs mesh and turn together by the gear ratio without jamming ({sim_scope})."
                if mesh_ok
                else f"A gear pair does not mesh: {mesh_failure}."
            ),
            expected="teeth stay in a meshing-contact band over a tooth cycle (no gross jam)",
            actual=sim_scope if mesh_ok else mesh_failure,
        )
    )
    if not mesh_ok:
        fixes.append(
            Fix(
                reason="Re-clock the meshing gear (tooth phase) or fix the module / tooth counts so "
                "the teeth interleave instead of colliding."
            )
        )


_MOUNT_POS_TOL_MM = 0.6  # measured hole axis must pass this close to the cited point


def _world_concave_cylinders(
    body: PlacedBody,
) -> list[tuple[float, NDArray[np.float64], NDArray[np.float64]]]:
    """World-placed (radius, axis_point, axis_dir) of the body's MEASURED-CONCAVE
    cylindrical faces only -- hole walls. A boss (convex) at the cited point must NOT
    satisfy a mount hole, so concavity is required, not optional (no-false-PASS)."""
    if body.cad is None:
        return []
    out = []
    for c in extract_cylinders(body.cad.part):
        if c.concave is not True:
            continue
        out.append(
            (
                float(c.radius),
                _apply_point(body.transform, c.axis_point),
                _unit(_apply_dir(body.transform, c.axis_dir)),
            )
        )
    return out


def _mount_pattern_checks(model: AssemblyModel, checks: list[Check], fixes: list[Fix]) -> None:
    """Every printed part bolted to a hardware port must carry that hardware's CITED
    hole pattern, measured on its compiled B-rep: each cited point needs a concave
    cylinder of the cited diameter, parallel to the cited axis (within
    ``_AXIS_PARALLEL_TOL_DEG``), whose axis line passes within ``_MOUNT_POS_TOL_MM``.
    Hardware with nothing joined at a pattern port claims nothing there."""
    for hw in model.hardware:
        for pattern in patterns_for(hw.kind):
            target: str | None = None
            for j in model.joints:
                if (
                    j.a_part() == hw.id
                    and j.a_port() == pattern.port
                    and model.find_hardware(j.b_part()) is None
                ):
                    target = j.b_part()
                elif (
                    j.b_part() == hw.id
                    and j.b_port() == pattern.port
                    and model.find_hardware(j.a_part()) is None
                ):
                    target = j.a_part()
            if target is None:
                continue
            body = model.body(target)
            cyls = _world_concave_cylinders(body)
            axis_w = _unit(_apply_dir(hw.transform, pattern.axis_local))
            missing: list[str] = []
            for point_local, (d_lo, d_hi) in pattern.holes:
                p_w = _apply_point(hw.transform, point_local)
                hit = any(
                    d_lo <= 2.0 * r <= d_hi
                    and _angle_deg(d, axis_w) <= _AXIS_PARALLEL_TOL_DEG
                    and _line_offset(cp, d, p_w) <= _MOUNT_POS_TOL_MM
                    for r, cp, d in cyls
                )
                if not hit:
                    missing.append(
                        f"({point_local[0]:.1f},{point_local[1]:.1f},{point_local[2]:.1f})"
                        f" d {d_lo:.1f}-{d_hi:.1f}"
                    )
            ok = not missing
            checks.append(
                Check(
                    id=f"assembly.mount_pattern.{hw.id}.{pattern.port}",
                    tier=Tier.ASSEMBLY,
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    message=(
                        f"[{target}] carries the cited {pattern.label}."
                        if ok
                        else f"[{target}] is missing cited mount hole(s) for {pattern.label}: "
                        + "; ".join(missing)
                    ),
                    expected=f"{len(pattern.holes)} hole(s) per {pattern.source}",
                    actual=("all present" if ok else f"{len(missing)} missing"),
                    tolerance=f"pos <= {_MOUNT_POS_TOL_MM} mm, axis <= {_AXIS_PARALLEL_TOL_DEG} deg",
                )
            )
            if not ok:
                fixes.append(
                    Fix(
                        reason=(
                            f"{target} must carry the {pattern.label} ({pattern.source}); "
                            "move/size the mount holes to the cited pattern."
                        )
                    )
                )


def gear_coupled_pairs(joints: list[Joint]) -> list[tuple[Joint, Joint, float]]:
    """(driver_revolute, follower_revolute, ratio) for every GEAR_MESH joint whose two
    gears are each carried by their own revolute joint -- the coupled-finger topology
    (ADR-0044). The gear joint's ``a`` side is the driver by the joint's own
    convention ("a turns, b counter-rotates"); ratio = teeth_a / teeth_b."""
    revolutes = {j.b_part(): j for j in joints if j.type is JointType.REVOLUTE}
    out: list[tuple[Joint, Joint, float]] = []
    for g in joints:
        if g.type is not JointType.GEAR_MESH:
            continue
        driver = revolutes.get(g.a_part())
        follower = revolutes.get(g.b_part())
        if driver is None or follower is None or g.teeth_a is None or g.teeth_b is None:
            continue
        out.append((driver, follower, float(g.teeth_a) / float(g.teeth_b)))
    return out


def _pad_region_vertices(body: PlacedBody, op_id: str) -> NDArray[np.float64] | None:
    """The body's LOCAL mesh vertices inside the named op's local AABB (the grip-pad
    region) -- measured geometry, located by the op the builder tagged."""
    from forge_mind.dsl.primitives import Box

    op = next((o for o in body.program.operations if o.op_id == op_id), None)
    if op is None or not isinstance(op, Box):
        return None
    half = np.array([op.width / 2.0, op.depth / 2.0, op.height / 2.0]) + 0.1
    center = np.array([op.position.x, op.position.y, op.position.z])
    v = np.asarray(body.mesh.vertices, dtype=float)
    inside = np.all(np.abs(v - center) <= half, axis=1)
    pts = v[inside]
    return pts if len(pts) >= 3 else None


def _gripper_payload_checks(
    model: AssemblyModel,
    checks: list[Check],
    fixes: list[Fix],
    payload_key: str,
    sweeper: ChainSweeper | None,
) -> None:
    """The payload-scoped claims (ADR-0044), run ONLY when the request DECLARED a
    payload object:

    * ``assembly.gripper_aperture`` -- MAY GATE (FAIL): measured pad-to-pad span at
      the coupled OPEN and CLOSED extremes (the same FK transforms the sweeper poses
      with) vs the CITED payload diameter. Open must clear the payload by >= 8 mm;
      closed must undercut it by >= 2 mm (the pads can actually clamp). Measured
      geometry against a cited registry fact -- the same epistemic class as the
      mount-pattern gate. Unmeasurable = FAIL (a declared claim we cannot verify is
      never silently passed).
    * ``assembly.payload_torque`` -- INFO ONLY, never a gate in either direction:
      the cited stall torque vs the payload moment at the measured reach. "Can lift"
      depends on unmodeled friction/voltage/dynamics; the arithmetic is reported,
      and the fitness label (not the validator) carries any claim reduction.
    """
    from forge_mind.assembly.servo import (
        MG996R_STALL_TORQUE_KG_CM_4V8,
        MG996R_STALL_TORQUE_KG_CM_6V0,
    )
    from forge_mind.dsl.standards import payload_object as _payload_object
    from forge_mind.validator.motion_sweep import _rotation_about_line

    try:
        payload = _payload_object(payload_key)
    except ValueError:
        checks.append(
            Check(
                id="assembly.gripper_aperture",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message=f"Declared payload object {payload_key!r} is not in the cited registry.",
                expected="a registered payload object",
                actual=payload_key,
            )
        )
        return

    pairs = gear_coupled_pairs(model.joints)
    if sweeper is None or not pairs:
        checks.append(
            Check(
                id="assembly.gripper_aperture",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message=(
                    "The declared payload claim cannot be verified: "
                    + ("no coupled finger pair found" if not pairs else "geometry not sweepable")
                    + " -- a claim that cannot be measured is never passed."
                ),
                expected="a measurable coupled finger pair",
                actual="unmeasurable",
            )
        )
        return

    driver, follower, ratio = pairs[0]
    lo, hi = driver.range_deg or (0.0, 0.0)
    d_body = model.body(driver.b_part())
    f_body = model.body(follower.b_part())
    d_pad = _pad_region_vertices(d_body, "grip_pad")
    f_pad = _pad_region_vertices(f_body, "grip_pad")
    if d_pad is None or f_pad is None:
        checks.append(
            Check(
                id="assembly.gripper_aperture",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message="A finger carries no measurable 'grip_pad' region; aperture unverifiable.",
                expected="grip_pad ops with mesh vertices on both fingers",
                actual="missing pad region",
            )
        )
        return

    d_pivot_o, d_pivot_d = sweeper.joint_pivot(driver, {})
    f_pivot_o, f_pivot_d = sweeper.joint_pivot(follower, {})
    grip_dir = _unit(f_pivot_o - d_pivot_o)

    def aperture_at(theta: float) -> float:
        dm = _rotation_about_line(d_pivot_o, d_pivot_d, theta)
        fm = _rotation_about_line(f_pivot_o, f_pivot_d, -theta * ratio)
        dv = trimesh.transform_points(d_pad, dm @ d_body.transform)
        fv = trimesh.transform_points(f_pad, fm @ f_body.transform)
        return float(np.min(fv @ grip_dir) - np.max(dv @ grip_dir))

    a_lo, a_hi = aperture_at(lo), aperture_at(hi)
    open_mm, closed_mm = max(a_lo, a_hi), min(a_lo, a_hi)
    open_ok = open_mm >= payload.grip_diameter_mm + 8.0
    closed_ok = closed_mm <= payload.grip_diameter_mm - 2.0
    ok = open_ok and closed_ok
    checks.append(
        Check(
            id="assembly.gripper_aperture",
            tier=Tier.ASSEMBLY,
            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
            message=(
                f"The measured aperture admits the cited {payload.grip_diameter_mm:.0f} mm "
                f"payload ({payload_key}): open {open_mm:.1f} mm, closed {closed_mm:.1f} mm."
                if ok
                else (
                    f"The measured aperture does not grip the cited payload ({payload_key}): "
                    f"open {open_mm:.1f} mm (need >= {payload.grip_diameter_mm + 8.0:.1f}), "
                    f"closed {closed_mm:.1f} mm (need <= {payload.grip_diameter_mm - 2.0:.1f})."
                )
            ),
            expected=(
                f"open >= {payload.grip_diameter_mm + 8.0:.1f} mm, "
                f"closed <= {payload.grip_diameter_mm - 2.0:.1f} mm"
            ),
            actual=f"open {open_mm:.1f} mm, closed {closed_mm:.1f} mm",
        )
    )
    if not ok:
        fixes.append(
            Fix(
                reason=(
                    "Resize the finger reach / pad faces so the open span clears the cited "
                    "payload diameter and the closed span undercuts it."
                )
            )
        )

    # --- payload torque: REPORTED arithmetic, INFO in both directions (never a gate). ---
    driver_hw = model.find_hardware(driver.a_part())
    if driver_hw is not None and driver_hw.kind == "mg996r_servo":
        pad_centroid = trimesh.transform_points(d_pad, d_body.transform).mean(axis=0)
        reach_cm = float(np.linalg.norm(pad_centroid - d_pivot_o)) / 10.0
        moment_kg_cm = (payload.mass_full_g / 1000.0) * reach_cm
        checks.append(
            Check(
                id="assembly.payload_torque",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.INFO,
                message=(
                    f"Reported arithmetic (never a pass criterion): holding the full "
                    f"{payload.mass_full_g:.0f} g payload at the measured {reach_cm:.1f} cm pad "
                    f"reach loads ~{moment_kg_cm:.2f} kg.cm against the cited MG996R stall "
                    f"torque {MG996R_STALL_TORQUE_KG_CM_4V8:.1f} kg.cm @4.8V / "
                    f"{MG996R_STALL_TORQUE_KG_CM_6V0:.1f} kg.cm @6.0V (TowerPro datasheet, "
                    "provisional). Grip force and friction on the payload surface are NOT "
                    "modeled; lifting is not claimed."
                ),
                expected="reported only",
                actual=f"{moment_kg_cm:.2f} kg.cm at {reach_cm:.1f} cm",
            )
        )


def validate_assembly(
    model: AssemblyModel, *, payload_object: str | None = None
) -> ValidationReport:
    """Run the relational + kinematic tier and assemble a report.

    ``payload_object`` names a CITED payload-object registry key when the request
    DECLARED one (e.g. a gripper asked to grip a soda can) -- it arms the
    payload-scoped checks; None (the default) makes no payload claim."""
    checks: list[Check] = []
    fixes: list[Fix] = []
    payload_sweeper: ChainSweeper | None = None

    _mesh_checks_per_body(model, checks, fixes)
    _shaft_coupling_checks(model, checks, fixes)
    _print_in_place_gap_check(model, checks, fixes)
    _gear_mesh_checks(model, checks, fixes)
    _mount_pattern_checks(model, checks, fixes)

    coax = [j for j in model.joints if j.type == JointType.COAXIAL_INSERT]
    revolute = [j for j in model.joints if j.type == JointType.REVOLUTE]
    prismatic = [j for j in model.joints if j.type == JointType.PRISMATIC]
    fixed = [j for j in model.joints if j.type == JointType.FIXED]
    gear_mesh = [j for j in model.joints if j.type == JointType.GEAR_MESH]

    # Hardware-anchored coaxial fits computed up front (per joint): a validated
    # hardware joint (e.g. a hub on the motor shaft) is a real coaxial relationship.
    hw_fits = [f for f in (_hardware_coaxial(model, j) for j in model.joints) if f is not None]

    # Are there ALL-PRINTED coaxial-class joints? If so, a missing pin fit is a real
    # printed-mechanism defect and must FAIL even when an unrelated hardware coaxial
    # joint exists (Codex P2b: a hardware joint must not mask a malformed printed one).
    def _is_hw_joint(j: Joint) -> bool:
        return (
            model.find_hardware(j.a_part()) is not None
            or model.find_hardware(j.b_part()) is not None
        )

    printed_coaxial_joints = [j for j in (coax + prismatic + fixed) if not _is_hw_joint(j)]

    # --- per-pin coaxial + clearance: each inserted rod (pin/piston) measured vs ITS
    # OWN bore at its joint port. A hinge has one pin; a linkage has several. ---
    # The witness NAMES its pair (L100, review of the F2 postmortem): an aggregate
    # "a pin somewhere" left a 39-part run's bind unactionable -- and the old FAIL
    # message claimed inversion ("bore not larger") when F2's clearance was positive
    # but under the floor. `actual` format is a parsed interface; messages carry the
    # location instead.
    fits = [(j, f) for j in (coax + prismatic + fixed) if (f := _pin_fit(model, j)) is not None]
    if fits:
        worst_j, (worst_offset, _pin_clear_unused) = max(fits, key=lambda t: t[1][0])
        min_j, (_pin_off_unused, min_clearance) = min(fits, key=lambda t: t[1][1])
        coax_ok = worst_offset <= _COAXIAL_OFFSET_TOL_MM
        checks.append(
            Check(
                id="assembly.pin_coaxial",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.PASS if coax_ok else CheckStatus.FAIL,
                message=(
                    f"All {len(fits)} pin(s) are coaxial with their bores."
                    if coax_ok
                    else f"Pin {worst_j.b_part()} is not coaxial with its bore in "
                    f"{worst_j.a_part()}."
                ),
                expected=f"<= {_COAXIAL_OFFSET_TOL_MM} mm axis offset",
                actual=f"{worst_offset:.3f} mm",
            )
        )
        clear_ok = min_clearance >= _MIN_CLEARANCE_MM
        checks.append(
            Check(
                id="assembly.pin_clearance",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.PASS if clear_ok else CheckStatus.FAIL,
                message=(
                    "Every pin has positive running-fit clearance in its bore."
                    if clear_ok
                    else f"Pin {min_j.b_part()} does not run in its bore in "
                    f"{min_j.a_part()}: radial clearance is under the "
                    f"{_MIN_CLEARANCE_MM} mm floor."
                ),
                expected=f">= {_MIN_CLEARANCE_MM} mm radial clearance",
                actual=f"{min_clearance:.3f} mm",
            )
        )
        if not clear_ok:
            fixes.append(
                Fix(
                    reason="A pin radius >= its bore radius; enlarge the bore or shrink the pin.",
                    suggested_change={"min_clearance_mm": _MIN_CLEARANCE_MM},
                )
            )
    elif not hw_fits or printed_coaxial_joints:
        # FAIL when there is nothing coaxial to verify at all (no hw joint either), OR
        # when an ALL-PRINTED coaxial-class joint exists but produced no rod-in-bore
        # fit (a malformed printed mechanism -- a hardware joint elsewhere must not
        # mask it, Codex P2b). Only a purely hardware-coaxial assembly suppresses this.
        checks.append(
            Check(
                id="assembly.pin_coaxial",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message="No inserted pin / rod-in-bore found to verify.",
                expected="a pin inserted coaxially into a bore",
                actual="no rod-in-bore joint",
            )
        )

    # --- hardware-anchored coaxial: a part mated to a hardware frame (e.g. the motor
    # shaft) must be coaxial with it. Frame-level (no mesh), so it covers the
    # non-printed node the pin_coaxial/mesh path above cannot. ---
    if hw_fits:
        worst_angle = max(angle for angle, _ in hw_fits)
        worst_offset = max(offset for _, offset in hw_fits)
        hw_ok = worst_angle <= _AXIS_PARALLEL_TOL_DEG and worst_offset <= _COAXIAL_OFFSET_TOL_MM
        checks.append(
            Check(
                id="assembly.hardware_coaxial",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.PASS if hw_ok else CheckStatus.FAIL,
                message=(
                    f"All {len(hw_fits)} hardware-anchored joint(s) are coaxial with their frame."
                    if hw_ok
                    else "A part mated to a hardware frame is not coaxial with it."
                ),
                expected=f"<= {_AXIS_PARALLEL_TOL_DEG} deg, <= {_COAXIAL_OFFSET_TOL_MM} mm offset",
                actual=f"{worst_angle:.2f} deg, {worst_offset:.3f} mm",
            )
        )
        if not hw_ok:
            fixes.append(
                Fix(reason="A hardware-anchored part is off-axis; mate it to the correct frame.")
            )

    # --- range of motion: CHAIN-AWARE swept collision (the assembly claim gate's
    # motion half). Each moving joint sweeps its full DOWNSTREAM subtree against the
    # entire static rest -- printed bodies AND hardware envelopes -- over its whole
    # declared range, plus corner poses for chained joints. "Swept" is EARNED: if any
    # hardware node lacks an envelope mesh (legacy ADR-0012 shape) or any solid cannot
    # be sampled, NO joint claims swept and the check WARNs (never silently clear). ---
    moving_joints = revolute + prismatic
    if moving_joints:
        failure: str | None = None
        swept = 0
        skipped_reason = ""
        meshless = sorted(meshless_hardware_ids(model))
        sweeper: ChainSweeper | None = None
        if meshless:
            skipped_reason = "hardware without an envelope mesh: " + ", ".join(meshless)
        else:
            sweeper = ChainSweeper(model)
            if not sweeper.all_sampled_ok():
                skipped_reason = "un-sampleable solid(s), fail closed: " + ", ".join(
                    sweeper.unsampleable_ids()
                )
                sweeper = None
        if sweeper is None:
            # The home-pose clearance check rides the same sweeper, so without it the
            # `assembly.static_clearance` row simply VANISHED from the report -- the one
            # genuinely silent hole here (the motion claim itself already WARNs below and
            # names the cause). A check that disappears is indistinguishable from a check
            # that passed, so say WARN with the reason instead. Refusing to CLAIM clearance
            # stays correct: an un-meshed envelope is a body we cannot measure against, and
            # "clear" would be over-claiming exactly what we cannot see.
            checks.append(
                Check(
                    id="assembly.static_clearance",
                    tier=Tier.ASSEMBLY,
                    status=CheckStatus.WARN,
                    message=(
                        "Home-pose interpenetration was NOT measured: "
                        f"{skipped_reason}. No clearance claim is earned for this assembly."
                    ),
                    expected="no interior overlap between any pair",
                    actual=f"not measured ({skipped_reason})",
                )
            )
        if sweeper is not None:
            payload_sweeper = sweeper
            # The parts must actually assemble: no interior overlap at the HOME pose.
            overlapping = sweeper.static_overlaps()
            checks.append(
                Check(
                    id="assembly.static_clearance",
                    tier=Tier.ASSEMBLY,
                    status=CheckStatus.PASS if not overlapping else CheckStatus.FAIL,
                    message=(
                        "No two parts interpenetrate at the assembled (home) pose."
                        if not overlapping
                        else "Parts interpenetrate at the assembled pose: " + ", ".join(overlapping)
                    ),
                    expected="no interior overlap between any pair",
                    actual=("clear" if not overlapping else "; ".join(overlapping)),
                )
            )
            if overlapping:
                fixes.append(
                    Fix(reason="Overlapping pairs at home pose: " + ", ".join(overlapping))
                )
            # Gear-coupled revolute pairs (ADR-0044) sweep their REAL trajectory
            # together (driver theta, follower -theta*ratio): sweeping either alone
            # visits poses the mechanism never reaches -- meshing teeth at wrong
            # relative angles -- so the independent sweep can neither earn nor
            # honestly deny the claim. Each pair is handled ONCE; both joints' check
            # ids are issued from the one coupled run.
            coupled = gear_coupled_pairs(model.joints)
            coupled_member_ids = {id(j) for pair in coupled for j in pair[:2]}
            # Closed-loop ownership (rows 59-61): loops are structurally UNSOUND under
            # the one-joint-at-a-time tree sweep. Detection runs on the PIN-COLLAPSED
            # graph (the-pin-is-a-part rule hides classic loop shapes behind pin parts);
            # slider-crank and four-bar loops get coupled loop-closure drivers (pins
            # riding their host links); any remaining unowned cycle has its motion claim
            # WITHHELD (WARN) -- never tree-swept into a phantom verdict.
            collapsed_joints, pin_hosts, origin_map = collapse_pin_edges(model.joints)
            sc_systems = slider_crank_systems(collapsed_joints)
            sc_synth_ids = {
                id(j)
                for (j0, _frame, _crank, loops) in sc_systems
                for j in (j0, *(jj for lp in loops for jj in lp.joints()))
            }
            fb_systems = four_bar_systems(
                [j for j in collapsed_joints if id(j) not in sc_synth_ids]
            )

            def _originals(j: Joint) -> tuple[Joint, ...]:
                return origin_map.get(id(j), (j,))

            owned_original_ids: set[int] = set()
            for _j0, _f, _c, _loops in sc_systems:
                for j in (_j0, *(jj for lp in _loops for jj in lp.joints())):
                    owned_original_ids.update(id(oj) for oj in _originals(j))
            for _j0, _f, _c, _fb_loops in fb_systems:
                for j in (_j0, *(jj for lp in _fb_loops for jj in lp.joints())):
                    owned_original_ids.update(id(oj) for oj in _originals(j))
            # Epicyclic trains (row 69, the planetary north star): gear meshes riding a
            # moving carrier -- previously the 'coupled pair within a moving chain'
            # withhold; now driven closed-form by sweep_epicyclic. Detected AFTER
            # sc/fb so a shared joint stays with its first owner (conservative: an
            # intersecting candidate is dropped back to the withhold).
            epi_systems = [
                sysm
                for sysm in epicyclic_systems(collapsed_joints)
                if not any(
                    id(oj) in owned_original_ids
                    for j in (sysm[0], sysm[3][0].carrier_joint, *sysm[3][0].owned_joints)
                    for oj in _originals(j)
                )
            ]
            epi_owned_ids: set[int] = set()
            for _j0, _f, _c, _sets in epi_systems:
                for st in _sets:
                    for j in (_j0, st.carrier_joint, *st.owned_joints):
                        epi_owned_ids.update(id(oj) for oj in _originals(j))
                    for m in (*st.sun_meshes, *st.ring_meshes):
                        epi_owned_ids.update(id(oj) for oj in _originals(m))
            owned_original_ids |= epi_owned_ids
            withheld_ids = undriven_loop_joint_ids(
                model.joints, owned_original_ids | coupled_member_ids
            )
            if withheld_ids and not skipped_reason:
                skipped_reason = (
                    f"{len(withheld_ids)} moving joint(s) sit on a closed kinematic loop "
                    "no coupled driver owns (e.g. an X-linkage): motion claim withheld "
                    "rather than tree-swept into a phantom verdict"
                )
            in_chains = {id(j) for pc in chained_pairs(model.joints) for j in pc}
            for joint in moving_joints:
                if id(joint) in coupled_member_ids:
                    continue  # handled below as a coupled pair
                if id(joint) in owned_original_ids:
                    continue  # handled below by a loop-closure driver
                if id(joint) in withheld_ids:
                    continue  # unowned closed loop: claim withheld (WARN via skipped)
                if joint.type is JointType.REVOLUTE:
                    lo, hi = joint.range_deg or (0.0, 180.0)
                else:
                    lo, hi = joint.range_mm or (0.0, 10.0)
                ok, detail, cols = sweeper.sweep_diag(joint, lo, hi)
                # SEQUENCED-DOF retry (row 61, the ladder's lock): when EVERY collision
                # partner belongs to other moving joints' subtrees (an engaged lock arm,
                # a stowed latch), re-sweep with those joints posed at their extremes.
                # Clear there = the mechanism works IN SEQUENCE (unlock, then fold) --
                # a PASS that names the required order, not a phantom defect. Still
                # colliding at both extremes = an honest FAIL as before.
                if not ok and cols:
                    blockers = {static for (_mv, static) in cols}
                    cands = []
                    for k in moving_joints:
                        if k is joint or id(k) in coupled_member_ids:
                            continue
                        if id(k) in owned_original_ids or id(k) in withheld_ids:
                            continue
                        k_family = downstream_ids(model.joints, k) | {k.b_part()}
                        if blockers & k_family:
                            cands.append((k, k_family))
                    covered: set[str] = set.union(*(fam for _k, fam in cands)) if cands else set()
                    if cands and blockers <= covered and len(cands) <= 4:
                        for pick in ("hi", "lo"):
                            prefix: dict[str, Any] = {}
                            seq_desc = []
                            for k, _fam in cands:
                                if k.type is JointType.REVOLUTE:
                                    k_lo, k_hi = k.range_deg or (0.0, 180.0)
                                    unit_k = "deg"
                                else:
                                    k_lo, k_hi = k.range_mm or (0.0, 10.0)
                                    unit_k = "mm"
                                ext = k_hi if pick == "hi" else k_lo
                                prefix.update(sweeper.corner_prefix(k, ext))
                                seq_desc.append(f"{k.b_part()} at {ext:.0f} {unit_k}")
                            ok2, detail2 = sweeper.sweep(joint, lo, hi, prefix=prefix)
                            if ok2:
                                ok = True
                                detail = (
                                    f"SEQUENCED: {detail2} once {', '.join(seq_desc)} "
                                    f"(at home the motion is blocked by design: {detail})"
                                )
                                break
                swept += 1
                checks.append(
                    Check(
                        id=f"assembly.sweep.{joint.a_part()}__{joint.b_part()}",
                        tier=Tier.ASSEMBLY,
                        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                        message=(
                            f"{joint.a_part()}<->{joint.b_part()} ({joint.type.value}): {detail}."
                        ),
                        expected="full declared range collision-free (chain vs rest)",
                        actual=detail,
                        tolerance=f"{_MOTION_SAMPLES} poses/joint",
                        evidence=None if ok else _sweep_evidence(joint, cols),
                    )
                )
                if not ok and failure is None:
                    failure = f"{joint.a_part()}<->{joint.b_part()} {detail}"
            for driver, follower, ratio in coupled:
                if id(driver) in epi_owned_ids or id(follower) in epi_owned_ids:
                    continue  # the epicyclic driver owns this pair's full trajectory
                if id(driver) in in_chains or id(follower) in in_chains:
                    # A coupled pair inside a moving chain (e.g. a gripper riding an
                    # articulated arm) needs coupled corner poses this v1 does not
                    # implement -- fail CLOSED (not swept) rather than certify a
                    # trajectory that was never driven.
                    skipped_reason = (
                        "a gear-coupled pair sits within a moving chain; coupled corner "
                        "poses are not implemented (v1) -- claim not earned"
                    )
                    continue
                lo, hi = driver.range_deg or (0.0, 0.0)
                ok, detail = sweeper.sweep_coupled(driver, follower, ratio, lo, hi)
                swept += 2  # the one coupled run tests BOTH joints' trajectories
                for j in (driver, follower):
                    checks.append(
                        Check(
                            id=f"assembly.sweep.{j.a_part()}__{j.b_part()}",
                            tier=Tier.ASSEMBLY,
                            status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                            message=(
                                f"{j.a_part()}<->{j.b_part()} (revolute, gear-coupled "
                                f"{driver.b_part()}<->{follower.b_part()}): {detail}."
                            ),
                            expected="full coupled trajectory collision-free (pair vs rest AND vs each other)",
                            actual=detail,
                            tolerance=f"{_MOTION_SAMPLES} poses/pair",
                        )
                    )
                if not ok and failure is None:
                    failure = f"{driver.b_part()}<->{follower.b_part()} {detail}"
            # Loop-closure drivers: ONE coupled sweep per system certifies every member
            # joint (spoken in the ORIGINAL graph -- a collapsed pivot re-expands to its
            # two pin joints for check ids). Pins ride their host link. A non-drivable
            # loop earns NO claim: WARN via skipped_reason, never a phantom verdict.
            driver_runs: list[tuple[str, Any, Any]] = [
                ("slider-crank", sc_systems, sweeper.sweep_slider_crank),
                ("four-bar", fb_systems, sweeper.sweep_four_bar),
                ("epicyclic", epi_systems, sweeper.sweep_epicyclic),
            ]
            for kind, systems, drive in driver_runs:
                for dj, _frame_id, crank_id, dloops in systems:
                    lo, hi = dj.range_deg or (0.0, 360.0)
                    d_ok, d_detail = drive(dj, crank_id, dloops, lo, hi, riders=pin_hosts)
                    member_orig = [
                        oj
                        for j in (dj, *(jj for lp in dloops for jj in lp.joints()))
                        for oj in _originals(j)
                    ]
                    if d_ok is None:
                        skipped_reason = d_detail
                        continue
                    swept += sum(
                        1
                        for j in member_orig
                        if j.type in (JointType.REVOLUTE, JointType.PRISMATIC)
                    )
                    for j in member_orig:
                        checks.append(
                            Check(
                                id=f"assembly.sweep.{j.a_part()}__{j.b_part()}",
                                tier=Tier.ASSEMBLY,
                                status=CheckStatus.PASS if d_ok else CheckStatus.FAIL,
                                message=(
                                    f"{j.a_part()}<->{j.b_part()} ({j.type.value}, {kind} "
                                    f"loop of {crank_id}): {d_detail}."
                                ),
                                expected=(
                                    "full crank range collision-free (loop-closure driven: "
                                    "movers vs rest AND vs each other)"
                                ),
                                actual=d_detail,
                                tolerance=f"{_MOTION_SAMPLES} poses/system",
                            )
                        )
                    if not d_ok and failure is None:
                        failure = f"{dj.a_part()}<->{dj.b_part()} {d_detail}"
            # Corner poses: sweep each chained child while its parent holds an extreme
            # -- the combined poses a one-joint-at-a-time sweep cannot see. Coupled
            # members never sweep independently here (their fail-closed handling above
            # already withheld the claim); slider-crank members likewise (their corner
            # poses would be the same phantom trajectories the system sweep replaced).
            if failure is None:
                for parent, child in chained_pairs(model.joints):
                    if id(parent) in coupled_member_ids or id(child) in coupled_member_ids:
                        continue
                    if id(parent) in owned_original_ids or id(child) in owned_original_ids:
                        continue
                    if id(parent) in withheld_ids or id(child) in withheld_ids:
                        continue
                    if parent.type is JointType.REVOLUTE:
                        p_lo, p_hi = parent.range_deg or (0.0, 180.0)
                    else:
                        p_lo, p_hi = parent.range_mm or (0.0, 10.0)
                    if child.type is JointType.REVOLUTE:
                        c_lo, c_hi = child.range_deg or (0.0, 180.0)
                    else:
                        c_lo, c_hi = child.range_mm or (0.0, 10.0)
                    for extreme in (p_lo, p_hi):
                        ok, detail = sweeper.sweep(
                            child, c_lo, c_hi, prefix=sweeper.corner_prefix(parent, extreme)
                        )
                        if not ok:
                            failure = (
                                f"{child.a_part()}<->{child.b_part()} {detail} with "
                                f"{parent.b_part()} held at {extreme:.0f}"
                            )
                            checks.append(
                                Check(
                                    id=(
                                        f"assembly.sweep_corner.{parent.b_part()}__{child.b_part()}"
                                    ),
                                    tier=Tier.ASSEMBLY,
                                    status=CheckStatus.FAIL,
                                    message=f"Corner pose collision: {failure}.",
                                    expected="child range clear at parent extremes",
                                    actual=detail,
                                )
                            )
                            break
                    if failure is not None:
                        break
        skipped = len(moving_joints) - swept
        if failure is not None:
            motion_status, motion_actual, motion_msg = (
                CheckStatus.FAIL,
                failure,
                f"Joint {failure}.",
            )
        elif skipped > 0:
            # The sweep could not be run for every moving joint. Do NOT report
            # "clear": WARN so an untested joint is never silently passed, and say
            # exactly why (which hardware lacks an envelope / which solid failed to
            # sample). Static placement is still validated above (hardware_coaxial).
            motion_status, motion_actual, motion_msg = (
                CheckStatus.WARN,
                f"{skipped} joint(s) not swept ({skipped_reason})",
                f"{skipped} of {len(moving_joints)} moving joint(s) were not swept: "
                f"{skipped_reason}. The motion claim is NOT earned for this assembly.",
            )
        else:
            motion_status, motion_actual, motion_msg = (
                CheckStatus.PASS,
                "clear",
                f"All {len(moving_joints)} moving joint(s) sweep their full range "
                f"collision-free, whole chain vs rest incl. hardware envelopes "
                f"({_MOTION_SAMPLES} poses/joint + corner poses).",
            )
        checks.append(
            Check(
                id="assembly.range_of_motion",
                tier=Tier.ASSEMBLY,
                status=motion_status,
                message=motion_msg,
                expected="every joint collision-free over its range",
                actual=motion_actual,
                tolerance=f"{_MOTION_SAMPLES} samples/joint",
            )
        )
        if motion_status is CheckStatus.FAIL:
            fixes.append(
                Fix(
                    reason=f"Joint {failure}; check fit clearance, knuckle interleave / link layering, and travel."
                )
            )
    elif gear_mesh:
        # A GEAR_MESH-only mechanism HAS a degree of freedom: the coupled gear rotation. Its motion
        # is proved by assembly.gear_mesh + assembly.gear_center_distance (above), so the
        # revolute/prismatic requirement does not apply. CRITICAL (no-false-PASS): range_of_motion
        # PASSes ONLY when those gear checks actually ran AND passed -- a mis-spaced / mis-clocked /
        # under-specified gear joint (no gear check ran, or one failed) must NOT report a moving DOF.
        gear_checks = [
            c for c in checks if c.id in ("assembly.gear_mesh", "assembly.gear_center_distance")
        ]
        gear_ok = bool(gear_checks) and all(c.status is CheckStatus.PASS for c in gear_checks)
        checks.append(
            Check(
                id="assembly.range_of_motion",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.PASS if gear_ok else CheckStatus.FAIL,
                message=(
                    f"The mechanism's degree of freedom is the gear mesh ({len(gear_mesh)} pair(s)); "
                    "its motion is validated by assembly.gear_mesh."
                    if gear_ok
                    else "The gear mesh is not validated (the gears are mis-spaced, mis-clocked, or "
                    "under-specified), so the coupled-rotation DOF is not proven."
                ),
                expected="a validated moving joint (revolute, prismatic, or a meshing gear pair)",
                actual=f"{len(gear_mesh)} gear-mesh DOF, gears_validated={gear_ok}",
            )
        )
        if not gear_ok:
            fixes.append(
                Fix(
                    reason="Space the gears at the pitch-radii-sum centre distance with matching "
                    "module/tooth counts (and a meshing clock) so the gear mesh validates."
                )
            )
    else:
        reason = "joint is FIXED (0 DOF)" if fixed else "no revolute or prismatic joint present"
        checks.append(
            Check(
                id="assembly.range_of_motion",
                tier=Tier.ASSEMBLY,
                status=CheckStatus.FAIL,
                message=f"A mechanism needs a revolute or prismatic joint, but {reason}.",
                expected="a revolute or prismatic joint with a range",
                actual=reason,
            )
        )
        fixes.append(
            Fix(reason=f"{reason}; declare a revolute or prismatic joint with a clearance fit.")
        )

    if payload_object is not None:
        # A DECLARED payload claim is always answered: measured aperture vs the cited
        # object (may gate) + the reported torque arithmetic (never gates). The
        # sweeper from the motion block is reused so the aperture poses are exactly
        # the swept FK transforms; without one the claim fails closed inside.
        _gripper_payload_checks(model, checks, fixes, payload_object, payload_sweeper)

    return build_report(checks, fixes)


__all__ = ["validate_assembly"]
