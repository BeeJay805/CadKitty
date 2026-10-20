"""Chain-aware swept-collision: the motion half of the assembly claim gate.

The original range-of-motion check swept ONLY the joint's two endpoint bodies against
each other. That under-claims an articulated mechanism: rotating a shoulder moves the
WHOLE downstream chain (links, the elbow servo bolted to a link, the wrist...), and a
distal link grazing the base is exactly the failure "each part passes alone but the
assembly interpenetrates" -- the false-validated the gate exists to kill.

This module sweeps the joint's full downstream SUBTREE against the entire static rest
of the assembly (printed bodies + hardware envelopes), through the joint's WHOLE
declared range, plus corner poses for chained joints (the child swept while the parent
holds each extreme). "Swept" therefore means: every part that moves was tested against
every part that does not, over the full range, with hardware included.

Honesty rules:
  * the claim level "swept" is EARNED only when EVERY hardware node carries an
    envelope mesh -- one invisible (mesh-less) part anywhere and no joint may claim
    "swept" (a sweep that cannot see a part proves nothing about it);
  * interior-point sampling is seeded (deterministic verdicts, same as the original
    pairwise sweep);
  * a degenerate mesh that cannot be sampled fails CLOSED (counts as not-swept),
    never as silently clear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray

from forge_mind.assembly.joints import Joint, JointType
from forge_mind.compiler.compile_assembly import AssemblyModel

_SEED = 0
_POINTS_PER_PART = 800  # interior samples cached per part (transformed per pose, cheap)
_MIN_OVERLAP_POINTS = 2  # > this many CONFIRMED-deep points inside another part = overlap
# A pin in a bore / a body in a clearance pocket touches its partner across whole
# surfaces; chord tessellation puts near-surface sample points AMBIGUOUSLY inside by up
# to the sagitta. A candidate point only counts as interpenetration when it sits inside
# the other solid DEEPER than this -- far above tessellation noise, far below any real
# defect (the mutation fixtures interpenetrate by millimetres).
_MIN_DEPTH_MM = 0.15
_SAMPLES = 19  # poses per sweep (matches the original per-joint sweep density)

Vec3T = tuple[float, float, float]


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _apply_point(m: NDArray[np.float64], p: Vec3T) -> NDArray[np.float64]:
    return np.asarray((m @ np.array([p[0], p[1], p[2], 1.0]))[:3], dtype=np.float64)


def _apply_dir(m: NDArray[np.float64], d: Vec3T) -> NDArray[np.float64]:
    return np.asarray(m[:3, :3] @ np.asarray(d, dtype=float), dtype=np.float64)


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


def _format_collisions(collisions: dict[tuple[str, str], tuple[float, float]], unit: str) -> str:
    """Worst-first collision detail: '<mover> hits <static> at <pose> <unit> (depth D mm)'
    plus a count of the remaining colliding pairs. The 'X hits Y at V' prefix is the frozen
    grammar the repair loop parses; depth extends it (never replaces it)."""
    ranked = sorted(collisions.items(), key=lambda kv: kv[1][0], reverse=True)
    (mover, static), (depth, pose) = ranked[0]
    depth_txt = "depth unmeasurable" if math.isinf(depth) else f"depth {depth:.2f} mm"
    head = f"{mover} hits {static} at {pose:.0f} {unit} ({depth_txt})"
    if len(ranked) > 1:
        rest = ", ".join(f"{m}~{s}" for (m, s), _ in ranked[1:4])
        head += f"; +{len(ranked) - 1} more colliding pair(s): {rest}"
    return head


def _directed_downstream(joints: list[Joint], joint: Joint) -> set[str]:
    """The transitive closure of the placement tree's DIRECTED edges (a -> b) from the
    joint's b side. Correct only when the graph is directed outward from ground; kept as
    the fail-closed fallback for joints that sit on a closed loop (where a subtree is not
    defined -- those are owned by the coupled/loop drivers, not this tree sweep)."""
    adj: dict[str, set[str]] = {}
    for j in joints:
        if j.type is JointType.GEAR_MESH:
            continue
        adj.setdefault(j.a_part(), set()).add(j.b_part())
    out: set[str] = set()
    frontier = [joint.b_part()]
    while frontier:
        cur = frontier.pop()
        if cur in out:
            continue
        out.add(cur)
        frontier.extend(adj.get(cur, ()))
    return out


def downstream_ids(joints: list[Joint], joint: Joint) -> set[str]:
    """Every part that MOVES when ``joint`` articulates: the connected component on the
    far side of the cut. Build the UNDIRECTED joint graph, remove this joint's edge, and
    take the component holding ``joint.b_part()``.

    Why undirected, not the directed a->b closure: the placement adapter directs a
    pin-in-bore revolute pin -> link for BOTH links a pin joins, so every pin is a SOURCE
    and every link a SINK. A directed closure from a moving link then reaches NOTHING it
    carries -- the moving set collapses to the single b-link, the pins and further links
    it physically rides get left in the static "rest", and the lone link sweeps into its
    own neighbours -> phantom self-collisions (the M2 arm: yaw "hit" the shoulder_pin it
    carries at 150 deg, the elbow "hit" its own gripper_mount at 7 deg). The undirected
    cut carries the whole outboard subtree, matching kinematic reality. Collision is
    invariant to which side is called "moving" (the sweep tests only part-vs-part, no
    world/bed frame), so the b-side component is verdict-correct without a ground flag.

    GEAR_MESH edges are EXCLUDED (a mesh couples two ROTATIONS, it does not carry one on
    the other; ``ChainSweeper.sweep_coupled`` owns that trajectory). If removing the edge
    does NOT split the graph (``a`` still reaches ``b`` -> the joint lies on a closed
    loop), no subtree is defined: fall back to the directed closure (unchanged behaviour;
    loops are driven by the coupled/loop drivers and cycle-withheld elsewhere)."""
    a, b = joint.a_part(), joint.b_part()
    adj: dict[str, set[str]] = {}
    for j in joints:
        if j.type is JointType.GEAR_MESH:
            continue
        adj.setdefault(j.a_part(), set()).add(j.b_part())
        adj.setdefault(j.b_part(), set()).add(j.a_part())
    adj.get(a, set()).discard(b)
    adj.get(b, set()).discard(a)
    out: set[str] = set()
    frontier = [b]
    while frontier:
        cur = frontier.pop()
        if cur in out:
            continue
        out.add(cur)
        frontier.extend(adj.get(cur, ()))
    if a in out:  # the cut did not separate a from b: this joint sits on a loop
        return _directed_downstream(joints, joint)
    return out


def meshless_hardware_ids(model: AssemblyModel) -> set[str]:
    return {h.id for h in model.hardware if h.mesh is None}


@dataclass
class SliderCrankLoop:
    """One cylinder's closed chain hanging off a crank: crankpin revolute -> rod -> wrist
    revolute -> piston -> prismatic back to the crank's frame. Pure topology (which joints
    form the loop); the geometry is resolved from ports at sweep time."""

    pin_joint: Joint  # revolute: crank.crankpin <-> rod.big_end (either side order)
    wrist_joint: Joint  # revolute: rod.small_end <-> piston.wrist (either side order)
    slide_joint: Joint  # prismatic: frame.slide <-> piston.slide (either side order)
    rod: str
    piston: str

    def joints(self) -> tuple[Joint, Joint, Joint]:
        return (self.pin_joint, self.wrist_joint, self.slide_joint)


def collapse_pin_edges(
    joints: list[Joint],
) -> tuple[list[Joint], dict[str, str], dict[int, tuple[Joint, ...]]]:
    """Pin-transparent view of the joint graph (row 61, the scissor lesson): the
    decompose rule 'THE PIN IS A PART' turns every pivot into TWO joints through a pin
    part (linkA <-> pin <-> linkB), which hides classic loop shapes from the coupled-
    driver detectors. Collapse each degree-2 all-revolute/coaxial pin into ONE synthetic
    revolute (linkA <-> linkB, carrying the outer ports), so detection sees the
    mechanism. Returns (collapsed_joints, pin_hosts, origin_map):
      * pin_hosts: pin part id -> the host part whose rigid motion the pin rides in a
        coupled sweep (the a-side partner; a coaxial pin is symmetric about its axis, so
        either host keeps it physically coincident);
      * origin_map: id(synthetic joint) -> the ORIGINAL joints it replaces (check ids
        and exclusion sets must speak the original graph).
    Pins with more than two partners (a radial's shared crankpin) or any non-pivot
    joint are left untouched -- collapsing is strictly conservative."""
    by_part: dict[str, list[Joint]] = {}
    for j in joints:
        by_part.setdefault(j.a_part(), []).append(j)
        by_part.setdefault(j.b_part(), []).append(j)
    pin_types = (JointType.REVOLUTE, JointType.COAXIAL_INSERT)
    pins: dict[str, list[Joint]] = {}
    for part, js in by_part.items():
        partners = {j.a_part() for j in js} | {j.b_part() for j in js}
        partners.discard(part)
        # the structural tell separating a PIN from a LINK of equal degree: a pin's two
        # joints share ONE port on the pin (one axis through everything); a link pivots
        # at two DIFFERENT ports (two stations). Without this, every bar of a pure
        # pin-jointed loop matches the degree test and the loop collapses to nothing.
        own_ports = {(j.a_port() if j.a_part() == part else j.b_port()) for j in js}
        if (
            len(js) == 2
            and len(partners) == 2
            and len(own_ports) == 1
            and all(j.type in pin_types for j in js)
            and any(j.type is JointType.REVOLUTE for j in js)
        ):
            pins[part] = js
    if not pins:
        return list(joints), {}, {}
    collapsed: list[Joint] = []
    consumed: set[int] = set()
    pin_hosts: dict[str, str] = {}
    origin_map: dict[int, tuple[Joint, ...]] = {}
    for pin, (j1, j2) in pins.items():
        if id(j1) in consumed or id(j2) in consumed:
            continue  # a partner was itself a collapsed pin: keep conservative
        side1 = (j1.a, j1.a_part()) if j1.a_part() != pin else (j1.b, j1.b_part())
        side2 = (j2.a, j2.a_part()) if j2.a_part() != pin else (j2.b, j2.b_part())
        if side1[1] == side2[1]:
            continue  # both joints reach the same partner: not a link-pin-link pivot
        synth = Joint(
            type=JointType.REVOLUTE,
            a=side1[0],
            b=side2[0],
            range_deg=j1.range_deg or j2.range_deg,
            clearance_class=j1.clearance_class,
        )
        collapsed.append(synth)
        consumed.update((id(j1), id(j2)))
        pin_hosts[pin] = side1[1]
        origin_map[id(synth)] = (j1, j2)
    collapsed.extend(j for j in joints if id(j) not in consumed)
    return collapsed, pin_hosts, origin_map


@dataclass
class FourBarLoop:
    """A planar four-bar hanging off a crank: crank pivots on the frame, coupler spans
    crank to rocker, rocker pivots back on the frame. Pure topology; geometry resolves
    from ports at sweep time (the same contract as SliderCrankLoop)."""

    coupler_joint: Joint  # revolute: crank <-> coupler
    rocker_joint: Joint  # revolute: coupler <-> rocker
    ground_joint: Joint  # revolute: rocker <-> frame
    coupler: str
    rocker: str

    def joints(self) -> tuple[Joint, Joint, Joint]:
        return (self.coupler_joint, self.rocker_joint, self.ground_joint)


def four_bar_systems(joints: list[Joint]) -> list[tuple[Joint, str, str, list[FourBarLoop]]]:
    """Detect four-bar linkages: rev(frame,crank) + rev(crank,coupler) + rev(coupler,
    rocker) + rev(rocker,frame), four distinct parts, side-order agnostic. Run on the
    PIN-COLLAPSED joint list so pin-jointed linkages match. Same ownership contract as
    slider_crank_systems: detected loops are driven coupled and excluded from the
    (unsound) independent tree sweep."""
    revs = [j for j in joints if j.type is JointType.REVOLUTE]
    # A pure four-bar is topologically SYMMETRIC (any bar can be ground), so the frame
    # is disambiguated by full-graph degree: the real frame anchors more of the assembly
    # (other loops, sliders, fixed mounts) than the crank does. Ties fall back to the
    # schema convention (joint.a = the reference side).
    degree: dict[str, int] = {}
    for j in joints:
        degree[j.a_part()] = degree.get(j.a_part(), 0) + 1
        degree[j.b_part()] = degree.get(j.b_part(), 0) + 1
    systems: list[tuple[Joint, str, str, list[FourBarLoop]]] = []
    claimed: set[int] = set()
    for j0 in revs:
        if id(j0) in claimed:
            continue
        a, b = j0.a_part(), j0.b_part()
        orientations = (
            ((a, b), (b, a)) if degree.get(a, 0) >= degree.get(b, 0) else ((b, a), (a, b))
        )
        for frame, crank in orientations:
            loops: list[FourBarLoop] = []
            for jc in revs:
                if jc is j0 or id(jc) in claimed:
                    continue
                c_parts = {jc.a_part(), jc.b_part()}
                if crank not in c_parts or len(c_parts) != 2:
                    continue
                coupler = next(p for p in c_parts if p != crank)
                if coupler == frame:
                    continue
                for jr in revs:
                    if jr in (j0, jc) or id(jr) in claimed:
                        continue
                    r_parts = {jr.a_part(), jr.b_part()}
                    if coupler not in r_parts or len(r_parts) != 2:
                        continue
                    rocker = next(p for p in r_parts if p != coupler)
                    if rocker in (crank, frame):
                        continue
                    ground = next(
                        (
                            jg
                            for jg in revs
                            if jg not in (j0, jc, jr)
                            and id(jg) not in claimed
                            and {jg.a_part(), jg.b_part()} == {frame, rocker}
                        ),
                        None,
                    )
                    if ground is not None:
                        loops.append(FourBarLoop(jc, jr, ground, coupler, rocker))
                        break
            if loops:
                claimed.add(id(j0))
                for lp in loops:
                    claimed.update(id(j) for j in lp.joints())
                systems.append((j0, frame, crank, loops))
                break
    return systems


def undriven_loop_joint_ids(joints: list[Joint], owned_ids: set[int]) -> set[int]:
    """Moving joints that sit on a kinematic CYCLE no coupled driver owns (row 61, the
    scissor lesson): the one-joint-at-a-time tree sweep is structurally unsound on any
    closed loop, so an unowned loop's moving joints must have their motion claim
    WITHHELD (WARN), never tree-swept into a phantom verdict.

    Cycles are found as the fundamental cycles of a spanning forest over ALL joints
    (a loop closed through fixed/coaxial edges is still rigid closure); a cycle counts
    only when it carries >= 2 moving joints (one over-constrained mover is a different,
    static defect the fit checks own)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: dict[str, list[tuple[str, Joint]]] = {}
    non_tree: list[Joint] = []
    for j in joints:
        a, b = j.a_part(), j.b_part()
        ra, rb = find(a), find(b)
        if ra == rb:
            non_tree.append(j)
        else:
            parent[ra] = rb
            adj.setdefault(a, []).append((b, j))
            adj.setdefault(b, []).append((a, j))
    if not non_tree:
        return set()
    moving = (JointType.REVOLUTE, JointType.PRISMATIC)
    out: set[int] = set()
    for closer in non_tree:
        # tree path a..b + the closing edge = one fundamental cycle
        start, goal = closer.a_part(), closer.b_part()
        prev: dict[str, tuple[str, Joint] | None] = {start: None}
        frontier = [start]
        while frontier and goal not in prev:
            cur = frontier.pop()
            for nxt, edge in adj.get(cur, ()):
                if nxt not in prev:
                    prev[nxt] = (cur, edge)
                    frontier.append(nxt)
        cycle: list[Joint] = [closer]
        node = goal
        while node != start:
            step = prev.get(node)
            if step is None:
                cycle = [closer]  # disconnected walk (defensive): treat closer alone
                break
            node, edge = step
            cycle.append(edge)
        movers = [j for j in cycle if j.type in moving and id(j) not in owned_ids]
        if len([j for j in cycle if j.type in moving]) >= 2 and movers:
            out.update(id(j) for j in movers)
    return out


def slider_crank_systems(
    joints: list[Joint],
) -> list[tuple[Joint, str, str, list[SliderCrankLoop]]]:
    """Detect slider-crank systems in the joint graph: a revolute (frame<->crank) whose
    crank carries >=1 closed chain crank->rod->piston->frame (revolute, revolute,
    prismatic). Returns (crank_joint, frame_id, crank_id, loops) per system. Detection is
    pure topology -- side order agnostic -- so a plan's a/b conventions cannot hide a loop.
    These loops are exactly what the tree sweep is UNSOUND on (it would spin rods full-
    circle about their pins and never move a piston); the caller must route them to
    ``sweep_slider_crank`` and exclude their joints from independent sweeps."""
    revs = [j for j in joints if j.type is JointType.REVOLUTE]
    pris = [j for j in joints if j.type is JointType.PRISMATIC]
    systems: list[tuple[Joint, str, str, list[SliderCrankLoop]]] = []
    claimed: set[int] = set()  # joints already in a detected system (one system per joint)
    for j0 in revs:
        if id(j0) in claimed:
            continue
        for frame, crank in ((j0.a_part(), j0.b_part()), (j0.b_part(), j0.a_part())):
            loops: list[SliderCrankLoop] = []
            for jp in revs:
                if jp is j0 or id(jp) in claimed:
                    continue
                pin_parts = {jp.a_part(), jp.b_part()}
                if crank not in pin_parts or len(pin_parts) != 2:
                    continue
                rod = next(p for p in pin_parts if p != crank)
                if rod == frame:
                    continue
                for jw in revs:
                    if jw in (j0, jp) or id(jw) in claimed:
                        continue
                    w_parts = {jw.a_part(), jw.b_part()}
                    if rod not in w_parts or len(w_parts) != 2:
                        continue
                    piston = next(p for p in w_parts if p != rod)
                    if piston in (crank, frame):
                        continue
                    slide = next(
                        (
                            js
                            for js in pris
                            if id(js) not in claimed
                            and {js.a_part(), js.b_part()} == {frame, piston}
                        ),
                        None,
                    )
                    if slide is not None:
                        loops.append(SliderCrankLoop(jp, jw, slide, rod, piston))
                        break  # one loop per pin joint
            if loops:
                claimed.add(id(j0))
                for lp in loops:
                    claimed.update(id(j) for j in lp.joints())
                systems.append((j0, frame, crank, loops))
                break  # orientation resolved; do not re-detect swapped
    return systems


@dataclass
class _PartGeom:
    """Per-part cached collision data, all in LOCAL coords (posed by matrices per pose)."""

    pid: str
    base_transform: NDArray[np.float64]  # solved world transform (home pose)
    mesh_local: trimesh.Trimesh
    interior_local: NDArray[np.float64]  # (n,3) seeded interior samples, local frame
    sampled_ok: bool


class ChainSweeper:
    """Sweeps subtrees against the static rest with cached interior point clouds.

    Per pose, point clouds are transformed by 4x4s (cheap) and containment is tested
    in the OTHER part's local frame via its (cached) mesh.contains -- no per-pose mesh
    copies or re-sampling. Overlap in either direction (> _MIN_OVERLAP_POINTS interior
    points inside the other solid) is a real interpenetration; mere surface contact
    shares no interior volume and passes.
    """

    def __init__(self, model: AssemblyModel) -> None:
        self._model = model
        self._parts: dict[str, _PartGeom] = {}
        for body in model.bodies:
            self._parts[body.id] = self._prep(body.id, body.transform, body.mesh)
        for hw in model.hardware:
            if hw.mesh is not None:
                self._parts[hw.id] = self._prep(hw.id, hw.transform, hw.mesh)

    @staticmethod
    def _prep(pid: str, transform: NDArray[np.float64], mesh: trimesh.Trimesh) -> _PartGeom:
        rng_state = np.random.get_state()
        try:
            np.random.seed(_SEED)
            pts = np.asarray(trimesh.sample.volume_mesh(mesh, _POINTS_PER_PART), dtype=float)
            ok = len(pts) > 0
        except Exception:  # degenerate mesh: fail CLOSED (not silently clear)
            pts, ok = np.empty((0, 3)), False
        finally:
            np.random.set_state(rng_state)
        return _PartGeom(
            pid=pid,
            base_transform=np.asarray(transform, dtype=float),
            mesh_local=mesh,
            interior_local=pts,
            sampled_ok=ok,
        )

    def all_sampled_ok(self) -> bool:
        return all(p.sampled_ok for p in self._parts.values())

    def unsampleable_ids(self) -> list[str]:
        return sorted(p.pid for p in self._parts.values() if not p.sampled_ok)

    # ---- core pair test at a pose -------------------------------------------------

    def _pair_overlap_depth(
        self,
        a: _PartGeom,
        a_world: NDArray[np.float64],
        b: _PartGeom,
        b_world: NDArray[np.float64],
    ) -> float | None:
        """Max confirmed interpenetration depth (mm) between the posed pair, or None when
        clear. The depth was always computed for the confirmation threshold and then thrown
        away (the audit's 'info exists but discarded'); now it feeds the failure detail so a
        repair knows HOW DEEP, not just THAT. ``inf`` = a backend failure: cannot certify ->
        treated as an (unmeasurable) overlap, fail-closed as before."""
        # bbox cull in world space (corners of the local AABB, transformed)
        a_corners = trimesh.bounds.corners(a.mesh_local.bounds)  # type: ignore[no-untyped-call]
        b_corners = trimesh.bounds.corners(b.mesh_local.bounds)  # type: ignore[no-untyped-call]
        a_bounds = trimesh.transform_points(a_corners, a_world)
        b_bounds = trimesh.transform_points(b_corners, b_world)
        if (np.max(a_bounds, axis=0) < np.min(b_bounds, axis=0)).any() or (
            np.max(b_bounds, axis=0) < np.min(a_bounds, axis=0)
        ).any():
            return None
        worst: float | None = None
        for src, src_world, dst, dst_world in (
            (a, a_world, b, b_world),
            (b, b_world, a, a_world),
        ):
            if len(src.interior_local) == 0:
                continue
            pts_world = trimesh.transform_points(src.interior_local, src_world)
            pts_dst_local = trimesh.transform_points(pts_world, np.linalg.inv(dst_world))
            try:
                inside = dst.mesh_local.contains(pts_dst_local)
            except Exception:  # containment backend failure: cannot certify -> overlap
                return float("inf")
            if int(np.count_nonzero(inside)) > _MIN_OVERLAP_POINTS:
                # Depth confirmation: intended intimate contact (pin-in-bore, body-in-
                # pocket) yields a few AMBIGUOUS near-surface hits from tessellation
                # sagitta; real interpenetration goes deep. Count only points inside by
                # more than _MIN_DEPTH_MM (trimesh signed distance: positive = inside).
                try:
                    sd = np.asarray(
                        trimesh.proximity.signed_distance(  # type: ignore[no-untyped-call]
                            dst.mesh_local, pts_dst_local[inside]
                        )
                    )
                except Exception:  # depth backend failure: cannot certify -> overlap
                    return float("inf")
                deep = sd[sd > _MIN_DEPTH_MM]
                if len(deep) > _MIN_OVERLAP_POINTS:
                    d = float(np.max(deep))
                    worst = d if worst is None else max(worst, d)
        return worst

    def _pair_overlaps(
        self,
        a: _PartGeom,
        a_world: NDArray[np.float64],
        b: _PartGeom,
        b_world: NDArray[np.float64],
    ) -> bool:
        return self._pair_overlap_depth(a, a_world, b, b_world) is not None

    # ---- sweeps ---------------------------------------------------------------------

    def _world(self, pid: str, extra: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        g = self._parts[pid]
        e = extra.get(pid)
        return (e @ g.base_transform) if e is not None else g.base_transform

    def joint_pivot(
        self, joint: Joint, prefix: dict[str, NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The joint's pivot point + axis in world, honoring any prefix pose on its
        reference side (a corner sweep moves the parent first; the child pivot rides)."""
        a_tf, a_ports = self._model.placed_ports(joint.a_part())
        a_port = next(p for p in a_ports if p.name == joint.a_port())
        e = prefix.get(joint.a_part())
        world = (e @ np.asarray(a_tf, dtype=float)) if e is not None else np.asarray(a_tf)
        return (
            _apply_point(world, a_port.origin.as_tuple()),
            _unit(_apply_dir(world, a_port.axis.as_tuple())),
        )

    def sweep(
        self,
        joint: Joint,
        lo: float,
        hi: float,
        prefix: dict[str, NDArray[np.float64]] | None = None,
    ) -> tuple[bool, str]:
        """Sweep ``joint`` over [lo, hi]: ALL downstream parts vs ALL static parts.
        ``prefix`` poses parts before the sweep (corner poses of an upstream joint)."""
        ok, detail, _collisions = self.sweep_diag(joint, lo, hi, prefix)
        return ok, detail

    def sweep_diag(
        self,
        joint: Joint,
        lo: float,
        hi: float,
        prefix: dict[str, NDArray[np.float64]] | None = None,
    ) -> tuple[bool, str, dict[tuple[str, str], tuple[float, float]]]:
        """``sweep`` plus the raw per-pair collision map ((mover, static) -> (max depth,
        pose)) so callers can reason about WHO blocks the motion -- the sequenced-DOF
        retry (row 61, the ladder's lock) needs the colliding partners, not a string."""
        prefix = dict(prefix or {})
        moving = downstream_ids(self._model.joints, joint)
        pivot_o, pivot_d = self.joint_pivot(joint, prefix)
        movers = [g for pid, g in self._parts.items() if pid in moving]
        statics = [
            (g, self._world(g.pid, prefix)) for pid, g in self._parts.items() if pid not in moving
        ]
        unit = "deg" if joint.type is JointType.REVOLUTE else "mm"
        # Prismatic engagement: the sliding endpoint must STAY ENGAGED with its rail
        # (>= half its length overlapping along the axis), or it has escaped -- a
        # mechanism defect a pure collision test cannot see.
        rail = self._parts.get(joint.a_part())
        slider = self._parts.get(joint.b_part())
        rail_span: tuple[float, float] | None = None
        slider_len = 0.0
        if joint.type is JointType.PRISMATIC and rail is not None and slider is not None:
            rail_v = trimesh.transform_points(
                rail.mesh_local.vertices, self._world(rail.pid, prefix)
            )
            proj = rail_v @ pivot_d
            rail_span = (float(proj.min()), float(proj.max()))
        # FULL-RANGE scan (audit: the old first-hit early-return reported ONE pair at the
        # LOWEST colliding sample, starving the repair loop). Every colliding pair is
        # accumulated with its worst depth + the pose where it happens; the detail names
        # the worst pair first so a repair targets the dominant collision.
        collisions: dict[
            tuple[str, str], tuple[float, float]
        ] = {}  # (mover, static) -> (depth, pose)
        for i in range(_SAMPLES):
            v = lo + (hi - lo) * i / (_SAMPLES - 1)
            motion = (
                _rotation_about_line(pivot_o, pivot_d, v)
                if joint.type is JointType.REVOLUTE
                else _translation(pivot_d, v)
            )
            for g in movers:
                g_world = motion @ self._world(g.pid, prefix)
                for s, s_world in statics:
                    depth = self._pair_overlap_depth(g, g_world, s, s_world)
                    if depth is not None:
                        key = (g.pid, s.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, v)
            if rail_span is not None and slider is not None:
                s_v = trimesh.transform_points(
                    slider.mesh_local.vertices, motion @ self._world(slider.pid, prefix)
                )
                proj = s_v @ pivot_d
                slider_len = float(proj.max() - proj.min())
                overlap = min(float(proj.max()), rail_span[1]) - max(
                    float(proj.min()), rail_span[0]
                )
                if overlap < 0.5 * slider_len:
                    # past this travel the slider has ESCAPED its rail; further poses are
                    # not the mechanism -- stop scanning and report what was measured.
                    detail = f"{slider.pid} slides out at {v:.0f} {unit}"
                    if collisions:
                        detail += "; " + _format_collisions(collisions, unit)
                    return False, detail, collisions
        if collisions:
            return False, _format_collisions(collisions, unit), collisions
        return (
            True,
            f"clear {lo:.0f}..{hi:.0f} {unit} ({_SAMPLES} poses, chain vs rest)",
            collisions,
        )

    def sweep_coupled(
        self,
        driver: Joint,
        follower: Joint,
        ratio: float,
        lo: float,
        hi: float,
        prefix: dict[str, NDArray[np.float64]] | None = None,
    ) -> tuple[bool, str]:
        """Sweep a gear-coupled pair through its REAL trajectory: the driver joint at
        theta, the follower joint at ``-theta * ratio`` about its OWN pivot (the
        external-mesh constraint -- the same convention ``validate_gear_mesh_kinematic``
        drives). Tests, at every sample: each moving subtree vs the static rest, AND the
        two moving subtrees vs EACH OTHER (the both-fingers-closed pose a one-joint
        sweep structurally cannot see).

        This is the ONLY sweep that may earn "swept" for a coupled pair: sweeping either
        joint alone visits poses the mechanism never reaches (meshing teeth at wrong
        relative angles read as collision, while the real trajectory goes untested).
        """
        prefix = dict(prefix or {})
        d_moving = downstream_ids(self._model.joints, driver)
        f_moving = downstream_ids(self._model.joints, follower)
        if d_moving & f_moving:
            return False, (
                f"coupled subtrees overlap ({sorted(d_moving & f_moving)}): the gear pair "
                "does not form two independent branches; cannot certify"
            )
        d_pivot_o, d_pivot_d = self.joint_pivot(driver, prefix)
        f_pivot_o, f_pivot_d = self.joint_pivot(follower, prefix)
        d_movers = [g for pid, g in self._parts.items() if pid in d_moving]
        f_movers = [g for pid, g in self._parts.items() if pid in f_moving]
        statics = [
            (g, self._world(g.pid, prefix))
            for pid, g in self._parts.items()
            if pid not in d_moving and pid not in f_moving
        ]
        collisions: dict[tuple[str, str], tuple[float, float]] = {}
        for i in range(_SAMPLES):
            theta = lo + (hi - lo) * i / (_SAMPLES - 1)
            d_motion = _rotation_about_line(d_pivot_o, d_pivot_d, theta)
            f_motion = _rotation_about_line(f_pivot_o, f_pivot_d, -theta * ratio)
            d_posed = [(g, d_motion @ self._world(g.pid, prefix)) for g in d_movers]
            f_posed = [(g, f_motion @ self._world(g.pid, prefix)) for g in f_movers]
            for g, g_world in d_posed + f_posed:
                for s, s_world in statics:
                    depth = self._pair_overlap_depth(g, g_world, s, s_world)
                    if depth is not None:
                        key = (g.pid, s.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
            for g, g_world in d_posed:
                for h, h_world in f_posed:
                    depth = self._pair_overlap_depth(g, g_world, h, h_world)
                    if depth is not None:
                        key = (g.pid, h.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
        if collisions:
            return False, _format_collisions(collisions, "deg") + " (coupled)"
        return True, (
            f"clear {lo:.0f}..{hi:.0f} deg coupled (driver + follower at -{ratio:g}x, "
            f"{_SAMPLES} poses, subtrees vs rest AND vs each other)"
        )

    # ---- slider-crank coupled driver (audit fix: the tree sweep is UNSOUND on closed
    # loops -- it rigidly spins a crank's rods about the main axis and full-circles a rod
    # about its pin, poses the mechanism never reaches, while the pistons never move at
    # all. This driver parametrizes the WHOLE loop by crank angle and derives every
    # body's real pose from loop closure, the same way sweep_coupled owns gears.) -------

    def _port_world(
        self, part_id: str, port_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """(origin, axis) of a named port in world coords, or None if unresolvable."""
        try:
            tf, ports = self._model.placed_ports(part_id)
            port = next(p for p in ports if p.name == port_name)
        except Exception:
            return None
        world = np.asarray(tf, dtype=float)
        return _apply_point(world, port.origin.as_tuple()), _unit(
            _apply_dir(world, port.axis.as_tuple())
        )

    def _loop_geometry(
        self, crank_id: str, loop: SliderCrankLoop, crank_axis: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float] | str:
        """Resolve one loop's driving geometry: (pin_home, wrist_home, bore_axis, rod_len).
        Returns a human-readable reason string when the loop is NOT drivable (fail-closed:
        the caller withholds the motion claim rather than sweep an unsound trajectory)."""
        pin_side = (
            (loop.pin_joint.a_part(), loop.pin_joint.a_port())
            if loop.pin_joint.a_part() == crank_id
            else (loop.pin_joint.b_part(), loop.pin_joint.b_port())
        )
        pin = self._port_world(*pin_side)
        if pin is None:
            return f"crankpin port '{pin_side[1]}' unresolvable on '{pin_side[0]}'"
        wrist_side = (
            (loop.wrist_joint.a_part(), loop.wrist_joint.a_port())
            if loop.wrist_joint.a_part() == loop.piston
            else (loop.wrist_joint.b_part(), loop.wrist_joint.b_port())
        )
        wrist = self._port_world(*wrist_side)
        if wrist is None:
            return f"wrist port '{wrist_side[1]}' unresolvable on '{wrist_side[0]}'"
        slide_side = (
            (loop.slide_joint.a_part(), loop.slide_joint.a_port())
            if loop.slide_joint.b_part() == loop.piston
            else (loop.slide_joint.b_part(), loop.slide_joint.b_port())
        )
        slide = self._port_world(*slide_side)
        if slide is None:
            return f"slide port '{slide_side[1]}' unresolvable on '{slide_side[0]}'"
        pin_o, _pin_axis = pin
        wrist_o, _wrist_axis = wrist
        slide_o, bore_axis = slide
        rod_len = float(np.linalg.norm(wrist_o - pin_o))
        if rod_len < 1e-6:
            return f"rod '{loop.rod}': wrist and crankpin coincide (zero rod length)"
        # the rod-swing pose is exact only for a PLANAR loop (bore perpendicular to the
        # crank axis -- every real reciprocating engine). A skewed bore would need an
        # out-of-plane rod tilt this driver does not model: withhold the claim.
        if abs(float(np.dot(bore_axis, crank_axis))) > 0.05:
            return (
                f"piston '{loop.piston}' bore axis is not perpendicular to the crank axis "
                "(non-planar loop; not modeled)"
            )
        # the wrist must sit ON the bore line (the piston slides along it): off-line wrist
        # means the declared geometry cannot close and the drive would be fiction.
        off = (wrist_o - slide_o) - float(np.dot(wrist_o - slide_o, bore_axis)) * bore_axis
        if float(np.linalg.norm(off)) > 2.0:
            return (
                f"piston '{loop.piston}' wrist sits {float(np.linalg.norm(off)):.1f} mm off "
                "its bore axis -- the loop does not close as declared"
            )
        return pin_o, wrist_o, bore_axis, rod_len

    def sweep_slider_crank(
        self,
        crank_joint: Joint,
        crank_id: str,
        loops: list[SliderCrankLoop],
        lo: float,
        hi: float,
        riders: dict[str, str] | None = None,
    ) -> tuple[bool | None, str]:
        """Drive the crank through [lo, hi] deg and pose every rod + piston from loop
        closure: pin rides the crank; the piston translates along its bore axis to keep
        wrist-to-pin distance = rod length; the rod swings about the pin axis to span them.
        Collision-tests all movers vs the static rest AND movers vs movers (depth-confirmed,
        so intended pin-in-bore contact passes). Returns (None, reason) when the loop is not
        drivable -- the caller reports the motion claim NOT EARNED (fail-closed), never a
        phantom verdict either way."""
        pivot = self.joint_pivot(crank_joint, {})
        pivot_o, pivot_d = pivot
        geoms: list[
            tuple[
                SliderCrankLoop,
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
            ]
        ] = []
        for loop in loops:
            geo = self._loop_geometry(crank_id, loop, pivot_d)
            if isinstance(geo, str):
                return None, f"slider-crank loop ({loop.rod}/{loop.piston}) not drivable: {geo}"
            pin_o, wrist_o, bore_axis, rod_len = geo
            r = pin_o - pivot_o
            r_perp = r - float(np.dot(r, pivot_d)) * pivot_d
            if float(np.linalg.norm(r_perp)) < 1e-6:
                return None, (
                    f"slider-crank loop ({loop.rod}/{loop.piston}) not drivable: crankpin "
                    "sits on the main axis (zero throw)"
                )
            geoms.append((loop, pin_o, wrist_o, bore_axis, rod_len))
        mover_ids = {crank_id} | {lp.rod for lp in loops} | {lp.piston for lp in loops}
        riders = {
            pid: host
            for pid, host in (riders or {}).items()
            if host in mover_ids and pid in self._parts
        }
        mover_ids |= set(riders)
        missing = sorted(m for m in mover_ids if m not in self._parts)
        if missing:
            return None, f"slider-crank not drivable: mover(s) without geometry: {missing}"
        statics = [
            (g, self._world(g.pid, {})) for pid, g in self._parts.items() if pid not in mover_ids
        ]
        collisions: dict[tuple[str, str], tuple[float, float]] = {}
        max_obliquity = 0.0  # C1: worst rod angle off the bore axis (side-load proxy)
        for i in range(_SAMPLES):
            theta = lo + (hi - lo) * i / (_SAMPLES - 1)
            m_crank = _rotation_about_line(pivot_o, pivot_d, theta)
            host_m: dict[str, NDArray[np.float64]] = {crank_id: m_crank}
            posed: list[tuple[_PartGeom, NDArray[np.float64]]] = [
                (self._parts[crank_id], m_crank @ self._world(crank_id, {}))
            ]
            for loop, pin_o, wrist_o, bore_axis, rod_len in geoms:
                p_theta = _apply_point(m_crank, (pin_o[0], pin_o[1], pin_o[2]))
                # piston travel t: |wrist_o + b*t - p(theta)| = rod_len  (quadratic in t)
                d = wrist_o - p_theta
                b_half = float(np.dot(bore_axis, d))
                c = float(np.dot(d, d)) - rod_len * rod_len
                disc = b_half * b_half - c
                if disc < 0.0:
                    return None, (
                        f"slider-crank loop ({loop.rod}/{loop.piston}) not drivable: rod too "
                        f"short to close the loop at {theta:.0f} deg (throw + geometry "
                        "inconsistent)"
                    )
                root = math.sqrt(disc)
                # branch that keeps the piston continuous with its authored home (t(home)=0)
                t = -b_half + root if b_half >= 0.0 else -b_half - root
                w_theta = wrist_o + bore_axis * t
                # rod pose: big end rides the pin; swing angle about the crank axis spans
                # home direction (wrist-pin at home) to current direction.
                u_home = (wrist_o - pin_o) - float(np.dot(wrist_o - pin_o, pivot_d)) * pivot_d
                u_now = (w_theta - p_theta) - float(np.dot(w_theta - p_theta, pivot_d)) * pivot_d
                n_home, n_now = float(np.linalg.norm(u_home)), float(np.linalg.norm(u_now))
                if n_home < 1e-9 or n_now < 1e-9:
                    return None, (
                        f"slider-crank loop ({loop.rod}/{loop.piston}) not drivable: rod is "
                        "parallel to the crank axis (degenerate swing plane)"
                    )
                phi = math.degrees(
                    math.atan2(
                        float(np.dot(np.cross(u_home / n_home, u_now / n_now), pivot_d)),
                        float(np.dot(u_home / n_home, u_now / n_now)),
                    )
                )
                max_obliquity = max(max_obliquity, abs(phi))
                m_rod = np.eye(4)
                m_rod[:3, 3] = p_theta - pin_o
                m_rod = m_rod @ _rotation_about_line(pin_o, pivot_d, phi)
                m_piston = np.eye(4)
                m_piston[:3, 3] = bore_axis * t
                host_m[loop.rod] = m_rod
                host_m[loop.piston] = m_piston
                posed.append((self._parts[loop.rod], m_rod @ self._world(loop.rod, {})))
                posed.append((self._parts[loop.piston], m_piston @ self._world(loop.piston, {})))
            for pid, host in riders.items():  # collapsed pivot pins ride their host body
                posed.append((self._parts[pid], host_m[host] @ self._world(pid, {})))
            for g, g_world in posed:
                for s, s_world in statics:
                    depth = self._pair_overlap_depth(g, g_world, s, s_world)
                    if depth is not None:
                        key = (g.pid, s.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
            for ix in range(len(posed)):
                for jx in range(ix + 1, len(posed)):
                    g, g_world = posed[ix]
                    h, h_world = posed[jx]
                    depth = self._pair_overlap_depth(g, g_world, h, h_world)
                    if depth is not None:
                        key = (g.pid, h.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
        if collisions:
            return False, _format_collisions(collisions, "deg") + " (slider-crank coupled)"
        obliq = f"; max rod obliquity {max_obliquity:.0f} deg"
        if max_obliquity > 60.0:
            obliq += " (POOR -- extreme side load on the bores)"
        return True, (
            f"clear {lo:.0f}..{hi:.0f} deg slider-crank coupled ({len(loops)} loop(s), "
            f"{_SAMPLES} poses, loop-closure driven, movers vs rest AND vs each other)" + obliq
        )

    def sweep_four_bar(
        self,
        crank_joint: Joint,
        crank_id: str,
        loops: list[FourBarLoop],
        lo: float,
        hi: float,
        riders: dict[str, str] | None = None,
    ) -> tuple[bool | None, str]:
        """Drive a planar four-bar by crank angle: the coupler's far end is the circle-
        circle intersection (radius=coupler length about the crankpin, radius=rocker
        length about the rocker's frame pivot), branch chosen by continuity from home.
        A range the linkage cannot close (non-Grashof crank pushed past its limit, or
        non-parallel pivots) earns NO claim: (None, reason) -- the caller withholds
        (fail-closed), never a phantom verdict. Same collision contract as the other
        coupled drivers: movers vs statics AND movers vs movers, depth-confirmed."""
        pivot_o, pivot_d = self.joint_pivot(crank_joint, {})

        def perp(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return x - float(np.dot(x, pivot_d)) * pivot_d

        geoms = []
        for loop in loops:
            pin_side = (
                (loop.coupler_joint.a_part(), loop.coupler_joint.a_port())
                if loop.coupler_joint.a_part() == crank_id
                else (loop.coupler_joint.b_part(), loop.coupler_joint.b_port())
            )
            pin = self._port_world(*pin_side)
            re_side = (
                (loop.rocker_joint.a_part(), loop.rocker_joint.a_port())
                if loop.rocker_joint.a_part() == loop.coupler
                else (loop.rocker_joint.b_part(), loop.rocker_joint.b_port())
            )
            rend = self._port_world(*re_side)
            g_side = (
                (loop.ground_joint.a_part(), loop.ground_joint.a_port())
                if loop.ground_joint.b_part() == loop.rocker
                else (loop.ground_joint.b_part(), loop.ground_joint.b_port())
            )
            ground = self._port_world(*g_side)
            if pin is None or rend is None or ground is None:
                return None, (
                    f"four-bar loop ({loop.coupler}/{loop.rocker}) not drivable: a pivot "
                    "port is unresolvable"
                )
            pin_o, _ = pin
            re_o, _ = rend
            r0_o, r0_axis = ground
            if abs(float(np.dot(r0_axis, pivot_d))) < 0.95:
                return None, (
                    f"four-bar loop ({loop.coupler}/{loop.rocker}) not drivable: rocker "
                    "pivot not parallel to the crank axis (non-planar)"
                )
            lc = float(np.linalg.norm(perp(re_o - pin_o)))
            lr = float(np.linalg.norm(perp(re_o - r0_o)))
            if lc < 1e-6 or lr < 1e-6:
                return None, (
                    f"four-bar loop ({loop.coupler}/{loop.rocker}) not drivable: degenerate "
                    "link length"
                )
            geoms.append((loop, pin_o, re_o, r0_o, lc, lr))
        mover_ids = {crank_id} | {lp.coupler for lp in loops} | {lp.rocker for lp in loops}
        riders = {
            pid: host
            for pid, host in (riders or {}).items()
            if host in mover_ids and pid in self._parts
        }
        mover_ids |= set(riders)
        missing = sorted(m for m in mover_ids if m not in self._parts)
        if missing:
            return None, f"four-bar not drivable: mover(s) without geometry: {missing}"
        statics = [
            (g, self._world(g.pid, {})) for pid, g in self._parts.items() if pid not in mover_ids
        ]
        collisions: dict[tuple[str, str], tuple[float, float]] = {}
        prev_re: dict[int, NDArray[np.float64]] = {}
        # C1 (row-61 research): transmission angle = coupler-vs-rocker angle at their
        # shared joint. A linkage can clear every collision and still be mechanically
        # USELESS near a toggle (force transmission -> 0). Tracked per pose, reported
        # worst-first; outside ~[40, 140] deg the detail carries an explicit flag.
        worst_trans: tuple[float, float] | None = None  # (angle_from_90, theta)
        for i in range(_SAMPLES):
            theta = lo + (hi - lo) * i / (_SAMPLES - 1)
            m_crank = _rotation_about_line(pivot_o, pivot_d, theta)
            host_m: dict[str, NDArray[np.float64]] = {crank_id: m_crank}
            posed: list[tuple[_PartGeom, NDArray[np.float64]]] = [
                (self._parts[crank_id], m_crank @ self._world(crank_id, {}))
            ]
            for gix, (loop, pin_o, re_o, r0_o, lc, lr) in enumerate(geoms):
                p_theta = _apply_point(m_crank, (pin_o[0], pin_o[1], pin_o[2]))
                # circle-circle intersection in the plane perpendicular to the crank axis
                c12 = perp(r0_o - p_theta)
                d12 = float(np.linalg.norm(c12))
                if d12 > lc + lr - 1e-9 or d12 < abs(lc - lr) + 1e-9 or d12 < 1e-9:
                    return None, (
                        f"four-bar loop ({loop.coupler}/{loop.rocker}) cannot close at "
                        f"{theta:.0f} deg (declared crank range exceeds the linkage's "
                        "feasible range)"
                    )
                a_len = (lc * lc - lr * lr + d12 * d12) / (2 * d12)
                h = math.sqrt(max(0.0, lc * lc - a_len * a_len))
                e1 = c12 / d12
                e2 = np.cross(pivot_d, e1)
                base = p_theta + e1 * a_len
                cand = (base + e2 * h, base - e2 * h)
                ref = prev_re.get(gix)
                if ref is None:
                    ref = re_o  # home continuity: start at the authored pose's branch
                re_theta = min(cand, key=lambda c: float(np.linalg.norm(c - ref)))
                prev_re[gix] = re_theta
                # rocker: pure rotation about its frame pivot
                u0, u1 = perp(re_o - r0_o), perp(re_theta - r0_o)
                n0, n1 = float(np.linalg.norm(u0)), float(np.linalg.norm(u1))
                phi_r = math.degrees(
                    math.atan2(
                        float(np.dot(np.cross(u0 / n0, u1 / n1), pivot_d)),
                        float(np.dot(u0 / n0, u1 / n1)),
                    )
                )
                m_rocker = _rotation_about_line(r0_o, pivot_d, phi_r)
                # coupler: swings about the crankpin, translated with it
                v0, v1 = perp(re_o - pin_o), perp(re_theta - p_theta)
                m0, m1 = float(np.linalg.norm(v0)), float(np.linalg.norm(v1))
                phi_c = math.degrees(
                    math.atan2(
                        float(np.dot(np.cross(v0 / m0, v1 / m1), pivot_d)),
                        float(np.dot(v0 / m0, v1 / m1)),
                    )
                )
                m_coupler = np.eye(4)
                m_coupler[:3, 3] = p_theta - pin_o
                m_coupler = m_coupler @ _rotation_about_line(pin_o, pivot_d, phi_c)
                trans = abs(
                    math.degrees(
                        math.atan2(
                            float(np.linalg.norm(np.cross(v1 / m1, u1 / n1))),
                            float(np.dot(v1 / m1, u1 / n1)),
                        )
                    )
                )
                dev = abs(trans - 90.0)
                if worst_trans is None or dev > worst_trans[0]:
                    worst_trans = (dev, theta)
                host_m[loop.coupler] = m_coupler
                host_m[loop.rocker] = m_rocker
                posed.append((self._parts[loop.coupler], m_coupler @ self._world(loop.coupler, {})))
                posed.append((self._parts[loop.rocker], m_rocker @ self._world(loop.rocker, {})))
            for pid, host in riders.items():
                posed.append((self._parts[pid], host_m[host] @ self._world(pid, {})))
            for g, g_world in posed:
                for s, s_world in statics:
                    depth = self._pair_overlap_depth(g, g_world, s, s_world)
                    if depth is not None:
                        key = (g.pid, s.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
            for ix in range(len(posed)):
                for jx in range(ix + 1, len(posed)):
                    g, g_world = posed[ix]
                    h2, h_world = posed[jx]
                    depth = self._pair_overlap_depth(g, g_world, h2, h_world)
                    if depth is not None:
                        key = (g.pid, h2.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
        trans_note = ""
        if worst_trans is not None:
            t_min = 90.0 - worst_trans[0]
            trans_note = f"; min transmission angle {t_min:.0f} deg at {worst_trans[1]:.0f} deg"
            if t_min < 40.0:
                trans_note += " (POOR -- near toggle, force transmission collapses)"
        if collisions:
            return False, _format_collisions(collisions, "deg") + " (four-bar coupled)"
        return True, (
            f"clear {lo:.0f}..{hi:.0f} deg four-bar coupled ({len(loops)} loop(s), "
            f"{_SAMPLES} poses, loop-closure driven, movers vs rest AND vs each other)" + trans_note
        )

    def sweep_epicyclic(
        self,
        sun_joint: Joint,
        sun_id: str,
        systems: list[EpicyclicSet],
        lo: float,
        hi: float,
        riders: dict[str, str] | None = None,
    ) -> tuple[bool | None, str]:
        """Drive a FIXED-RING epicyclic train through the sun's range: carrier at
        theta*zs/(zs+zr) about the shared central axis, each planet counter-spinning
        about its own carrier-riding pin at the Willis-derived relative rate (the v1
        'gear-coupled pair within a moving chain' class). Poses are EXACT closed-form --
        correctly-backlashed teeth never interpenetrate, so a clear sweep is earned, and
        a mis-spaced/oversized member shows real depth. Returns (None, reason) when the
        train is not drivable (mis-specified tooth counts, non-coaxial sun/carrier,
        missing geometry) -- the motion claim stays honestly NOT EARNED, never a phantom
        verdict."""
        riders = dict(riders or {})
        if len(systems) != 1:
            return None, "epicyclic not drivable: exactly one train per driver supported (v1)"
        sysm = systems[0]
        z_s, z_p, z_r = sysm.z_sun, sysm.z_planet, sysm.z_ring
        if z_r != z_s + 2 * z_p:
            return None, (
                f"epicyclic not drivable: tooth counts do not close the train "
                f"(ring {z_r} != sun {z_s} + 2*planet {z_p})"
            )
        pivot_o, pivot_d = self.joint_pivot(sun_joint, {})
        car_o, car_d = self.joint_pivot(sysm.carrier_joint, {})
        axis_dot = abs(float(np.dot(pivot_d, car_d)))
        offset = float(np.linalg.norm(np.cross(car_o - pivot_o, pivot_d)))
        if axis_dot < 0.999 or offset > 1.0:
            return None, (
                f"epicyclic not drivable: sun and carrier axes are not coaxial "
                f"(angle cos {axis_dot:.3f}, offset {offset:.1f} mm)"
            )
        k_c, k_p = epicyclic_ratios(z_s, z_p, z_r)
        carrier_id = sysm.carrier_joint.b_part()
        planet_ids = [j.b_part() for j in sysm.planet_joints]
        sun_moving = downstream_ids(self._model.joints, sun_joint)
        carrier_all = downstream_ids(self._model.joints, sysm.carrier_joint)
        planet_moving: dict[str, set[str]] = {
            j.b_part(): downstream_ids(self._model.joints, j) for j in sysm.planet_joints
        }
        planet_union: set[str] = set().union(*planet_moving.values()) if planet_moving else set()
        planet_union |= set(planet_moving)
        carrier_only = ((carrier_all - planet_union) | {carrier_id}) - {sun_id}
        # DISJOINT groups: a carrier journaled ON the sun puts the whole train inside
        # the sun's placement subtree -- each member belongs to its MOST SPECIFIC driver.
        sun_moving = sun_moving - carrier_all - {carrier_id} - planet_union
        pin_axes: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
            j.b_part(): self.joint_pivot(j, {}) for j in sysm.planet_joints
        }
        groups: dict[str, set[str]] = {"sun": sun_moving | {sun_id}, "carrier": carrier_only}
        for pid_, members in planet_moving.items():
            groups[pid_] = members | {pid_}
        mover_ids: set[str] = set().union(*groups.values())
        for rid, host in riders.items():
            if rid in self._parts and rid not in mover_ids:
                for members in groups.values():
                    if host in members:
                        members.add(rid)
                        mover_ids.add(rid)
                        break
        missing = sorted(m for members in groups.values() for m in members if m not in self._parts)
        if missing:
            return None, f"epicyclic not drivable: mover(s) without geometry: {missing}"
        statics = [
            (g, self._world(g.pid, {})) for pid, g in self._parts.items() if pid not in mover_ids
        ]
        # FALSE-PASS guard (adversarial review): sweeping the SUN's declared range covers
        # only k_c x of the CARRIER orbit -- 'clear 0..360' certified a reducer whose
        # planets never reached a frame boss at 165 deg (struck on the next crank turn in
        # real use). A train is a continuous-rotation device: when the sun range allows a
        # full turn, drive ONE FULL CARRIER REVOLUTION (sun theta to 360/k_c) and report
        # the CARRIER span. Denser sampling (10 deg carrier steps) so an orbital-scale
        # feature cannot alias between poses.
        span = hi - lo
        theta_hi = lo + 360.0 / k_c if span >= 360.0 - 1e-9 else hi
        car_span = k_c * (theta_hi - lo)
        n_samples = max(_SAMPLES, 37)
        collisions: dict[tuple[str, str], tuple[float, float]] = {}
        for i in range(n_samples):
            theta = lo + (theta_hi - lo) * i / (n_samples - 1)
            m_sun = _rotation_about_line(pivot_o, pivot_d, theta)
            m_car = _rotation_about_line(pivot_o, pivot_d, k_c * theta)
            posed: list[tuple[_PartGeom, NDArray[np.float64]]] = []
            for m in groups["sun"]:
                posed.append((self._parts[m], m_sun @ self._world(m, {})))
            for m in groups["carrier"]:
                posed.append((self._parts[m], m_car @ self._world(m, {})))
            for pid_ in planet_ids:
                pin_o, pin_d = pin_axes[pid_]
                m_planet = m_car @ _rotation_about_line(pin_o, pin_d, k_p * theta)
                for m in groups[pid_]:
                    posed.append((self._parts[m], m_planet @ self._world(m, {})))
            for g, g_world in posed:
                for s, s_world in statics:
                    depth = self._pair_overlap_depth(g, g_world, s, s_world)
                    if depth is not None:
                        key = (g.pid, s.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
            for ix in range(len(posed)):
                for jx in range(ix + 1, len(posed)):
                    g, g_world = posed[ix]
                    h2, h_world = posed[jx]
                    if g.pid in groups["sun"] and h2.pid in groups["sun"]:
                        continue  # rigidly co-moving: relative pose never changes
                    if g.pid in groups["carrier"] and h2.pid in groups["carrier"]:
                        continue
                    depth = self._pair_overlap_depth(g, g_world, h2, h_world)
                    if depth is not None:
                        key = (g.pid, h2.pid)
                        prev = collisions.get(key)
                        if prev is None or depth > prev[0]:
                            collisions[key] = (depth, theta)
        if collisions:
            # Actionable keep-out for the repair round: the measured swept envelope of
            # the ORBITING members (carrier + planets) about the central axis -- the
            # static housing must be void exactly there. Derived from the posed bbox
            # corners across all samples (conservative hull, deterministic).
            r_lo, r_hi = float("inf"), 0.0
            z_lo, z_hi = float("inf"), -float("inf")
            orbit_ids = set(groups["carrier"]) | {m for pid_ in planet_ids for m in groups[pid_]}
            for i in range(n_samples):
                theta = lo + (theta_hi - lo) * i / (n_samples - 1)
                m_car = _rotation_about_line(pivot_o, pivot_d, k_c * theta)
                for pid_ in ("carrier", *planet_ids):
                    if pid_ == "carrier":
                        motion = m_car
                        members = groups["carrier"]
                    else:
                        pin_o, pin_d = pin_axes[pid_]
                        motion = m_car @ _rotation_about_line(pin_o, pin_d, k_p * theta)
                        members = groups[pid_]
                    for m in members:
                        g = self._parts[m]
                        world = motion @ self._world(m, {})
                        corners = trimesh.bounds.corners(g.mesh_local.bounds)  # type: ignore[no-untyped-call]
                        pts = (world[:3, :3] @ corners.T).T + world[:3, 3]
                        rel = pts - pivot_o
                        along = rel @ pivot_d
                        radial = np.linalg.norm(rel - np.outer(along, pivot_d), axis=1)
                        r_lo = min(r_lo, float(radial.min()))
                        r_hi = max(r_hi, float(radial.max()))
                        z_lo = min(z_lo, float(along.min()))
                        z_hi = max(z_hi, float(along.max()))
            keep_out = (
                f"; ORBIT KEEP-OUT: the carrier+planets sweep an annulus r 0..{r_hi:.1f} mm"
                f" (members reach in to r {r_lo:.1f}), axial {z_lo:.1f}..{z_hi:.1f} mm about"
                " the central axis -- every static part must be VOID in that volume"
                if orbit_ids
                else ""
            )
            return False, _format_collisions(collisions, "deg") + " (epicyclic coupled)" + keep_out
        return True, (
            f"clear carrier {k_c * lo:.0f}..{k_c * lo + car_span:.0f} deg epicyclic "
            f"(sun driven {lo:.0f}..{theta_hi:.0f} deg at {k_c:.3f}x, {len(planet_ids)} "
            f"planet(s) counter-spinning at {k_p:.3f}x relative, {n_samples} poses, "
            "movers vs rest AND vs each other; pitch-envelope sweep -- tooth meshing is "
            "owned by the gear checks)"
        )

    def static_overlaps(self) -> list[str]:
        """Interpenetrating pairs at the HOME pose -- 'the parts actually assemble'.
        Every mesh-bearing pair is tested once; pins inside bores and bodies inside
        clearance pockets share no interior volume and pass."""
        ids = sorted(self._parts)
        bad: list[str] = []
        for i, pa in enumerate(ids):
            for pb in ids[i + 1 :]:
                a, b = self._parts[pa], self._parts[pb]
                if self._pair_overlaps(a, a.base_transform, b, b.base_transform):
                    bad.append(f"{pa}<->{pb}")
        return bad

    def corner_prefix(self, parent: Joint, extreme: float) -> dict[str, NDArray[np.float64]]:
        """The pose map for 'parent held at extreme': every part downstream of parent
        gets the parent's rotation/translation about its HOME pivot."""
        pivot_o, pivot_d = self.joint_pivot(parent, {})
        motion = (
            _rotation_about_line(pivot_o, pivot_d, extreme)
            if parent.type is JointType.REVOLUTE
            else _translation(pivot_d, extreme)
        )
        return dict.fromkeys(downstream_ids(self._model.joints, parent), motion)


def chained_pairs(joints: list[Joint]) -> list[tuple[Joint, Joint]]:
    """(parent, child) pairs of MOVING joints where the child's reference part moves
    under the parent -- the combinations a one-joint-at-a-time sweep cannot see."""
    moving = [j for j in joints if j.type in (JointType.REVOLUTE, JointType.PRISMATIC)]
    out: list[tuple[Joint, Joint]] = []
    for parent in moving:
        down = downstream_ids(joints, parent)
        for child in moving:
            if child is not parent and child.a_part() in down:
                out.append((parent, child))
    return out


@dataclass(frozen=True)
class EpicyclicSet:
    """One fixed-ring epicyclic train: the members ``sweep_epicyclic`` drives.

    ``planet_joints`` are ORIENTED copies (b = the planet, the mover convention the
    sweep + check ids use); ``owned_joints`` are the ORIGINAL graph objects (their
    ``id()`` keys the pin-collapse origin_map, so ownership expands to the real pin
    joints -- an oriented copy would silently break that)."""

    carrier_joint: Joint  # carrier revolute ORIENTED b=carrier (host = frame OR sun)
    planet_joints: list[Joint]  # carrier<->planet revolutes, oriented b=planet
    owned_joints: list[Joint]  # ALL original graph objects (sun+carrier+planet revolutes)
    emit_joints: list[Joint]  # originals for member/check emission (carrier+planets --
    # the driver tuple's sun joint emits separately; originals key the pin-collapse
    # origin_map so a pin-decomposed planet re-expands and `swept` counts its real pins)
    sun_meshes: list[Joint]  # gear_mesh sun<->planet_i (external)
    ring_meshes: list[Joint]  # gear_mesh ring<->planet_i (internal)
    z_sun: int
    z_planet: int
    z_ring: int

    def joints(self) -> list[Joint]:
        return list(self.emit_joints)


def epicyclic_ratios(z_sun: int, z_planet: int, z_ring: int) -> tuple[float, float]:
    """(carrier/sun ratio, planet-spin-RELATIVE-TO-CARRIER/sun ratio) for a FIXED ring,
    from the Willis equation: w_c = w_s * zs/(zs+zr); (w_p - w_c) = -(zs/zp)(w_s - w_c).
    Cross-check (12/18/48): carrier 1/5 (the 5:1 the chief promises), planet -8/15 --
    consistent through BOTH the sun-mesh and the ring-mesh routes."""
    k_c = z_sun / (z_sun + z_ring)
    k_p = -(z_sun / z_planet) * (z_ring / (z_sun + z_ring))
    return k_c, k_p


def epicyclic_systems(
    joints: list[Joint],
) -> list[tuple[Joint, str, str, list[EpicyclicSet]]]:
    """Detect FIXED-RING epicyclic trains from the (pin-collapsed) joint graph, in the
    (driver_joint, frame_id, driver_part_id, [set]) shape the driver_runs loop eats.

    The signature (all conditions required -- anything else stays with the honest
    withhold): a part S with EXTERNAL gear meshes to N>=2 planets P_i; a part R with
    INTERNAL meshes to the same planets; a common carrier C with a revolute to every
    P_i; revolutes F<->S and F<->C on the same frame F; and R is F itself (the ring
    integral to the frame, v1). Tooth counts/module must be declared and consistent
    (the gear checks own their correctness; detection only needs them present)."""
    meshes = [j for j in joints if j.type is JointType.GEAR_MESH]
    revs = [j for j in joints if j.type is JointType.REVOLUTE]
    if not meshes or not revs:
        return []

    def _teeth(j: Joint, part: str) -> int | None:
        if j.teeth_a is None or j.teeth_b is None:
            return None
        return int(j.teeth_a) if j.a_part() == part else int(j.teeth_b)

    external: dict[str, list[Joint]] = {}
    internal: dict[str, list[Joint]] = {}
    for m in meshes:
        for side in (m.a_part(), m.b_part()):
            (internal if m.internal else external).setdefault(side, []).append(m)

    rigid = [j for j in joints if j.type in (JointType.COAXIAL_INSERT, JointType.FIXED)]
    moving_count: dict[str, int] = {}
    for j in joints:
        if j.type in (JointType.REVOLUTE, JointType.PRISMATIC):
            moving_count[j.a_part()] = moving_count.get(j.a_part(), 0) + 1
            moving_count[j.b_part()] = moving_count.get(j.b_part(), 0) + 1

    def _deref_pin(host: str) -> str:
        """A press-fit PIN (exactly one rigid mount + one planet revolute, iteration 6's
        coax+rev shape that collapse_pin_edges does not fold) is rigid with its mount
        host: the EFFECTIVE carrier is one level up. Conservative: any other topology
        keeps the host as-is."""
        if moving_count.get(host, 0) != 1:
            return host
        mounts = [
            (j.b_part() if j.a_part() == host else j.a_part())
            for j in rigid
            if host in (j.a_part(), j.b_part())
        ]
        return mounts[0] if len(mounts) == 1 else host

    out: list[tuple[Joint, str, str, list[EpicyclicSet]]] = []
    claimed: set[int] = set()
    for sun, sun_meshes in external.items():
        if any(id(m) in claimed for m in sun_meshes):
            continue
        planets = [m.b_part() if m.a_part() == sun else m.a_part() for m in sun_meshes]
        if len(planets) < 2 or len(set(planets)) != len(planets):
            continue
        ring_candidates = [
            r
            for r, rms in internal.items()
            if r not in planets
            and r != sun
            and {(rm.b_part() if rm.a_part() == r else rm.a_part()) for rm in rms} >= set(planets)
        ]
        if len(ring_candidates) != 1:
            continue
        ring = ring_candidates[0]
        ring_meshes = [
            rm
            for rm in internal[ring]
            if (rm.b_part() if rm.a_part() == ring else rm.a_part()) in planets
        ]
        planet_joints: list[Joint] = []
        owned_joints: list[Joint] = []
        carriers: set[str] = set()
        for pid in planets:
            pj = [
                r
                for r in revs
                if (r.b_part() == pid and r.a_part() not in (sun, ring))
                or (r.a_part() == pid and r.b_part() not in (sun, ring))
            ]
            if len(pj) != 1:
                planet_joints = []
                break
            j = pj[0]
            host = _deref_pin(j.a_part() if j.b_part() == pid else j.b_part())
            carriers.add(host)
            owned_joints.append(j)  # the original object: its id() keys origin_map
            # oriented copy so b is the planet (the mover convention the sweep uses)
            planet_joints.append(
                j if j.b_part() == pid else j.model_copy(update={"a": j.b, "b": j.a})
            )
        if not planet_joints or len(carriers) != 1:
            continue
        carrier = next(iter(carriers))

        def _oriented(mover: str, hosts: tuple[str, ...]) -> tuple[Joint, Joint] | None:
            """(original, copy-oriented-b=mover) for the revolute mounting ``mover`` on
            one of ``hosts`` -- either declared orientation (iteration 6: the chief
            legitimately journaled the CARRIER on the SUN's front journal, and the
            rigid b==mover/a==ring lookup missed the whole train)."""
            for r in revs:
                if r.b_part() == mover and r.a_part() in hosts:
                    return r, r
                if r.a_part() == mover and r.b_part() in hosts:
                    return r, r.model_copy(update={"a": r.b, "b": r.a})
            return None

        sun_pair = _oriented(sun, (ring,))
        carrier_pair = _oriented(carrier, (ring, sun))
        if sun_pair is None or carrier_pair is None:
            continue
        sun_orig, sun_joint = sun_pair
        carrier_orig, carrier_joint = carrier_pair
        z_s = _teeth(sun_meshes[0], sun)
        z_p = _teeth(sun_meshes[0], planets[0])
        z_r = _teeth(ring_meshes[0], ring)
        if z_s is None or z_p is None or z_r is None:
            continue
        eps = EpicyclicSet(
            carrier_joint=carrier_joint,
            planet_joints=planet_joints,
            owned_joints=[sun_orig, carrier_orig, *owned_joints],
            emit_joints=[carrier_orig, *owned_joints],
            sun_meshes=list(sun_meshes),
            ring_meshes=ring_meshes,
            z_sun=z_s,
            z_planet=z_p,
            z_ring=z_r,
        )
        for j in (sun_orig, carrier_orig, *owned_joints, *sun_meshes, *ring_meshes):
            claimed.add(id(j))
        out.append((sun_joint, ring, sun, [eps]))
    return out


__all__ = [
    "ChainSweeper",
    "FourBarLoop",
    "SliderCrankLoop",
    "chained_pairs",
    "collapse_pin_edges",
    "downstream_ids",
    "epicyclic_ratios",
    "epicyclic_systems",
    "four_bar_systems",
    "meshless_hardware_ids",
    "slider_crank_systems",
    "undriven_loop_joint_ids",
]
