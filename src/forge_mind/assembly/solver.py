"""ConstraintSolver — the deterministic core that owns every world coordinate.

Given an ``AssemblyGraph`` (parts + typed joints between ports), the solver places
each part by *frame-mating* its port onto the already-placed reference part's port.
Mating uses the convention that connected ports face each other (axes anti-parallel,
normals opposed), which yields, for two identical leaves, a coplanar open-flat
relationship and, for a pin, a coaxial insertion — all without the LLM (or the
graph) ever stating a coordinate. This is "structure derives metric" lifted from
intra-part (the LayoutSpec interpreters) to inter-part.

Tree assemblies only (no loop closure): a part is placed once its reference is
placed; the spike's hinge is a 2-edge tree rooted at the fixed leaf.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from forge_mind.assembly.archetypes import build_node
from forge_mind.assembly.fits import clearance_mm
from forge_mind.assembly.hardware import build_hardware_node
from forge_mind.assembly.joints import AssemblyGraph, JointType, PartNode, Port
from forge_mind.assembly.program import AssemblyPart, AssemblyProgram, Transform
from forge_mind.dsl.program import PartProgram

Vec3T = tuple[float, float, float]


def _unit(v: Vec3T | NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def _apply_point(m: NDArray[np.float64], p: Vec3T) -> NDArray[np.float64]:
    return np.asarray((m @ np.array([p[0], p[1], p[2], 1.0]))[:3], dtype=np.float64)


def _apply_dir(m: NDArray[np.float64], d: Vec3T) -> NDArray[np.float64]:
    return np.asarray(m[:3, :3] @ np.asarray(d, dtype=float), dtype=np.float64)


def _frame_mate(
    a_origin: NDArray[np.float64],
    a_axis: NDArray[np.float64],
    a_normal: NDArray[np.float64],
    b_origin: Vec3T,
    b_axis: Vec3T,
    b_normal: Vec3T,
    *,
    oppose: bool = True,
) -> NDArray[np.float64]:
    """World transform placing part b so its port frame mates onto a's.

    ``oppose`` (default): ports face each other -- axis -> -a_axis, normal -> -a_normal
    (a pin into a bore, hinge leaves). ``oppose=False``: ports align parallel -- axis ->
    +a_axis (stacked links in a serial chain). Builds an orthonormal basis from each
    frame and solves R = T @ B^T.
    """
    sign = -1.0 if oppose else 1.0
    t_axis = sign * _unit(a_axis)
    t_normal = sign * _unit(a_normal)
    t_bi = np.cross(t_axis, t_normal)
    target = np.column_stack([t_axis, t_normal, t_bi])

    b_ax = _unit(b_axis)
    b_n = _unit(b_normal)
    b_bi = np.cross(b_ax, b_n)
    basis = np.column_stack([b_ax, b_n, b_bi])

    rot = target @ basis.T
    trans = a_origin - rot @ np.asarray(b_origin, dtype=float)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def _port(ports: list[Port], name: str) -> Port:
    return next(p for p in ports if p.name == name)


def _resolved_params(graph: AssemblyGraph) -> dict[str, dict[str, Any]]:
    """Inject the inter-part metric the structure determines (not the graph).

    Currently: a rod inserted (coaxial_insert) into a bored part is sized to that
    part's bore minus the joint's clearance class -- so the pin's fit is DERIVED
    from the mating bore, never hand-set. An explicit ``radius`` on the rod wins
    (escape hatch), but the references omit it so the fit comes from the structure.
    """
    params: dict[str, dict[str, Any]] = {node.id: dict(node.params) for node in graph.nodes}
    archetype = {node.id: node.archetype for node in graph.nodes}
    for joint in graph.joints:
        # A rod mated into a bored part (pin, piston -- ANY joint type, including a
        # welded/fixed one) is sized to that bore minus the fit class: the fit is a
        # structural fact, not a property of the DOF.
        inserted, receiver = joint.b_part(), joint.a_part()
        if archetype.get(inserted) == "rod" and "radius" not in params[inserted]:
            bore_r = params[receiver].get("bore_r")
            if bore_r is not None:
                params[inserted]["radius"] = float(bore_r) - clearance_mm(joint.clearance_class)
    return params


def _build(node: PartNode, params: dict[str, Any]) -> tuple[PartProgram | None, list[Port]]:
    """Build a node to ``(program, ports)``. A non-printed hardware node yields
    ``(None, ports)`` (frames only, no geometry); a printed node yields its program.
    The solver only ever reads the ports, so both kinds place identically."""
    if not node.printed:
        return build_hardware_node(node.id, params)
    return build_node(node.archetype, node.id, params)


def solve(graph: AssemblyGraph) -> AssemblyProgram:
    """Resolve every part's world transform from the graph's joints. Deterministic."""
    resolved = _resolved_params(graph)
    built = {node.id: _build(node, resolved[node.id]) for node in graph.nodes}

    transforms: dict[str, NDArray[np.float64]] = {}
    root = graph.root or (graph.joints[0].a_part() if graph.joints else graph.nodes[0].id)
    transforms[root] = np.eye(4)

    # Repeated relaxation: place the unplaced side of each joint once the other side
    # is placed. The mate convention is SYMMETRIC for both modes (oppose: axes
    # anti-parallel; align: axes parallel), so a -> b and b -> a use the same
    # frame-mate with the roles swapped -- without the reverse direction, rooting a
    # graph at a part that only appears on b-sides silently left the rest at identity
    # (latent until the hardware envelopes made the wrong placement collide).
    # Bounded by joint count (tree).
    for _ in range(len(graph.joints) + 1):
        for joint in graph.joints:
            if joint.type == JointType.GEAR_MESH:
                # A gear mesh COUPLES two already-placed parts' rotations; it is a
                # constraint edge, never a placement edge. Frame-mating across it
                # would stack one gear ONTO the other's axis (wrong by a full centre
                # distance) -- both gears place through their own pivot joints.
                continue
            a_part, b_part = joint.a_part(), joint.b_part()
            placed, mover = None, None
            if a_part in transforms and b_part not in transforms:
                placed, placed_port = a_part, joint.a_port()
                mover, mover_port = b_part, joint.b_port()
            elif b_part in transforms and a_part not in transforms:
                placed, placed_port = b_part, joint.b_port()
                mover, mover_port = a_part, joint.a_port()
            if placed is None or mover is None:
                continue
            p_port = _port(built[placed][1], placed_port)
            m_port = _port(built[mover][1], mover_port)
            m_placed = transforms[placed]
            p_origin = _apply_point(m_placed, p_port.origin.as_tuple())
            p_axis = _apply_dir(m_placed, p_port.axis.as_tuple())
            p_normal = _apply_dir(m_placed, p_port.normal.as_tuple())
            transforms[mover] = _frame_mate(
                p_origin,
                p_axis,
                p_normal,
                m_port.origin.as_tuple(),
                m_port.axis.as_tuple(),
                m_port.normal.as_tuple(),
                oppose=(joint.mate == "oppose"),
            )

    parts = [
        AssemblyPart(
            id=node.id,
            program=built[node.id][0],
            transform=Transform.from_matrix(transforms.get(node.id, np.eye(4))),
            ports=built[node.id][1],
            printed=node.printed,
            hardware_kind=(str(resolved[node.id].get("hardware")) if not node.printed else None),
        )
        for node in graph.nodes
    ]
    return AssemblyProgram(parts=parts, joints=list(graph.joints))


__all__ = ["solve"]
