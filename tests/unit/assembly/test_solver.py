"""ConstraintSolver: graph -> world transforms (no compile, fast).

Verifies the solver's frame-mate owns the inter-part metric the LLM never sets:
the pin lands coaxial in the bore, the moving leaf mates coplanar-opposite, the
root is identity, and the solve is deterministic.
"""

from __future__ import annotations

import numpy as np

from forge_mind.assembly.program import AssemblyPart
from forge_mind.assembly.solver import solve
from tests.fixtures.assembly.hinge_graphs import good_hinge_graph, sub_clearance_hinge_graph


def _port_world(part: AssemblyPart, name: str) -> tuple[np.ndarray, np.ndarray]:
    m = part.transform.as_matrix()
    port = next(p for p in part.ports if p.name == name)
    o = (m @ np.array([*port.origin.as_tuple(), 1.0]))[:3]
    ax = m[:3, :3] @ np.array(port.axis.as_tuple())
    return o, ax / np.linalg.norm(ax)


def test_solver_places_all_three_bodies() -> None:
    asm = solve(good_hinge_graph())
    assert {p.id for p in asm.parts} == {"leafA", "leafB", "pin"}


def test_root_leaf_is_world_identity() -> None:
    asm = solve(good_hinge_graph())
    assert np.allclose(asm.part("leafA").transform.as_matrix(), np.eye(4))


def test_pin_is_coaxial_with_the_bore() -> None:
    # The solver -- not the graph -- derives where the pin goes.
    asm = solve(good_hinge_graph())
    bore_o, bore_ax = _port_world(asm.part("leafA"), "bore")
    pin_o, pin_ax = _port_world(asm.part("pin"), "shaft")
    assert abs(abs(float(np.dot(bore_ax, pin_ax))) - 1.0) < 1e-6  # axes colinear
    offset = pin_o - bore_o
    perp = offset - np.dot(offset, bore_ax) * bore_ax
    assert float(np.linalg.norm(perp)) < 1e-6  # pin origin sits on the bore line


def test_moving_leaf_mates_coplanar_opposite() -> None:
    # leafB's plate normal flips to face leafA (opposed) while staying in-plane,
    # and the placement is a proper rotation (no reflection).
    asm = solve(good_hinge_graph())
    m = asm.part("leafB").transform.as_matrix()
    plate_normal = m[:3, :3] @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(plate_normal, [0.0, 1.0, 0.0], atol=1e-6)
    assert abs(float(np.linalg.det(m[:3, :3])) - 1.0) < 1e-6


def test_solver_is_deterministic() -> None:
    a = solve(good_hinge_graph())
    b = solve(good_hinge_graph())
    for pa, pb in zip(a.parts, b.parts, strict=True):
        assert np.allclose(pa.transform.as_matrix(), pb.transform.as_matrix())


def _shaft_radius(asm: object) -> float:
    pin = asm.part("pin")  # type: ignore[attr-defined]
    shaft = next(op for op in pin.program.operations if op.op_id == "shaft")
    return float(shaft.radius)  # type: ignore[attr-defined]


def test_pin_radius_is_derived_from_bore_minus_running_fit() -> None:
    # The graph never states the pin radius; the solver sizes it from the bore
    # (2.6) minus the coaxial running_fit (0.4) = 2.2. The inter-part metric is
    # owned by deterministic code, not the LLM-facing graph.
    assert abs(_shaft_radius(solve(good_hinge_graph())) - 2.2) < 1e-9


def test_interference_fit_derives_an_oversize_pin() -> None:
    # An interference coaxial fit (-0.1) sizes the pin to 2.7 > bore 2.6, which the
    # validator will then catch from geometry as a clearance failure.
    assert abs(_shaft_radius(solve(sub_clearance_hinge_graph())) - 2.7) < 1e-9
