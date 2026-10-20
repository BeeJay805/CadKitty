"""Hand-authored reference slide AssemblyGraphs (ADR-0010): a prismatic piston-in-tube.

Proves the assembly spine generalizes past the hinge: a NEW joint type (prismatic)
and a NEW archetype (tube), while REUSING the rod archetype, the cylinder clearance
check, and the bore-minus-fit derivation. The slide EMERGES from generic pieces --
there is no `slide` object-template, exactly as there is no `hinge` one.

Defect variants (each isolates a failure mode):
  * interference -> piston fatter than the bore (no clearance; also binds)
  * welded       -> a FIXED joint, no translational DOF
  * over_travel  -> the declared travel slides the piston out of the tube
"""

from __future__ import annotations

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode

_TUBE = {"length_x": 40.0, "outer_r": 8.0, "bore_r": 5.0}
_PISTON_LENGTH = 16.0


def _graph(
    *,
    fit: str = "running_fit",
    pivot: JointType = JointType.PRISMATIC,
    travel: tuple[float, float] = (0.0, 10.0),
) -> AssemblyGraph:
    nodes = [
        PartNode(id="tube", archetype="tube", params=dict(_TUBE)),
        # No radius: the solver sizes the piston from the tube bore minus the fit.
        PartNode(id="piston", archetype="rod", params={"length_x": _PISTON_LENGTH}),
    ]
    joints = [
        Joint(
            type=pivot,
            a="tube.bore",
            b="piston.shaft",
            range_mm=travel if pivot is JointType.PRISMATIC else None,
            clearance_class=fit,
        )
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="tube")


def good_slide_graph() -> AssemblyGraph:
    """A correct linear slide: running-fit piston (solver-sized), prismatic travel within the tube."""
    return _graph()


def interference_slide_graph() -> AssemblyGraph:
    """Defect: an interference fit sizes the piston fatter than the bore -> it cannot slide."""
    return _graph(fit="interference")


def welded_slide_graph() -> AssemblyGraph:
    """Defect: a FIXED joint -> no translational DOF (a seized slide)."""
    return _graph(pivot=JointType.FIXED)


def over_travel_slide_graph() -> AssemblyGraph:
    """Defect: the declared travel slides the piston out of the tube (loses engagement)."""
    return _graph(travel=(0.0, 40.0))


__all__ = [
    "good_slide_graph",
    "interference_slide_graph",
    "over_travel_slide_graph",
    "welded_slide_graph",
]
