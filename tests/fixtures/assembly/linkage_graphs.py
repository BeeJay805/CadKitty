"""Hand-authored reference 2-bar linkage AssemblyGraphs (ADR-0010): a kinematic CHAIN.

The third mechanism family, proving the spine handles SERIAL chains (a link pivots on
a MOVING link), not just single joints. New generic pieces: the ``flat_link`` archetype
(a bar with a pivot bore at each end) and the ``align`` mate (parallel stacking, so
chained links sit on different Z-layers and articulate freely). Everything else is
reused -- the rod archetype, the per-pin cylinder clearance check, the bore-minus-fit
derivation, and the per-joint motion check. No ``linkage`` object-template.

Defects: interference (a pin fatter than its bore), seized (both pivots fixed -> no DOF).
"""

from __future__ import annotations

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode

_LINK = {"length_x": 40.0, "width_y": 14.0, "thickness": 4.0, "bore_r": 2.6}
_PIN_LENGTH = 16.0


def _graph(*, fit: str = "running_fit", seized: bool = False) -> AssemblyGraph:
    pivot = JointType.FIXED if seized else JointType.REVOLUTE
    nodes = [
        PartNode(id="base", archetype="flat_link", params=dict(_LINK)),
        PartNode(id="bar1", archetype="flat_link", params=dict(_LINK)),
        PartNode(id="bar2", archetype="flat_link", params=dict(_LINK)),
        PartNode(id="pin1", archetype="rod", params={"length_x": _PIN_LENGTH}),
        PartNode(id="pin2", archetype="rod", params={"length_x": _PIN_LENGTH}),
    ]
    joints = [
        Joint(type=JointType.COAXIAL_INSERT, a="base.pivot_b", b="pin1.shaft", clearance_class=fit),
        Joint(type=pivot, a="base.pivot_b", b="bar1.pivot_a", range_deg=(0.0, 90.0), mate="align"),
        Joint(type=JointType.COAXIAL_INSERT, a="bar1.pivot_b", b="pin2.shaft"),
        Joint(type=pivot, a="bar1.pivot_b", b="bar2.pivot_a", range_deg=(0.0, 90.0), mate="align"),
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="base")


def good_linkage_graph() -> AssemblyGraph:
    """A correct 2-bar serial linkage: base -> bar1 -> bar2, two pinned revolute pivots."""
    return _graph()


def interference_linkage_graph() -> AssemblyGraph:
    """Defect: an interference fit sizes a pin fatter than its bore -> it cannot pivot."""
    return _graph(fit="interference")


def seized_linkage_graph() -> AssemblyGraph:
    """Defect: both pivots are FIXED joints -> the chain has no degree of freedom."""
    return _graph(seized=True)


__all__ = ["good_linkage_graph", "interference_linkage_graph", "seized_linkage_graph"]
