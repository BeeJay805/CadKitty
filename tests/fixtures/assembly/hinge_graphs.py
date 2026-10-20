"""Hand-authored reference HingeGraphs for the assembly spike (ADR-0010).

NOT a src-level hinge template: these are reference fixtures -- one known-good
2-leaf hinge plus three single-defect variants -- that exercise the GENERIC
assembly spine (knuckled_plate + rod + ports + joints + solver + validator), in
the same spirit as tests/fixtures/handwritten/*.partdsl.json. The hinge is
composed in the graph; nothing here is a hinge-specific archetype.

Defect variants (each isolates ONE failure mode, like the Phase-2 broken fixtures):
  * welded        -> the pivot joint is FIXED (no rotational DOF)
  * sub_clearance -> the pin is fatter than the bore (no running-fit clearance)
  * binding       -> leafB lacks its interleave notches, so its plate hits leafA's barrels
"""

from __future__ import annotations

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode

# Geometry shared by both leaves. Loose-ish running fit so the first print
# definitely rotates; tighten only after physical calibration.
_LEAF = {
    "length_x": 40.0,
    "width_y": 18.0,
    "thickness": 4.0,
    "knuckle_outer_r": 5.0,
    "bore_r": 2.6,
    "knuckle_width": 8.0,
}
_LEAFA_KNUCKLES = [-8.5, 8.5]  # two outer knuckles
_LEAFB_KNUCKLES = [0.0]  # one knuckle, interleaved into leafA's gap
_PIN_LENGTH = 25.0


def _graph(
    *, coaxial_class: str = "running_fit", leafb_notches: list[float], pivot: JointType
) -> AssemblyGraph:
    leaf_a = dict(
        _LEAF, knuckle_centers_x=list(_LEAFA_KNUCKLES), clearance_notches_x=list(_LEAFB_KNUCKLES)
    )
    leaf_b = dict(
        _LEAF, knuckle_centers_x=list(_LEAFB_KNUCKLES), clearance_notches_x=list(leafb_notches)
    )
    nodes = [
        PartNode(id="leafA", archetype="knuckled_plate", params=leaf_a),
        PartNode(id="leafB", archetype="knuckled_plate", params=leaf_b),
        # No radius: the solver sizes the pin from the bore minus the coaxial fit class.
        PartNode(id="pin", archetype="rod", params={"length_x": _PIN_LENGTH}),
    ]
    joints = [
        Joint(
            type=JointType.COAXIAL_INSERT,
            a="leafA.bore",
            b="pin.shaft",
            clearance_class=coaxial_class,
        ),
        Joint(
            type=pivot,
            a="leafA.bore",
            b="leafB.bore",
            range_deg=(0.0, 180.0) if pivot is JointType.REVOLUTE else None,
            clearance_class="running_fit",
        ),
    ]
    return AssemblyGraph(nodes=nodes, joints=joints, root="leafA")


def good_hinge_graph() -> AssemblyGraph:
    """A correct 2-leaf hinge: interleaved, running-fit pin (solver-sized), revolute pivot."""
    return _graph(leafb_notches=_LEAFA_KNUCKLES, pivot=JointType.REVOLUTE)


def welded_hinge_graph() -> AssemblyGraph:
    """Defect: the pivot is a FIXED joint -> no rotational DOF (a seized/welded hinge)."""
    return _graph(leafb_notches=_LEAFA_KNUCKLES, pivot=JointType.FIXED)


def sub_clearance_hinge_graph() -> AssemblyGraph:
    """Defect: an interference coaxial fit -> the solver sizes the pin fatter than the bore."""
    return _graph(
        coaxial_class="interference", leafb_notches=_LEAFA_KNUCKLES, pivot=JointType.REVOLUTE
    )


def binding_hinge_graph() -> AssemblyGraph:
    """Defect: leafB lacks interleave notches -> its plate collides with leafA's barrels."""
    return _graph(leafb_notches=[], pivot=JointType.REVOLUTE)


__all__ = [
    "binding_hinge_graph",
    "good_hinge_graph",
    "sub_clearance_hinge_graph",
    "welded_hinge_graph",
]
