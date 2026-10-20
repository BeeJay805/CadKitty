"""The servo-arm family: a deterministic serial-chain generator (MG996R, 1-4 DOF).

One generator covers the whole class -- pan-tilt heads (dof=2), desktop arms (dof=3/4),
single-axis turntables (dof=1): a printed base cradles servo 1 shaft-up; each servo's
horn drives a printed link; each non-terminal link cradles the next servo (the first
stage turns the axis horizontal via the yoke, later stages keep axes parallel -- the
classic pan/tilt/elbow/wrist topology). Geometry comes from ``assembly.arm_parts``;
placement from the deterministic solver; every claim (static fit, full-range swept
collision incl. servo envelopes, cited mount patterns, coaxiality) is the assembly
gate's to issue -- this module never self-certifies anything.

The LLM's only legitimate inputs here are the SPEC fields (dof, link lengths); every
derived number traces to the registry facts in ``assembly.servo``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode
from forge_mind.assembly.servo import MG996R_RANGE_DEG

_DEFAULT_LINK_LENGTHS: dict[int, list[float]] = {
    1: [55.0],
    2: [24.0, 60.0],
    3: [24.0, 70.0, 55.0],
    4: [24.0, 70.0, 60.0, 55.0],
}


@dataclass(frozen=True)
class ServoArmSpec:
    """The family's parametric surface. ``dof`` = number of MG996R servos = number of
    revolute joints. ``link_lengths_mm[i]`` is link i+1's hub-to-far-feature length
    (the first entry is the yoke offset when dof >= 2)."""

    dof: int = 2
    link_lengths_mm: list[float] = field(default_factory=list)
    name: str = "servo_arm"

    def lengths(self) -> list[float]:
        if self.link_lengths_mm:
            if len(self.link_lengths_mm) != self.dof:
                raise ValueError(
                    f"link_lengths_mm needs exactly dof={self.dof} entries, "
                    f"got {len(self.link_lengths_mm)}"
                )
            return [float(x) for x in self.link_lengths_mm]
        return list(_DEFAULT_LINK_LENGTHS[self.dof])


def build_arm_graph(spec: ServoArmSpec) -> AssemblyGraph:
    """The serial chain as an AssemblyGraph: base -> servo_1 -> link_1 -> servo_2 ->
    ... -> link_dof. FIXED joints bolt servos into cradles (mount pattern gated);
    REVOLUTE joints ride each horn at the registry's range."""
    if not 1 <= spec.dof <= 4:
        raise ValueError(f"dof must be 1..4, got {spec.dof}")
    lengths = spec.lengths()

    nodes: list[PartNode] = [
        PartNode(id="base", archetype="servo_base", params={}),
    ]
    joints: list[Joint] = []
    prev_cradle = "base.cradle_mount"
    for i in range(1, spec.dof + 1):
        servo_id = f"servo_{i}"
        link_id = f"link_{i}"
        if i == 1 and spec.dof >= 2:
            end = "cradle_perp"
        elif i < spec.dof:
            end = "cradle_parallel"
        else:
            end = "tool"
        nodes.append(
            PartNode(
                id=servo_id,
                archetype="hardware",
                params={"hardware": "mg996r_servo"},
                printed=False,
            )
        )
        nodes.append(
            PartNode(
                id=link_id,
                archetype="servo_link",
                params={"length_x": lengths[i - 1], "end": end},
            )
        )
        joints.append(
            Joint(
                type=JointType.FIXED,
                a=prev_cradle,
                b=f"{servo_id}.mount",
                mate="align",
            )
        )
        joints.append(
            Joint(
                type=JointType.REVOLUTE,
                a=f"{servo_id}.horn",
                b=f"{link_id}.hub",
                range_deg=MG996R_RANGE_DEG,
                clearance_class="running_fit",
                mate="align",
            )
        )
        prev_cradle = f"{link_id}.cradle_mount"

    return AssemblyGraph(nodes=nodes, joints=joints, root="base")


__all__ = ["ServoArmSpec", "build_arm_graph"]
