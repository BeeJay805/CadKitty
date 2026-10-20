"""Typed ports + joints — the coordinate-free vocabulary the LLM will eventually emit.

A ``Port`` is a connectable frame (origin + axis + an in-plane normal) declared by
a part's archetype in the part's LOCAL coordinates. A ``Joint`` connects two ports
by reference (``"<part_id>.<port>"``) with a categorical type and clearance class.

Crucially, NOTHING here carries a world coordinate or a placement transform: the
graph says only *which ports mate and how*. The solver derives every metric.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.types import Vec3


class Port(BaseModel):
    """A connectable frame on a part, in the part's local coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    origin: Vec3 = Field(..., description="Point on the port axis (local mm).")
    axis: Vec3 = Field(..., description="Port axis direction (e.g. a pivot/insert axis).")
    normal: Vec3 = Field(..., description="An in-plane reference direction, perpendicular to axis.")
    # Wave A (row 61 research): without a diameter the fit arithmetic cannot run until
    # parts are BUILT (~90 min too late). Optional so every existing graph stays valid.
    diameter_mm: float | None = Field(
        default=None,
        description="The mating feature's diameter at this port (bore ID or pin OD, mm). "
        "Declare it on BOTH mated ports so the fit is checkable at PLAN time.",
    )


class JointType(StrEnum):
    """The joint vocabulary the spike needs (extend only when a benchmark demands)."""

    COAXIAL_INSERT = (
        "coaxial_insert"  # slide part b's axis onto part a's axis (e.g. a pin into a bore)
    )
    REVOLUTE = "revolute"  # 1 rotational DOF about the shared axis (a real pivot)
    PRISMATIC = "prismatic"  # 1 translational DOF along the shared axis (a linear slide)
    FIXED = "fixed"  # 0 DOF: parts rigidly mated (a welded/seized joint)
    GEAR_MESH = "gear_mesh"  # coupled rotation: a turns, b counter-rotates by the gear ratio


class Joint(BaseModel):
    """A typed relationship between two ports. No coordinates: the solver resolves them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = Field(
        default=None,
        description="Stable joint identity (ADR-0130). When a MechArchitecture is in "
        "force, the plan joint REALIZING an architecture joint carries that "
        "MechJoint.name verbatim; otherwise null.",
    )
    type: JointType
    a: str = Field(..., description="Fixed/reference port, '<part_id>.<port>'.")
    b: str = Field(..., description="Mated/moving port, '<part_id>.<port>'.")
    range_deg: tuple[float, float] | None = Field(
        default=None, description="Rotational range for a revolute joint (e.g. (0, 180))."
    )
    range_mm: tuple[float, float] | None = Field(
        default=None, description="Translational range for a prismatic joint (e.g. (0, 20))."
    )
    clearance_class: str = Field(
        default="running_fit", description="Categorical fit; resolves to a metric in code."
    )
    mate: Literal["oppose", "align"] = Field(
        default="oppose",
        description="How the ports meet: 'oppose' (anti-parallel -- a pin into a bore, or hinge "
        "leaves) or 'align' (parallel -- stacked links in a serial chain).",
    )
    # --- GEAR_MESH fields (all default None so every non-gear joint is byte-identical) ---
    module: float | None = Field(
        default=None, description="GEAR_MESH: shared gear module (mm pitch dia per tooth)."
    )
    teeth_a: int | None = Field(default=None, description="GEAR_MESH: tooth count of gear a.")
    teeth_b: int | None = Field(default=None, description="GEAR_MESH: tooth count of gear b.")
    clock_a_deg: float = Field(
        default=0.0, description="GEAR_MESH: tooth phase of gear a (deg); a is the reference."
    )
    clock_b_deg: float | None = Field(
        default=None,
        description="GEAR_MESH: tooth phase of gear b (deg). None = the meshing clock for b.",
    )
    internal: bool = Field(
        default=False,
        description="GEAR_MESH: True when one gear is an INTERNAL (ring) gear -- the mesh "
        "centre distance is then the pitch-radius DIFFERENCE |za-zb|*m/2, not the sum.",
    )

    def a_part(self) -> str:
        return self.a.split(".", 1)[0]

    def a_port(self) -> str:
        return self.a.split(".", 1)[1]

    def b_part(self) -> str:
        return self.b.split(".", 1)[0]

    def b_port(self) -> str:
        return self.b.split(".", 1)[1]


class PartNode(BaseModel):
    """A node = one parametric archetype instance. The LLM picks archetype + categorical params.

    ``printed`` defaults True (a normal printed archetype). A non-printed HARDWARE node
    (ADR-0012: a bought motor/bearing) sets ``printed=False`` and uses the ``hardware``
    archetype: it contributes named frames + facts but no geometry. The default keeps
    every existing all-printed graph byte-identical (the field is simply absent).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    archetype: str
    params: dict[str, Any] = Field(default_factory=dict)
    printed: bool = Field(
        default=True, description="False = a non-printed hardware node (no geometry)."
    )


class AssemblyGraph(BaseModel):
    """The coordinate-free IR: parts as nodes, typed joints as edges."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[PartNode] = Field(..., min_length=1)
    joints: list[Joint] = Field(default_factory=list)
    root: str | None = Field(
        default=None, description="Part placed at world identity (default: first joint's a)."
    )


__all__ = ["AssemblyGraph", "Joint", "JointType", "PartNode", "Port"]
