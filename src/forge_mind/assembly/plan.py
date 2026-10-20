"""FunctionalPlan — the semantic layer above the AssemblyGraph (ADR-0010, Path 4).

Before any geometry, the LLM states WHAT it is building functionally: the parts and
their roles, and the functional relationships between them (what rotates about what,
what slides along what, what is fixed to what). This is generic -- there is no closed
"mechanism" menu; the family (hinge / slide / ...) emerges from the relations.

Two checks make the plan load-bearing rather than decorative:
  * ``plan_issues``     -- the plan is internally coherent (relations reference real
    parts; at least one moving relationship exists).
  * ``plan_graph_issues`` -- the GEOMETRY realizes the stated function: a
    ``rotates_about`` relation requires a revolute joint, ``slides_along`` a prismatic
    one. A part that claims to move but whose graph has no matching DOF is rejected --
    the structure cannot contradict the function.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.assembly.joints import AssemblyGraph, JointType

RelationType = Literal["rotates_about", "slides_along", "fixed_to", "meshes_with"]


class PlanPart(BaseModel):
    """A part named by its FUNCTION, not its shape."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str = Field(
        ..., description="Functional role, e.g. 'fixed_leaf', 'pivot_pin', 'rail', 'slider'."
    )


class PlanRelation(BaseModel):
    """A functional relationship between two parts (the mechanism's behaviour)."""

    model_config = ConfigDict(extra="forbid")

    type: RelationType
    moving: str = Field(..., description="The part that moves / is constrained.")
    reference: str = Field(..., description="The part it moves relative to / is fixed to.")


class FunctionalPlan(BaseModel):
    """The LLM's semantic decomposition, emitted before the geometry."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    parts: list[PlanPart] = Field(..., min_length=1)
    relations: list[PlanRelation] = Field(..., min_length=1)


_MOVING = {"rotates_about", "slides_along", "meshes_with"}
_RELATION_JOINT = {
    "rotates_about": JointType.REVOLUTE,
    "slides_along": JointType.PRISMATIC,
    "meshes_with": JointType.GEAR_MESH,
}


def plan_issues(plan: FunctionalPlan) -> list[str]:
    """Internal coherence problems with the plan (empty list = coherent)."""
    ids = {p.id for p in plan.parts}
    issues: list[str] = []
    for r in plan.relations:
        if r.moving not in ids:
            issues.append(f"relation references unknown part {r.moving!r}")
        if r.reference not in ids:
            issues.append(f"relation references unknown part {r.reference!r}")
        if r.moving == r.reference:
            issues.append(f"relation relates part {r.moving!r} to itself")
    if not any(r.type in _MOVING for r in plan.relations):
        issues.append("plan has no moving relationship (a mechanism needs a degree of freedom)")
    return issues


def plan_graph_issues(plan: FunctionalPlan, graph: AssemblyGraph) -> list[str]:
    """Ways the GEOMETRY fails to realize the stated function (empty list = consistent)."""
    joint_types = {j.type for j in graph.joints}
    issues: list[str] = []
    for r in plan.relations:
        if r.type in _RELATION_JOINT and _RELATION_JOINT[r.type] not in joint_types:
            issues.append(
                f"plan says {r.moving!r} {r.type} {r.reference!r} but the assembly has no "
                f"{_RELATION_JOINT[r.type].value} joint to realize it"
            )
    return issues


__all__ = [
    "FunctionalPlan",
    "PlanPart",
    "PlanRelation",
    "plan_graph_issues",
    "plan_issues",
]
