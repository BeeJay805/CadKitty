"""FunctionalPlan checks (no LLM): coherence + plan-realizes-geometry consistency."""

from __future__ import annotations

from forge_mind.assembly.joints import AssemblyGraph, Joint, JointType, PartNode
from forge_mind.assembly.plan import (
    FunctionalPlan,
    PlanPart,
    PlanRelation,
    plan_graph_issues,
    plan_issues,
)


def _plan(rel_type: str = "rotates_about") -> FunctionalPlan:
    return FunctionalPlan(
        summary="x",
        parts=[PlanPart(id="a", role="fixed"), PlanPart(id="b", role="moving")],
        relations=[PlanRelation(type=rel_type, moving="b", reference="a")],
    )


def test_coherent_plan_has_no_issues() -> None:
    assert plan_issues(_plan()) == []


def test_relation_to_unknown_part_is_flagged() -> None:
    plan = FunctionalPlan(
        summary="x",
        parts=[PlanPart(id="a", role="fixed")],
        relations=[PlanRelation(type="slides_along", moving="ghost", reference="a")],
    )
    assert any("unknown part" in i for i in plan_issues(plan))


def test_plan_with_no_moving_relationship_is_flagged() -> None:
    plan = FunctionalPlan(
        summary="x",
        parts=[PlanPart(id="a", role="fixed"), PlanPart(id="b", role="bracket")],
        relations=[PlanRelation(type="fixed_to", moving="b", reference="a")],
    )
    assert any("degree of freedom" in i for i in plan_issues(plan))


def _graph(joint_type: JointType) -> AssemblyGraph:
    return AssemblyGraph(
        nodes=[PartNode(id="a", archetype="rod", params={"length_x": 10.0, "radius": 2.0})],
        joints=[Joint(type=joint_type, a="a.bore", b="b.shaft")],
        root="a",
    )


def test_geometry_must_realize_the_stated_function() -> None:
    # A plan that says "rotates_about" but a graph with only a prismatic joint is rejected.
    issues = plan_graph_issues(_plan("rotates_about"), _graph(JointType.PRISMATIC))
    assert any("revolute" in i for i in issues)
    # A matching revolute joint clears it.
    assert plan_graph_issues(_plan("rotates_about"), _graph(JointType.REVOLUTE)) == []


def test_slide_plan_requires_a_prismatic_joint() -> None:
    assert any(
        "prismatic" in i
        for i in plan_graph_issues(_plan("slides_along"), _graph(JointType.REVOLUTE))
    )
    assert plan_graph_issues(_plan("slides_along"), _graph(JointType.PRISMATIC)) == []
