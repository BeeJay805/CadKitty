"""The requirements compiler (ADR-0087 S4), golden SSD scenario: every leaf closes through the
honest order (user -> registry -> research -> ask), the SFF hole-coordinate honesty propagates into
a question, thermal touches its ceiling verbatim, checks compile only when their facts resolved."""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.provenance import Tier, parse_stated_dimensions
from forge_mind.invention.compile_requirements import compile_requirements
from forge_mind.invention.decompose import ProposedNode, ProposedTree
from forge_mind.invention.requirements import Fact
from forge_mind.invention.research import (
    Document,
    FakeResearchProvider,
    ProposedFact,
    ResearchProposal,
)
from forge_mind.llm.cost import LLMUsage

_NR200_DOC = Document(
    url="https://example.com/nr200-spec",
    title="Cooler Master NR200 specifications",
    text="The NR200 mini-ITX case: the drive bay depth is 4 inches behind the front panel.",
    fetched_at="2026-07-05",
)


def _ssd_tree() -> ProposedTree:
    n = ProposedNode
    return ProposedTree(
        nodes=[
            n(node_id="r", parent_id=None, statement="Mount one more 2.5in SSD in the NR200"),
            n(
                node_id="fit",
                parent_id="r",
                statement="must fit the available space",
                domain="fit_envelope",
                closure_kind="fact",
                closure_detail="available_height",
            ),
            n(node_id="mnt", parent_id="r", statement="carry the SSD", domain="mount_interface"),
            n(
                node_id="w",
                parent_id="mnt",
                statement="span the SSD width",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_width",
            ),
            n(
                node_id="t",
                parent_id="mnt",
                statement="match the SSD thickness",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_thickness",
            ),
            n(
                node_id="hp",
                parent_id="mnt",
                statement="bolt to the SSD's holes",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="mount_hole_pitch",
            ),
            n(
                node_id="th",
                parent_id="r",
                statement="keep everything working",
                domain="thermal_airflow",
            ),
            n(
                node_id="th1",
                parent_id="th",
                statement="do not block the intake",
                domain="thermal_airflow",
                closure_kind="check",
                closure_detail="keepout_clear",
            ),
            n(
                node_id="cab",
                parent_id="r",
                statement="cables must route",
                domain="cable_routing",
                closure_kind="not_applicable",
                closure_detail="user said cables route on the open side",
            ),
            n(
                node_id="acc",
                parent_id="r",
                statement="screws reachable",
                domain="access_service",
                closure_kind="design_freedom",
                closure_detail="engine orients heads outward",
            ),
        ]
    )


class _Client:
    """Dispatches on output_model: the canned tree for decompose, the canned proposal for research."""

    def __init__(self, tree: ProposedTree, proposal: ResearchProposal) -> None:
        self.tree, self.proposal = tree, proposal

    def generate_structured(
        self, *, output_model: type, model: str, **_: Any
    ) -> tuple[Any, LLMUsage]:
        usage = LLMUsage(model=model, input_tokens=1, output_tokens=1)
        if output_model is ProposedTree:
            return self.tree, usage
        if output_model is ResearchProposal:
            return self.proposal, usage
        raise AssertionError(f"unexpected output_model {output_model}")


def _compile(user_facts: tuple[Fact, ...] = ()) -> Any:
    proposal = ResearchProposal(
        facts=[
            ProposedFact(
                name="available_height",
                value=4.0,
                unit="inches",
                quote="drive bay depth is 4 inches",
                url=_NR200_DOC.url,
            )
        ]
    )
    compiled, _usages = compile_requirements(
        "Mount one more 2.5in SSD in my NR200; printable bracket.",
        client=_Client(_ssd_tree(), proposal),
        model="m",
        user_facts=(
            Fact(name="payload_thickness", value=9.5, unit="mm", source="user"),
            *user_facts,
        ),
        entity="NR200",
        registry_keys={"drive_form": "2.5in_drive"},
        provider=FakeResearchProvider({"nr200": [_NR200_DOC]}),
    )
    return compiled


def test_golden_ssd_closure_routes() -> None:
    c = _compile()
    by = {f.name: f for f in c.ledger.facts}
    # registry: the SSD width from SFF-8201, cited
    assert by["payload_width"].source == "registry" and "SFF-8201" in (
        by["payload_width"].citation or ""
    )
    # user: the stated thickness
    assert by["payload_thickness"].source == "user"
    # researched: 4 inches verified against the fetched quote, normalized to mm downstream
    assert by["available_height"].source == "researched"
    assert by["available_height"].as_mm_or_g == (101.6, "mm")


def test_sff_hole_honesty_propagates_to_a_question() -> None:
    # the 2.5in bottom_hole_pitch_mm is honestly None in the registry -> the compiler must ASK,
    # never invent the pattern (the S2 honesty flowing through S4)
    c = _compile()
    assert any("mounting-hole spacing" in q.question for q in c.open_questions)
    assert any("not sourced" in n for n in c.notes)
    assert c.status == "needs_input"


def test_thermal_ceiling_rides_verbatim_and_check_demotes() -> None:
    c = _compile()
    # touched != validated: the ceiling is on the card even though the thermal check compiled nothing
    assert any("Temperatures and CFM are NOT simulated" in a for a in c.assumptions)
    assert any("keepout_clear" in a and "NOT compiled" in a for a in c.assumptions)
    assert not any(ch.check_id == "keepout_clear" for ch in c.checks)


def test_checks_compile_when_facts_resolve() -> None:
    c = _compile(
        user_facts=(
            Fact(name="fan_frame_size", value=120.0, unit="mm", source="user"),
            Fact(name="airflow_clearance", value=15.0, unit="mm", source="user"),
            Fact(name="mount_hole_pitch", value=76.2, unit="mm", source="user"),
            # the base gather (S7): an active fit_envelope domain always gathers its core
            # facts, so the remaining envelope scalars must be stated for a fully-ready spec
            Fact(name="available_length", value=160.0, unit="mm", source="user"),
            Fact(name="available_width", value=110.0, unit="mm", source="user"),
        )
    )
    (check,) = [ch for ch in c.checks if ch.check_id == "keepout_clear"]
    assert check.params == {"fan_frame_size": 120.0, "airflow_clearance": 15.0}
    assert c.status == "ready"  # every question closed -> the build may proceed


def test_not_applicable_and_design_freedom_ask_nothing() -> None:
    # the over-templating guard: a n/a branch and an engine-owned branch generate NO questions
    c = _compile()
    assert not any("cable" in q.question.lower() for q in c.open_questions)
    assert not any("screw" in q.question.lower() for q in c.open_questions)
    assert "[n/a]" in c.card and "[engine's choice]" in c.card


def test_extra_context_round_trips_honest_tiers() -> None:
    c = _compile()
    dims = parse_stated_dimensions(c.extra_context)
    tiers = {d.line.split(" =")[0].replace("dimension: ", ""): d.tier for d in dims}
    assert tiers["available_height"] is Tier.SOURCED  # researched, never STATED
    assert tiers["payload_thickness"] is Tier.STATED
    assert tiers["payload_width"] is Tier.SOURCED


def test_prompt_carries_constraints_and_heuristics() -> None:
    c = _compile()
    assert "payload_width: 69.85 mm" in c.prompt
    assert "Design guidance (advisory):" in c.prompt
    assert any("upstream of an intake fan" in h for h in c.heuristics)


def test_card_shows_provenance_per_leaf() -> None:
    c = _compile()
    assert "[ok: 69.85 mm, standard]" in c.card
    assert "[ok: 9.5 mm, you said]" in c.card
    assert "[ok: 101.6 mm, researched - verify]" in c.card
    assert "[needs answer]" in c.card  # the hole-pitch ask


# --- S7 anti-quiz fixes: reviewed defaults, relevance-gated base gather, leaf_only vocabulary ---


def _compile_tree(nodes: list[ProposedNode], problem: str, **kw: Any) -> Any:
    from forge_mind.invention.compile_requirements import compile_from_tree
    from forge_mind.invention.decompose import _validate

    decomposed = _validate(ProposedTree(nodes=nodes))
    return compile_from_tree(
        problem,
        decomposed,
        client=_Client(ProposedTree(nodes=[]), ResearchProposal()),
        model="m",
        **kw,
    )


def test_default_policy_closes_as_visible_assumption_not_a_question() -> None:
    n = ProposedNode
    c = _compile_tree(
        [
            n(node_id="r", parent_id=None, statement="route the cable"),
            n(
                node_id="cc",
                parent_id="r",
                statement="leave connector room",
                domain="cable_routing",
                closure_kind="fact",
                closure_detail="connector_clearance",
            ),
        ],
        "route a cable cleanly",
    )
    assert not any(q.fact_name == "connector_clearance" for q in c.open_questions)
    (fact,) = [f for f in c.ledger.facts if f.name == "connector_clearance"]
    assert fact.source == "derived" and fact.value == 25.0
    assert fact.citation == "reviewed playbook default"
    assert any("connector_clearance assumed 25 mm" in a for a in c.assumptions)
    assert "assumed - default" in c.card  # visible provenance, never silent
    # derived is NOT an anchor: it must never enter the stated-facts block (no laundering)
    assert "connector_clearance" not in c.extra_context


def test_base_gather_is_relevance_gated_no_fan_facts_for_a_router() -> None:
    # the run-4 smoking gun: a router bracket was asked fan dimensions because its tree
    # brushed thermal_airflow — the relevance gate makes that impossible
    n = ProposedNode
    c = _compile_tree(
        [
            n(node_id="r", parent_id=None, statement="hold the router under the shelf"),
            n(
                node_id="w",
                parent_id="r",
                statement="span the router",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_width",
            ),
            n(
                node_id="th",
                parent_id="r",
                statement="do not block ventilation",
                domain="thermal_airflow",
                closure_kind="check",
                closure_detail="keepout_clear",
            ),
        ],
        "I need a bracket to hold my router to the underside of a shelf",
    )
    asked = {q.fact_name for q in c.open_questions if q.fact_name}
    assert "fan_frame_size" not in asked and "airflow_clearance" not in asked
    # the un-parameterized thermal check demotes to a visible assumption — the honest outcome
    assert any("keepout_clear" in a and "NOT compiled" in a for a in c.assumptions)


def test_leaf_only_vocabulary_never_auto_gathered_but_asks_when_leaf_named() -> None:
    n = ProposedNode
    base = [
        n(node_id="r", parent_id=None, statement="mount the SSD"),
        n(
            node_id="t",
            parent_id="r",
            statement="thickness",
            domain="mount_interface",
            closure_kind="fact",
            closure_detail="payload_thickness",
        ),
    ]
    c = _compile_tree(base, "mount one more SSD on the desk")
    assert not any(q.fact_name == "attachment_hole_pitch" for q in c.open_questions)
    named = [
        *base,
        n(
            node_id="ap",
            parent_id="r",
            statement="bolt into the desk's existing holes",
            domain="mount_interface",
            closure_kind="fact",
            closure_detail="attachment_hole_pitch",
        ),
    ]
    c2 = _compile_tree(named, "mount one more SSD on the desk")
    assert any(q.fact_name == "attachment_hole_pitch" for q in c2.open_questions)
