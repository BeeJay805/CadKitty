"""Support Copilot KB v0 (ADR-0048, P5): a deterministic grounded KB + a read-only QA seam that
REFUSES safety/load questions, labels ungrounded questions 'unknown' (never fabricates), and lets an
injected model phrase ONLY from the retrieved grounding. No autonomous chatbot."""

from __future__ import annotations

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import build_invention_package
from forge_mind.invention.mission import EventType, MissionLog
from forge_mind.invention.package import InventionPackage
from forge_mind.invention.support import (
    KBSnippet,
    answer_question,
    build_support_kb,
    retrieve,
)
from forge_mind.kit import build_kit
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _package(features: list[HardwareFeature] | None = None) -> InventionPackage:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=features or [],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(1),
        problem_statement="route two cables",
    )


def _hw(category: str, key: str) -> HardwareFeature:
    return HardwareFeature(category=category, key=key, nominal_mm=0.0, fit=None, where="t")


class _StubAnswerer:
    def __init__(self) -> None:
        self.seen: list[KBSnippet] | None = None

    def answer(self, query: str, snippets: list[KBSnippet]) -> str:
        self.seen = snippets
        return "ASSEMBLED ANSWER"


def test_kb_assembles_grounded_sections() -> None:
    kb = build_support_kb(_package())
    assert kb.product_purpose == "route two cables"
    assert any(p.kind == "printed" for p in kb.parts)
    assert kb.assembly_steps  # grounded in the kit instructions
    assert any("not physically" in limit.lower() for limit in kb.known_limitations)
    assert any("feasibility" in limit.lower() for limit in kb.known_limitations)
    assert "NOT PHYSICALLY VALIDATED" in kb.validation_status.banner
    assert kb.troubleshooting
    assert any("not rely" in w.lower() or "not a rated" in w.lower() for w in kb.safety_warnings)


def test_kb_with_a_fastener_lists_a_bom_and_substitute_note() -> None:
    kb = build_support_kb(_package([_hw("clearance_hole", "M3")]))
    assert kb.bom
    assert any("M3" in item.component for item in kb.bom)
    assert any("M3" in s for s in kb.substitute_constraints)


def test_empty_bom_substitute_note_is_honest() -> None:
    kb = build_support_kb(_package())
    assert kb.substitute_constraints == ["No purchased parts -- nothing to substitute."]


def test_retrieve_matches_sections_or_returns_empty() -> None:
    kb = build_support_kb(_package())
    assert any(s.section == "assembly" for s in retrieve(kb, "how do I assemble it?"))
    assert any(s.section == "parts" for s in retrieve(kb, "what part is this"))
    assert any(s.section == "troubleshooting" for s in retrieve(kb, "it does not fit, how to fix"))
    assert retrieve(kb, "what is the meaning of life") == []  # ungrounded -> nothing


def test_answer_refuses_safety_and_load_questions() -> None:
    kb = build_support_kb(_package())
    answer = answer_question(kb, "can this safely hold 5 kg?")
    assert answer.status == "refused"
    assert answer.answer is not None and "not" in answer.answer.lower()
    assert answer.citations == ["safety_warnings"]


def test_answer_is_unknown_when_ungrounded() -> None:
    """An ungrounded question is labelled unknown -- never fabricated."""
    kb = build_support_kb(_package())
    answer = answer_question(kb, "what is the weather today")
    assert answer.status == "unknown"
    assert answer.answer is None
    assert answer.citations == []


def test_answer_grounded_without_a_model_returns_the_snippets() -> None:
    kb = build_support_kb(_package())
    answer = answer_question(kb, "how do I assemble it")
    assert answer.status == "answered"
    assert "assembly" in answer.citations
    assert answer.answer is not None and "[assembly]" in answer.answer


def test_answer_uses_the_injected_model_on_grounding_only() -> None:
    kb = build_support_kb(_package())
    stub = _StubAnswerer()
    answer = answer_question(kb, "how do I assemble it", answerer=stub)
    assert answer.status == "answered"
    assert answer.answer == "ASSEMBLED ANSWER"
    # the model saw ONLY the retrieved grounding snippets (no raw KB, no free rein)
    assert stub.seen is not None and stub.seen[0].section == "assembly"


def test_revision_history_comes_from_the_mission_timeline() -> None:
    log = MissionLog("m1", "route two cables")
    log.append(EventType.REVISION_CREATED, summary="moved the jaw offset", stage="cad")
    kb = build_support_kb(_package(), events=log.events())
    assert any("moved the jaw offset" in r for r in kb.revision_history)
