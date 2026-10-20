"""Planner agent behaviour: classification, smart (not fluffy) questions, standards, refusal,
the image-scale rule, and a mockable brain (no live LLM)."""

from __future__ import annotations

from forge_mind.intake.agent import (
    HeuristicPlannerBrain,
    PlannerAgent,
    PlannerOutcome,
    classify_part_type,
)
from forge_mind.intake.schema import PlannerState
from forge_mind.intake.session import DesignSession
from forge_mind.intake.standards import detect_standards


def _session(text: str | None) -> DesignSession:
    return DesignSession(original_text_prompt=text)


def test_classify_part_type() -> None:
    assert classify_part_type("a motor mount for a stepper", []) == "motor_mount"
    assert classify_part_type("a bracket to hold a sensor", []) == "bracket"
    assert classify_part_type("an enclosure for a pcb", []) == "enclosure"
    assert classify_part_type("a bearing block", []) == "bearing_block"
    assert classify_part_type("a thingamajig", []) == "generic"
    # A NEMA mention forces motor_mount regardless of loose words.
    assert classify_part_type("a holder", detect_standards("NEMA-17")) == "motor_mount"


def test_classify_part_type_resolves_shape_primitives() -> None:
    # Plain shape primitives must get a concrete type instead of falling through to 'generic',
    # so "a 1 m cube" gets the minimal primitive question set (not the generic catch-all).
    assert classify_part_type("a 1 m cube", []) == "cube"
    assert classify_part_type("an 80 mm plate", []) == "plate"
    assert classify_part_type("a 50 mm cylinder", []) == "cylinder"
    assert classify_part_type("a sphere", []) == "sphere"


def test_vague_motor_mount_asks_a_small_fit_critical_set() -> None:
    outcome = PlannerAgent().run(_session("make a motor mount"))
    assert outcome.ready is False
    ids = {q.id for q in outcome.questions}
    # The signature example: motor size, what it attaches to, fasteners -- not twelve questions.
    assert ids == {"q_motor_size", "q_mounting_surface", "q_fastener_type"}
    assert len(outcome.questions) <= 4
    assert all(q.priority == 1 for q in outcome.questions)


def test_fully_specified_request_is_ready_with_defaults_marked() -> None:
    outcome = PlannerAgent().run(
        _session("A NEMA-17 motor mount for a 2020 aluminum extrusion rail.")
    )
    assert outcome.ready is True
    assert outcome.questions == []
    assert outcome.brief is not None
    assert outcome.brief.ready_for_blueprint is True
    assert outcome.brief.part_type == "motor_mount"
    names = {s.name for s in outcome.brief.known_standards}
    assert "NEMA-17" in names and "2020 aluminum extrusion" in names
    # The plate thickness was not asked -- it is a clearly-marked default, never a silent guess.
    thickness = next(d for d in outcome.brief.critical_dimensions if d.name == "plate_thickness")
    assert thickness.source.value == "default"


def test_does_not_ask_for_what_a_standard_provides() -> None:
    outcome = PlannerAgent().run(_session("A NEMA-17 motor mount for a 2020 rail."))
    ids = {q.id for q in outcome.questions}
    # The motor size + surface + rail fastener all come from the standards -> not re-asked.
    assert "q_motor_size" not in ids
    assert "q_mounting_surface" not in ids


def test_contradiction_refuses_instead_of_asking() -> None:
    outcome = PlannerAgent().run(_session("a mount for both a NEMA-17 and a NEMA-23 motor"))
    assert outcome.ready is False
    assert outcome.refusal is not None
    assert "NEMA-17" in outcome.refusal and "NEMA-23" in outcome.refusal
    assert outcome.questions == []


def test_image_input_forces_a_scale_question_even_when_text_is_complete() -> None:
    session = _session("A NEMA-17 motor mount for a 2020 aluminum extrusion rail.")
    # Attach a (metadata-only) image asset -> shape known, scale not.
    from forge_mind.intake.schema import Asset

    session.uploaded_assets.append(Asset(id="img1", filename="part.png"))
    outcome = HeuristicPlannerBrain().analyze(session)
    assert outcome.ready is False
    assert "q_image_scale" in {q.id for q in outcome.questions}


def test_scale_confirmed_input_removes_the_scale_question() -> None:
    from forge_mind.intake.schema import Asset

    session = _session("A NEMA-17 motor mount for a 2020 aluminum extrusion rail.")
    session.uploaded_assets.append(Asset(id="img1", filename="part.png", scale_confirmed=True))
    outcome = HeuristicPlannerBrain().analyze(session)
    # With the text complete and the scale confirmed, nothing critical remains.
    assert outcome.ready is True


class _ScriptedBrain:
    """A mock brain (no LLM) -- proves the PlannerBrain Protocol is swappable for CI."""

    def __init__(self, outcome: PlannerOutcome) -> None:
        self._outcome = outcome

    def analyze(self, session: DesignSession) -> PlannerOutcome:
        return self._outcome


def test_agent_accepts_a_mock_brain() -> None:
    scripted = PlannerOutcome(state=PlannerState(part_type="bracket"), ready=True)
    agent = PlannerAgent(brain=_ScriptedBrain(scripted))
    assert agent.run(_session("anything")) is scripted
