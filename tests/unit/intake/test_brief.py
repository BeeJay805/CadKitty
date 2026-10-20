"""DesignBrief assembly, the readiness gate, and brief -> prompt rendering."""

from __future__ import annotations

from forge_mind.intake.agent import HeuristicPlannerBrain
from forge_mind.intake.brief import (
    brief_to_prompt,
    readiness,
    resolved_fastener,
    resolved_thickness,
)
from forge_mind.intake.schema import (
    Answer,
    CriticalDimension,
    DesignBrief,
    DimensionSource,
    Option,
    PlannerState,
    Question,
    QuestionType,
)
from forge_mind.intake.session import DesignSession
from forge_mind.intake.standards import detect_standards


def _ready_brief(session_id: str = "s1") -> DesignBrief:
    return DesignBrief(
        session_id=session_id,
        part_name="p",
        user_goal="g",
        part_type="motor_mount",
        critical_dimensions=[
            CriticalDimension(name="plate_thickness", value=5.0, source=DimensionSource.DEFAULT)
        ],
        ready_for_blueprint=True,
    )


def test_readiness_blocks_without_a_brief() -> None:
    ok, reasons = readiness(DesignSession())
    assert ok is False
    assert reasons


def test_readiness_blocks_when_brief_not_ready() -> None:
    s = DesignSession()
    s.final_design_brief = _ready_brief(s.id).model_copy(update={"ready_for_blueprint": False})
    ok, _ = readiness(s)
    assert ok is False


def test_readiness_blocks_on_unanswered_required_question() -> None:
    s = DesignSession()
    s.final_design_brief = _ready_brief(s.id)
    s.questions = [
        Question(
            id="q_motor_size",
            type=QuestionType.DROPDOWN,
            title="motor?",
            options=[Option(label="N17", value="NEMA-17")],
        )
    ]
    ok, reasons = readiness(s)
    assert ok is False
    assert any("q_motor_size" in r for r in reasons)


def test_readiness_blocks_on_unknown_source_dimension() -> None:
    s = DesignSession()
    brief = _ready_brief(s.id)
    brief.critical_dimensions.append(CriticalDimension(name="width", value=0.0))  # UNKNOWN source
    s.final_design_brief = brief
    ok, _ = readiness(s)
    assert ok is False


def test_readiness_passes_when_complete() -> None:
    s = DesignSession()
    s.final_design_brief = _ready_brief(s.id)
    ok, reasons = readiness(s)
    assert ok is True
    assert reasons == []


def test_resolved_thickness_default_vs_user() -> None:
    assert resolved_thickness(DesignSession()) == (5.0, DimensionSource.DEFAULT)

    user = DesignSession()
    user.answers.append(Answer(question_id="q_plate_thickness", value=8.0))
    assert resolved_thickness(user) == (8.0, DimensionSource.USER)

    accepted_default = DesignSession()
    accepted_default.answers.append(
        Answer(question_id="q_plate_thickness", value=5.0, used_default=True)
    )
    assert resolved_thickness(accepted_default) == (5.0, DimensionSource.DEFAULT)


def test_image_scale_accepted_default_is_not_user_provenance() -> None:
    # audit M7: an accepted-default image scale must carry DEFAULT provenance, not be recorded as a
    # USER-confirmed dimension (mirror resolved_thickness). A real pick stays USER.
    from forge_mind.intake.brief import _critical_dimensions

    state = PlannerState()
    picked = DesignSession()
    picked.answers.append(Answer(question_id="q_image_scale", value=42.0))
    width = next(d for d in _critical_dimensions(picked, state) if d.name == "part_width")
    assert width.value == 42.0 and width.source == DimensionSource.USER

    accepted = DesignSession()
    accepted.answers.append(Answer(question_id="q_image_scale", value=42.0, used_default=True))
    width2 = next(d for d in _critical_dimensions(accepted, state) if d.name == "part_width")
    assert width2.source == DimensionSource.DEFAULT


def test_resolved_fastener_from_standard_and_answer() -> None:
    nema = PlannerState(known_standards=detect_standards("NEMA-17"))
    assert resolved_fastener(DesignSession(), nema) == ("M3", DimensionSource.STANDARD)

    extrusion = PlannerState(known_standards=detect_standards("2020 extrusion"))
    assert resolved_fastener(DesignSession(), extrusion) == ("M5", DimensionSource.STANDARD)

    picked = DesignSession()
    picked.answers.append(Answer(question_id="q_fastener_type", value="quarter_20"))
    value, source = resolved_fastener(picked, extrusion)
    assert value == "1/4-20" and source == DimensionSource.USER


def test_brief_to_prompt_is_precise_and_marks_defaults() -> None:
    outcome = HeuristicPlannerBrain().analyze(
        DesignSession(
            original_text_prompt="A NEMA-17 motor mount for a 2020 aluminum extrusion rail."
        )
    )
    assert outcome.brief is not None
    prompt = brief_to_prompt(outcome.brief)
    assert "motor mount" in prompt
    assert "NEMA-17" in prompt
    assert "(default)" in prompt  # the plate thickness default is surfaced, not hidden
    assert "Constraints:" in prompt


def test_brief_to_prompt_dedups_the_part_prefix() -> None:
    # A re-rendered brief can already carry the "A <part>:" prefix; rendering must not double it
    # into "A bracket: A bracket: ...".
    brief = DesignBrief(
        session_id="s1",
        part_name="p",
        user_goal="A bracket: hold a sensor",
        part_type="bracket",
        ready_for_blueprint=True,
    )
    prompt = brief_to_prompt(brief)
    assert "A bracket: A bracket:" not in prompt
    assert prompt.startswith("A bracket: hold a sensor")


def test_brief_to_prompt_generic_uses_the_raw_goal() -> None:
    # part_type 'generic' must not render the ugly "A generic:" phrasing -- state the goal plainly.
    brief = DesignBrief(
        session_id="s1",
        part_name="p",
        user_goal="a 1 m cube",
        part_type="generic",
        ready_for_blueprint=True,
    )
    prompt = brief_to_prompt(brief)
    assert "A generic:" not in prompt
    assert prompt.startswith("a 1 m cube")
