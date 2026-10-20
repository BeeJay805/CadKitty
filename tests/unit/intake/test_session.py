"""DesignSession helpers + the in-memory and JSON-file stores."""

from __future__ import annotations

from pathlib import Path

from forge_mind.intake.schema import (
    Answer,
    DesignBrief,
    Option,
    Question,
    QuestionType,
    SessionStatus,
)
from forge_mind.intake.session import (
    DesignSession,
    InMemorySessionStore,
    JSONFileSessionStore,
)


def _q(qid: str, required: bool = True) -> Question:
    return Question(
        id=qid,
        type=QuestionType.MULTIPLE_CHOICE,
        title="t",
        required=required,
        options=[Option(label="a", value="a")],
    )


def test_in_memory_store_create_get_save_delete() -> None:
    store = InMemorySessionStore()
    session = store.create()
    assert store.get(session.id) is session
    session.original_text_prompt = "a bracket"
    store.save(session)
    assert store.get(session.id).original_text_prompt == "a bracket"
    assert store.delete(session.id) is True
    assert store.get(session.id) is None
    assert store.delete(session.id) is False


def test_unanswered_required_excludes_optional_and_answered() -> None:
    s = DesignSession()
    s.questions = [_q("q1"), _q("q2"), _q("q3", required=False)]
    s.answers = [Answer(question_id="q1", value="a")]
    unanswered = s.unanswered_required()
    assert [q.id for q in unanswered] == ["q2"]


def test_answer_for_returns_latest() -> None:
    s = DesignSession()
    s.answers = [Answer(question_id="q1", value="a"), Answer(question_id="q1", value="b")]
    assert s.answer_for("q1") is not None
    assert s.answer_for("q1").value == "b"
    assert s.answer_for("missing") is None


def test_json_file_store_round_trips_full_session(tmp_path: Path) -> None:
    store = JSONFileSessionStore(tmp_path)
    session = store.create()
    session.original_text_prompt = "A NEMA-17 motor mount."
    session.status = SessionStatus.READY_FOR_BLUEPRINT
    session.final_design_brief = DesignBrief(
        session_id=session.id,
        part_name="nema17_motor_mount",
        user_goal="mount a motor",
        part_type="motor_mount",
        ready_for_blueprint=True,
    )
    store.save(session)

    loaded = store.get(session.id)
    assert loaded is not None
    assert loaded.original_text_prompt == "A NEMA-17 motor mount."
    assert loaded.status == SessionStatus.READY_FOR_BLUEPRINT
    assert loaded.final_design_brief is not None
    assert loaded.final_design_brief.part_name == "nema17_motor_mount"
    assert store.delete(session.id) is True
    assert store.get(session.id) is None
