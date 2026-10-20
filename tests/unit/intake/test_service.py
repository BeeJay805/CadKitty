"""IntakeService: the guided flow end to end, with generation gated on brief readiness.

Uses the deterministic heuristic brain (no LLM) and an injected fake generation backend, so the whole
flow runs in CI with no API key and no CAD stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.intake.schema import Answer, AssetType, Sketch, SketchPoint, SketchStroke
from forge_mind.intake.service import (
    GenerationBlockedError,
    IntakeService,
    PlannerNotReadyError,
)


class _FakeGen:
    """Captures the prompt + dir the service hands to the pipeline."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, prompt: str, output_dir: Path) -> object:
        self.calls.append((prompt, output_dir))
        return {"ok": True, "prompt": prompt}


def _service(tmp_path: Path, gen: _FakeGen | None = None) -> IntakeService:
    return IntakeService(asset_dir=tmp_path, generate_fn=gen or _FakeGen())


def test_create_and_store_text_input(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    assert svc.get_session(session.id) is not None
    svc.add_input(session.id, text="a bracket for a sensor")
    assert svc.get_session(session.id).original_text_prompt == "a bracket for a sensor"


def test_store_image_asset_metadata(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    asset = svc.add_asset(
        session.id,
        filename="drawing.png",
        data=b"\x89PNG\r\n\x1a\nfake",
        asset_type=AssetType.TECHNICAL_DRAWING,
        content_type="image/png",
    )
    assert asset.scale_confirmed is False
    assert asset.size_bytes > 0
    assert len(svc.get_session(session.id).uploaded_assets) == 1


def test_store_sketch_json(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    sketch = Sketch(
        canvas_width=800,
        canvas_height=600,
        strokes=[SketchStroke(id="s1", points=[SketchPoint(x=1, y=2), SketchPoint(x=3, y=4)])],
    )
    sk = svc.add_sketch(session.id, sketch=sketch)
    assert sk.sketch.strokes[0].points[1].y == 4
    assert len(svc.get_session(session.id).sketch_assets) == 1


def test_answering_questions_records_answers(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    svc.add_input(session.id, text="make a motor mount")
    svc.run_planner(session.id)
    svc.answer_questions(
        session.id,
        [
            Answer(question_id="q_motor_size", value="NEMA-17"),
            Answer(question_id="q_mounting_surface", value="2020_extrusion"),
            Answer(question_id="q_fastener_type", value="recommended", used_default=True),
        ],
    )
    assert len(svc.get_session(session.id).answers) == 3


def test_generation_blocked_when_required_answers_missing(tmp_path: Path) -> None:
    gen = _FakeGen()
    svc = _service(tmp_path, gen)
    session = svc.create_session()
    svc.add_input(session.id, text="make a motor mount")  # missing motor/surface/fastener
    outcome = svc.run_planner(session.id)
    assert outcome.ready is False
    with pytest.raises(GenerationBlockedError):
        svc.start_generation(session.id)
    assert gen.calls == []  # the pipeline was never invoked


def test_finalize_raises_until_ready_then_returns_brief(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    svc.add_input(session.id, text="make a motor mount")
    with pytest.raises(PlannerNotReadyError):
        svc.finalize(session.id)

    # Answer the fit-critical questions, then it finalizes.
    svc.answer_questions(
        session.id,
        [
            Answer(question_id="q_motor_size", value="NEMA-17"),
            Answer(question_id="q_mounting_surface", value="2020_extrusion"),
            Answer(question_id="q_fastener_type", value="recommended", used_default=True),
        ],
    )
    brief = svc.finalize(session.id)
    assert brief.ready_for_blueprint is True


def test_generation_allowed_when_ready(tmp_path: Path) -> None:
    gen = _FakeGen()
    svc = _service(tmp_path, gen)
    session = svc.create_session()
    svc.add_input(session.id, text="A NEMA-17 motor mount for a 2020 aluminum extrusion rail.")
    outcome = svc.run_planner(session.id)
    assert outcome.ready is True

    result = svc.start_generation(session.id)
    assert isinstance(result, dict) and result["ok"] is True
    assert len(gen.calls) == 1
    prompt, _ = gen.calls[0]
    assert "NEMA-17" in prompt  # the synthesized, standards-resolved prompt
    fresh = svc.get_session(session.id)
    assert fresh.linked_generation_id is not None
    assert fresh.status.value == "complete"


def test_image_without_scale_blocks_until_a_dimension_is_given(tmp_path: Path) -> None:
    gen = _FakeGen()
    svc = _service(tmp_path, gen)
    session = svc.create_session()
    svc.add_input(session.id, text="A NEMA-17 motor mount for a 2020 aluminum extrusion rail.")
    svc.add_asset(session.id, filename="part.png", data=b"\x89PNGfake", content_type="image/png")

    outcome = svc.run_planner(session.id)
    assert outcome.ready is False
    assert "q_image_scale" in {q.id for q in outcome.questions}
    with pytest.raises(GenerationBlockedError):
        svc.start_generation(session.id)

    # Supplying one real dimension confirms the scale and unblocks.
    svc.answer_questions(session.id, [Answer(question_id="q_image_scale", value=60.0)])
    assert svc.get_session(session.id).uploaded_assets[0].scale_confirmed is True
    outcome2 = svc.run_planner(session.id)
    assert outcome2.ready is True
    assert outcome2.brief is not None
    widths = [d for d in outcome2.brief.critical_dimensions if d.name == "part_width"]
    assert widths and widths[0].source.value == "user"
    svc.start_generation(session.id)
    assert len(gen.calls) == 1


def test_asset_file_path_guards_traversal(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    session = svc.create_session()
    asset = svc.add_asset(session.id, filename="d.png", data=b"abc", content_type="image/png")
    assert asset.path is not None
    name = Path(asset.path).name
    assert svc.asset_file_path(session.id, name) is not None
    assert svc.asset_file_path(session.id, "../secret.txt") is None
    assert svc.asset_file_path(session.id, "missing.png") is None
