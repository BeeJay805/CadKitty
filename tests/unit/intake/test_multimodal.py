"""Multimodal capture (ADR-0025): speech/scale/viewpoint reach the planner, sketch constraints reach
the spec, and the role-separated user message is built from the whole bundle."""

from __future__ import annotations

from pathlib import Path

from forge_mind.intake.agent import HeuristicPlannerBrain, _compose_planner_user
from forge_mind.intake.brief import brief_to_prompt
from forge_mind.intake.schema import (
    AssetType,
    CaptureContext,
    ScaleRef,
    Sketch,
    SketchAsset,
    SketchPoint,
    SketchStroke,
    Viewpoint,
)
from forge_mind.intake.service import IntakeService
from forge_mind.intake.session import DesignSession


def _l_bracket_sketch() -> Sketch:
    return Sketch(
        canvas_width=800,
        canvas_height=600,
        strokes=[
            SketchStroke(
                id="h", type="line", points=[SketchPoint(x=100, y=300), SketchPoint(x=300, y=300)]
            ),
            SketchStroke(
                id="v", type="line", points=[SketchPoint(x=100, y=300), SketchPoint(x=100, y=120)]
            ),
        ],
    )


def test_speech_narration_reaches_brief_and_prompt() -> None:
    cap = CaptureContext(speech_transcript="It clips onto a 12 mm rail and holds a sensor.")
    outcome = HeuristicPlannerBrain().analyze(
        DesignSession(original_text_prompt="a clip", capture=cap)
    )
    assert outcome.brief is not None
    assert outcome.brief.intent_narration == cap.speech_transcript
    assert "clips onto a 12 mm rail" in brief_to_prompt(outcome.brief)


def test_sketch_constraints_reach_the_spec() -> None:
    session = DesignSession(
        original_text_prompt="an L bracket",
        sketch_assets=[SketchAsset(id="sk1", sketch=_l_bracket_sketch())],
    )
    outcome = HeuristicPlannerBrain().analyze(session)
    assert outcome.constraints, "the right-angle L should infer constraints even with no LLM"
    assert outcome.brief is not None
    assert outcome.brief.geometry_constraints, "constraints should render into the build spec"


def test_compose_user_separates_channels_by_role() -> None:
    session = DesignSession(
        original_text_prompt="a phone stand",
        capture=CaptureContext(
            speech_transcript="lean it back 15 degrees",
            viewpoint=Viewpoint.FRONT,
            scale=ScaleRef(mm=60.0, ref_label="base"),
        ),
        sketch_assets=[SketchAsset(id="sk1", sketch=Sketch(canvas_width=400, canvas_height=400))],
    )
    msg = _compose_planner_user(session)
    assert "SPEECH TRANSCRIPT (PRIMARY INTENT" in msg
    assert "lean it back 15 degrees" in msg
    assert "FORM and TOPOLOGY" in msg  # the sketch is tagged form-only
    assert "front view" in msg  # viewpoint hint names the missing axis
    assert "60 mm" in msg  # scale calibration surfaced


def test_service_add_input_capture_confirms_scale(tmp_path: Path) -> None:
    svc = IntakeService(asset_dir=tmp_path)
    sid = svc.create_session().id
    svc.add_asset(
        sid, filename="d.png", data=b"\x89PNGfake", asset_type=AssetType.TECHNICAL_DRAWING
    )
    svc.add_input(sid, capture=CaptureContext(scale=ScaleRef(mm=60.0)))
    session = svc.get_session(sid)
    assert session.capture is not None and session.capture.scale is not None
    # A declared scale is a real in-plane dimension -> the drawing is no longer pixels-only.
    assert all(a.scale_confirmed for a in session.uploaded_assets)
