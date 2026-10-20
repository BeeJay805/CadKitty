"""Planner depth: classify shape primitives, fold capture labels/speech measurements into dimensions,
parse measurements server-side (unit-aware), and suppress questions for anything already stated in ANY
capture channel. All deterministic / no-LLM."""

from __future__ import annotations

from forge_mind.intake.agent import HeuristicPlannerBrain, classify_part_type
from forge_mind.intake.brief import extract_dimensions, parse_measurements
from forge_mind.intake.schema import Asset, CaptureContext, CriticalDimension, DimensionSource
from forge_mind.intake.session import DesignSession


def _dim(dims: list[CriticalDimension], name: str) -> CriticalDimension | None:
    return next((d for d in dims if d.name == name), None)


# --- Gap 1: classify shape primitives -------------------------------------------------------------


def test_classify_shape_primitives() -> None:
    assert classify_part_type("a 1 meter cube", []) == "cube"
    assert classify_part_type("an 80 mm plate", []) == "plate"
    assert classify_part_type("a 20 mm diameter cylinder", []) == "cylinder"
    assert classify_part_type("a tube", []) == "tube"
    assert classify_part_type("a washer", []) == "ring"
    # Functional keywords still win over a primitive word in the same request.
    assert classify_part_type("an adapter plate", []) == "adapter_plate"
    assert classify_part_type("a bracket plate to hold a sensor", []) == "bracket"
    # Truly unknown still falls through to generic.
    assert classify_part_type("a thingamajig", []) == "generic"


def test_primitive_with_a_stated_size_generates_with_no_questions() -> None:
    outcome = HeuristicPlannerBrain().analyze(DesignSession(original_text_prompt="a 1 meter cube"))
    assert outcome.ready is True
    assert outcome.questions == []
    assert outcome.brief is not None
    assert outcome.brief.part_type == "cube"
    overall = _dim(outcome.brief.critical_dimensions, "overall_size")
    assert overall is not None and overall.value == 1000.0  # 1 meter -> 1000 mm


# --- Gap 3: unit-aware server-side measurement parsing --------------------------------------------


def test_parse_measurements_is_unit_aware() -> None:
    assert parse_measurements("1 meter") == [1000.0]
    assert parse_measurements("5 cm") == [50.0]
    assert parse_measurements("30 mm") == [30.0]
    assert parse_measurements("2 inches") == [50.8]
    assert parse_measurements("a plain cube") == []
    # A designation like M3 or NEMA-17 is NOT a measurement (no unit) -> not parsed.
    assert parse_measurements("a NEMA-17 mount with M3 screws") == []


def test_extract_dimensions_binds_feature_words() -> None:
    assert extract_dimensions("30 mm center bore") == [("bore_d", 30.0)]
    assert extract_dimensions("make it 40 mm tall") == [("height", 40.0)]
    assert extract_dimensions("a 1 meter cube") == [(None, 1000.0)]  # unbound -> overall size


# --- Gap 2: fold capture labels / speech into critical dimensions ---------------------------------


def test_label_measurement_becomes_a_feature_dimension() -> None:
    session = DesignSession(
        original_text_prompt="a plate",
        capture=CaptureContext(labels=["30 mm center bore"]),
    )
    outcome = HeuristicPlannerBrain().analyze(session)
    assert outcome.brief is not None
    bore = _dim(outcome.brief.critical_dimensions, "bore_d")
    # A label is on the drawing -> IMAGE provenance, not invented.
    assert bore is not None and bore.value == 30.0 and bore.source == DimensionSource.IMAGE


def test_spoken_measurement_becomes_a_dimension() -> None:
    session = DesignSession(
        original_text_prompt="a plate",
        capture=CaptureContext(speech_transcript="make it 40 mm tall"),
    )
    outcome = HeuristicPlannerBrain().analyze(session)
    assert outcome.brief is not None
    height = _dim(outcome.brief.critical_dimensions, "height")
    assert height is not None and height.value == 40.0 and height.source == DimensionSource.USER


# --- Gap 4: suppress questions for anything already stated in ANY channel -------------------------


def test_size_in_a_label_suppresses_the_image_scale_question() -> None:
    session = DesignSession(
        original_text_prompt="a bracket",
        capture=CaptureContext(labels=["60 mm wide"]),
    )
    session.uploaded_assets.append(Asset(id="img1", filename="part.png"))  # visual input
    outcome = HeuristicPlannerBrain().analyze(session)
    ids = {q.id for q in outcome.questions}
    # The width is already given (in the label), so we do NOT ask for a scale measurement again.
    assert "q_image_scale" not in ids


def test_screw_in_speech_suppresses_the_fastener_question() -> None:
    session = DesignSession(
        original_text_prompt="a bracket",
        capture=CaptureContext(speech_transcript="use M4 screws"),
    )
    outcome = HeuristicPlannerBrain().analyze(session)
    ids = {q.id for q in outcome.questions}
    assert "q_fastener_type" not in ids
