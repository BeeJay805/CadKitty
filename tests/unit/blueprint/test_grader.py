"""Unit tests for blueprint schemas + semantic grader (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.blueprint.grader import grade_blueprint
from forge_mind.blueprint.schema import Blueprint
from forge_mind.intent.schema import IntentSpec

REPO = Path(__file__).resolve().parents[3]
INTENT_FIXTURE = REPO / "tests" / "fixtures" / "intent" / "nema17_mount.intent.json"
BLUEPRINT_FIXTURE = REPO / "tests" / "fixtures" / "blueprints" / "nema17_mount.blueprint.json"

REQUIRED_TYPES = {"nema17_face", "extrusion_2020_slot"}
ENVELOPE_CAP = (60.0, 60.0, 40.0)
EXPECTED_WALL = 4.0


def _ref_blueprint() -> Blueprint:
    return Blueprint.model_validate(json.loads(BLUEPRINT_FIXTURE.read_text()))


def test_intent_fixture_loads() -> None:
    intent = IntentSpec.model_validate(json.loads(INTENT_FIXTURE.read_text()))
    assert intent.part_family == "motor_mount"
    assert {m.interface_type.value for m in intent.mating_interfaces} == REQUIRED_TYPES


def test_generator_raises_on_malformed_envelope() -> None:
    # audit M8: a present-but-malformed max_envelope_mm (2 elements -- a reachable LLM-extraction
    # state) must RAISE, not be silently coerced to None ("no cap"), which erased the user's max.
    from forge_mind.blueprint.generator import BlueprintGenerator
    from forge_mind.llm.cost import LLMUsage

    intent = IntentSpec.model_validate(json.loads(INTENT_FIXTURE.read_text()))
    bad = intent.model_copy(
        update={
            "constraints": intent.constraints.model_copy(update={"max_envelope_mm": [10.0, 20.0]})
        }
    )

    class _StubClient:
        def generate_structured(self, **_: object) -> tuple[Blueprint, LLMUsage]:
            return _ref_blueprint(), LLMUsage(model="stub", input_tokens=0, output_tokens=0)

    gen = BlueprintGenerator(client=_StubClient())
    with pytest.raises(ValueError, match="max_envelope_mm"):
        gen(bad)


def test_reference_blueprint_passes_grading() -> None:
    grade = grade_blueprint(
        _ref_blueprint(),
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )
    assert grade.passed, grade.failures


def test_grader_flags_missing_interface() -> None:
    bp = _ref_blueprint()
    bp = bp.model_copy(update={"required_interfaces": [bp.required_interfaces[0]]})
    grade = grade_blueprint(
        bp,
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )
    assert not grade.passed
    assert any("missing required interfaces" in f for f in grade.failures)


def test_grader_flags_oversized_envelope() -> None:
    bp = _ref_blueprint()
    from forge_mind.blueprint.schema import BlueprintVec3

    bp = bp.model_copy(update={"envelope_mm": BlueprintVec3(x=200.0, y=60.0, z=40.0)})
    grade = grade_blueprint(
        bp,
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )
    assert not grade.passed
    assert any("does not fit the cap" in f for f in grade.failures)


def test_grader_flags_wrong_wall() -> None:
    bp = _ref_blueprint().model_copy(update={"min_wall_mm": 2.0})
    grade = grade_blueprint(
        bp,
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )
    assert not grade.passed
    assert any("min_wall_mm" in f for f in grade.failures)


def test_grader_flags_interface_on_unknown_feature() -> None:
    bp = _ref_blueprint()
    bad = bp.required_interfaces[0].model_copy(update={"on_feature": "nonexistent"})
    bp = bp.model_copy(update={"required_interfaces": [bad, bp.required_interfaces[1]]})
    grade = grade_blueprint(
        bp,
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )
    assert not grade.passed
    assert any("unknown feature" in f for f in grade.failures)
