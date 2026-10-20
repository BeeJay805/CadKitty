"""Unit tests for the intent grader (no LLM)."""

from __future__ import annotations

from forge_mind.intent.grader import grade_intent
from forge_mind.intent.schema import IntentConstraints, IntentSpec, MatingInterfaceRequest

REQUIRED = {"nema17_face", "extrusion_2020_slot"}


def _spec(
    *,
    interfaces: list[str],
    wall: float | None = 4.0,
    envelope: list[float] | None = None,
    material: str | None = "PLA",
) -> IntentSpec:
    return IntentSpec(
        part_family="motor_mount",
        summary="test",
        mating_interfaces=[
            MatingInterfaceRequest(interface_type=t, target="x")
            for t in interfaces  # type: ignore[arg-type]
        ],
        constraints=IntentConstraints(
            min_wall_mm=wall,
            material=material,
            max_envelope_mm=envelope,
        ),
    )


def test_good_intent_passes() -> None:
    spec = _spec(
        interfaces=["nema17_face", "extrusion_2020_slot"],
        envelope=[60.0, 60.0, 40.0],
    )
    grade = grade_intent(
        spec,
        required_interface_types=REQUIRED,
        expected_min_wall_mm=4.0,
        expected_envelope_mm=(60.0, 60.0, 40.0),
        expected_material="PLA",
    )
    assert grade.passed, grade.failures


def test_missing_interface_fails() -> None:
    spec = _spec(interfaces=["nema17_face"], envelope=[60.0, 60.0, 40.0])
    grade = grade_intent(spec, required_interface_types=REQUIRED)
    assert not grade.passed
    assert any("missing interfaces" in f for f in grade.failures)


def test_wrong_wall_fails() -> None:
    spec = _spec(interfaces=list(REQUIRED), wall=2.0, envelope=[60.0, 60.0, 40.0])
    grade = grade_intent(spec, required_interface_types=REQUIRED, expected_min_wall_mm=4.0)
    assert not grade.passed
    assert any("min_wall_mm" in f for f in grade.failures)


def test_envelope_orientation_independent() -> None:
    # Same dims in a different axis order still passes.
    spec = _spec(interfaces=list(REQUIRED), envelope=[40.0, 60.0, 60.0])
    grade = grade_intent(
        spec,
        required_interface_types=REQUIRED,
        expected_envelope_mm=(60.0, 60.0, 40.0),
    )
    assert grade.passed, grade.failures


def test_missing_envelope_fails_when_expected() -> None:
    spec = _spec(interfaces=list(REQUIRED), envelope=None)
    grade = grade_intent(
        spec,
        required_interface_types=REQUIRED,
        expected_envelope_mm=(60.0, 60.0, 40.0),
    )
    assert not grade.passed
    assert any("envelope" in f for f in grade.failures)
