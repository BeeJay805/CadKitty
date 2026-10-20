"""End-to-end proof that the A4 hardware-conformance gate actually RUNS and BITES.

The trifecta this guards: the composable interpreter must CARRY a placed hardware feature on the
PartProgram (``hardware_features``), and ``validate()`` must CHECK it (``check_conformance_all``).
Before this was wired, ``check_conformance`` was dead code outside its own module, so a bearing
seat validated with ZERO dimensional verification -- a silent false-PASS. These tests fail if any
link of that chain regresses.

Both compile a solid, so they are @slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.types import Vec3
from forge_mind.validator.report import CheckStatus
from forge_mind.validator.run import validate

pytestmark = pytest.mark.slow

_ENVELOPE = Vec3(x=300.0, y=200.0, z=120.0)  # generous; the interpreter derives the tight plate
_MIN_WALL = 4.0


def _bearing_plate() -> ComposablePlateSpec:
    """A flat plate hosting a press-fit 608 bearing seat (a real, validating composed part)."""
    return ComposablePlateSpec(
        name="bearing_seat_proof",
        envelope_mm=_ENVELOPE,
        material="PLA",
        min_wall_mm=_MIN_WALL,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            )
        ],
    )


def test_bearing_seat_conformance_passes_end_to_end(tmp_path: Path) -> None:
    program = interpret_composable_plate(_bearing_plate())
    # (a) CARRY: the placed bearing seat rode through onto the PartProgram.
    assert any(f.category == "bearing_seat" and f.key == "608" for f in program.hardware_features)

    model = compile_program(program, tmp_path / "brg")
    report = validate(model, program, material="PLA")

    # (c) VERIFY: validate() actually emitted the conformance check, and it PASSES (the built
    # dimension matches the registry, by construction).
    conf = [c for c in report.checks if c.id == "conformance.bearing_seat.608"]
    assert conf, "conformance check did not run end-to-end (the gate is not wired)"
    assert conf[0].status == CheckStatus.PASS
    assert report.passed  # a real bearing-seat plate validates clean overall


def test_corrupted_bearing_nominal_fails_end_to_end(tmp_path: Path) -> None:
    # The geometry is fine (compiled from the good program); we corrupt only the DECLARED built
    # dimension (as if the placement helper had drifted from the registry). The gate MUST bite:
    # if this does not FAIL, the conformance check is not really gating (a false-PASS).
    program = interpret_composable_plate(_bearing_plate())
    model = compile_program(program, tmp_path / "brg")  # good geometry

    good = program.hardware_features[0]
    bad_feature = good.model_copy(update={"nominal_mm": good.nominal_mm + 5.0})
    corrupted = program.model_copy(update={"hardware_features": [bad_feature]})

    report = validate(model, corrupted, material="PLA")
    conf = [c for c in report.checks if c.id.startswith("conformance.bearing_seat")]
    assert conf and conf[0].status == CheckStatus.FAIL
    assert not report.passed  # a non-conformant declared dimension makes the whole report non-PASS
