"""Smoke test for the Streamlit UI: the app boots and renders its form with NO API call.

Uses Streamlit's headless AppTest harness, so no server and no paid call -- the pipeline only
runs when the Generate button is pressed, which this test never does. Skipped entirely when the
optional ``streamlit`` dependency is absent (the ``ui`` extra), so the default gate is unaffected.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from forge_mind.compiler.compile import compile_program
from forge_mind.kit import build_kit
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.pipeline import PipelineResult
from forge_mind.types import Vec3
from forge_mind.validator.run import validate

_APP = Path(__file__).resolve().parents[3] / "src" / "forge_mind" / "ui" / "streamlit_app.py"


def test_app_boots_and_renders_form_without_api_call() -> None:
    app = AppTest.from_file(str(_APP)).run(timeout=30)
    # Boots clean (imports the pipeline, sets up the page) with no uncaught exception.
    assert not app.exception
    # The request form is present; the button exists but was never pressed -> no paid call ran.
    assert len(app.text_input) >= 1
    assert any(button.label == "Generate" for button in app.button)


def _real_result(tmp_path: Path) -> PipelineResult:
    """A real PipelineResult (real part + report + kit) built OFFLINE -- no API call."""
    spec = ComposablePlateSpec(
        name="ui-bearing-plate",
        envelope_mm=Vec3(x=200.0, y=200.0, z=20.0),
        material="PLA",
        min_wall_mm=4.0,
        base=BaseArchetype.FLAT_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[
                InterfacePlacement(
                    ref=InterfaceRef.CLEARANCE_HOLE,
                    face=Face.PLATE_TOP,
                    component_id="M3",
                    fit="normal",
                )
                for _ in range(4)
            ],
        ],
    )
    program = interpret_composable_plate(spec)
    model = compile_program(program, tmp_path / "ui-part")
    report = validate(model, program, material="PLA")
    kit = build_kit(program, material="PLA", report=report)
    return PipelineResult(
        prompt="a flat plate with a 608 bearing seat and four M3 clearance holes",
        program=program,
        model=model,
        report=report,
        kit=kit,
        gen_path="layoutspec",
    )


def test_app_renders_a_real_result_without_api_call(tmp_path: Path) -> None:
    # Patch the pipeline so pressing Generate returns a real, pre-built result -- $0, no network.
    # Exercises the full display path (verdict + tiers + image + BOM table + steps + download).
    fake = _real_result(tmp_path)
    with patch("forge_mind.pipeline.run_pipeline", return_value=fake):
        app = AppTest.from_file(str(_APP)).run(timeout=60)
        app.text_input[0].set_value(fake.prompt)
        app.button[0].click()
        app.run(timeout=60)

    assert not app.exception  # st.image + st.download_button + st.table all executed cleanly
    text = " ".join(
        str(getattr(el, "value", ""))
        for group in (app.markdown, app.success, app.info, app.caption)
        for el in group
    )
    assert "PASS" in text  # the validated verdict surfaced
    assert "Press the" in text  # a make-it step (the bearing press) -> the kit reached the page
