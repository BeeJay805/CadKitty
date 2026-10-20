"""Smoke test for the live monitor UI (ADR-0112): the app boots and renders its form with NO engine
call. Uses Streamlit's headless AppTest; `produce_part` only runs when Generate is pressed, which this
test never does. Skipped when the optional `streamlit` dep is absent, so the default gate is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

_APP = Path(__file__).resolve().parents[3] / "src" / "forge_mind" / "ui" / "monitor_app.py"


def test_monitor_boots_and_renders_form_without_engine_call() -> None:
    app = AppTest.from_file(str(_APP)).run(timeout=30)
    assert not app.exception  # imports produce_part, builds the page, no engine call
    assert len(app.text_input) >= 1
    assert any(button.label == "Generate" for button in app.button)
    # the engine selector (default vs CAD-Coder local) is present
    assert any("CAD-Coder" in str(opt) for radio in app.radio for opt in radio.options)
