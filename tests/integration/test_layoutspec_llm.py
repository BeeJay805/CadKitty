"""LLM smoke: Blueprint -> LayoutSpecGenerator -> interpret -> valid part.

Proves the LLM-generated LayoutSpec (not hand-authored) deterministically becomes
a part the existing validator passes. Marked `llm`; skipped without a key. Run:

    uv run pytest -m llm tests/integration/test_layoutspec_llm.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forge_mind.blueprint.schema import Blueprint
from forge_mind.layout.composable_spec import BaseArchetype, ComposablePlateSpec
from forge_mind.layout.generator import LayoutSpecGenerator
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.layout.spec import Pattern
from forge_mind.llm.client import api_key_available

pytestmark = pytest.mark.llm

REPO = Path(__file__).resolve().parents[2]
BLUEPRINT = REPO / "tests" / "fixtures" / "blueprints" / "nema17_mount.blueprint.json"


@pytest.mark.skipif(not api_key_available(), reason="no ANTHROPIC_API_KEY")
def test_layoutspec_generator_produces_valid_part() -> None:
    blueprint = Blueprint.model_validate(json.loads(BLUEPRINT.read_text()))

    spec, usage = LayoutSpecGenerator()(blueprint)
    # The NEMA17 mount blueprint must route to the composable plate, l_bracket_bend base
    # (ADR-0014: the plate family is one composable spec; the LLM picks base + interfaces).
    assert isinstance(spec, ComposablePlateSpec)
    assert spec.pattern is Pattern.COMPOSABLE_PLATE
    assert spec.base is BaseArchetype.L_BRACKET_BEND
    assert usage.cost_usd > 0

    result = run_layout_pipeline(spec, Path(tempfile.mkdtemp()))
    # The LLM only chose sizes + constraints; the interpreter placed all geometry.
    assert result.error is None, result.error
    assert result.report is not None
    assert result.passed, f"failed checks: {result.report.failed_check_ids()}"
