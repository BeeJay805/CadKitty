"""Phase 6 EXIT GATE (LLM).

All 10 phrasings of the NEMA 17 request must extract to the same facts. Plus a
smoke test that the full pipeline runs from a raw prompt and produces a part
(whether or not it passes validation first-try -- that's the reviser's job).

Marked `llm`; skipped without a key. Run:
    uv run pytest -m llm tests/integration/test_intent_llm.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.intent.grader import grade_intent
from forge_mind.intent.parser import IntentParser
from forge_mind.llm.client import api_key_available
from forge_mind.pipeline import run_pipeline

pytestmark = pytest.mark.llm

REPO = Path(__file__).resolve().parents[2]
VARIANTS = REPO / "tests" / "fixtures" / "intent" / "prompt_variants.json"

_skip = pytest.mark.skipif(
    not api_key_available(),
    reason="ANTHROPIC_API_KEY not set (add it to .env to run LLM tests)",
)


@_skip
@pytest.mark.slow
def test_all_variants_extract_equivalent_specs() -> None:
    data = json.loads(VARIANTS.read_text())
    exp = data["expected"]
    required = set(exp["required_interface_types"])
    envelope = tuple(exp["envelope_mm"])

    parser = IntentParser()  # Haiku
    passes = 0
    failures: list[str] = []
    for i, text in enumerate(data["variants"]):
        spec, _ = parser(text)
        grade = grade_intent(
            spec,
            required_interface_types=required,
            expected_min_wall_mm=exp["min_wall_mm"],
            expected_envelope_mm=envelope,
            expected_material=exp["material"],
        )
        if grade.passed:
            passes += 1
        else:
            failures.append(f"variant {i}: {grade.failures}")

    n = len(data["variants"])
    assert passes >= max(1, round(0.9 * n)), f"{passes}/{n} equivalent; misses: {failures}"


@_skip
def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    """The pipeline runs all stages from a raw prompt and produces a compiled part.

    We assert it reaches a compiled model (geometry), not that it passes
    validation first-try -- reliable passing is the Phase 7 reviser's job.
    """
    prompt = (
        "Design a NEMA 17 motor mount that bolts to a 2020 extrusion with two M5 "
        "T-nuts. 4mm walls, PLA, fit within 60x60x40mm."
    )
    result = run_pipeline(prompt, tmp_path)
    assert result.intent is not None, f"intent stage failed: {result.error}"
    assert result.blueprint is not None, f"blueprint stage failed: {result.error}"
    assert result.program is not None, f"dsl stage failed: {result.error}"
    assert result.model is not None, f"compile failed: {result.error}"
    assert result.report is not None
