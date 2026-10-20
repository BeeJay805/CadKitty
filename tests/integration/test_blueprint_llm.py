"""Phase 4 EXIT GATE (LLM).

Runs the real BlueprintGenerator against the hand-authored IntentSpec and grades
each output semantically. Target: >= 9/10 trials pass.

Marked `llm` and skipped when no ANTHROPIC_API_KEY is configured, so the fast
test suite + CI stay free and offline. Run explicitly:

    uv run pytest -m llm tests/integration/test_blueprint_llm.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge_mind.blueprint.generator import BlueprintGenerator
from forge_mind.blueprint.grader import grade_blueprint
from forge_mind.intent.schema import IntentSpec
from forge_mind.llm.client import api_key_available

pytestmark = pytest.mark.llm

REPO = Path(__file__).resolve().parents[2]
INTENT_FIXTURE = REPO / "tests" / "fixtures" / "intent" / "nema17_mount.intent.json"

REQUIRED_TYPES = {"nema17_face", "extrusion_2020_slot"}
ENVELOPE_CAP = (60.0, 60.0, 40.0)
EXPECTED_WALL = 4.0

_skip = pytest.mark.skipif(
    not api_key_available(),
    reason="ANTHROPIC_API_KEY not set (add it to .env to run LLM tests)",
)


def _intent() -> IntentSpec:
    return IntentSpec.model_validate(json.loads(INTENT_FIXTURE.read_text()))


def _grade(bp) -> object:  # type: ignore[no-untyped-def]
    return grade_blueprint(
        bp,
        required_interface_types=REQUIRED_TYPES,
        envelope_cap_mm=ENVELOPE_CAP,
        expected_min_wall_mm=EXPECTED_WALL,
    )


@_skip
def test_blueprint_generation_smoke() -> None:
    """One real call produces a gradeable Blueprint."""
    gen = BlueprintGenerator()
    bp, usage = gen(_intent())
    assert bp.required_interfaces
    assert usage.total_tokens > 0
    # Cost should be small for a single blueprint call.
    assert usage.cost_usd < 0.10


@_skip
@pytest.mark.slow
def test_blueprint_pass_rate() -> None:
    """>= 9/10 generations must pass the semantic grader."""
    n = int(os.environ.get("FORGE_LLM_TRIALS", "10"))
    gen = BlueprintGenerator()
    intent = _intent()

    passes = 0
    failures: list[list[str]] = []
    total_cost = 0.0
    for _ in range(n):
        bp, usage = gen(intent)
        total_cost += usage.cost_usd
        grade = _grade(bp)
        if grade.passed:  # type: ignore[attr-defined]
            passes += 1
        else:
            failures.append(grade.failures)  # type: ignore[attr-defined]

    assert passes >= max(1, round(0.9 * n)), (
        f"{passes}/{n} passed (need >=90%); cost ${total_cost:.4f}; failures: {failures}"
    )
