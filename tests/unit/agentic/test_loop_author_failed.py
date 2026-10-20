"""An author-stage failure ends the part GRACEFULLY (`stopped_reason="author_failed"`), never as a
crash through ``run_agentic_pipeline`` -- the loop's "never raises on a stage failure" contract.
Measured: a truncated author response killed a whole bench run mid-flight before this guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.llm.client import LLMError


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="test bracket",
        summary="a bracket",
        visual_description="an L-shaped bracket",
        overall_envelope_mm="40 x 40 x 10",
        features=[
            BriefFeature(
                name="bolt holes", purpose="mounting", approx_dims_mm="M4", method="bolt_holes"
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat side down",
    )


def test_initial_author_failure_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise LLMError("structured output failed to parse twice (stub)")

    monkeypatch.setattr(loop_mod, "author_initial", _boom)
    result = loop_mod.run_agentic_pipeline(
        "a bracket",
        tmp_path,
        max_iterations=2,
        brief=_brief(),  # pre-researched: no network for the brief stage
        client=object(),  # never actually called past the stubbed author
    )
    assert result.stopped_reason == "author_failed"
    assert result.final_code is None
    assert result.geometry_passed is False
