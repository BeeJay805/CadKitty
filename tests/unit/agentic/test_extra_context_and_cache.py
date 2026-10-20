"""The intent contract, engine side (ADR-0036 + ADR-0035 B8): the typed STATED-FACTS block
reaches the brief researcher, and the build cache round-trips the brief so a cache hit re-runs
the acceptance judge instead of fabricating a "rejected" disposition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.agentic import result as result_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.build_cache import compute_key, load, save
from forge_mind.agentic.loop import LoopResult
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.cost import LLMUsage


def _brief(target: str = "cosmetic") -> DesignBrief:
    return DesignBrief(
        object_name="test part",
        summary="s",
        visual_description="v",
        overall_envelope_mm="40 x 40 x 10",
        features=[BriefFeature(name="body", purpose="p", approx_dims_mm="40", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
        target=target,  # type: ignore[arg-type]
    )


def _loop(prompt: str, brief: DesignBrief | None) -> LoopResult:
    return LoopResult(
        prompt=prompt,
        brief=brief,
        final_code=None,
        final_stl=None,
        final_step=None,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=False,
        critic_approved=False,
        stopped_reason="test",
        iterations=[],
        total_cost_usd=0.0,
        total_duration_s=0.0,
    )


def test_produce_part_forwards_extra_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The research seam (ADR-0036): produce_part threads extra_context into research_brief,
    where it is injected as reference material the model copies instead of re-guessing."""
    captured: dict[str, Any] = {}

    def _fake_research(prompt: str, **kw: Any) -> tuple[DesignBrief, LLMUsage]:
        captured["extra_context"] = kw.get("extra_context")
        return _brief(), LLMUsage()

    monkeypatch.setattr(result_mod, "research_brief", _fake_research)
    monkeypatch.setattr(
        result_mod, "run_agentic_pipeline", lambda prompt, work, **kw: _loop(prompt, _brief())
    )

    # client= skips LLMClient() construction (which needs ANTHROPIC_API_KEY -- absent in CI);
    # every stage that would consume it is monkeypatched, so a sentinel is honest here.
    result_mod.produce_part(
        "a test part",
        tmp_path,
        extra_context="STATED FACTS\n- dimension: width = 40 mm",
        client=object(),  # type: ignore[arg-type]
    )
    assert captured["extra_context"] == "STATED FACTS\n- dimension: width = 40 mm"


def test_produce_part_default_context_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_research(prompt: str, **kw: Any) -> tuple[DesignBrief, LLMUsage]:
        captured["extra_context"] = kw.get("extra_context")
        return _brief(), LLMUsage()

    monkeypatch.setattr(result_mod, "research_brief", _fake_research)
    monkeypatch.setattr(
        result_mod, "run_agentic_pipeline", lambda prompt, work, **kw: _loop(prompt, _brief())
    )
    result_mod.produce_part("a test part", tmp_path, client=object())  # type: ignore[arg-type]
    assert captured["extra_context"] is None


def test_build_cache_round_trips_the_brief(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR-0035 B8: a v2 cache hit restored brief=None, which skipped the acceptance judge and
    fabricated disposition='rejected' for a previously-SHIPPED identical part. v3 persists the
    brief, so a hit re-runs the judge against the SAME brief."""
    monkeypatch.setenv("FORGE_BUILD_CACHE", str(tmp_path / "cache"))
    stl = tmp_path / "final.stl"
    stl.write_bytes(b"solid x\nendsolid x\n")
    brief = _brief()
    result = _loop("a cached part", brief)
    result = LoopResult(**{**result.__dict__, "final_stl": stl, "geometry_passed": True})

    key = compute_key(
        "a cached part",
        material=None,
        n_drafts=1,
        max_iterations=4,
        fabrication_mode=FabricationMode.GENERAL,
        author_model=None,
        critic_model=None,
        brief_model=None,
    )
    save(key, result)
    restored = load(key, tmp_path / "work")
    assert restored is not None
    assert restored.brief is not None  # the v2 lie: this was None
    assert restored.brief.object_name == "test part"
    assert restored.brief.features[0].name == "body"
