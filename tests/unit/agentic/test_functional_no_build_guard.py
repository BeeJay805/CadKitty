"""produce_functional_part never raises on stage-1 LLM failures (engine-probe regression).

The 5-part engine probe found ``research_brief`` OUTSIDE the never-raises guard: its
DesignBrief response truncated at max_tokens (deterministic, so the client's one parse
retry cannot recover) and the raised LLMError killed the whole run. Both stage-1 calls
(research + decompose) must degrade to an honest ``no_build`` instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.agentic import functional as functional_mod
from forge_mind.agentic.functional import produce_functional_part
from forge_mind.llm.client import LLMError


def _boom(*_a: object, **_k: object) -> object:
    raise LLMError("structured output failed to parse twice -- truncated at max_tokens")


def test_research_brief_failure_degrades_to_no_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(functional_mod, "research_brief", _boom)
    res = produce_functional_part("a five-part demonstration engine", tmp_path, client=object())
    assert res.disposition == "no_build"
    assert res.whole_brief is None
    assert res.object_name.startswith("a five-part demonstration engine")


def test_token_ceilings_hold_the_big_vision_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Row-54 regression class: deterministic truncation of structured stages. These
    ceilings are load-bearing for dense multi-part prompts -- they must never drift
    back down (founder direction 2026-07-10)."""
    from forge_mind.agentic.author import author_max_tokens
    from forge_mind.agentic.critic import CRITIC_MAX_TOKENS
    from forge_mind.agentic.judge import JUDGE_MAX_TOKENS
    from forge_mind.config import BRIEF_MAX_TOKENS, PLAN_MAX_TOKENS

    monkeypatch.delenv("FORGE_AUTHOR_MAX_TOKENS", raising=False)
    assert BRIEF_MAX_TOKENS >= 16000
    assert PLAN_MAX_TOKENS >= 16000
    assert author_max_tokens() >= 16000
    assert CRITIC_MAX_TOKENS >= 4000
    assert JUDGE_MAX_TOKENS >= 4000


def test_decompose_failure_still_no_build_with_the_brief_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_brief = SimpleNamespace(object_name="demonstration engine")
    monkeypatch.setattr(
        functional_mod,
        "research_brief",
        lambda *_a, **_k: (fake_brief, SimpleNamespace(cost_usd=0.01)),
    )
    monkeypatch.setattr(functional_mod, "decompose_functional", _boom)
    res = produce_functional_part("a five-part demonstration engine", tmp_path, client=object())
    assert res.disposition == "no_build"
    assert res.object_name == "demonstration engine"
    assert res.whole_brief is fake_brief


def test_plan_is_saved_and_reusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Row-57 fix: the accepted plan persists (plan.json) and FORGE_REUSE_PLAN loads it
    instead of re-decomposing -- a re-run starts from the surviving plan."""
    from forge_mind.agentic import functional as fm

    plan = fm.FunctionalPlan(
        object_name="engine",
        parts=[
            fm.PartPlan(name="a", prompt="a card"),
            fm.PartPlan(name="b", prompt="b card"),
        ],
        assembly_note="a on b",
        joints=[],
    )
    calls = {"decompose": 0}

    def _fake_decompose(*_a: object, **_k: object):
        calls["decompose"] += 1
        return plan, SimpleNamespace(cost_usd=0.0)

    def _fake_pipeline(prompt: str, out_dir: Path, **_k: object):
        out_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            final_stl=None,
            final_step=None,
            final_render=None,
            total_cost_usd=0.0,
            prompt=prompt,
            validation=None,
            geometry_passed=False,
            critic_approved=False,
            stopped_reason="no_build",
            iterations=[],
            critique=None,
            brief=None,
        )

    monkeypatch.setattr(fm, "decompose_functional", _fake_decompose)
    monkeypatch.setattr(fm, "run_agentic_pipeline", _fake_pipeline)
    monkeypatch.setattr(
        fm, "acceptance_judge", lambda *a, **k: (None, SimpleNamespace(cost_usd=0.0))
    )
    monkeypatch.delenv("FORGE_REUSE_PLAN", raising=False)

    brief = SimpleNamespace(object_name="engine", material=None, visual_description="x")
    run1 = tmp_path / "r1"
    fm.produce_functional_part("an engine", run1, client=object(), brief=brief)
    assert (run1 / "plan.json").exists()
    assert calls["decompose"] == 1

    monkeypatch.setenv("FORGE_REUSE_PLAN", str(run1 / "plan.json"))
    run2 = tmp_path / "r2"
    res2 = fm.produce_functional_part("an engine", run2, client=object(), brief=brief)
    assert calls["decompose"] == 1  # NOT called again
    assert (run2 / "plan.json").exists()
    assert res2.object_name == "engine"


def test_transient_author_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part that author_fails with 0 iterations (an intermittent CLI blip) is retried;
    the radial lost 9/16 parts to this with no retry. A real result is NOT retried."""
    from forge_mind.agentic import functional as fm

    monkeypatch.setattr(fm.time, "sleep", lambda _s: None)  # no real backoff in the test

    plan = fm.FunctionalPlan(
        object_name="thing",
        parts=[fm.PartPlan(name="a", prompt="a"), fm.PartPlan(name="b", prompt="b")],
        assembly_note="a+b",
        joints=[],
    )
    monkeypatch.setattr(
        fm, "decompose_functional", lambda *a, **k: (plan, SimpleNamespace(cost_usd=0.0))
    )
    monkeypatch.setattr(
        fm, "acceptance_judge", lambda *a, **k: (None, SimpleNamespace(cost_usd=0.0))
    )

    calls: dict[str, int] = {}

    def _pipeline(prompt, out_dir, **_k):
        out_dir.mkdir(parents=True, exist_ok=True)
        calls[prompt] = calls.get(prompt, 0) + 1
        # part "a" fails transiently the first 2 tries, succeeds on the 3rd; "b" succeeds at once
        if prompt == "a" and calls[prompt] < 3:
            return SimpleNamespace(
                final_stl=None,
                final_step=None,
                final_render=None,
                total_cost_usd=0.0,
                prompt=prompt,
                validation=None,
                geometry_passed=False,
                critic_approved=False,
                stopped_reason="author_failed",
                iterations=[],
                critique=None,
                brief=None,
            )
        stl = out_dir / "p.stl"
        stl.write_text("x")
        return SimpleNamespace(
            final_stl=stl,
            final_step=None,
            final_render=None,
            total_cost_usd=0.0,
            prompt=prompt,
            validation=None,
            geometry_passed=True,
            critic_approved=True,
            stopped_reason="accepted",
            iterations=[object()],
            critique=None,
            brief=None,
        )

    monkeypatch.setattr(fm, "run_agentic_pipeline", _pipeline)
    brief = SimpleNamespace(object_name="thing", material=None, visual_description="x")
    fm.produce_functional_part("thing", tmp_path, client=object(), brief=brief)
    assert calls["a"] == 3  # retried twice then succeeded
    assert calls["b"] == 1  # a clean result is never retried
