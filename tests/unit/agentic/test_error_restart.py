"""Error-path restarts (ADR-0117): (1) an iteration-1 failure a verified hint recognizes gets
ONE fresh massing with the hint as its strategy (a patch-repair inherits a structurally doomed
draft -- measured: revolve(angle=) recurs at iteration 1 and its repair lands in Null TopoDS);
(2) rung 4 -- when the salted best-of-3 repair (rung 3) fired and the SAME error came back,
the next attempt authors 3 fresh massings via the draft selector (measured: repair cannot
leave the basin; 6 whole pulley runs burned on identical Null TopoDS)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import author as author_mod
from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.author import AuthoredCode
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.drafts import Draft, DraftSelection
from forge_mind.llm.cost import LLMUsage

_USAGE = LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0)


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="test block",
        summary="a block",
        visual_description="a solid block",
        overall_envelope_mm="10 x 10 x 10",
        features=[BriefFeature(name="body", purpose="mass", approx_dims_mm="10", method="box")],
        min_wall_mm=1.0,
        print_orientation_note="flat",
    )


def _failing_sandbox(message: str):
    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(ok=False, error_type="exec", error_message=message)

    return _sandbox


def test_iter1_hinted_error_gets_a_fresh_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_calls: list[str] = []
    revision_calls: list[str] = []

    def _initial(*_a: object, strategy: str = "", **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        initial_calls.append(strategy)
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), _USAGE

    def _revision(
        _b: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        revision_calls.append(feedback)
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), _USAGE

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(
        loop_mod,
        "run_build123d_code",
        _failing_sandbox("TypeError: revolve() got an unexpected keyword argument 'angle'"),
    )

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=2, brief=_brief(), budget_usd=1.0, client=object()
    )
    # the iteration-1 failure carries the revolve hint -> the SECOND author call is a fresh
    # start whose strategy names the error and the verified guidance, NOT a patch repair
    assert len(initial_calls) >= 2
    assert "FRESH RESTART" in initial_calls[1]
    assert "revolution_arc" in initial_calls[1]  # the triggered hint rides along


def test_iter1_unhinted_error_stays_on_the_repair_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_calls: list[str] = []
    revision_calls: list[str] = []

    def _initial(*_a: object, strategy: str = "", **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        initial_calls.append(strategy)
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), _USAGE

    def _revision(
        _b: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        revision_calls.append(feedback)
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), _USAGE

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(
        loop_mod, "run_build123d_code", _failing_sandbox("NameError: name 'boxx' is not defined")
    )

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=2, brief=_brief(), budget_usd=1.0, client=object()
    )
    # no hint matches a plain NameError -> the normal repair path runs, no fresh start
    assert len(initial_calls) == 1
    assert len(revision_calls) >= 1


def test_rung4_routes_through_fresh_drafts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    draft_calls: list[str] = []
    salted_calls: list[int] = []

    def _initial(*_a: object, strategy: str = "", **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), _USAGE

    def _revision(
        _b: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), _USAGE

    def _best_revision(*_a: object, **_k: object) -> tuple[str, float, int]:
        salted_calls.append(1)
        return "result = Box(3, 3, 3)", 0.0, 3

    def _select(
        _brief_arg: object, n: int, _wd: object, *, strategy_suffix: str = "", **_k: object
    ) -> DraftSelection:
        draft_calls.append(strategy_suffix)
        return DraftSelection(
            best_code="result = Box(4, 4, 4)",
            best_approach="fresh",
            drafts=[Draft(0, "result = Box(4, 4, 4)", "fresh", True, 1, 64.0, None)],
            cost_usd=0.0,
        )

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(loop_mod, "best_revision", _best_revision)
    monkeypatch.setattr(loop_mod, "select_best_draft", _select)
    # an error no hint recognizes, so the iter-1 fresh start stays out of the way and the
    # streak machinery is what routes: rung 3 (salted) at streak 3, rung 4 (drafts) at 4+
    monkeypatch.setattr(
        loop_mod, "run_build123d_code", _failing_sandbox("RuntimeError: mystery kernel state")
    )

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=6, brief=_brief(), budget_usd=1.0, client=object()
    )
    assert len(salted_calls) == 1  # rung 3 fired exactly once (streak 3)
    assert len(draft_calls) >= 1  # rung 4 took over when the error persisted
    assert "FRESH RESTART" in draft_calls[0]
    assert "mystery kernel state" in draft_calls[0]
