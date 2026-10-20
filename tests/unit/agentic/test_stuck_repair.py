"""Stuck-repair escalation (ADR-0116 follow-up, measured in the 2026-07-18 e2e A/B): when a
repair produces the IDENTICAL sandbox error again, the next revision feedback must say so and
demand a structurally different fix -- a bare traceback re-roll reproduces the same greedy
completion (the same ImportError was re-authored unchanged 4 iterations straight)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import author as author_mod
from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.author import AuthoredCode
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.llm.cost import LLMUsage


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


def test_identical_error_escalates_the_next_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feedbacks: list[str] = []

    def _initial(*_a: object, **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _revision(
        _brief_arg: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        feedbacks.append(feedback)
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(
            ok=False, error_type="exec", error_message="ImportError: no clearance_hole_d"
        )

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=3, brief=_brief(), client=object()
    )
    # Repair 1 follows the FIRST failure -- no repeat yet, no escalation.
    assert len(feedbacks) == 2
    assert "REPAIR NON-PROGRESS" not in feedbacks[0]
    # Repair 2 follows the identical error repeating -- the escalation must be present.
    assert "REPAIR NON-PROGRESS" in feedbacks[1]


def test_distinct_errors_do_not_escalate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feedbacks: list[str] = []
    errors = iter(["TypeError: a", "ValueError: b", "NameError: c"])

    def _initial(*_a: object, **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _revision(
        _brief_arg: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        feedbacks.append(feedback)
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(ok=False, error_type="exec", error_message=next(errors))

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=3, brief=_brief(), client=object()
    )
    assert len(feedbacks) == 2
    assert all("REPAIR NON-PROGRESS" not in fb for fb in feedbacks)


def test_third_identical_error_escalates_to_salted_best_of_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured: note + hint alone let an identical error run 3-4 iterations. On the THIRD
    identical failure the loop must route the repair through best_revision (k=3 salted)."""
    calls: list[int] = []

    def _initial(*_a: object, **_k: object) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="v0", code="result = Box(1, 1, 1)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _revision(
        _brief_arg: object, _base: object, feedback: str, **_k: object
    ) -> tuple[AuthoredCode, LLMUsage]:
        return AuthoredCode(approach="retry", code="result = Box(2, 2, 2)"), LLMUsage(
            model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0
        )

    def _spy_best_revision(*_a: object, k: int = 0, **_k: object) -> tuple[str, float, int]:
        calls.append(k)
        return "result = Box(3, 3, 3)", 0.0, 0

    def _sandbox(code: str, *_a: object, **_k: object) -> object:
        from forge_mind.agentic.sandbox import SandboxResult

        return SandboxResult(
            ok=False, error_type="exec", error_message="ValueError: same thing every time"
        )

    monkeypatch.setattr(loop_mod, "author_initial", _initial)
    monkeypatch.setattr(author_mod, "author_revision", _revision)
    monkeypatch.setattr(loop_mod, "best_revision", _spy_best_revision)
    monkeypatch.setattr(loop_mod, "run_build123d_code", _sandbox)

    loop_mod.run_agentic_pipeline(
        "a block", tmp_path, max_iterations=4, brief=_brief(), client=object()
    )
    assert calls == [3]  # exactly one escalation, at k=3, on the third identical failure
