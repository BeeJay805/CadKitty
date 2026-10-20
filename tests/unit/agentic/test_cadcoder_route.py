"""CAD-Coder live-loop route (ADR-0112): the flag gate, per-candidate runner selection, and
fail-soft first-pass. All $0 -- no model, no server (the client is stubbed / never constructed)."""

from __future__ import annotations

import pytest

from forge_mind.agentic.cadcoder_route import cadcoder_enabled, cadcoder_initial, runner_for

_CADQUERY_RUNNER = "forge_mind.agentic._cadquery_runner"
_BUILD123D_RUNNER = "forge_mind.agentic._sandbox_runner"


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_GENERATOR", raising=False)
    assert cadcoder_enabled() is False


def test_enabled_only_for_exact_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_GENERATOR", "cadcoder")
    assert cadcoder_enabled() is True
    monkeypatch.setenv("FORGE_GENERATOR", "author")
    assert cadcoder_enabled() is False


def test_runner_for_routes_cadquery_to_the_bridge() -> None:
    assert runner_for("import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)") == (
        _CADQUERY_RUNNER
    )


def test_runner_for_keeps_build123d_on_the_default_runner() -> None:
    # A build123d author draft must NOT be rerouted -- the default path stays byte-identical.
    assert runner_for("result = Box(10, 10, 5)") == _BUILD123D_RUNNER


def test_initial_fails_soft_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model/server failure returns None so the loop falls through to the build123d author."""
    import forge_mind.llm.cadcoder_client as mod

    def _boom(_prompt: str) -> tuple[str, object]:
        raise RuntimeError("ollama down")

    monkeypatch.setattr(mod, "generate_cadquery_coder", _boom)
    assert cadcoder_initial("a cube") is None


def test_initial_returns_code_when_the_model_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge_mind.llm.cadcoder_client as mod

    def _ok(_prompt: str) -> tuple[str, object]:
        return "import cadquery as cq\nresult = cq.Workplane('XY').box(2,2,2)", object()

    monkeypatch.setattr(mod, "generate_cadquery_coder", _ok)
    code = cadcoder_initial("a cube")
    assert code is not None and "cq.Workplane" in code
