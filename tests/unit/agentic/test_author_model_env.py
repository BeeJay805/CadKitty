"""Author model is runtime-overridable via FORGE_AUTHOR_MODEL (ADR-0072): an alias
(opus/fable/sonnet/haiku) or a full model id, default OPUS. An explicit ``model=`` arg wins.
Makes the Fable A/B a one-line ENV flip -- no code change, same pattern as FORGE_BRAIN_MODEL.
"""

from __future__ import annotations

import pytest

from forge_mind.agentic.author import author_default_model, author_revision
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.config import FABLE, HAIKU, OPUS, OPUS_5, SONNET
from forge_mind.llm.cost import LLMUsage


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="block",
        summary="a block",
        visual_description="a block",
        overall_envelope_mm="10 x 10 x 10",
        features=[BriefFeature(name="body", purpose="mass", approx_dims_mm="10", method="box")],
        min_wall_mm=1.0,
        print_orientation_note="flat",
    )


class _ModelCapture:
    def __init__(self) -> None:
        self.model: str | None = None

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: object,
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[object, LLMUsage]:
        self.model = model
        return (
            output_model(approach="x", code="result = Box(1, 1, 1)"),  # type: ignore[operator]
            LLMUsage(model=model, input_tokens=1, output_tokens=1, cost_usd=0.0),
        )


def test_default_is_fable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_AUTHOR_MODEL", raising=False)
    assert author_default_model() == FABLE  # founder decision 2026-07-02 (ADR-0076)


def test_aliases_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    for alias, expected in [
        ("fable", FABLE),
        ("opus", OPUS),
        ("opus5", OPUS_5),
        ("sonnet", SONNET),
        ("haiku", HAIKU),
    ]:
        monkeypatch.setenv("FORGE_AUTHOR_MODEL", alias)
        assert author_default_model() == expected
    monkeypatch.setenv("FORGE_AUTHOR_MODEL", "FaBLe")  # case-insensitive
    assert author_default_model() == FABLE


def test_opus5_is_selectable_but_moves_no_default() -> None:
    """Opus 5 is wired as an A/B ARM only (founder 2026-07-25: "wire both, don't test yet").

    `opus` must still resolve to Opus 4.8: that constant is the judge/critic model AND the default
    BRAIN_MODEL, so rebinding it would silently flip three stages at once. ADR-0076's
    Fable-over-Opus author verdict also stands until an A/B says otherwise -- it was measured
    against Opus 4.x, which is a reason to RE-TEST, not a reason to assume the new model wins.
    """
    from forge_mind.config import _resolve_brain_model

    assert OPUS_5 == "claude-opus-5"
    assert OPUS != OPUS_5, "the `opus` pin must not be rebound to Opus 5"
    assert _resolve_brain_model("opus5") == OPUS_5, "planner arm is selectable"
    assert _resolve_brain_model("opus") == OPUS, "planner `opus` alias unchanged"
    assert _resolve_brain_model("") == OPUS, "planner default unchanged (still Opus 4.8)"


def test_full_id_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_AUTHOR_MODEL", "claude-some-future-1")
    assert author_default_model() == "claude-some-future-1"


def test_env_override_reaches_the_author_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_AUTHOR_MODEL", "fable")
    cap = _ModelCapture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix", client=cap)
    assert cap.model == FABLE  # the env default flowed into the generate call


def test_explicit_model_arg_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_AUTHOR_MODEL", "fable")
    cap = _ModelCapture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix", client=cap, model=OPUS)
    assert cap.model == OPUS  # an explicit model= always wins
