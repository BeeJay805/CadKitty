"""Config resolution tests — the BRAIN_MODEL swap seam (ADR-0055).

The invention lab + chief-engineer brain defaults to Opus; FORGE_BRAIN_MODEL swaps it (e.g. to Fable
when it returns) with no code change. ``_resolve_brain_model`` is the pure resolver, so the seam is
tested directly without reimporting the module.
"""

from __future__ import annotations

from forge_mind.config import FABLE, HAIKU, OPUS, SONNET, _resolve_brain_model


def test_brain_model_defaults_to_opus_when_unset() -> None:
    assert _resolve_brain_model("") == OPUS
    assert _resolve_brain_model("   ") == OPUS


def test_brain_model_aliases_resolve_case_insensitively() -> None:
    assert _resolve_brain_model("fable") == FABLE
    assert _resolve_brain_model("FABLE") == FABLE
    assert _resolve_brain_model("Opus") == OPUS
    assert _resolve_brain_model("sonnet") == SONNET
    assert _resolve_brain_model(" haiku ") == HAIKU


def test_brain_model_passes_through_an_unknown_full_id() -> None:
    # A future model id needs no code change: a non-alias value is used verbatim.
    assert _resolve_brain_model("claude-future-9") == "claude-future-9"
