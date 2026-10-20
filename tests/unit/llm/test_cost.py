"""Unit tests for cost accounting (no LLM)."""

from __future__ import annotations

import logging
import math

import pytest

from forge_mind.llm.cost import MODEL_PRICING, LLMUsage, ModelPrice, cost_of


def test_sonnet_cost_basic() -> None:
    # 1M input @ $3, 1M output @ $15.
    cost = cost_of("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert math.isclose(cost, 18.0, rel_tol=1e-9)


def test_cache_read_is_cheaper_than_input() -> None:
    full = cost_of("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0)
    cached = cost_of(
        "claude-sonnet-4-6", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
    )
    assert cached < full
    assert math.isclose(cached, full * 0.10, rel_tol=1e-9)


def test_unknown_model_is_not_zero() -> None:
    # An unknown model must NEVER cost 0.0 -- a silent zero defeats the loop's
    # dollar-budget stop. Fail-closed: bill SOMETHING nonzero. (Old behavior was
    # 0.0; this test FAILS before the fail-closed change and PASSES after.)
    cost = cost_of("not-a-model", 1000, 1000)
    assert cost > 0.0


def test_unknown_model_bills_at_least_the_most_expensive_known_tier() -> None:
    # The unknown fallback must not UNDER-bill: for the same token counts it should
    # cost at least as much as the priciest known model, so spend is never silently
    # cheaper than reality.
    tokens_in, tokens_out = 1_000_000, 1_000_000
    unknown = cost_of("not-a-model", tokens_in, tokens_out)
    priciest_known = max(cost_of(m, tokens_in, tokens_out) for m in MODEL_PRICING)
    assert unknown >= priciest_known


def test_unknown_model_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="forge_mind.llm.cost"):
        cost_of("not-a-model", 1000, 1000)
    assert any("unknown model" in r.message.lower() for r in caplog.records)


def test_known_model_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="forge_mind.llm.cost"):
        cost_of("claude-sonnet-4-6", 1000, 1000)
    assert not caplog.records


def test_usage_total_and_cache_hit() -> None:
    u = LLMUsage(
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=200,
    )
    assert u.total_tokens == 350
    assert u.cache_hit is True


def test_every_pinned_config_model_is_priced() -> None:
    """No pinned model may fall through to the unknown-model rate.

    The fallback is $15/$75, deliberately set ABOVE the priciest known tier so a typo can never
    UNDER-bill. The cost of that safety is that a legitimately-new model is over-billed 3x until
    registered -- which silently starves it of iterations under the per-part dollar budget and
    rigs any A/B against it. Opus 5 shipped unpriced and this test exists so the next one cannot.
    """
    from forge_mind import config

    for name in ("SONNET", "HAIKU", "OPUS", "OPUS_5", "FABLE"):
        model_id = getattr(config, name)
        assert model_id in MODEL_PRICING, f"config.{name} = {model_id!r} is not priced"


def test_opus5_is_priced_at_the_opus_tier() -> None:
    """Verified 2026-07-25 against Anthropic's published pricing: $5/$25, unchanged from the
    previous Opus generation. Equal to the Fable and Opus 4.8 rates, which is what lets the
    per-part dollar budget grant every A/B arm the same iteration allowance."""
    from forge_mind.config import FABLE, OPUS, OPUS_5

    assert MODEL_PRICING[OPUS_5] == ModelPrice(5.0, 25.0)
    assert MODEL_PRICING[OPUS_5] == MODEL_PRICING[OPUS] == MODEL_PRICING[FABLE]
