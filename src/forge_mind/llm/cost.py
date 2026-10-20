"""Token + cost accounting for LLM calls.

Prices are USD per 1M tokens (Anthropic list pricing, cached 2026-02-17). Cache
writes are billed at 1.25x input; cache reads at 0.10x input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conservative per-1M-token fallback for a model not in MODEL_PRICING. Set ABOVE
# the most expensive known tier so an unknown model is NEVER under-billed: a
# missing price must spend MORE than reality, never silently zero. This keeps the
# loop's dollar-budget stop honest -- spend can never be zeroed by a typo'd or new
# model id.
_UNKNOWN_INPUT_PER_MTOK = 15.0
_UNKNOWN_OUTPUT_PER_MTOK = 75.0


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * 1.25

    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * 0.10


MODEL_PRICING: dict[str, ModelPrice] = {
    # Opus 5 (2026-07-24): $5/$25, VERIFIED against Anthropic's published pricing -- unchanged from
    # the previous Opus generation. Registering it is not cosmetic: unpriced, it would fall back to
    # $15/$75 (3x), which both inflates reported spend and, worse, CONFOUNDS the very A/B it exists
    # for -- the per-part dollar budget would grant the Opus 5 arm ~1/3 the iterations of the Fable
    # arm, for the same reason the Fable entry below is deliberately pinned to the Opus tier.
    "claude-opus-5": ModelPrice(5.0, 25.0),
    # Opus 4.8 pricing is an estimate (same tier as 4.6) pending verification.
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-4-6": ModelPrice(5.0, 25.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
    # Fable 5 billed at the Opus tier DELIBERATELY (estimate pending verification): equal rates
    # mean the per-part $-budget grants both A/B arms the same iteration allowance, so the
    # author comparison is not confounded by pricing. Revisit with list pricing before any
    # production default swap.
    "claude-fable-5": ModelPrice(5.0, 25.0),
}


@dataclass(frozen=True)
class LLMUsage:
    """Per-call usage + cost, normalized across stages."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0


def cost_of(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute USD cost.

    Fail-closed on an unknown model: an unpriced model must NOT cost 0.0, because
    a silent zero defeats the loop's dollar-budget stop (spend would never reach
    the ceiling). Instead we bill at the most expensive known tier and log a
    warning, so the id gets added to ``MODEL_PRICING`` but spend is never silently
    zeroed in the meantime.
    """
    price = MODEL_PRICING.get(model)
    if price is None:
        logger.warning(
            "unknown model %r not in MODEL_PRICING; billing at conservative "
            "fallback rate ($%.0f/$%.0f per Mtok). Add it to MODEL_PRICING.",
            model,
            _UNKNOWN_INPUT_PER_MTOK,
            _UNKNOWN_OUTPUT_PER_MTOK,
        )
        return (
            (input_tokens + cache_read_tokens + cache_write_tokens) * _UNKNOWN_INPUT_PER_MTOK
            + output_tokens * _UNKNOWN_OUTPUT_PER_MTOK
        ) / 1_000_000.0
    return (
        input_tokens * price.input_per_mtok
        + output_tokens * price.output_per_mtok
        + cache_read_tokens * price.cache_read_per_mtok
        + cache_write_tokens * price.cache_write_per_mtok
    ) / 1_000_000.0


__all__ = ["MODEL_PRICING", "LLMUsage", "ModelPrice", "cost_of"]
