"""Stripe price -> (plan, monthly credits) mapping. The unknown-price default is a deliberate
fail-safe: a mismapped or renamed price still grants *a* paid allotment (Pro) rather than crashing
the webhook or silently granting nothing."""

from __future__ import annotations

import pytest

from forge_mind.server import billing


def test_plan_credits_unknown_or_missing_price_defaults_to_pro() -> None:
    assert billing._plan_credits_for_price(None) == ("pro", billing.PRO_MONTHLY_CREDITS)
    assert billing._plan_credits_for_price("price_not_configured") == (
        "pro",
        billing.PRO_MONTHLY_CREDITS,
    )


def test_plan_credits_max_price_maps_to_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(billing, "STRIPE_PRICE_MAX", "price_max_123")
    assert billing._plan_credits_for_price("price_max_123") == ("max", billing.MAX_MONTHLY_CREDITS)
