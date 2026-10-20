"""Startup environment validation for the deployed backend.

Additive and safe to import: defines helpers only, no side effects at import time. Call
``check_required_env()`` at startup to surface missing production secrets. By default it logs
a warning; set ``FORGE_STRICT_ENV=1`` (or pass ``strict=True``) to make a missing required
var fatal. Never logs values.

See ENVIRONMENT.md for the full list and meanings.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# (name, why) for vars the deployed backend needs to function at all.
_REQUIRED_CORE: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "LLM calls"),
    ("SUPABASE_URL", "auth + credit RPC base"),
    ("SUPABASE_ANON_KEY", "verify caller tokens"),
]

# Required for the billing/credits surface (Stripe + admin credit grants).
_REQUIRED_BILLING: list[tuple[str, str]] = [
    ("SUPABASE_SERVICE_ROLE_KEY", "admin credit grants"),
    ("STRIPE_SECRET_KEY", "Stripe API"),
    ("STRIPE_WEBHOOK_SECRET", "verify Stripe webhooks"),
    ("STRIPE_PRICE_PRO", "pro plan price id"),
    ("STRIPE_PRICE_MAX", "max plan price id"),
    ("APP_URL", "Stripe checkout/portal redirects"),
]


def missing_env(*, include_billing: bool = True) -> list[str]:
    """Return the names of required env vars that are unset or empty."""
    checks = list(_REQUIRED_CORE)
    if include_billing:
        checks += _REQUIRED_BILLING
    return [name for name, _ in checks if not os.environ.get(name)]


def check_required_env(*, include_billing: bool = True, strict: bool | None = None) -> list[str]:
    """Check required env vars; log (or raise) if any are missing. Returns the missing list.

    ``strict`` defaults to the ``FORGE_STRICT_ENV`` env var (``"1"`` => True). When strict and
    something is missing, raises ``RuntimeError`` (fail-fast at boot). Otherwise logs a warning
    so local/dev runs are unaffected. Values are never logged.
    """
    missing = missing_env(include_billing=include_billing)
    if strict is None:
        strict = os.environ.get("FORGE_STRICT_ENV") == "1"
    if missing:
        message = "missing required environment variables: " + ", ".join(missing)
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    return missing


__all__ = ["check_required_env", "missing_env"]
