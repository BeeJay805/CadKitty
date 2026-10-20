"""Resolve the configured author model without implying measured runtime identity."""

from __future__ import annotations

import os
from typing import Any


def author_model(client: Any) -> str:
    """Respect a client's effective model, including Codex's existing setting.

    Injected clients can own their model. Clients without a fixed model receive
    the same explicit FORGE_CODEX_MODEL selection as the default workspace client.
    Returned usage remains the provider's report, not independent attestation.
    """
    fixed = getattr(client, "model", None)
    if isinstance(fixed, str) and fixed.strip():
        return fixed.strip()
    return os.environ.get("FORGE_CODEX_MODEL", "gpt-6-astra").strip() or "gpt-6-astra"


def staged_native_planning() -> bool:
    """Opt in to private staged planning without changing the deployed default."""
    return os.environ.get("FORGE_NATIVE_STAGED_PLANNING", "0") == "1"


def native_call_allowance(remaining: float | None) -> float:
    """Bound an inference attempt, retaining the enclosing episode deadline."""
    return 300.0 if remaining is None else min(300.0, remaining)


def generation_configuration(client_factory: Any) -> dict[str, Any]:
    """Describe configuration without constructing a client or attesting inference."""
    from forge_mind.llm.codex_client import CodexClient

    if client_factory is CodexClient:
        return {
            "provider": "codex",
            "model": os.environ.get("FORGE_CODEX_MODEL", "gpt-6-astra"),
            "model_identity": "configured_only",
        }
    return {"provider": "custom", "model": None, "model_identity": "unknown"}
