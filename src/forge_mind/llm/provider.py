"""The LLM provider factory -- ONE place that decides which client the engine bills.

Every stage that used to fall back to ``client or LLMClient()`` now falls back to
``client or make_client()``, so the whole engine (agentic author/critic/judge, the
deterministic lane's intent/blueprint/dsl/layout/router/patch stages, intake planner,
narration) honors the same two environment switches:

* ``FORGE_LLM_PROVIDER=claude_code`` -- route EVERY internal LLM call through the
  subscription-billed headless Claude Code client (``ClaudeCodeClient``, $0 API spend).
  An unknown provider value raises ``LLMError`` (fail closed -- never silently bill
  the metered key because of a typo).
* ``FORGE_FORBID_API_SPEND=1`` -- a hard guard: constructing the metered API client is
  an ERROR unless a subscription provider is explicitly selected. With the guard set,
  nothing in the engine can silently bill the still-present ``ANTHROPIC_API_KEY``.

Explicitly-injected clients (the ``client`` parameter every stage already accepts) are
untouched: tests keep passing sentinels/mocks, and callers that constructed a specific
client keep full control. The factory only owns the FALLBACK path.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

from forge_mind.llm.claude_code_client import ClaudeCodeClient
from forge_mind.llm.client import LLMClient, LLMError
from forge_mind.llm.cost import LLMUsage

T = TypeVar("T", bound=BaseModel)

_PROVIDER_ENV = "FORGE_LLM_PROVIDER"
_FORBID_ENV = "FORGE_FORBID_API_SPEND"

# ROLE-BASED routing (ADR-0113, the local-first roster): each engine role can pick its own provider,
# so Fable(API) can plan while CAD-Coder / qwen3-coder / Qwen2.5-VL (all local, $0) build, repair, and
# see. A role's provider is read from ``FORGE_PROVIDER_<ROLE>`` (e.g. FORGE_PROVIDER_PLANNER=anthropic);
# unset falls back to the global FORGE_LLM_PROVIDER, then the default. A None role = legacy behavior.
_ROLE_PROVIDER_ENV_PREFIX = "FORGE_PROVIDER_"

# The NARROW billing allow-list (founder 2026-07-15, reversing the 2026-07-10 subscription-only rule for
# ONE role): with FORGE_FORBID_API_SPEND=1 still set, the metered API is constructible ONLY for a role
# named in FORGE_API_ROLES (comma-separated, e.g. "planner"). Every other role stays hard-blocked from
# spend -- so a misconfigured coder/repair/vision can never silently bill. Fail-closed by default (empty
# list = nothing may bill under the guard).
_API_ROLES_ENV = "FORGE_API_ROLES"


class LLMClientLike(Protocol):
    """The structural contract every provider satisfies (``LLMClient`` natively,
    ``ClaudeCodeClient`` by duck-typing). Stages annotate against THIS so a
    subscription client threads through the same code paths as the API client."""

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]: ...

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[T],
        model: str,
        max_tokens: int,
    ) -> tuple[T, LLMUsage]: ...


def _api_allowed_roles() -> frozenset[str]:
    """Roles permitted to construct the metered API client under the forbid guard."""
    raw = os.environ.get(_API_ROLES_ENV, "")
    return frozenset(r.strip().lower() for r in raw.split(",") if r.strip())


def _resolve_provider(role: str | None) -> str:
    """Provider for ``role``: the per-role override, else the global default, else ""."""
    if role is not None:
        override = os.environ.get(f"{_ROLE_PROVIDER_ENV_PREFIX}{role.upper()}", "").strip()
        if override:
            return override
    return os.environ.get(_PROVIDER_ENV, "").strip()


def make_client(role: str | None = None) -> LLMClientLike:
    """The engine's one fallback constructor, now role-aware (ADR-0113).

    ``role`` (e.g. ``"planner"``, ``"vision"``) selects a provider via ``FORGE_PROVIDER_<ROLE>``,
    falling back to the global ``FORGE_LLM_PROVIDER`` then the default. ``role=None`` is the legacy
    path (global provider only) -- every existing call site is byte-identical.

    Provider resolution (fail closed):
      1. ``claude_code`` -> ``ClaudeCodeClient`` (subscription, $0).
      2. ``ollama`` -> ``OllamaClient`` (local structured, $0).
      3. ``ollama_vision`` -> ``OllamaVisionClient`` (local Qwen2.5-VL, $0, real image input).
      4. ``anthropic`` (or the empty default) -> the metered ``LLMClient``, but ONLY if spend is not
         forbidden OR this ``role`` is in ``FORGE_API_ROLES`` (the narrow allow-list). Otherwise
         ``LLMError`` -- a non-allow-listed role can never bill.
      5. Any other non-empty value -> ``LLMError`` (unknown provider; never guess a billing path).
    """
    provider = _resolve_provider(role)
    if provider == "codex":
        from forge_mind.llm.codex_client import CodexClient

        return CodexClient()
    if provider == "claude_code":
        return ClaudeCodeClient()
    if provider == "ollama":
        # LOCAL GPU (ADR-0111): a local Ollama model. $0, no API/subscription -- satisfies the
        # forbid-spend guard, and is the target for the 4090 fine-tune.
        from forge_mind.llm.ollama_client import OllamaClient

        return OllamaClient()
    if provider == "ollama_vision":
        # LOCAL vision (ADR-0113): Qwen2.5-VL over Ollama, real image input. $0.
        from forge_mind.llm.ollama_vision_client import OllamaVisionClient

        return OllamaVisionClient()
    if provider not in ("", "anthropic"):
        raise LLMError(
            f"unknown provider {provider!r} for role={role!r} (known: 'codex', 'claude_code', 'ollama', "
            "'ollama_vision', 'anthropic'); refusing to guess a billing path"
        )
    # Metered API path (provider is "" default or explicit "anthropic").
    if os.environ.get(_FORBID_ENV, "").strip() == "1":
        role_key = (role or "").lower()
        if role_key not in _api_allowed_roles():
            raise LLMError(
                f"{_FORBID_ENV}=1 forbids the metered API client for role={role!r}; add it to "
                f"{_API_ROLES_ENV} to allow spend for this role, or select a local/subscription "
                f"provider ({_PROVIDER_ENV}=claude_code / ollama)"
            )
    return LLMClient()


__all__ = ["LLMClientLike", "make_client"]
