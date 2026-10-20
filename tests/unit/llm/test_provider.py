"""The provider factory: env-driven billing-path resolution, fail closed (money-free).

The factory is the ONE fallback constructor for every engine stage, so these tests pin
the exact resolution order: explicit subscription provider > unknown-provider error >
spend-guard error > metered default. No network, no key use -- construction only.
"""

from __future__ import annotations

import pytest

from forge_mind.llm.claude_code_client import ClaudeCodeClient
from forge_mind.llm.client import LLMClient, LLMError
from forge_mind.llm.provider import make_client


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("FORGE_FORBID_API_SPEND", raising=False)


def _fake_claude_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI runners have no claude CLI; the factory must still RESOLVE to the
    subscription client there (construction needs an exe path, not a working CLI)."""
    monkeypatch.setattr(
        "forge_mind.llm.claude_code_client.shutil.which", lambda name: "claude-cli-stub"
    )


def test_default_is_the_metered_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    assert isinstance(make_client(), LLMClient)


def test_claude_code_provider_returns_subscription_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    _fake_claude_cli(monkeypatch)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "claude_code")
    # No API key needed at all on the subscription path.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_client(), ClaudeCodeClient)


def test_spend_guard_blocks_the_metered_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")  # present but FORBIDDEN
    with pytest.raises(LLMError, match="FORGE_FORBID_API_SPEND"):
        make_client()


def test_spend_guard_allows_the_subscription_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    _fake_claude_cli(monkeypatch)
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "claude_code")
    assert isinstance(make_client(), ClaudeCodeClient)


def test_unknown_provider_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "claude-code")  # typo'd value must NOT bill
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    with pytest.raises(LLMError, match="unknown provider"):  # role-aware message (ADR-0113)
        make_client()


def test_missing_key_still_raises_on_the_metered_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_client()


def test_both_clients_satisfy_the_protocol_surface() -> None:
    """The two providers expose identical structured-generation entry points, so the
    Protocol annotation is real, not aspirational."""
    for method in ("generate_structured", "generate_structured_with_image"):
        assert callable(getattr(LLMClient, method))
        assert callable(getattr(ClaudeCodeClient, method))
