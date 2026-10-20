"""Role-based provider routing + the narrow billing allow-list (ADR-0113). The billing logic is
security-sensitive (it gates real money), so it is tested exhaustively and $0 -- no client is ever
actually called, only constructed/refused."""

from __future__ import annotations

import pytest

from forge_mind.llm.claude_code_client import ClaudeCodeClient
from forge_mind.llm.client import LLMError
from forge_mind.llm.ollama_client import OllamaClient
from forge_mind.llm.ollama_vision_client import OllamaVisionClient
from forge_mind.llm.provider import make_client


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FORGE_LLM_PROVIDER", "FORGE_FORBID_API_SPEND", "FORGE_API_ROLES"):
        monkeypatch.delenv(var, raising=False)
    for role in ("PLANNER", "CODER", "REPAIR", "VISION"):
        monkeypatch.delenv(f"FORGE_PROVIDER_{role}", raising=False)


def test_per_role_provider_overrides_global(monkeypatch: pytest.MonkeyPatch) -> None:
    # This checks routing, not whether a developer happens to have the optional CLI.
    monkeypatch.setattr(
        "forge_mind.llm.claude_code_client.shutil.which", lambda _: "claude-cli-stub"
    )
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "claude_code")
    monkeypatch.setenv("FORGE_PROVIDER_VISION", "ollama_vision")
    assert isinstance(make_client(role="vision"), OllamaVisionClient)
    # a role with no override falls back to the global provider.
    assert isinstance(make_client(role="repair"), ClaudeCodeClient)


def test_role_none_is_legacy_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_LLM_PROVIDER", "ollama")
    assert isinstance(make_client(), OllamaClient)


def test_forbid_blocks_metered_for_a_non_allowlisted_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")
    monkeypatch.setenv("FORGE_PROVIDER_CODER", "anthropic")
    with pytest.raises(LLMError, match="forbids the metered API client for role='coder'"):
        make_client(role="coder")


def test_forbid_allows_metered_only_for_the_allowlisted_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")
    monkeypatch.setenv("FORGE_API_ROLES", "planner")
    monkeypatch.setenv("FORGE_PROVIDER_PLANNER", "anthropic")
    # planner is allow-listed -> the metered client constructs (needs a key, but that is a later error,
    # not the billing gate; we only assert the gate opened for planner).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    client = make_client(role="planner")
    assert client is not None
    # a DIFFERENT role with the same guard is still blocked -- the allow-list is per-role.
    monkeypatch.setenv("FORGE_PROVIDER_REPAIR", "anthropic")
    with pytest.raises(LLMError):
        make_client(role="repair")


def test_unknown_provider_refuses_to_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_PROVIDER_PLANNER", "gpt5")
    with pytest.raises(LLMError, match="unknown provider"):
        make_client(role="planner")


def test_local_roles_never_bill_even_without_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # coder=ollama and vision=ollama_vision are $0 regardless of the forbid flag.
    monkeypatch.setenv("FORGE_PROVIDER_CODER", "ollama")
    monkeypatch.setenv("FORGE_PROVIDER_VISION", "ollama_vision")
    assert isinstance(make_client(role="coder"), OllamaClient)
    assert isinstance(make_client(role="vision"), OllamaVisionClient)
