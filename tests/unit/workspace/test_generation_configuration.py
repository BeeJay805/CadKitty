"""Capability reads report configuration without invoking inference."""

import pytest

from forge_mind.llm.codex_client import CodexClient
from forge_mind.workspace.generation_policy import generation_configuration


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "", " unknown "])
def test_default_factory_reports_configured_override_without_construction(monkeypatch, model):
    monkeypatch.setenv("FORGE_CODEX_MODEL", model)

    def forbidden(*args, **kwargs):
        raise AssertionError("Capability reads must not construct a model client")

    monkeypatch.setattr(CodexClient, "__init__", forbidden)
    assert generation_configuration(CodexClient) == {
        "provider": "codex",
        "model": model,
        "model_identity": "configured_only",
    }


def test_default_model_is_configuration_not_attestation(monkeypatch):
    monkeypatch.delenv("FORGE_CODEX_MODEL", raising=False)
    assert generation_configuration(CodexClient)["model"] == "gpt-6-astra"
    assert generation_configuration(CodexClient)["model_identity"] == "configured_only"


def test_custom_factory_does_not_inherit_default_identity(monkeypatch):
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-6-astra")

    def custom(**kwargs):
        raise AssertionError("Capability reads must not invoke an injected factory")

    assert generation_configuration(custom) == {
        "provider": "custom",
        "model": None,
        "model_identity": "unknown",
    }
