"""The /generate <- guided-intake handoff (ADR-0036): design_session_id resolves to the typed
STATED-FACTS block; every miss degrades silently to prompt-only behavior."""

from __future__ import annotations

import pytest

from forge_mind.intake.brief import assemble_brief
from forge_mind.intake.schema import PlannerState
from forge_mind.intake.session import DesignSession
from forge_mind.server import app as server_app


def test_no_session_id_is_none() -> None:
    assert server_app._brief_context_for(None) is None
    assert server_app._brief_context_for("") is None


def test_unknown_session_id_degrades_silently() -> None:
    assert server_app._brief_context_for("definitely-not-a-session") is None


def test_session_without_ready_brief_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    session = DesignSession()
    monkeypatch.setattr(server_app._intake_service.store, "get", lambda sid: session)
    assert server_app._brief_context_for(session.id) is None


def test_ready_brief_renders_stated_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    session = DesignSession()
    session.original_text_prompt = "a NEMA-17 mount"
    session.fabrication = "general"
    session.final_design_brief = assemble_brief(session, PlannerState(), ready=True)
    monkeypatch.setattr(server_app._intake_service.store, "get", lambda sid: session)
    ctx = server_app._brief_context_for(session.id)
    assert ctx is not None and ctx.startswith("STATED FACTS")
    assert "- object: a NEMA-17 mount" in ctx
    assert "general fabrication" in ctx
