"""Honest live-progress streaming (evaluator #6b): the SSE worker no longer scripts a fixed
intent->plan->cad heartbeat. The engine drives REAL phase transitions via ``on_phase`` as each stage
actually starts and ends, and the keepalive re-asserts the CURRENT phase rather than a faked "cad".

These pin the wiring three ways, all no-spend: the deterministic implicit (lattice) lane and the
agentic lane (its two LLM stages stubbed) prove the engine emits the real transitions, and a
TestClient drive of the ACTUAL /generate/stream route proves the SSE worker serializes them in order.
Only the live LLM behind the deployed endpoint is smoke-tested manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from forge_mind.agentic import result as result_mod
from forge_mind.api import core
from forge_mind.server import app as server_app


def test_generate_part_drives_real_phases_on_implicit_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deterministic lattice lane fires plan->cad in order; export only on a successful build."""
    calls: list[tuple[str, str]] = []

    def _stub_implicit(*_a: Any, **_k: Any) -> Any:
        return core.GenerateResult(
            status="error", error=core.GenerateError(stage="x", message="stub")
        )

    monkeypatch.setattr(core, "_generate_implicit", _stub_implicit)
    core.generate_part(
        "a gyroid lattice 30mm cube", tmp_path, on_phase=lambda k, s: calls.append((k, s))
    )

    # plan + cad bracket the (stubbed) build; export is NOT emitted because it returned an error.
    assert calls == [("plan", "active"), ("plan", "done"), ("cad", "active"), ("cad", "done")]


def test_produce_part_drives_real_phases_on_agentic_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agentic lane brackets brief research as "plan" and author/build as "cad" -- proven with
    both LLM stages stubbed so there is no network call or spend."""
    calls: list[tuple[str, str]] = []

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("brief research is stubbed in this test")

    fake_loop = result_mod.LoopResult(
        prompt="x",
        brief=None,
        final_code="result = None",
        final_stl=None,
        final_step=None,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=False,
        critic_approved=False,
        stopped_reason="no_build",
        iterations=[],
        total_cost_usd=0.0,
        total_duration_s=0.0,
    )

    monkeypatch.setattr(result_mod, "research_brief", _no_network)
    monkeypatch.setattr(result_mod, "run_agentic_pipeline", lambda *_a, **_k: fake_loop)
    monkeypatch.setattr(result_mod, "_write_artifacts", lambda *_a, **_k: {})

    result_mod.produce_part(
        "x", tmp_path, client=object(), on_phase=lambda k, s: calls.append((k, s))
    )

    assert calls == [("plan", "active"), ("plan", "done"), ("cad", "active"), ("cad", "done")]


def test_generate_stream_route_serializes_phases_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the REAL /generate/stream route with auth + balance + engine stubbed (test-only), so the
    worker thread, the on_phase emit, the honest keepalive, and the SSE serialization all run with no
    network or spend. The stream must carry intent->plan->cad->export in order, then a result event."""

    def _stub_user(_authorization: str | None) -> tuple[str, str]:
        return ("test-token", "stream-test-user")

    def _stub_entitlement(_token: str) -> dict[str, Any]:
        return {
            "plan_id": "pro",
            "is_unlimited": False,
            "is_complimentary": False,
            "credits_balance": 999,
            "allowed_qualities": ["standard", "studio"],
            "per_quality_cost": {"standard": 1, "studio": 5},
            "upgrade_to": "max",
        }

    def _stub_run_agentic(
        prompt: str,
        quality: str,
        token: str,
        cost: int,
        user_id: str,
        charge: bool = True,
        on_phase: Any = None,
        *,
        printable: bool = False,
        brief_context: str | None = None,
    ) -> dict[str, Any]:
        # Stand in for the engine: drive the same phases generate_part/produce_part would, instantly.
        for key, status in (
            ("plan", "active"),
            ("plan", "done"),
            ("cad", "active"),
            ("cad", "done"),
            ("export", "active"),
        ):
            if on_phase is not None:
                on_phase(key, status)
        return {
            "status": "ok",
            "files": {"stl": "files/x/part.stl"},
            "validation": {"passed": True},
        }

    monkeypatch.setattr(server_app, "_require_user", _stub_user)
    monkeypatch.setattr(server_app, "_get_entitlement", _stub_entitlement)
    monkeypatch.setattr(server_app, "_run_agentic", _stub_run_agentic)

    client = TestClient(server_app.app)
    chunks: list[str] = []
    with client.stream(
        "POST",
        "/generate/stream",
        json={"prompt": "a bracket"},
        headers={"Authorization": "Bearer test-token"},
    ) as resp:
        assert resp.status_code == 200
        chunks.extend(resp.iter_text())
    body = "".join(chunks)

    phases: list[tuple[str, str]] = []
    results: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        event: str | None = None
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        if event is None or data is None:
            continue
        if event == "phase":
            payload = json.loads(data)
            phases.append((payload["key"], payload["status"]))
        elif event == "result":
            results.append(json.loads(data))

    # intent (worker) -> plan/cad/export (engine via on_phase) -> export done (worker); no faked cad.
    assert phases == [
        ("intent", "active"),
        ("intent", "done"),
        ("plan", "active"),
        ("plan", "done"),
        ("cad", "active"),
        ("cad", "done"),
        ("export", "active"),
        ("export", "done"),
    ]
    assert len(results) == 1 and results[0]["status"] == "ok"
