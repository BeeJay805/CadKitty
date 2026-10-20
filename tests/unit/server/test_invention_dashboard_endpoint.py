"""GET /invention/dashboard/{mission_id} (ADR-0052): the lab dashboard read-model endpoint. Flag-gated
behind FORGE_INVENTION (404 when off, so the deployed contract is byte-identical by default), 404 for an
unknown mission, and a valid MissionDashboard folded from a persisted mission's timeline. Money-free: the
mission loader is patched out / fed canned REST rows, so no live Supabase and no spend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forge_mind.invention.mission import (
    ArtifactRecord,
    EventType,
    MissionLog,
    ValidationRecord,
)
from forge_mind.learning.mission_records import rows_for_log
from forge_mind.server import app as server_app


def _built_log() -> MissionLog:
    """A small but realistic mission: a concept picked, an artifact produced, the gates passed."""
    log = MissionLog("m-1", "route two cables under a desk")
    log.append(
        EventType.CONCEPT_SELECTED, summary="picked a clip", concept_id="c1", stage="concept"
    )
    log.append(
        EventType.ARTIFACT_CREATED,
        summary="built the clip",
        stage="build",
        artifact=ArtifactRecord(artifact_id="a1", kind="glb", ref="parts/clip.glb"),
    )
    log.append(
        EventType.VALIDATION_PASSED,
        summary="geometry gates passed",
        stage="validate",
        validation=ValidationRecord(passed=True),
    )
    return log


def _get(mission_id: str) -> object:
    return TestClient(server_app.app).get(f"/invention/dashboard/{mission_id}")


@pytest.fixture()
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "invention_enabled", lambda: True)
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))


def test_dashboard_404_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "invention_enabled", lambda: False)
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    # The flag gate must short-circuit BEFORE any load, so the deployed contract is byte-identical.
    monkeypatch.setattr(server_app, "_load_mission_log", lambda mid: _built_log())
    assert _get("m-1").status_code == 404


def test_dashboard_404_when_mission_not_found(
    enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_app, "_load_mission_log", lambda mid: None)
    assert _get("missing").status_code == 404


def test_dashboard_returns_read_model(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    log = _built_log()
    monkeypatch.setattr(server_app, "_load_mission_log", lambda mid: log if mid == "m-1" else None)
    r = _get("m-1")
    assert r.status_code == 200
    body = r.json()
    assert body["mission_id"] == "m-1"
    assert body["problem"] == "route two cables under a desk"
    assert body["current_stage"] == "validate"
    assert body["validation_state"] == {"passed": 1, "failed": 0, "latest_passed": True}
    assert {"kind": "glb", "ref": "parts/clip.glb"} in body["artifact_previews"]
    assert body["latest_finding"]["event_type"] == "validation_passed"
    # The moat: the NOT-PHYSICALLY-VALIDATED risk is always present.
    assert any("Not physically validated" in risk for risk in body["unresolved_risks"])
    # v0: no InventionPackage persisted -> no package-derived maturity / evidence / banner.
    assert body["maturity"] is None
    assert body["evidence_tier"] is None
    assert body["banner"] is None


def test_load_mission_log_folds_rest_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real prod read path, money-free: canned PostgREST rows fold back into the exact log."""
    log = _built_log()
    rows = rows_for_log(log)

    class _Resp:
        status_code = 200

        def json(self) -> list[dict[str, object]]:
            return rows

    monkeypatch.setattr(server_app, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(server_app, "SUPABASE_SERVICE_ROLE_KEY", "svc-role")
    monkeypatch.setattr(server_app.httpx, "get", lambda *a, **k: _Resp())

    loaded = server_app._load_mission_log("m-1")
    assert loaded is not None
    assert loaded.state() == log.state()


def test_load_mission_log_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "SUPABASE_URL", "")
    monkeypatch.setattr(server_app, "SUPABASE_SERVICE_ROLE_KEY", "")
    assert server_app._load_mission_log("m-1") is None
