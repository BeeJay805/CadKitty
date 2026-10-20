"""POST /blueprint-fields (ADR-0060): the editable-field catalog for a stored LayoutSpec, so the
web UI can render the structured blueprint's controls. Authed read; no LLM, no credit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from forge_mind.server import app as server_app

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


def _spec() -> dict[str, Any]:
    return json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    return TestClient(server_app.app)


def test_returns_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    resp = client.post(
        "/blueprint-fields", json={"spec": _spec()}, headers={"Authorization": "Bearer x"}
    )
    assert resp.status_code == 200
    fields = resp.json()["fields"]
    assert any(f["path"] == "extrusion_2020.m5_spacing_mm" for f in fields)


def test_invalid_spec_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch)
    resp = client.post(
        "/blueprint-fields",
        json={"spec": {"pattern": "nope"}},
        headers={"Authorization": "Bearer x"},
    )
    assert resp.status_code == 422
