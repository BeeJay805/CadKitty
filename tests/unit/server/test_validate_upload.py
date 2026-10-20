"""POST /validate-upload (ADR-0056 Flavor A) — proven money-free against the ACTUAL FastAPI app.

The honest contract: a real STL comes back with an intent-free pre-print verdict that NEVER claims
fit/function; junk / wrong-type / oversized uploads are rejected (fail-closed at the boundary)."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

from forge_mind.physics.fea import StressField
from forge_mind.server import app as server_app


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(server_app, "_upload_artifact", lambda *a, **k: None)
    # FEA is validated in test_fea.py; keep these endpoint tests fast + deterministic (no solver).
    monkeypatch.setattr(server_app, "estimate_stress", lambda *a, **k: None)
    return TestClient(server_app.app)


def _stl_bytes() -> bytes:
    data = trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(file_type="stl")
    return data if isinstance(data, bytes) else bytes(data, "utf-8")


def _hdr() -> dict[str, str]:
    return {"Authorization": "Bearer x"}


def test_valid_stl_returns_intrinsic_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _client(monkeypatch, tmp_path)
    resp = client.post(
        "/validate-upload", files={"file": ("part.stl", _stl_bytes(), "model/stl")}, headers=_hdr()
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    report = body["report"]
    # honest ceiling — never claims fit/function, never says "safe".
    assert report["fit_function_verified"] is False
    assert "NOT verified" in report["verdict"]
    assert "safe" not in report["verdict"].lower()
    assert report["printable"] in {"pass", "warn", "fail"}
    # a viewer GLB is produced so the Wave 1 lenses can render it.
    assert "glb" in body["files"]
    # viewer metrics in the PartMetrics shape (Y-up CoM + stand-or-tip) so the lenses fire on the upload.
    assert body["metrics"] is not None
    assert body["metrics"]["center_of_mass_mm"] is not None
    assert body["metrics"]["stability"] is not None
    assert body["metrics"]["stability"]["stable"] is True
    # the FEA stress summary key is wired (null here since the solver is patched off in _client).
    assert "stress" in body and body["stress"] is None


def test_stress_summary_is_wired_when_fea_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When estimate_stress returns a field, the endpoint surfaces the honest summary (never a verdict).
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(server_app, "_upload_artifact", lambda *a, **k: None)
    fake = StressField(
        von_mises=(1.0, 2.0, 3.0),
        von_mises_max=3.0,
        von_mises_p95=2.9,
        max_location_mm=(1.0, 2.0, 3.0),
        load_case="base fixed, unit downward push on the top face",
        material_e_mpa=3500.0,
        note="Relative stress under a generic load — NOT a safety rating.",
    )
    monkeypatch.setattr(server_app, "estimate_stress", lambda *a, **k: fake)
    client = TestClient(server_app.app)
    resp = client.post(
        "/validate-upload", files={"file": ("part.stl", _stl_bytes(), "model/stl")}, headers=_hdr()
    )
    assert resp.status_code == 200
    stress = resp.json()["stress"]
    assert stress is not None and stress["available"] is True
    assert stress["von_mises_max"] == 3.0
    assert "safe" not in stress["load_case"].lower()
    assert "NOT a safety rating" in stress["note"]


def test_non_stl_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    resp = client.post(
        "/validate-upload", files={"file": ("notes.txt", b"hello", "text/plain")}, headers=_hdr()
    )
    assert resp.status_code == 400


def test_garbage_stl_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    resp = client.post(
        "/validate-upload",
        files={"file": ("bad.stl", b"this is not an stl mesh", "model/stl")},
        headers=_hdr(),
    )
    assert resp.status_code == 400


def test_oversized_upload_is_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(server_app, "_MAX_UPLOAD_BYTES", 16)
    resp = client.post(
        "/validate-upload", files={"file": ("part.stl", _stl_bytes(), "model/stl")}, headers=_hdr()
    )
    assert resp.status_code == 413


def _demo_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    # No _require_user patch -- the demo endpoint is public by design.
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(server_app, "estimate_stress", lambda *a, **k: None)
    monkeypatch.setattr(server_app, "_DEMO_HITS", {})  # reset the in-memory limiter per test
    return TestClient(server_app.app)


def test_demo_no_signin_returns_honest_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _demo_client(monkeypatch, tmp_path)
    # NO Authorization header -- the public funnel.
    resp = client.post(
        "/validate-upload-demo", files={"file": ("part.stl", _stl_bytes(), "model/stl")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("demo") is True
    report = body["report"]
    assert report["fit_function_verified"] is False  # same honest ceiling as the authed path
    assert "safe" not in report["verdict"].lower()


def test_demo_is_rate_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server_app, "_DEMO_MAX_PER_HOUR", 2)
    client = _demo_client(monkeypatch, tmp_path)

    def _hit() -> int:
        return client.post(
            "/validate-upload-demo", files={"file": ("part.stl", _stl_bytes(), "model/stl")}
        ).status_code

    assert _hit() == 200
    assert _hit() == 200
    assert _hit() == 429  # third request from the same client exceeds the 2/hour demo cap
