"""The blueprint pregen + job-scoped parametric API + guest taste-build (founder, 2026-07-02).

The heavy pieces (LLM author, OCC sandbox) are stubbed at their seams; what IS proven here:
the endpoint contracts (shapes, auth, 404/422/429 behavior), the REAL PARAMS extract/override
path against a stored program.py, the fail-closed unknown-override rejection, and that the
demo route neither stores nor charges.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from forge_mind.api import draft as draft_mod
from forge_mind.api.draft import DraftError, build_into_job, job_params, rebuild_job
from forge_mind.server import app as server_app

CODE = (
    "PARAMS = {'plate_width_mm': 60.0, 'plate_thickness_mm': 6.0, 'bolt_count': 4}\n"
    "from build123d import *\n"
    "result = Box(PARAMS['plate_width_mm'], PARAMS['plate_width_mm'], PARAMS['plate_thickness_mm'])\n"
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(server_app, "_require_user", lambda authorization: ("tok", "user-1"))
    monkeypatch.setattr(server_app, "WORK_DIR", tmp_path)
    monkeypatch.setattr(server_app, "_TASTE_HITS", {})
    return TestClient(server_app.app)


def _fake_sandbox_ok(code: str, job_dir: Path, **_kw: Any) -> SimpleNamespace:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "part.stl").write_text("solid fake", encoding="utf-8")
    return SimpleNamespace(ok=True, error_type=None)


def _fake_glb(stl: Path, glb: Path) -> bool:
    glb.write_bytes(b"glTF-fake")
    return True


# --- draft core (real extract/override; sandbox faked) ------------------------------------------


def test_build_into_job_leaves_parametric_job_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(draft_mod, "run_build123d_code", _fake_sandbox_ok)
    built = build_into_job(CODE, tmp_path, glb_writer=_fake_glb)
    assert built.params == {"plate_width_mm": 60.0, "plate_thickness_mm": 6.0, "bolt_count": 4}
    assert (tmp_path / "program.py").read_text(encoding="utf-8") == CODE
    assert (tmp_path / "part.glb").is_file()
    assert job_params(tmp_path) == built.params


def test_rebuild_job_applies_float_overrides_and_freezes_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(draft_mod, "run_build123d_code", _fake_sandbox_ok)
    build_into_job(CODE, tmp_path, glb_writer=_fake_glb)
    built = rebuild_job(tmp_path, {"plate_width_mm": 80.0}, glb_writer=_fake_glb)
    assert built.params["plate_width_mm"] == 80.0
    assert built.params["plate_thickness_mm"] == 6.0
    assert built.params["bolt_count"] == 4  # ints = frozen structure
    # The stored script now carries the new value — the next rebuild starts from it.
    assert job_params(tmp_path)["plate_width_mm"] == 80.0


def test_rebuild_job_rejects_unknown_and_int_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(draft_mod, "run_build123d_code", _fake_sandbox_ok)
    build_into_job(CODE, tmp_path, glb_writer=_fake_glb)
    with pytest.raises(DraftError, match="nope_mm"):
        rebuild_job(tmp_path, {"nope_mm": 1.0}, glb_writer=_fake_glb)
    with pytest.raises(DraftError, match="bolt_count"):
        rebuild_job(tmp_path, {"bolt_count": 6.0}, glb_writer=_fake_glb)


def test_author_draft_fails_closed_without_params(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def generate_structured(self, **_kw: Any) -> tuple[Any, Any]:
            return SimpleNamespace(code="result = 1\n"), None

    with pytest.raises(DraftError, match="PARAMS"):
        draft_mod.author_draft("a plate", client=FakeLLM())


# --- endpoints ------------------------------------------------------------------------------------


def test_draft_endpoint_projects_contract(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(draft_mod, "run_build123d_code", _fake_sandbox_ok)
    monkeypatch.setattr(server_app, "_draft_glb_writer", _fake_glb)
    monkeypatch.setattr(
        draft_mod, "author_draft", lambda prompt, **_kw: (CODE, draft_mod.extract_params(CODE))
    )
    r = client.post("/draft", json={"prompt": "a NEMA-17 plate"})
    assert r.status_code == 200
    body = r.json()
    assert body["draft"] is True
    assert body["params"]["plate_width_mm"] == 60.0
    assert body["files"]["glb"] == f"files/{body['job']}/part.glb"


def test_job_params_and_rebuild_endpoints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(draft_mod, "run_build123d_code", _fake_sandbox_ok)
    monkeypatch.setattr(server_app, "_draft_glb_writer", _fake_glb)
    job = "abc123"
    (tmp_path / job).mkdir()
    (tmp_path / job / "program.py").write_text(CODE, encoding="utf-8")
    from forge_mind.server.job_ownership import claim

    claim(tmp_path, job, "user-1")

    r = client.get(f"/jobs/{job}/params")
    assert r.status_code == 200
    assert r.json()["params"]["plate_thickness_mm"] == 6.0

    r = client.post(f"/jobs/{job}/rebuild", json={"overrides": {"plate_thickness_mm": 9.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["params"]["plate_thickness_mm"] == 9.0
    assert body["stl_url"] == f"files/{job}/part.stl"
    assert body["validation"] is None  # NO validator ran on the rebuilt mesh — never inherited

    r = client.post(f"/jobs/{job}/rebuild", json={"overrides": {"bogus_mm": 1.0}})
    assert r.status_code == 422

    assert client.get("/jobs/missing00/params").status_code == 404
    assert client.get("/jobs/../etc/params").status_code == 404


def test_job_edit_rejects_another_owner(client: TestClient, tmp_path: Path) -> None:
    from forge_mind.server.job_ownership import claim

    job = "anotherusersjob"
    (tmp_path / job).mkdir()
    (tmp_path / job / "program.py").write_text(CODE, encoding="utf-8")
    claim(tmp_path, job, "user-2")
    assert client.get(f"/jobs/{job}/params").status_code == 404
    assert (
        client.post(f"/jobs/{job}/rebuild", json={"overrides": {"plate_width_mm": 1.0}}).status_code
        == 404
    )
    assert (tmp_path / job / "program.py").read_text(encoding="utf-8") == CODE


def test_generate_demo_rate_limits_and_never_stores(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored: list[Any] = []
    monkeypatch.setattr(server_app, "_store_generation", lambda *a, **k: stored.append(a))
    monkeypatch.setattr(
        server_app,
        "_run_agentic",
        lambda prompt, quality, token, cost, user_id, charge, printable, store: {
            "status": "ok",
            "echo": {
                "token": token,
                "cost": cost,
                "charge": charge,
                "store": store,
                "quality": quality,
            },
        },
    )
    r = client.post("/generate-demo", json={"prompt": "a bracket"})
    assert r.status_code == 200
    body = r.json()
    assert body["demo"] is True
    assert body["echo"] == {
        "token": None,
        "cost": 0,
        "charge": False,
        "store": False,
        "quality": "standard",
    }
    assert stored == []  # nothing written to the generations record

    # Second call from the same IP inside the window: the free build is spent.
    assert client.post("/generate-demo", json={"prompt": "another"}).status_code == 429
