"""HTTP contract for the design-intake flow. Skipped entirely if the optional ``api`` extra
(fastapi + httpx) is not installed, so the default suite never requires it."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from forge_mind.intake.api import create_app
from forge_mind.intake.service import IntakeService


class _FakeGen:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, prompt: str, output_dir: Path) -> object:
        self.calls.append((prompt, output_dir))
        return {"ok": True}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    svc = IntakeService(asset_dir=tmp_path, generate_fn=_FakeGen())
    return TestClient(create_app(svc))


def _create(client: TestClient) -> str:
    r = client.post("/api/design-sessions")
    assert r.status_code == 200
    return r.json()["id"]


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_session_404(client: TestClient) -> None:
    assert client.get("/api/design-sessions/nope").status_code == 404


def test_create_get_and_input(client: TestClient) -> None:
    sid = _create(client)
    assert client.get(f"/api/design-sessions/{sid}").status_code == 200
    r = client.post(f"/api/design-sessions/{sid}/input", json={"text": "a bracket"})
    assert r.status_code == 200
    assert r.json()["original_text_prompt"] == "a bracket"


def test_asset_upload_base64(client: TestClient) -> None:
    sid = _create(client)
    payload = {
        "filename": "drawing.png",
        "content_type": "image/png",
        "data_base64": base64.b64encode(b"\x89PNGfake").decode(),
        "asset_type": "technical_drawing",
    }
    r = client.post(f"/api/design-sessions/{sid}/assets", json=payload)
    assert r.status_code == 200
    assert r.json()["scale_confirmed"] is False


def test_asset_upload_rejects_oversize_body() -> None:
    """Regression (audit H4): the base64 asset field is capped at ~25 MB so the open (unauthenticated)
    assets endpoint can't be used to POST an unbounded body and exhaust disk/RAM. Asserted at the
    model boundary so the test doesn't push 30+ MB through the HTTP client."""
    from pydantic import ValidationError

    from forge_mind.intake.api import _MAX_ASSET_B64_CHARS, AssetBody

    with pytest.raises(ValidationError):
        AssetBody(filename="bomb.bin", data_base64="A" * (_MAX_ASSET_B64_CHARS + 4))
    # a normal small upload still validates
    AssetBody(filename="ok.png", data_base64=base64.b64encode(b"\x89PNGok").decode())


def test_sketch_endpoint(client: TestClient) -> None:
    sid = _create(client)
    sketch = {
        "canvas_width": 800,
        "canvas_height": 600,
        "strokes": [{"id": "s1", "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}],
    }
    r = client.post(f"/api/design-sessions/{sid}/sketch", json={"sketch": sketch})
    assert r.status_code == 200


def test_planner_run_returns_questions_for_vague_request(client: TestClient) -> None:
    sid = _create(client)
    client.post(f"/api/design-sessions/{sid}/input", json={"text": "make a motor mount"})
    r = client.post(f"/api/design-sessions/{sid}/planner/run")
    body = r.json()
    assert body["ready"] is False
    assert {q["id"] for q in body["questions"]} == {
        "q_motor_size",
        "q_mounting_surface",
        "q_fastener_type",
    }


def test_generate_blocked_then_allowed(client: TestClient) -> None:
    sid = _create(client)
    client.post(f"/api/design-sessions/{sid}/input", json={"text": "make a motor mount"})
    client.post(f"/api/design-sessions/{sid}/planner/run")
    blocked = client.post(f"/api/design-sessions/{sid}/generate")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["error"] == "blocked"

    # A fully specified request goes ready -> finalize -> generate.
    sid2 = _create(client)
    client.post(
        f"/api/design-sessions/{sid2}/input",
        json={"text": "A NEMA-17 motor mount for a 2020 aluminum extrusion rail."},
    )
    run = client.post(f"/api/design-sessions/{sid2}/planner/run").json()
    assert run["ready"] is True
    assert run["prompt"] and "NEMA-17" in run["prompt"]  # standards-resolved prompt for the client
    fin = client.post(f"/api/design-sessions/{sid2}/planner/finalize")
    assert fin.status_code == 200
    assert fin.json()["ready_for_blueprint"] is True
    gen = client.post(f"/api/design-sessions/{sid2}/generate")
    assert gen.status_code == 200
    assert gen.json()["status"] == "complete"


def test_intake_routes_mounted_on_deployed_backend() -> None:
    """The same backend that serves /generate also serves the design-intake routes (router mounted
    in forge_mind.server.app), and the existing /generate route is untouched."""
    from forge_mind.server.app import app as deployed_app

    deployed = TestClient(deployed_app)
    r = deployed.post("/api/design-sessions")
    assert r.status_code == 200
    sid = r.json()["id"]
    run = deployed.post(f"/api/design-sessions/{sid}/input", json={"text": "make a motor mount"})
    assert run.status_code == 200
    paths = {getattr(route, "path", "") for route in deployed_app.routes}
    assert "/generate" in paths  # the existing prompt -> generation flow is intact
    assert "/api/design-sessions" in paths


def test_uploaded_file_is_served(client: TestClient) -> None:
    sid = _create(client)
    payload = {
        "filename": "drawing.png",
        "content_type": "image/png",
        "data_base64": base64.b64encode(b"\x89PNGhello").decode(),
        "asset_type": "technical_drawing",
    }
    asset = client.post(f"/api/design-sessions/{sid}/assets", json=payload).json()
    assert asset["preview_url"]  # populated now that the router passes base_url
    got = client.get(asset["preview_url"])
    assert got.status_code == 200
    assert got.content == b"\x89PNGhello"
    assert client.get(f"/api/design-sessions/{sid}/files/nope.png").status_code == 404


def test_auth_dependency_gates_only_generate(tmp_path: Path) -> None:
    """An auth_dependency protects the costly generate endpoint; planning stays open."""
    from fastapi import FastAPI, HTTPException

    from forge_mind.intake.api import create_router

    def deny() -> None:
        raise HTTPException(status_code=401, detail="sign in")

    app = FastAPI()
    svc = IntakeService(asset_dir=tmp_path, generate_fn=_FakeGen())
    app.include_router(create_router(svc, auth_dependency=deny))
    c = TestClient(app)

    sid = c.post("/api/design-sessions").json()["id"]
    ready = c.post(
        f"/api/design-sessions/{sid}/input",
        json={"text": "A NEMA-17 motor mount for a 2020 aluminum extrusion rail."},
    )
    assert ready.status_code == 200  # planning is open
    assert c.post(f"/api/design-sessions/{sid}/planner/run").json()["ready"] is True
    assert c.post(f"/api/design-sessions/{sid}/generate").status_code == 401  # generation gated
