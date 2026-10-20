from __future__ import annotations

from fastapi.testclient import TestClient

from forge_mind.server.workspace_runtime import create_app
from forge_mind.workspace.hosting import HostConfig


def test_candidate_liveness_never_implies_ready_or_admission(tmp_path):
    config = HostConfig(
        tmp_path / "store",
        tmp_path / "spool",
        tmp_path / "runner.sock",
        "sha256:" + "a" * 64,
        "private1",
        "b" * 64,
        ("https://cadkitty.com",),
        "https://auth.example/auth/v1",
    )
    # No lifespan or process starts. This tests startup-negative ASGI behavior only.
    app = create_app(config)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["scope"] == "process_liveness_only"
    assert client.get("/ready").status_code == 503
    assert client.post("/workspace/customer-projects", json={}).status_code == 503
    assert client.get("/version").json()["scope"] == "private_candidate"
