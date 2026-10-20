"""HTTP photo routing and capture ownership; no real graphics or CAD execution."""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.project_delivery import create_project_export, read_project_export


@pytest.fixture
def capture(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    receipt = create_project_export(service, sid, "owner", RID)
    base = f"/workspace/sessions/{sid}/project-exports/{receipt['package_id']}/{receipt['sha256']}"
    return service, sid, receipt, base


def client_for(service, owner):
    app = FastAPI()
    app.include_router(product_router(service.store, service, lambda: owner))
    return TestClient(app)


def test_missing_captured_mesh_preserves_owned_download(capture):
    service, sid, receipt, base = capture
    with client_for(service, "owner") as client:
        result = client.post(base + "/photo", json={})
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "unavailable"
        assert "image_base64" not in result.json()
        downloaded = client.get(base + ".zip")
        assert downloaded.status_code == 200
        assert hashlib.sha256(downloaded.content).hexdigest() == receipt["sha256"]
    with client_for(service, "other") as client:
        denied = client.post(base + "/photo", json={})
        assert denied.status_code == 409
        assert "image_base64" not in denied.text
    assert (
        read_project_export(service, sid, "owner", receipt["package_id"], receipt["sha256"])[1]
        == receipt
    )


@pytest.mark.parametrize(
    "body",
    [
        {"preset": "invented"},
        {"preset": True},
        {"part_id": "../reference"},
        {"part_id": 7},
        {"extra": "unsupported"},
    ],
)
def test_invalid_photo_request_does_not_invoke_renderer(capture, monkeypatch, body):
    from forge_mind.workspace import project_photo

    monkeypatch.setattr(
        project_photo, "create_project_photo", lambda *a, **k: pytest.fail("invalid request ran")
    )
    service, _, _, base = capture
    with client_for(service, "owner") as client:
        assert client.post(base + "/photo", json=body).status_code == 422


def test_photo_download_and_cleanup_keep_owner_and_package_binding(capture, monkeypatch):
    from forge_mind.workspace import project_photo

    service, sid, receipt, base = capture
    photo_id, image_sha = "a" * 32, "b" * 64
    calls = []

    def read(svc, project_id, owner, package_id, package_sha, requested_photo, requested_hash):
        read_project_export(svc, project_id, owner, package_id, package_sha)
        assert (requested_photo, requested_hash) == (photo_id, image_sha)
        return b"verified image supplied by service", {}

    monkeypatch.setattr(project_photo, "read_project_photo", read)
    monkeypatch.setattr(project_photo, "discard_project_photo", lambda *args: calls.append(args))
    image_route = f"{base}/photos/{photo_id}/{image_sha}.png"
    cleanup_route = f"{base}/photos/{photo_id}"
    with client_for(service, "other") as client:
        assert client.get(image_route).status_code == 409
        assert client.delete(cleanup_route).status_code == 409
        assert not calls
    with client_for(service, "owner") as client:
        image = client.get(image_route)
        assert image.status_code == 200
        assert image.content == b"verified image supplied by service"
        assert image.headers["content-type"] == "image/png"
        assert image.headers["cache-control"] == "private, no-store"
        assert client.delete(cleanup_route.replace(receipt["sha256"], "c" * 64)).status_code == 409
        assert not calls
        assert client.delete(cleanup_route).status_code == 204
    assert calls == [(service, sid, "owner", receipt["package_id"], photo_id)]
