"""Exercise dependency opt-in across the actual product HTTP request boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.store import Store

RID = "b" * 32


@pytest.mark.parametrize("option", [False, True])
def test_product_http_forwards_explicit_dependency_option(tmp_path, monkeypatch, option):
    store = Store(tmp_path / "store")
    sid = store.create("owner", "Fitted reference", {})["id"]
    service = SimpleNamespace(store=store)
    calls = []

    def capture(actual_service, project_id, owner, revision_id, **options):
        actual_service.store.session_record(project_id, owner)
        calls.append((project_id, owner, revision_id, options))
        return {"include_import_assets": options["include_import_assets"]}

    monkeypatch.setattr("forge_mind.workspace.project_delivery.create_project_export", capture)
    app = FastAPI()
    app.include_router(product_router(store, service, lambda: "owner"))
    with TestClient(app) as client:
        response = client.post(
            f"/workspace/sessions/{sid}/project-exports",
            json={"revision_id": RID, "include_import_assets": option},
        )
    assert response.status_code == 200
    assert response.json() == {"include_import_assets": option}
    assert calls == [(sid, "owner", RID, {"include_stl": False, "include_import_assets": option})]


@pytest.mark.parametrize("option", ["true", 1, None])
def test_product_http_rejects_nonboolean_dependency_option(tmp_path, monkeypatch, option):
    store = Store(tmp_path / "store")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid request must not reach package creation")

    monkeypatch.setattr("forge_mind.workspace.project_delivery.create_project_export", unexpected)
    app = FastAPI()
    app.include_router(product_router(store, SimpleNamespace(store=store), lambda: "owner"))
    with TestClient(app) as client:
        response = client.post(
            f"/workspace/sessions/{'a' * 32}/project-exports",
            json={"revision_id": RID, "include_import_assets": option},
        )
    assert response.status_code == 422
