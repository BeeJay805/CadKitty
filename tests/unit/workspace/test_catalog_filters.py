from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

from forge_mind.agentic.catalog.registry import catalog_counts
from forge_mind.workspace.engineering_api import engineering_router
from forge_mind.workspace.hardware import search


def test_filters_apply_before_limit_and_preserve_sources():
    results = search("", family="screw", subtype="socket_head_cap_screw", thread_d=3, length=20)
    assert results
    assert all(
        p["family"] == "screw"
        and p["subtype"] == "socket_head_cap_screw"
        and p["dims_mm"]["thread_d"] == 3
        and p["dims_mm"]["length"] == 20
        and p["source"]
        for p in results
    )
    assert (
        search("", family="screw", thread_d=3, length=20, limit=1)
        == search("", family="screw", thread_d=3, length=20)[:1]
    )
    assert search("", family="screw", thread_d=3.001, length=20) == []
    assert search("", family="washer", thread_d=3) == []
    washers = search("m3", family="washer", limit=100)
    assert len(washers) > 1
    assert all(part["family"] == "washer" for part in washers)


def test_catalog_api_preserves_list_and_exposes_counts():
    app = FastAPI()
    app.include_router(engineering_router(Mock(), Mock(), lambda: "owner"))
    with TestClient(app) as client:
        assert client.get("/workspace/catalog").json() == jsonable_encoder(search("bearing"))
        assert client.get("/workspace/catalog?q=washer").json() == jsonable_encoder(
            search("washer")
        )
        response = client.get(
            "/workspace/catalog",
            params={"family": "screw", "thread_d": 3, "length": 20, "limit": 2},
        )
        assert response.status_code == 200
        assert response.json() == jsonable_encoder(
            search("", family="screw", thread_d=3, length=20, limit=2)
        )
        assert client.get("/workspace/catalog-counts").json() == catalog_counts()
        assert client.get("/workspace/catalog?limit=501").status_code == 422
        assert client.get("/workspace/catalog?length=-1").status_code == 422
        assert client.get("/workspace/catalog?thread_d=nan").status_code == 422
