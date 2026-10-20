from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.product_service import product_review, run_tool
from forge_mind.workspace.product_sources import _append
from forge_mind.workspace.store import Store, WorkspaceError


def test_isolated_tool_receipt_staleness_and_integrity(tmp_path: Path) -> None:
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("owner", "Axial fit", {"requirements": []})["id"]
    request = {
        "tool": "tolerance_stack",
        "units": "mm",
        "terms": [{"low": 2, "high": 3}, {"low": -1, "high": -0.5}],
        "allowable": {"low": 1, "high": 2.5},
    }
    receipt = run_tool(service, sid, "owner", None, request, timeout=30)
    assert receipt["result"]["status"] == "pass"
    review = product_review(service, sid, "owner", None)
    assert not review["checks"][0]["stale"]
    assert review["physical_readiness"] == "unverified"
    service.store.update(sid, contract={"requirements": [], "coverage_notes": ["Changed"]})
    assert product_review(service, sid, "owner", None)["checks"][0]["stale"]
    service.store.update(sid, contract={"requirements": []})
    _append(service.store, sid, "owner", {"kind": "artifact", "data": {"id": "source"}})
    assert product_review(service, sid, "owner", None)["checks"][0]["stale"]
    with pytest.raises(WorkspaceError, match="not found"):
        product_review(service, sid, "other-owner", None)
    with service.store.connect() as con:
        con.execute("UPDATE product_checks SET document='{}' WHERE id=?", (receipt["id"],))
    with pytest.raises(WorkspaceError, match="integrity"):
        product_review(service, sid, "owner", None)


def test_http_product_routes_are_registered_and_inputs_validated(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from forge_mind.server.workspace_app import create_app

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("local-owner", "Job", None)["id"]
    client = TestClient(create_app(service=service, local=True))
    assert client.get("/workspace/product-schema?query=fastener_stack").status_code == 200
    assert (
        client.get(f"/workspace/sessions/{sid}/product").json()["physical_readiness"]
        == "unverified"
    )
    assert (
        client.post(
            f"/workspace/sessions/{sid}/product-tools", json={"request": {"tool": "invented"}}
        ).status_code
        == 409
    )
    assert (
        client.get(
            f"/workspace/sessions/{sid}/product", headers={"Origin": "https://untrusted.example"}
        ).status_code
        == 403
    )
