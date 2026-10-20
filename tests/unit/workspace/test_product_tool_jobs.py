"""Product probes use durable jobs without replacing an unfinished design."""

import asyncio
import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.product_service import read_tool_receipt
from forge_mind.workspace.store import Store, WorkspaceError


def request():
    return {
        "tool": "tolerance_stack",
        "units": "mm",
        "terms": [{"low": 2, "high": 3}],
        "allowable": {"low": 2, "high": 3},
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    sid = service.store.create("local-owner", "Preserved candidate", {"requirements": []})["id"]
    baseline, candidate = "a" * 32, "b" * 32
    proposal = {
        "revision_id": candidate,
        "based_on": baseline,
        "eligible": True,
        "contract": {"requirements": [], "coverage_notes": ["Clamping unverified"]},
    }
    service.store.update(
        sid, selected_revision=baseline, accepted_revision=baseline, proposal=proposal
    )
    service.store.add_revision(
        sid, {"id": baseline, "contract": {"requirements": []}, "measurement": {}}
    )
    service.store.add_revision(
        sid,
        {"id": candidate, "contract": proposal["contract"], "measurement": {"artifact_hashes": {}}},
    )
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    return (
        TestClient(create_app(service=service, local=True)),
        service,
        sid,
        baseline,
        candidate,
        proposal,
    )


def test_probe_admission_does_not_execute_or_expand_geometry(workspace, monkeypatch):
    client, service, sid, baseline, candidate, proposal = workspace
    from forge_mind.workspace import product_service

    started, release = threading.Event(), threading.Event()

    def slow_probe(directory, payload, timeout, cancelled):
        assert timeout is None
        started.set()
        assert release.wait(5)
        return {
            "status": "pass",
            "qualification_status": "unverified",
            "results": {"sum_mm": [2, 3]},
        }

    monkeypatch.setattr(product_service, "process_revision", slow_probe)
    queued = client.post(
        f"/workspace/sessions/{sid}/product-tool-jobs",
        json={"revision_id": candidate, "request": request()},
    )
    assert queued.status_code == 202, queued.text
    assert not started.is_set()
    identity = queued.json()
    assert len(queued.content) < 1024
    job = service.store.claim()
    assert job["id"] == identity["job_id"]
    assert "preview_only" not in job
    thread = threading.Thread(target=service.run_job, args=(job,))
    thread.start()
    try:
        assert started.wait(5)
        response = client.get(f"/workspace/sessions/{sid}/overview")
        assert response.status_code == 200, response.text
        overview = response.json()
        assert overview["state"] == "running"
        assert overview["last_product_probe"]["check_id"] == identity["check_id"]
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    saved = service.store.get(sid, "local-owner")
    assert saved["state"] == "needs_review"
    assert saved["proposal"] == proposal
    assert saved["selected_revision"] == saved["accepted_revision"] == baseline
    assert saved["contract"] == {"requirements": []}
    result = client.get(f"/workspace/sessions/{sid}/product-checks/{identity['check_id']}")
    assert result.status_code == 200, result.text
    assert result.json()["job_id"] == identity["job_id"]
    assert result.json()["contract"] == proposal["contract"]
    assert result.json()["result"]["qualification_status"] == "unverified"
    with pytest.raises(WorkspaceError, match="not found"):
        read_tool_receipt(service, sid, "other-owner", identity["check_id"])
    with service.store.connect() as con:
        con.execute("UPDATE product_checks SET document='{}' WHERE id=?", (identity["check_id"],))
    assert (
        client.get(f"/workspace/sessions/{sid}/product-checks/{identity['check_id']}").status_code
        == 409
    )


def test_invalid_probe_is_rejected_before_queue_admission(workspace):
    client, service, sid, *_ = workspace
    response = client.post(
        f"/workspace/sessions/{sid}/product-tool-jobs", json={"request": {"tool": "invented"}}
    )
    assert response.status_code == 422
    assert service.store.claim() is None
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_late_cancellation_reaches_worker_and_prevents_receipt(workspace, monkeypatch):
    client, service, sid, baseline, candidate, proposal = workspace
    from forge_mind.workspace import product_service

    def cancelled_probe(directory, payload, timeout, cancelled):
        assert not cancelled()
        service.store.cancel(sid, "local-owner")
        assert cancelled()
        return {"status": "pass"}

    monkeypatch.setattr(product_service, "process_revision", cancelled_probe)
    queued = client.post(
        f"/workspace/sessions/{sid}/product-tool-jobs",
        json={"revision_id": candidate, "request": request()},
    ).json()
    service.run_job(service.store.claim())
    saved = service.store.get(sid, "local-owner")
    assert saved["state"] == "cancelled"
    assert saved["proposal"] == proposal
    assert saved["selected_revision"] == baseline
    assert (
        client.get(f"/workspace/sessions/{sid}/product-checks/{queued['check_id']}").status_code
        == 409
    )


def test_historical_receipt_distinguishes_current_design_from_staleness(workspace, monkeypatch):
    client, service, sid, baseline, candidate, proposal = workspace
    from forge_mind.workspace import product_service

    monkeypatch.setattr(product_service, "process_revision", lambda *args: {"status": "pass"})
    queued = client.post(
        f"/workspace/sessions/{sid}/product-tool-jobs",
        json={"revision_id": candidate, "request": request()},
    ).json()
    service.run_job(service.store.claim())

    def receipt():
        response = client.get(f"/workspace/sessions/{sid}/product-checks/{queued['check_id']}")
        assert response.status_code == 200, response.text
        return response.json()

    initial = receipt()
    assert initial["current_design_revision"] == candidate
    assert initial["matches_current_design"] is True
    assert initial["stale"] is False

    newer = "c" * 32
    service.store.add_revision(
        sid,
        {"id": newer, "contract": proposal["contract"], "measurement": {"artifact_hashes": {}}},
    )
    service.store.update(sid, proposal={**proposal, "revision_id": newer})
    historical = receipt()
    assert historical["revision_id"] == candidate
    assert historical["current_design_revision"] == newer
    assert historical["matches_current_design"] is False
    assert historical["stale"] is False

    service.store.update(sid, proposal=None)
    assert receipt()["current_design_revision"] == baseline
    assert receipt()["matches_current_design"] is False
    service.store.update(sid, selected_revision=candidate)
    assert receipt()["matches_current_design"] is True

    monkeypatch.setattr(product_service, "engine_fingerprint", lambda: "changed-engine")
    assert receipt()["stale"] is True
    assert receipt()["matches_current_design"] is True


def test_cancellation_between_callback_and_insert_prevents_receipt(workspace, monkeypatch):
    client, service, sid, baseline, candidate, proposal = workspace
    from forge_mind.workspace import product_service

    monkeypatch.setattr(product_service, "process_revision", lambda *args: {"status": "pass"})
    queued = client.post(
        f"/workspace/sessions/{sid}/product-tool-jobs",
        json={"revision_id": candidate, "request": request()},
    ).json()
    original_table = product_service._table
    reached_publication = []

    def cancel_before_insert(target):
        # This hook runs after the callback check and receipt serialization.
        original_table(target)
        reached_publication.append(True)
        target.store.cancel(sid, "local-owner")

    monkeypatch.setattr(product_service, "_table", cancel_before_insert)
    service.run_job(service.store.claim())
    assert reached_publication == [True]
    saved = service.store.get(sid, "local-owner")
    assert saved["state"] == "cancelled"
    assert saved["proposal"] == proposal
    assert saved["selected_revision"] == saved["accepted_revision"] == baseline
    assert saved["contract"] == {"requirements": []}
    with service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM product_checks").fetchone()[0] == 0
    monkeypatch.setattr(product_service, "_table", original_table)
    assert (
        client.get(f"/workspace/sessions/{sid}/product-checks/{queued['check_id']}").status_code
        == 409
    )


def test_product_probe_mcp_uses_immediate_queue_and_specific_receipt(tmp_path):
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    sid, rid, cid, jid = (c * 32 for c in "abcd")
    seen = []

    def backend(req):
        seen.append(req)
        if req.method == "POST":
            assert req.url.path.endswith("/product-tool-jobs")
            assert json.loads(req.content)["request"]["tool"] == "tolerance_stack"
            return httpx.Response(202, json={"check_id": cid, "job_id": jid, "state": "queued"})
        assert req.url.path.endswith("/product-checks/" + cid)
        return httpx.Response(200, json={"id": cid, "job_id": jid, "result": {"status": "pass"}})

    async def scenario():
        async with Client(
            create_server(
                WorkspaceAdapter(
                    "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
                )
            )
        ) as client:
            queued = await client.call_tool(
                "run_product_check", {"project_id": sid, "revision_id": rid, "request": request()}
            )
            assert not queued.is_error
            assert queued.structured_content["state"] == "queued"
            receipt = await client.call_tool(
                "get_product_check", {"project_id": sid, "check_id": cid}
            )
            assert not receipt.is_error
            assert receipt.structured_content["job_id"] == jid

    asyncio.run(scenario())
    assert len(seen) == 2
