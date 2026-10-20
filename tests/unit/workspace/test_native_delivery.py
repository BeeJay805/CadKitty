"""Automatic handoff guards use saved files; no CAD or author calls are needed."""

from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from forge_mind.workspace import native_delivery as delivery

RID = "a" * 32
SID = "b" * 32
CONTENT = b"owned saved package"


def prepared(monkeypatch, *, requested=True, mesh=False):
    document = {
        "name": "Saved part",
        "features": [],
        "production": {"role": "make", "deliver_stl": mesh},
    }
    revision = {
        "id": RID,
        "document": document,
        "measurement": {
            "geometry_valid": True,
            "engine_fingerprint": "export-engine",
            "artifact_hashes": {"part.step": "d" * 64},
        },
    }
    session = {
        "authoring_mode": "native_operations",
        "selected_revision": RID,
        "accepted_revision": None,
        "state": "needs_review",
        "engineering": {"version": 1, "requirements": []},
    }
    updates = []

    def get(sid, owner):
        assert (sid, owner) == (SID, "owner")
        return copy.deepcopy(session)

    def update(sid, **values):
        assert sid == SID
        updates.append(copy.deepcopy(values))
        session.update(copy.deepcopy(values))

    service = SimpleNamespace(
        store=SimpleNamespace(get=get, update=update),
        _load_revision=lambda sid, rid: copy.deepcopy(revision),
    )
    receipt = {
        "project_id": SID,
        "revision_id": RID,
        "package_id": "c" * 32,
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "bytes": len(CONTENT),
    }
    captures = []
    reads = []

    def capture(*args, **kwargs):
        captures.append((args, kwargs))
        return copy.deepcopy(receipt)

    def read(*args):
        reads.append(args)
        return CONTENT, copy.deepcopy(receipt)

    sources = {"fingerprint": "sources"}

    def snapshot(service, sid, owner, rid):
        current = service.store.get(sid, owner)
        return {
            "revision_id": rid,
            "engine_fingerprint": "export-engine",
            "artifact_hashes": copy.deepcopy(revision["measurement"]["artifact_hashes"]),
            "source_fingerprint": sources["fingerprint"],
            "input_version": current["engineering"]["version"],
            "contract_hash": "contract-hash",
            "session_contract_hash": "contract-hash",
            "plan_hash": "plan-hash",
        }

    monkeypatch.setattr(delivery, "_snapshot", snapshot)
    monkeypatch.setattr(delivery, "create_project_export", capture)
    monkeypatch.setattr(delivery, "read_project_export", read)
    job = {"id": "job", "session": SID, "owner": "owner", "delivery_requested": requested}
    return SimpleNamespace(
        service=service,
        job=job,
        session=session,
        revision=revision,
        receipt=receipt,
        captures=captures,
        reads=reads,
        updates=updates,
        sources=sources,
    )


@pytest.mark.parametrize("requested,mesh", [(True, False), (False, True)])
def test_requested_saved_capture_preserves_author_and_approval_state(monkeypatch, requested, mesh):
    fixture = prepared(monkeypatch, requested=requested, mesh=mesh)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "budget_exhausted"
    )
    assert report["status"] == "delivered"
    assert report["author_state"] == "budget_exhausted"
    assert report["readback_verified"] is True
    assert report["artifact_hashes"] == fixture.revision["measurement"]["artifact_hashes"]
    assert report["engine_fingerprint"] == "export-engine"
    assert fixture.captures[0][1]["include_stl"] is mesh
    assert fixture.session["state"] == "needs_review"
    assert fixture.session["accepted_revision"] is None
    assert all(set(update) == {"automatic_delivery"} for update in fixture.updates)


def test_guidance_update_after_capture_invalidates_delivery_readback(monkeypatch):
    fixture = prepared(monkeypatch)
    fixture.session["delivery_guidance_record"] = {"guidance": "first reviewed text"}

    def capture(*args, **kwargs):
        fixture.session["delivery_guidance_record"] = {"guidance": "new reviewed text"}
        return {**fixture.receipt, "guidance_status": "current"}

    monkeypatch.setattr(delivery, "create_project_export", capture)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert "changed after capture" in report["reason"]
    assert report["receipt"] is not None
    assert fixture.reads == []


@pytest.mark.parametrize(
    "requested,state,cancelled,deadline,expected",
    [
        (False, "needs_review", False, None, "not_requested"),
        (True, "needs_review", True, None, "cancelled"),
        (True, "needs_review", False, 0.0, "budget_exhausted"),
        (True, "error", False, None, "incomplete"),
        (True, "cancelled", False, None, "incomplete"),
    ],
)
def test_guarded_phases_do_not_capture(
    monkeypatch, requested, state, cancelled, deadline, expected
):
    fixture = prepared(monkeypatch, requested=requested)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, deadline, lambda: cancelled, state
    )
    assert report["status"] == expected
    assert fixture.captures == fixture.reads == []


def test_numeric_edit_without_new_candidate_does_not_deliver_old_files(monkeypatch):
    fixture = prepared(monkeypatch)
    fixture.job.update(parameters={"spacing": 44}, based_on=RID)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert fixture.captures == []


@pytest.mark.parametrize("after", ["capture", "readback"])
def test_cancellation_keeps_exact_package_receipt_without_claiming_delivery(monkeypatch, after):
    fixture = prepared(monkeypatch)
    report = delivery.capture_native_delivery(
        fixture.service,
        fixture.job,
        None,
        lambda: bool(fixture.captures if after == "capture" else fixture.reads),
        "needs_review",
    )
    assert report["status"] == "cancelled"
    assert report["receipt"] == fixture.receipt
    assert "readback_verified" not in report
    captured = [u["automatic_delivery"] for u in fixture.updates]
    assert any(r["status"] == "captured" and r["receipt"] == fixture.receipt for r in captured)


def test_changed_selected_revision_retains_capture_but_does_not_deliver(monkeypatch):
    fixture = prepared(monkeypatch)

    def capture(*args, **kwargs):
        fixture.session["selected_revision"] = "e" * 32
        return copy.deepcopy(fixture.receipt)

    monkeypatch.setattr(delivery, "create_project_export", capture)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert report["receipt"] == fixture.receipt
    assert fixture.reads == []


def test_readback_failure_retains_captured_identity(monkeypatch):
    fixture = prepared(monkeypatch)

    def read(*args):
        raise ValueError("Saved checksum changed")

    monkeypatch.setattr(delivery, "read_project_export", read)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "failed"
    assert report["receipt"] == fixture.receipt
    assert "Saved checksum changed" in report["reason"]


@pytest.mark.parametrize("change", [None, "source", "engineering"])
def test_existing_owned_package_services_work_without_generation(tmp_path, monkeypatch, change):
    from tests.unit.workspace.test_project_delivery import saved_project

    from forge_mind.workspace.store import Store

    create = Store.create

    def create_native(store, *args, **kwargs):
        return create(store, *args, **{**kwargs, "authoring_mode": "native_operations"})

    monkeypatch.setattr(Store, "create", create_native)
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    service.store.update(sid, state="needs_review")
    from forge_mind.workspace import (
        completion,
        delivery_guidance,
        product_sources,
        project_delivery,
    )

    monkeypatch.setattr(completion, "engine_fingerprint", lambda: "fixture-engine")
    monkeypatch.setattr(project_delivery, "engine_fingerprint", lambda: "fixture-engine")
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "fixture-engine")
    if change == "source":
        monkeypatch.setattr(
            product_sources,
            "retrieve",
            lambda *args: (b"Fixture thickness 3 mm", "text/plain", "https://fixture.example/a"),
        )
        sources = product_sources.fetch_source(service, sid, "owner", "https://fixture.example/a")
        claim = {
            "source_artifact": sources["artifacts"][0]["id"],
            "quantity": 3,
            "unit": "mm",
            "exact_quote": "Fixture thickness 3 mm",
            "configuration": "Test fixture",
            "datum": "Front to rear planes",
            "locator": "Retained fixture text",
        }
    capture = delivery.create_project_export

    def capture_then_change(*args, **kwargs):
        receipt = capture(*args, **kwargs)
        if change == "source":
            product_sources.propose_claim(service.store, sid, "owner", claim)
        elif change == "engineering":
            current = service.store.engineering(sid)
            service.store.save_engineering(sid, "owner", current["version"], current["inputs"])
        return receipt

    monkeypatch.setattr(delivery, "create_project_export", capture_then_change)
    report = delivery.capture_native_delivery(
        service,
        {"id": "job", "session": sid, "owner": "owner", "delivery_requested": True},
        None,
        lambda: False,
        "needs_review",
    )
    assert report["status"] == ("incomplete" if change else "delivered"), report["reason"]
    assert report["receipt"]["revision_id"] == service.store.get(sid, "owner")["selected_revision"]
    assert service.store.get(sid, "owner")["accepted_revision"] is None


@pytest.mark.parametrize("after", ["capture", "readback"])
def test_expired_budget_retains_package_without_reporting_delivery(monkeypatch, after):
    fixture = prepared(monkeypatch)
    monkeypatch.setattr(
        delivery,
        "expired",
        lambda deadline: bool(fixture.captures if after == "capture" else fixture.reads),
    )
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, 100.0, lambda: False, "needs_review"
    )
    assert report["status"] == "budget_exhausted"
    assert report["receipt"] == fixture.receipt
    assert "readback_verified" not in report


def test_no_valid_geometry_is_an_incomplete_delivery(monkeypatch):
    fixture = prepared(monkeypatch)
    fixture.revision["measurement"]["geometry_valid"] = False
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert fixture.captures == []


def test_engine_mismatch_refuses_capture(monkeypatch):
    fixture = prepared(monkeypatch)
    fixture.revision["measurement"]["engine_fingerprint"] = "old-engine"
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert "different engine" in report["reason"]
    assert fixture.captures == []


@pytest.mark.parametrize("changed", ["source", "engineering"])
def test_source_or_engineering_change_after_capture_keeps_package_but_not_delivery(
    monkeypatch, changed
):
    fixture = prepared(monkeypatch)

    def capture(*args, **kwargs):
        if changed == "source":
            fixture.sources["fingerprint"] = "updated-claim-set"
        else:
            fixture.session["engineering"]["requirements"].append("new input")
        return copy.deepcopy(fixture.receipt)

    monkeypatch.setattr(delivery, "create_project_export", capture)
    report = delivery.capture_native_delivery(
        fixture.service, fixture.job, None, lambda: False, "needs_review"
    )
    assert report["status"] == "incomplete"
    assert report["receipt"] == fixture.receipt
    assert fixture.reads == []
