from __future__ import annotations

import json
import time

import pytest

from forge_mind.workspace.hosting import HostConfig, HostingError
from forge_mind.workspace.hosting_budget import grant
from forge_mind.workspace.hosting_identity import canonical_principal
from forge_mind.workspace.hosting_store import HostingStore
from forge_mind.workspace.store import Store, WorkspaceError


def policy(tmp_path):
    return HostConfig(
        tmp_path / "store",
        tmp_path / "spool",
        tmp_path / "runner.sock",
        "sha256:" + "a" * 64,
        "private1",
        "b" * 64,
        ("https://cadkitty.com",),
        "https://auth.example/auth/v1",
    )


def funded(tmp_path):
    config = policy(tmp_path)
    store = HostingStore(config)
    owner = canonical_principal(config.issuer, "alice")
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        grant(con, owner, "operator", 100)
    return store, owner


def test_customer_request_and_reservation_are_one_atomic_admission(tmp_path):
    config = policy(tmp_path)
    store = HostingStore(config)
    owner = canonical_principal(config.issuer, "alice")
    request = {
        "request_id": "a" * 32,
        "prompt": "A nominal plate",
        "max_seconds": 10,
        "max_actions": 4,
    }
    with pytest.raises(WorkspaceError, match="Insufficient"):
        store.create_customer_project(owner, request)
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM customer_requests").fetchone()[0] == 0


def test_success_replay_and_original_lineage_survive_store_restart(tmp_path):
    store, owner = funded(tmp_path)
    request = {
        "request_id": "a" * 32,
        "prompt": "A nominal plate",
        "max_seconds": 10,
        "max_actions": 4,
    }
    session = store.create_customer_project(owner, request)
    assert store.create_customer_project(owner, request)["id"] == session["id"]
    job = store.claim()
    assert job is not None
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("UPDATE jobs SET started=? WHERE id=?", (time.time() - 2, job["id"]))
    store.finish(job["id"], session["id"], "budget_exhausted")
    restarted = HostingStore(store.host_config)
    continuation = restarted.enqueue(
        session["id"], owner, {"resume": True, "max_seconds": 10, "max_actions": 4}
    )
    with restarted.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (continuation,)).fetchone()[0]
        )
        assert payload["host_lineage_id"] == job["id"]
        assert 0 < payload["max_seconds"] <= 8
        assert con.execute("SELECT count(*) FROM host_grants").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM host_reservations").fetchone()[0] == 2


def test_uncertain_runner_authority_prevents_refund_and_next_claim(tmp_path):
    store, owner = funded(tmp_path)
    session = store.create(owner, "A part", None)
    jid = store.enqueue(session["id"], owner, {"max_seconds": 10, "max_actions": 4})
    store.claim()
    with store.active_job(jid):
        store.hold_uncertain("Lost runner response for operation " + "a" * 32)
    store.finish(jid, session["id"], "error")
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT used_seconds FROM host_reservations WHERE job_id=?", (jid,)
            ).fetchone()[0]
            is None
        )
    assert store.claim() is None


def test_private_host_never_silently_orphans_legacy_owners(tmp_path):
    config = policy(tmp_path)
    legacy = Store(config.root)
    legacy.create("raw-subject", "Existing project", None)
    with pytest.raises(HostingError, match="fresh Store"):
        HostingStore(config)
    assert legacy.list_sessions("raw-subject")


def test_direct_observer_uncertainty_is_durable_without_any_job(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from forge_mind.workspace.hosting import readiness

    store, owner = funded(tmp_path)
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    monkeypatch.setenv("FORGE_CODEX_MODEL", store.host_config.provider_model)
    operation_id = "a" * 32
    store.hold_uncertain("Lost observer response for operation " + operation_id)
    restarted = HostingStore(store.host_config)
    with pytest.raises(WorkspaceError, match="uncertainty"):
        restarted.create_customer_project(
            owner, {"request_id": "b" * 32, "prompt": "A plate", "max_seconds": 10}
        )
    assert restarted.claim() is None
    executor = SimpleNamespace(
        typed_container_image=store.host_config.image,
        typed_container_verified=True,
        customer_container_verified=True,
        runner_verified=True,
    )
    assert (
        readiness(
            store.host_config,
            current_engine=store.host_config.engine,
            executor=executor,
            worker_alive=True,
            provider_qualified=True,
            recovery_qualified=True,
        )["checks"]["runner_reconciled"]
        is False
    )
    receipt = tmp_path / "runner.json"
    native = {
        "version": 1,
        "image": store.host_config.image,
        "fence": store.host_config.runner_fence,
        "container_name": "cadkitty-" + operation_id,
        "operation_id": operation_id,
        "state": "uncertain",
    }
    receipt.write_text(json.dumps(native))
    with pytest.raises(HostingError, match="terminal"):
        restarted.clear_runner_hold(receipt)
    native["state"] = "interrupted"
    receipt.write_text(json.dumps(native))
    assert restarted.clear_runner_hold(receipt)["automatic_retry"] is False
    assert restarted.clear_runner_hold(receipt)["operation_id"] == operation_id


@pytest.mark.parametrize("where", ["before_commit", "after_commit", "before_marker_remove"])
def test_fenced_recovery_crash_boundaries_are_exact_and_idempotent(tmp_path, monkeypatch, where):
    from dataclasses import replace

    from forge_mind.workspace import hosting_backup

    source, owner = funded(tmp_path)
    session = source.create(owner, "A part", None)
    jid = source.enqueue(session["id"], owner, {"max_seconds": 10})
    source.claim()
    with source.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("UPDATE jobs SET worker_host='old-host' WHERE id=?", (jid,))
    monkeypatch.setattr(hosting_backup.socket, "gethostname", lambda: "old-host")
    captured, destination = tmp_path / "backup", tmp_path / "restored"
    hosting_backup.snapshot(source.root, captured)
    hosting_backup.restore(captured, destination)
    restored = HostingStore(replace(source.host_config, root=destination))
    evidence = tmp_path / "fencing.json"
    evidence.write_text(
        json.dumps(
            {
                "old_host": "old-host",
                "fenced": True,
                "operator": "test-operator",
                "evidence": "Synthetic host fencing fixture",
            }
        )
    )

    def failpoint(name):
        if name == where:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        restored.apply_fenced_recovery(
            old_host="old-host", job_ids=[jid], fencing_evidence=evidence, failpoint=failpoint
        )
    assert (destination / ".restore-required.json").exists()
    with restored.connect() as con:
        assert con.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()[0] == (
            "running" if where == "before_commit" else "interrupted"
        )
    assert (
        restored.apply_fenced_recovery(
            old_host="old-host", job_ids=[jid], fencing_evidence=evidence
        )["applied"]
        is True
    )
    assert not (destination / ".restore-required.json").exists()
    assert (
        restored.apply_fenced_recovery(
            old_host="old-host", job_ids=[jid], fencing_evidence=evidence
        )["automatic_retry"]
        is False
    )
