"""Operational recovery negatives from independent hosting review 001; no workers run."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests.unit.workspace.test_hosting_runner import host
from tests.unit.workspace.test_hosting_store import funded

from forge_mind.workspace.hosting import HostingError
from forge_mind.workspace.hosting_backup import restore, snapshot
from forge_mind.workspace.hosting_runner import (
    Runner,
    RunnerClient,
    RunnerRejectedError,
    RunnerUncertainError,
    request_digest,
)
from forge_mind.workspace.hosting_runner_executor import BrokerExecutor
from forge_mind.workspace.hosting_store import HostingStore
from forge_mind.workspace.store import WorkspaceError


class LocalClient(RunnerClient):
    def __init__(self, runner):
        super().__init__(
            runner.policy.root / "not-listening.sock",
            runner.policy.staging_root,
            runner.policy.fence,
        )
        self.runner = runner
        self.calls = 0

    def request(self, command, **payload):
        self.calls += 1
        message = {"version": 1, "fence": self.fence, "command": command, **payload}
        return self.decode_response(message, json.dumps(self.runner.respond(message)).encode())


@pytest.mark.parametrize("kind", ["busy", "manifest", "operation", "disk", "retention"])
def test_acknowledged_preaccept_refusals_do_not_create_launch_authority(
    tmp_path, monkeypatch, kind
):
    runner, backend, request = host(tmp_path)
    monkeypatch.setattr(runner, "_execute", lambda *args: None)
    if kind == "busy":
        runner.dispatch(request)
        request = {**request, "operation_id": "f" * 32}
    elif kind == "manifest":
        request = {**request, "inputs": {"probe.json": {"bytes": -1, "sha256": "a" * 64}}}
    elif kind == "operation":
        request = {**request, "operation_kind": "shell"}
    elif kind == "disk":
        monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    else:
        with runner._connect() as con:
            con.executemany(
                "INSERT INTO operations VALUES (?,?,?,?)",
                [
                    (f"{i:032x}", runner.policy.fence, "a" * 64, '{"state":"done"}')
                    for i in range(4096)
                ],
            )
    response = runner.respond(request)
    assert response["accepted"] is False
    assert response["request_hash"] == request_digest(request)
    with pytest.raises(RunnerRejectedError):
        RunnerClient.decode_response(request, json.dumps(response).encode())
    with runner._connect() as con:
        assert (
            con.execute(
                "SELECT 1 FROM operations WHERE id=?", (request["operation_id"],)
            ).fetchone()
            is None
        )
    if kind == "busy":
        assert runner._read("e" * 32)["state"] == "prepared"
    assert backend.created == []


def test_busy_startup_probe_does_not_mint_an_unreconcilable_store_hold(tmp_path, monkeypatch):
    runner, _, initial = host(tmp_path)
    monkeypatch.setattr(runner, "_execute", lambda *args: None)
    runner.dispatch(initial)
    store, _ = funded(tmp_path / "app")
    executor = BrokerExecutor(
        LocalClient(runner), runner.policy.image, on_uncertain=store.hold_uncertain
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text("{}")
    with pytest.raises(RunnerRejectedError, match="active"):
        executor._run_container(
            "ignored",
            probe,
            "forge_mind.workspace.container_probe",
            "probe.json",
            1,
            lambda: False,
            image=runner.policy.image,
        )
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM host_runner_holds").fetchone()[0] == 0
    assert runner._read(initial["operation_id"])["state"] == "prepared"


def test_committed_acceptance_failure_remains_uncertain_and_reconcilable(tmp_path, monkeypatch):
    runner, backend, _ = host(tmp_path)
    monkeypatch.setattr(
        "threading.Thread.start",
        lambda _: (_ for _ in ()).throw(RuntimeError("crash after commit")),
    )
    store, _ = funded(tmp_path / "app")
    executor = BrokerExecutor(
        LocalClient(runner), runner.policy.image, on_uncertain=store.hold_uncertain
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text("{}")
    with pytest.raises(RunnerUncertainError) as caught:
        executor._run_container(
            "ignored",
            probe,
            "forge_mind.workspace.container_probe",
            "probe.json",
            1,
            lambda: False,
            image=runner.policy.image,
        )
    operation_id = caught.value.operation_id
    assert runner._read(operation_id)["state"] == "prepared"
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM host_runner_holds").fetchone()[0] == 1
        original = json.loads(
            con.execute("SELECT document FROM host_runner_hold_evidence").fetchone()[0]
        )
        assert original["request_hash"] == caught.value.request_hash
    restarted = Runner(runner.policy, backend)
    assert restarted.reconcile(operation_id)["state"] == "interrupted"


def test_lost_launch_reply_retains_accepted_authority_without_retry(tmp_path, monkeypatch):
    runner, _, _ = host(tmp_path)
    monkeypatch.setattr(runner, "_execute", lambda *args: None)

    class Lost(LocalClient):
        def request(self, command, **payload):
            super().request(command, **payload)
            raise TimeoutError("response lost after acceptance")

    client = Lost(runner)
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text("{}")
    with pytest.raises(RunnerUncertainError) as caught:
        client.execute(probe, "probe", 1, lambda: False)
    assert client.calls == 1
    assert runner._read(caught.value.operation_id)["request_hash"] == caught.value.request_hash


@pytest.mark.parametrize(
    "response", [None, {}, {"state": "rejected"}, {"protocol_error": True, "error": "unclassified"}]
)
def test_unbound_and_malformed_responses_never_prove_nonacceptance(tmp_path, response):
    runner, _, _ = host(tmp_path)

    class Malformed(LocalClient):
        def request(self, command, **payload):
            return response

    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text("{}")
    with pytest.raises(RunnerUncertainError):
        Malformed(runner).execute(probe, "probe", 1, lambda: False)


def test_lost_unaccepted_response_requires_exact_offline_nonacceptance_evidence(tmp_path):
    runner, _, _ = host(tmp_path)
    with pytest.raises(HostingError, match="exact retained request hash"):
        runner.reconcile("1" * 32)
    receipt = runner.reconcile("1" * 32, unaccepted_request_hash="a" * 64)
    assert receipt["state"] == "rejected" and receipt["accepted"] is False
    assert runner._read("1" * 32) == receipt
    assert runner.reconcile("1" * 32, unaccepted_request_hash="a" * 64) == receipt
    with pytest.raises(HostingError, match="differs"):
        runner.reconcile("1" * 32, unaccepted_request_hash="b" * 64)


def test_lost_busy_refusal_can_clear_only_with_exact_retained_nonacceptance(tmp_path, monkeypatch):
    runner, backend, seed = host(tmp_path)
    monkeypatch.setattr(runner, "_execute", lambda *args: None)
    runner.dispatch(seed)

    class Lost(LocalClient):
        def request(self, command, **payload):
            self.calls += 1
            self.runner.respond({"version": 1, "fence": self.fence, "command": command, **payload})
            raise TimeoutError("refusal response lost")

    store, _ = funded(tmp_path / "app")
    store.host_config = replace(store.host_config, runner_fence=runner.policy.fence)
    executor = BrokerExecutor(Lost(runner), runner.policy.image, on_uncertain=store.hold_uncertain)
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text("{}")
    with pytest.raises(RunnerUncertainError) as caught:
        executor._run_container(
            "ignored",
            probe,
            "forge_mind.workspace.container_probe",
            "probe.json",
            1,
            lambda: False,
            image=runner.policy.image,
        )
    restarted = Runner(runner.policy, backend)
    receipt = restarted.reconcile(
        caught.value.operation_id, unaccepted_request_hash=caught.value.request_hash
    )
    path = tmp_path / "nonacceptance.json"
    path.write_text(json.dumps({**receipt, "request_hash": "0" * 64}))
    with pytest.raises(HostingError, match="original held authority"):
        store.clear_runner_hold(path)
    path.write_text(json.dumps(receipt))
    assert store.clear_runner_hold(path)["automatic_retry"] is False
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM host_runner_holds").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM host_runner_hold_evidence").fetchone()[0] == 1
    assert restarted._read(seed["operation_id"])["state"] == "uncertain"


def fencing_file(path, old_host):
    path.write_text(
        json.dumps(
            {
                "old_host": old_host,
                "fenced": True,
                "operator": "fixture operator",
                "evidence": "Synthetic host fence",
            }
        )
    )
    return path


@pytest.mark.parametrize("with_running_job", [False, True])
def test_second_restore_cycle_keeps_both_histories_and_rejects_old_authority(
    tmp_path, monkeypatch, with_running_job
):
    monkeypatch.setattr("socket.gethostname", lambda: "old-a")
    source, owner = funded(tmp_path / "source")
    a_snapshot, a_root = tmp_path / "snapshot-a", tmp_path / "host-a"
    snapshot(source.root, a_snapshot)
    monkeypatch.setattr("socket.gethostname", lambda: "host-a")
    a_identity = restore(a_snapshot, a_root)
    a = HostingStore(replace(source.host_config, root=a_root))
    a_evidence = fencing_file(tmp_path / "fence-a.json", "old-a")
    a_receipt = a.apply_fenced_recovery(old_host="old-a", job_ids=[], fencing_evidence=a_evidence)
    a_path = a_root / ".restore-history" / a_identity["restore_id"] / "recovery.json"
    a_bytes = a_path.read_bytes()
    jobs = []
    if with_running_job:
        session = a.create(owner, "A part", None)
        jobs.append(a.enqueue(session["id"], owner, {"max_seconds": 10}))
        a.claim()
    b_snapshot, b_root = tmp_path / "snapshot-b", tmp_path / "host-b"
    snapshot(a.root, b_snapshot)
    monkeypatch.setattr("socket.gethostname", lambda: "host-b")
    b_identity = restore(b_snapshot, b_root)
    b = HostingStore(replace(a.host_config, root=b_root))
    b_evidence = fencing_file(tmp_path / "fence-b.json", "host-a")
    with pytest.raises(HostingError, match="current restore identity"):
        b.apply_fenced_recovery(
            old_host="old-a",
            job_ids=[],
            fencing_evidence=a_evidence,
            restore_id=a_identity["restore_id"],
        )
    assert (b_root / ".restore-required.json").exists()
    b_receipt = b.apply_fenced_recovery(
        old_host="host-a",
        job_ids=jobs,
        fencing_evidence=b_evidence,
        restore_id=b_identity["restore_id"],
    )
    assert (
        b.apply_fenced_recovery(old_host="host-a", job_ids=jobs, fencing_evidence=b_evidence)
        == b_receipt
    )
    assert a_receipt["restore_id"] != b_receipt["restore_id"]
    assert (
        b_root / ".restore-history" / a_identity["restore_id"] / "recovery.json"
    ).read_bytes() == a_bytes
    assert not (b_root / ".restore-required.json").exists()


def restored_holds(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "old-host")
    source, owner = funded(tmp_path / "source")
    source.host_config = replace(source.host_config, runner_fence="a" * 32)
    session = source.create(owner, "Native part", None, "native_operations")
    jid = source.enqueue(session["id"], owner, {"max_seconds": 10})
    source.claim()
    for op in ("1" * 32, "2" * 32):
        with source.active_job(jid):
            source.hold_uncertain(
                RunnerUncertainError(
                    "Lost runner response for operation " + op,
                    operation_id=op,
                    request_hash="9" * 64,
                )
            )
    (source.root / "retained-publication.json").write_bytes(b"original publication evidence")
    capture, destination = tmp_path / "snapshot", tmp_path / "restored"
    snapshot(source.root, capture)
    monkeypatch.setattr("socket.gethostname", lambda: "replacement-host")
    identity = restore(capture, destination)
    store = HostingStore(replace(source.host_config, root=destination, runner_fence="b" * 32))
    store.apply_fenced_recovery(
        old_host="old-host",
        job_ids=[jid],
        fencing_evidence=fencing_file(tmp_path / "host-fence.json", "old-host"),
    )
    hold = next(
        row for row in identity["runner_holds"] if row["evidence"]["operation_id"] == "1" * 32
    )
    values = {
        "restore_id": identity["restore_id"],
        "hold_ids": [hold["hold_id"]],
        "operation_id": "1" * 32,
        "old_fence": "a" * 32,
        "old_image": source.host_config.image,
        "job_ids": [jid],
    }
    attestation = {
        **values,
        "old_host": "old-host",
        "fenced": True,
        "operator": "fixture operator",
        "evidence": "Old machine and runner destroyed; synthetic fixture",
    }
    evidence = tmp_path / "disposition.json"
    evidence.write_text(json.dumps(attestation))
    return store, owner, jid, identity, values, attestation, evidence


def test_rotated_fence_disposition_is_exact_conservative_and_preserves_other_hold(
    tmp_path, monkeypatch
):
    store, owner, jid, _identity, values, attestation, evidence = restored_holds(
        tmp_path, monkeypatch
    )
    before = store.session_record(store.list_sessions(owner)[0]["id"], owner)
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT used_seconds FROM host_reservations WHERE job_id=?", (jid,)
            ).fetchone()[0]
            is None
        )
    claimed_current = tmp_path / "false-current-fence.json"
    claimed_current.write_text(
        json.dumps(
            {
                "version": 1,
                "operation_id": values["operation_id"],
                "image": store.host_config.image,
                "fence": store.host_config.runner_fence,
                "container_name": "cadkitty-" + values["operation_id"],
                "state": "interrupted",
                "request_hash": "9" * 64,
            }
        )
    )
    with pytest.raises(HostingError, match="original held authority"):
        store.clear_runner_hold(claimed_current)
    receipt = store.dispose_restored_runner_hold(**values, fencing_evidence=evidence)
    assert receipt["new_fence"] == "b" * 32 and receipt["old_fence"] == "a" * 32
    assert receipt["automatic_retry"] is False
    assert store.dispose_restored_runner_hold(**values, fencing_evidence=evidence) == receipt
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT used_seconds FROM host_reservations WHERE job_id=?", (jid,)
            ).fetchone()[0]
            == 10
        )
        assert con.execute("SELECT count(*) FROM host_runner_holds").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM host_runner_hold_evidence").fetchone()[0] == 2
    assert store.session_record(before["id"], owner) == before
    assert (
        store.root / "retained-publication.json"
    ).read_bytes() == b"original publication evidence"
    with pytest.raises(WorkspaceError, match="uncertainty"):
        store.create_customer_project(
            owner, {"request_id": "f" * 32, "prompt": "Another plate", "max_seconds": 10}
        )
    attestation["evidence"] = "changed evidence"
    evidence.write_text(json.dumps(attestation))
    with pytest.raises(HostingError, match="replay changed"):
        store.dispose_restored_runner_hold(**values, fencing_evidence=evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("restore_id", "f" * 32),
        ("old_fence", "f" * 32),
        ("old_image", "sha256:" + "f" * 64),
        ("operation_id", "3" * 32),
        ("hold_ids", ["f" * 64]),
        ("job_ids", ["f" * 32]),
        ("fenced", False),
    ],
)
def test_wrong_or_unfenced_old_runner_disposition_cannot_release_allowance(
    tmp_path, monkeypatch, field, value
):
    store, _, jid, _, values, attestation, evidence = restored_holds(tmp_path, monkeypatch)
    if field == "fenced":
        attestation[field] = value
        evidence.write_text(json.dumps(attestation))
    else:
        values[field] = value
    with pytest.raises(HostingError):
        store.dispose_restored_runner_hold(**values, fencing_evidence=evidence)
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM host_runner_holds").fetchone()[0] == 2
        assert (
            con.execute(
                "SELECT used_seconds FROM host_reservations WHERE job_id=?", (jid,)
            ).fetchone()[0]
            is None
        )


@pytest.mark.parametrize("where", ["before_commit", "after_commit"])
def test_restored_hold_disposition_replays_across_commit_crash(tmp_path, monkeypatch, where):
    store, _, jid, _, values, _, evidence = restored_holds(tmp_path, monkeypatch)

    def fail(name):
        if name == where:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        store.dispose_restored_runner_hold(**values, fencing_evidence=evidence, failpoint=fail)
    assert (
        store.dispose_restored_runner_hold(**values, fencing_evidence=evidence)["automatic_retry"]
        is False
    )
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT used_seconds FROM host_reservations WHERE job_id=?", (jid,)
            ).fetchone()[0]
            == 10
        )
