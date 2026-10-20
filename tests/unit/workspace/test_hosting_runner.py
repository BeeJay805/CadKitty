from __future__ import annotations

import json
import os
import threading
import time

import pytest

from forge_mind.workspace.hosting import HostingError
from forge_mind.workspace.hosting_backup import inventory
from forge_mind.workspace.hosting_runner import (
    Runner,
    RunnerClient,
    RunnerPolicy,
    RunnerUncertainError,
)
from forge_mind.workspace.hosting_runner_executor import BrokerExecutor


class FakeDocker:
    def __init__(self):
        self.created = []
        self.removed = []

    def identity(self):
        return "sha256:" + "a" * 64

    def create(self, name, directory, kind, operation_id):
        self.created.append((name, directory, kind, operation_id))
        return "b" * 64

    def start(self, identity):
        pass

    def poll(self, identity):
        return 0

    def remove(self, identity):
        self.removed.append(identity)

    def reconcile(self, name, operation_id, expected):
        self.removed.append(expected)


def host(tmp_path):
    staging = tmp_path / "incoming"
    staging.mkdir()
    policy = RunnerPolicy(tmp_path / "private", staging, "sha256:" + "a" * 64, "c" * 32, 1234)
    backend = FakeDocker()
    runner = Runner(policy, backend)
    ticket = staging / ("d" * 32)
    ticket.mkdir()
    (ticket / "probe.json").write_text("{}")
    request = {
        "version": 1,
        "fence": policy.fence,
        "command": "launch",
        "operation_id": "e" * 32,
        "operation_kind": "probe",
        "staging_ticket": ticket.name,
        "inputs": inventory(ticket),
        "remaining_seconds": 2,
    }
    return runner, backend, request


@pytest.mark.parametrize(
    "patch",
    [
        {"fence": "f" * 32},
        {"image": "attacker:latest"},
        {"operation_kind": "shell"},
        {"staging_ticket": "../host"},
        {"remaining_seconds": float("inf")},
        {"remaining_seconds": True},
        {"command": "reconcile"},
        {"inputs": {"../outside": {"bytes": 1, "sha256": "a" * 64}}},
    ],
)
def test_public_envelope_cannot_choose_privileged_execution(tmp_path, patch):
    runner, backend, request = host(tmp_path)
    with pytest.raises(HostingError):
        runner.dispatch({**request, **patch})
    assert backend.created == []


def test_launch_receipt_is_durable_before_worker_and_replay_is_bound(tmp_path, monkeypatch):
    runner, backend, request = host(tmp_path)
    observed = threading.Event()

    def controlled_execute(envelope, receipt, cancelled):
        assert runner._read(request["operation_id"])["state"] == "prepared"
        observed.set()

    monkeypatch.setattr(runner, "_execute", controlled_execute)
    runner.dispatch(request)
    assert observed.wait(2)
    assert runner.dispatch(request)["state"] == "prepared"
    with pytest.raises(HostingError, match="changed its authority"):
        runner.dispatch({**request, "remaining_seconds": 1})
    # A process restart cannot infer whether create/start happened after prepared.
    restarted = Runner(runner.policy, backend)
    assert restarted._read(request["operation_id"])["state"] == "uncertain"
    with pytest.raises(HostingError, match="offline reconciliation"):
        restarted.dispatch(request)
    assert restarted.reconcile(request["operation_id"])["state"] == "interrupted"
    assert backend.created == []


@pytest.mark.skipif(
    os.name != "posix",
    reason="Production runner uses Linux openat and peer UID; Windows does not qualify it",
)
def test_fake_backend_exercises_anchored_staging_and_cleanup(tmp_path):
    runner, backend, request = host(tmp_path)
    runner.dispatch(request)
    deadline = time.monotonic() + 3
    while (receipt := runner._read(request["operation_id"]))["state"] not in {
        "done",
        "error",
        "uncertain",
    }:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert receipt["state"] == "done"
    assert backend.removed == ["b" * 64]
    assert len(backend.created) == 1


def test_broker_image_inspect_and_legacy_build_use_only_runner(tmp_path):
    class Client:
        fence = "c" * 32

        def request(self, command):
            assert command == "identity"
            return {
                "version": 1,
                "image": "sha256:" + "a" * 64,
                "fence": self.fence,
                "restricted": True,
            }

        def execute(self, directory, kind, timeout, cancelled):
            assert kind == "legacy_build"
            assert json.loads((directory / "job.json").read_text())["out_dir"] == "/work"
            (directory / "part.step").write_bytes(b"step")
            (directory / "part.stl").write_bytes(b"stl")
            (directory / "result.json").write_text('{"ok":true}')
            return {"operation_id": "d" * 32}

    executor = BrokerExecutor(Client(), "sha256:" + "a" * 64)
    assert executor._inspect_image("not-an-executable") == "sha256:" + "a" * 64
    assert executor.runner_verified
    assert executor.build("test source", tmp_path, 10, lambda: False).ok
    with pytest.raises(HostingError, match="allowlisted"):
        executor._run_container(
            "ignored",
            tmp_path,
            "os",
            "shell",
            1,
            lambda: False,
            image=executor.typed_container_image,
        )


def test_uncertain_launch_is_never_retried_or_represented_as_result(tmp_path):
    class LostResponse:
        fence = "c" * 32
        staging_root = tmp_path / "staging"
        calls = 0

        def request(self, command, **payload):
            self.calls += 1
            raise TimeoutError("response lost")

    client = LostResponse()
    client.staging_root.mkdir()
    (tmp_path / "probe.json").write_text("{}")
    with pytest.raises(RunnerUncertainError, match="response was lost"):
        RunnerClient.execute(client, tmp_path, "probe", 1, lambda: False)
    assert client.calls == 1
