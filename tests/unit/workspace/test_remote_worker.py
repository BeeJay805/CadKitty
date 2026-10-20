from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge_mind.workspace import executor
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")
    service = Executor()
    service._typed_container_verified = True
    (tmp_path / "program.py").write_text("# owned source\n", encoding="utf-8")
    (tmp_path / "document.json").write_text("{}", encoding="utf-8")
    payload = {
        "revision_id": "a" * 32,
        "contract": {},
        "document": {},
        "assets": {},
        "cache_source": "C:/secret",
    }
    return service, tmp_path, payload


def emit(scratch: Path, *, change=None):
    data = b"trusted CAD"
    (scratch / "part.step").write_bytes(data)
    measured = Measurement(
        revision_id="a" * 32,
        contract_hash=Contract().fingerprint(),
        artifact_hashes={"part.step": hashlib.sha256(data).hexdigest()},
        engine_fingerprint="worker",
        geometry_valid=True,
    ).model_dump()
    for name in ("program.py", "document.json"):
        measured["artifact_hashes"][name] = hashlib.sha256(
            (scratch / name).read_bytes()
        ).hexdigest()
    if change:
        change(measured, scratch)
    (scratch / "processed.json").write_text(json.dumps(measured), encoding="utf-8")


def test_default_guard_remains_closed_before_any_worker(tmp_path):
    with pytest.raises(ValueError, match="qualification"):
        Executor().process_document(tmp_path, {}, 10, lambda: False)
    with pytest.raises(AttributeError):
        Executor().typed_container_verified = True


def test_typed_worker_uses_disposable_stage_without_host_cache(worker, monkeypatch):
    service, directory, payload = worker
    calls = []

    def run(docker, scratch, module, request, timeout, cancelled, *, image):
        calls.append(scratch)
        staged = json.loads((scratch / request).read_text())
        assert scratch != directory
        assert staged["cache_source"] is None
        assert module == "forge_mind.workspace.processing"
        assert timeout <= 10
        emit(scratch)

    monkeypatch.setattr(service, "_run_container", run)
    measured = service.process_document(directory, payload, 10, lambda: False)
    assert measured["geometry_valid"]
    assert (directory / "part.step").read_bytes() == b"trusted CAD"
    assert not calls[0].exists()


@pytest.mark.parametrize(
    "case", ["traversal", "wrong-revision", "wrong-contract", "tamper", "link", "changed-input"]
)
def test_container_results_do_not_override_owned_revision(worker, monkeypatch, case):
    service, directory, payload = worker

    def change(data, scratch):
        if case == "traversal":
            data["artifact_hashes"]["../outside.step"] = "a" * 64
        elif case == "wrong-revision":
            data["revision_id"] = "b" * 32
        elif case == "wrong-contract":
            data["contract_hash"] = "b" * 64
        elif case == "tamper":
            (scratch / "part.step").write_bytes(b"changed")
        elif case == "link":
            import os

            os.link(scratch / "part.step", scratch / "other.step")
        elif case == "changed-input":
            (scratch / "program.py").write_text("changed", encoding="utf-8")
            data["artifact_hashes"] = {"program.py": hashlib.sha256(b"changed").hexdigest()}

    monkeypatch.setattr(
        service,
        "_run_container",
        lambda docker, scratch, *args, **kwargs: emit(scratch, change=change),
    )
    with pytest.raises(ValueError):
        service.process_document(directory, payload, 10, lambda: False)
    assert (directory / "program.py").read_text() == "# owned source\n"


@pytest.mark.parametrize("budget", [None, 0, 601])
def test_hosted_typed_execution_always_requires_bounded_budget(worker, budget):
    service, directory, payload = worker
    with pytest.raises(ValueError, match="deadline"):
        service.process_document(directory, payload, budget, lambda: False)


def test_qualification_does_not_fallback_to_local_without_docker(monkeypatch):
    monkeypatch.setattr(executor.shutil, "which", lambda _: None)
    service = Executor()
    with pytest.raises(ValueError, match="Docker"):
        service.qualify_typed_container()
    assert not service.typed_container_verified


@pytest.mark.parametrize(
    "name",
    [
        "../outside.step",
        "/secret.step",
        "other.brep",
        "part_xyz.brep",
        "contract.json",
        "assets.json",
        "worker.log",
    ],
)
def test_worker_artifact_allowlist_rejects_unknown_and_reserved_files(name):
    assert not executor.typed_artifact_name(name)


@pytest.mark.parametrize("cleanup,stopped", [("missing", False), ("ok", True), ("failed", True)])
def test_hardened_container_reap_is_idempotent_and_preserves_cancel(
    tmp_path, monkeypatch, cleanup, stopped
):
    from types import SimpleNamespace

    commands = []
    launched = []

    class Process:
        returncode = 0
        killed = False
        waited = False

        def poll(self):
            return None if stopped and not self.killed else 0

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            self.waited = True

    process = Process()

    def launch(command, **kwargs):
        commands.append(command)
        launched.append(True)
        return process

    def remove(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0 if cleanup == "ok" else 1,
            stderr=b"No such container" if cleanup == "missing" else b"daemon refused",
        )

    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    monkeypatch.setattr(executor.subprocess, "run", remove)
    if stopped:
        with pytest.raises(
            ValueError, match="cancelled" + (".*cleanup failed" if cleanup == "failed" else "")
        ):
            Executor._run_container(
                "docker",
                tmp_path,
                "module",
                "probe.json",
                1,
                lambda: bool(launched),
                image="sha256:" + "a" * 64,
            )
    else:
        Executor._run_container(
            "docker", tmp_path, "module", "probe.json", 1, lambda: False, image="sha256:" + "a" * 64
        )
    assert process.waited
    assert process.killed == stopped
    command = commands[0]
    for flag, value in [
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--user", "65534:65534"),
        ("--security-opt", "no-new-privileges"),
        ("--entrypoint", "/app/.venv/bin/python"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--read-only" in command
    assert command.count("--mount") == 1
    assert commands[1][1:3] == ["rm", "-f"]
