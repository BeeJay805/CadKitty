"""Failed isolated workers retain bounded diagnostics without retaining scratch logs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import executor
from forge_mind.workspace.executor import Executor


def failed_process(monkeypatch, diagnostic=b"ValueError: rod profile is not closed\n", code=7):
    calls = []

    class Process:
        returncode = code
        waited = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.waited = True

    process = Process()

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(diagnostic)
        return process

    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=b""),
    )
    return process, calls


def test_failed_worker_retains_exit_and_bounded_untrusted_tail(tmp_path, monkeypatch):
    process, calls = failed_process(monkeypatch, b"old output\n" * 2000 + b"ValueError: bad rod\n")
    with pytest.raises(ValueError) as raised:
        Executor._run_container(
            "docker", tmp_path, "module", "request.json", 10, lambda: False, image="qualified"
        )
    error = str(raised.value)
    assert "exit=7" in error
    assert "ValueError: bad rod" in error
    assert "untrusted" in error.lower()
    assert len(error.encode("utf-8")) < 7000
    assert process.waited and len(calls) == 1


def test_failure_survives_scratch_cleanup_and_reaches_next_author_context(tmp_path, monkeypatch):
    from forge_mind.workspace import document_service
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.models import Action, Contract
    from forge_mind.workspace.store import Store

    process, calls = failed_process(monkeypatch)
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")
    worker = Executor()
    worker._typed_container_verified = True
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Retain useful worker errors", contract.model_dump())["id"]
    contexts = []

    class Client:
        model = "scripted-diagnostic-test"

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return Action(
                kind="finish", explanation="Failed geometry remains unverified"
            ), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, worker, lambda **kwargs: Client())
    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "base", "name": "Base", "kind": "box", "params": {"x": 10, "y": 10, "z": 10}}
            ]
        }
    )
    revision = document_service.build_document(
        service,
        sid,
        "owner",
        document,
        contract,
        None,
        "Controlled failed worker",
        10,
        lambda: False,
    )
    error = revision["measurement"]["errors"][0]
    assert "rod profile is not closed" in error and "exit=7" in error
    assert revision["measurement"]["geometry_valid"] is False
    assert not store.get(sid, "owner").get("accepted_revision")
    scratch = calls[0][1]["stdout"].name
    assert not Path(scratch).exists()
    assert not (store.directory(sid, revision["id"]) / "execution.log").exists()
    store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": 60})
    service._run_agent_job(store.claim())
    assert contexts[0]["failed_attempt"]["measurement"]["errors"] == [error]
    assert process.waited


@pytest.mark.parametrize(
    "case", ["missing", "replacement", "directory", "hardlink", "oversized", "changed"]
)
def test_missing_linked_oversized_or_changed_logs_cannot_supply_a_tail(tmp_path, case):
    path = tmp_path / "execution.log"
    with path.open("x+b") as log:
        log.write(b"Untrusted log text")
        log.flush()
        reader = log
        if case == "missing":
            path = tmp_path / "missing.log"
        elif case == "replacement":
            path = tmp_path / "replacement.log"
            path.write_text("Forged different file", encoding="utf-8")
        elif case == "directory":
            path = tmp_path
        elif case == "hardlink":
            os.link(path, tmp_path / "alias.log")
        elif case == "oversized":
            log.truncate(executor.HOSTED_LIMITS["file_bytes"] + 1)
        else:

            class ChangingReader:
                def __getattr__(self, name):
                    return getattr(log, name)

                def read(self, count):
                    value = log.read(count)
                    log.write(b"Changed during bounded read")
                    log.flush()
                    return value

            reader = ChangingReader()
        with pytest.raises((ValueError, OSError)):
            executor._typed_worker_log_tail(reader, path)


def test_symlink_log_is_not_followed(tmp_path):
    original = tmp_path / "owned.log"
    linked = tmp_path / "execution.log"
    with original.open("x+b") as log:
        log.write(b"Untrusted log text")
        try:
            linked.symlink_to(original)
        except OSError:
            pytest.skip("Platform does not allow symlink creation")
        with pytest.raises(ValueError):
            executor._typed_worker_log_tail(log, linked)


def test_link_guard_is_enforced_without_platform_symlink_privileges(tmp_path, monkeypatch):
    path = tmp_path / "execution.log"
    with path.open("x+b") as log:
        log.write(b"Must not be read through a linked path")
        monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == path)
        with pytest.raises(ValueError, match="linked"):
            executor._typed_worker_log_tail(log, path)


def test_worker_logs_remain_outside_public_artifact_allowlist():
    assert not executor.typed_artifact_name("execution.log")


@pytest.mark.parametrize("failure", [OSError("do not expose local path"), ValueError("changed")])
def test_unavailable_diagnostic_cannot_mask_nonzero_exit(tmp_path, monkeypatch, failure):
    process, _ = failed_process(monkeypatch)

    def unreadable(*args):
        raise failure

    monkeypatch.setattr(executor, "_typed_worker_log_tail", unreadable)
    with pytest.raises(ValueError) as raised:
        Executor._run_container(
            "docker", tmp_path, "module", "request.json", 10, lambda: False, image="qualified"
        )
    error = str(raised.value)
    assert "exit=7" in error and "diagnostic unavailable" in error
    assert "do not expose local path" not in error
    assert process.waited


@pytest.mark.parametrize(
    "data", [b"\xff\xfe\x80" * 3000, "☃".encode() * 3000, b""], ids=["invalid", "unicode", "empty"]
)
def test_log_decoding_and_persisted_tail_are_bounded(tmp_path, data):
    path = tmp_path / "execution.log"
    with path.open("x+b") as log:
        log.write(data)
        tail = executor._typed_worker_log_tail(log, path)
    assert len(tail) <= 1600
    assert len(tail.encode("utf-8")) <= 6400


def test_success_does_not_read_or_persist_worker_diagnostic(tmp_path, monkeypatch):
    process, _ = failed_process(monkeypatch, code=0)

    def forbidden(*args):
        pytest.fail("A successful worker must not read diagnostic output")

    monkeypatch.setattr(executor, "_typed_worker_log_tail", forbidden)
    Executor._run_container(
        "docker", tmp_path, "module", "request.json", 10, lambda: False, image="qualified"
    )
    assert process.waited


def test_cleanup_error_does_not_drop_primary_worker_diagnostic(tmp_path, monkeypatch):
    process, _ = failed_process(monkeypatch)
    monkeypatch.setattr(
        executor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=b"daemon refused"),
    )
    with pytest.raises(ValueError) as raised:
        Executor._run_container(
            "docker", tmp_path, "module", "request.json", 10, lambda: False, image="qualified"
        )
    error = str(raised.value)
    assert "exit=7" in error and "rod profile is not closed" in error
    assert "container cleanup failed" in error and process.waited
