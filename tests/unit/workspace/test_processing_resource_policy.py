from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from forge_mind.agentic.sandbox import SandboxResult
from forge_mind.workspace import executor, processing
from forge_mind.workspace.models import Action, EditRequest, WorkspaceRequest

POLICIES = ("FILE_BYTES", "JOB_BYTES", "RSS_BYTES", "RESULT_BYTES")


@pytest.fixture(autouse=True)
def clear_operator_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in POLICIES:
        monkeypatch.delenv("FORGE_WORKSPACE_" + name, raising=False)


def test_unset_and_configured_policies_are_discoverable_without_reimport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = processing.resource_policy()
    assert all(value["value"] is None for value in policy["local"].values())
    for name in POLICIES:
        monkeypatch.setenv("FORGE_WORKSPACE_" + name, "123")
    current = processing.resource_policy()
    assert all(value["value"] == 123 for value in current["local"].values())
    assert current["local"]["rss_bytes"]["environment"] == "FORGE_WORKSPACE_RSS_BYTES"
    assert current["hosted"] == policy["hosted"]
    assert current["hosted"]["memory_bytes"] == 2 * 1024**3
    assert current["hosted"]["file_bytes"] == 64 * 1024**2
    assert isinstance(processing.MAX_JOB_BYTES, int)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "unlimited"])
def test_invalid_operator_policies_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FORGE_WORKSPACE_RSS_BYTES", value)
    with pytest.raises(ValueError, match=r"FORGE_WORKSPACE_RSS_BYTES.*positive integer"):
        processing.resource_policy()


def test_large_regular_files_and_outputs_are_valid_without_a_configured_size_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "part.step"
    size = processing.MAX_JOB_BYTES + 1
    with source.open("wb") as writer:
        writer.truncate(size)
    assert processing.regular_file(source)
    assert processing.directory_size(tmp_path) == size


def test_file_policy_keeps_exact_boundaries_and_explicit_control_record_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "part.step"
    source.write_bytes(b"123456")
    monkeypatch.setenv("FORGE_WORKSPACE_FILE_BYTES", "6")
    processing.copy_artifact(source, tmp_path / "copied.step")
    assert (tmp_path / "copied.step").read_bytes() == b"123456"
    assert not processing.regular_file(source, 5)
    monkeypatch.setenv("FORGE_WORKSPACE_FILE_BYTES", "5")
    assert not processing.regular_file(source, 20)
    with pytest.raises(ValueError, match="oversized"):
        processing.copy_artifact(source, tmp_path / "rejected.step")
    assert not (tmp_path / "rejected.step").exists()


def test_unlimited_file_policy_still_rejects_directories_and_hard_links(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep")
    hard_link = tmp_path / "part.step"
    os.link(sentinel, hard_link)
    assert not processing.regular_file(hard_link)
    assert not processing.regular_file(tmp_path)
    for invalid in (hard_link, tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError, match="linked, nonregular"):
            processing.copy_artifact(invalid, tmp_path / "copy")
    assert sentinel.read_bytes() == b"keep"
    assert not (tmp_path / "copy").exists()


class Worker:
    pid = 12345
    returncode = 0

    def __init__(self) -> None:
        self.polls = 0
        self.waited = False

    def poll(self) -> int | None:
        self.polls += 1
        return None if self.polls < 3 else self.returncode

    def wait(self, **kwargs: Any) -> int:
        self.waited = True
        return self.returncode


def fake_worker(
    monkeypatch: pytest.MonkeyPatch, rss: int = 3 * 1024**3
) -> tuple[Worker, list[int]]:
    worker, killed = Worker(), []
    monkeypatch.setattr(processing.subprocess, "Popen", lambda *a, **kw: worker)
    monitor = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=rss), children=lambda **kw: []
    )
    monkeypatch.setattr(processing.psutil, "Process", lambda _: monitor)
    monkeypatch.setattr(processing, "_kill_tree", killed.append)
    monkeypatch.setattr(processing.time, "sleep", lambda _: None)
    return worker, killed


def test_unlimited_worker_survives_previous_rss_and_output_ceilings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, killed = fake_worker(monkeypatch)
    with (tmp_path / "large.step").open("wb") as writer:
        writer.truncate(processing.MAX_JOB_BYTES + 1)
    processing.bounded_process(["worker"], tmp_path, None, lambda: False)
    assert worker.waited and killed == [worker.pid]


def test_configured_rss_policy_stops_and_reaps_worker_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, killed = fake_worker(monkeypatch)
    monkeypatch.setenv("FORGE_WORKSPACE_RSS_BYTES", str(2 * 1024**3))
    with pytest.raises(ValueError, match=r"operator memory limit.*FORGE_WORKSPACE_RSS_BYTES"):
        processing.bounded_process(["worker"], tmp_path, None, lambda: False)
    assert worker.waited and killed == [worker.pid]


def test_output_policy_catches_files_written_by_a_worker_that_exits_between_polls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker, killed = fake_worker(monkeypatch)
    monkeypatch.setenv("FORGE_WORKSPACE_JOB_BYTES", "5")

    def launch(*args: Any, **kwargs: Any) -> Worker:
        (tmp_path / "part.step").write_bytes(b"123456")
        worker.polls = 10
        return worker

    monkeypatch.setattr(processing.subprocess, "Popen", launch)
    with pytest.raises(ValueError, match=r"operator output-byte limit.*FORGE_WORKSPACE_JOB_BYTES"):
        processing.bounded_process(["worker"], tmp_path, None, lambda: False)
    assert worker.waited and killed == [worker.pid]


def test_processed_result_limit_is_optional_and_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = {"geometry": "x" * (2 * 1024**2 + 1)}

    def process(*args: Any, **kwargs: Any) -> None:
        (tmp_path / "processed.json").write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(processing, "bounded_process", process)
    assert processing.process_revision(tmp_path, {}, None, lambda: False) == data
    monkeypatch.setenv("FORGE_WORKSPACE_RESULT_BYTES", "100")
    with pytest.raises(ValueError, match="operator resource policy"):
        processing.process_revision(tmp_path, {}, None, lambda: False)


@pytest.mark.parametrize("limit", [None, 1234])
def test_local_executor_uses_parent_policy_and_has_no_hidden_result_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int | None
) -> None:
    if limit is not None:
        monkeypatch.setenv("FORGE_WORKSPACE_RSS_BYTES", str(limit))
    captured: list[dict[str, Any]] = []

    def build(*args: Any, **kwargs: Any) -> None:
        captured.append(json.loads((tmp_path / "job.json").read_text()))
        (tmp_path / "part.step").write_bytes(b"step")
        (tmp_path / "part.stl").write_bytes(b"stl")
        (tmp_path / "result.json").write_text(
            json.dumps({"ok": True, "trace": "x" * (1024**2 + 1)})
        )

    monkeypatch.setattr(executor, "bounded_process", build)
    assert executor.Executor("trusted_local").build("", tmp_path, None, lambda: False).ok
    assert "rss_limit_bytes" not in captured[0]
    assert "memory_mb" not in captured[0]
    monkeypatch.setenv("FORGE_WORKSPACE_RESULT_BYTES", "100")
    assert not executor.Executor("trusted_local").build("", tmp_path, None, lambda: False).ok


def test_hosted_isolation_envelope_is_preserved_when_local_limits_are_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command: list[str] = []
    worker = Worker()
    worker.polls = 10
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")

    def launch(args: list[str], **kwargs: Any) -> Worker:
        command.extend(args)
        return worker

    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    monkeypatch.setattr(executor.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(executor.Executor, "_result", lambda *a: SandboxResult(ok=True))
    assert executor.Executor("docker").build("", tmp_path, None, lambda: False).ok
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--memory") + 1] == str(2 * 1024**3)
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--ulimit") + 1] == "fsize=67108864:67108864"


def test_hosted_output_policy_catches_worker_that_exits_between_polls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = Worker()
    worker.polls = 10
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")

    def launch(*args: Any, **kwargs: Any) -> Worker:
        with (tmp_path / "oversized.bin").open("wb") as writer:
            writer.truncate(processing.MAX_JOB_BYTES + 1)
        return worker

    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    removed: list[list[str]] = []
    monkeypatch.setattr(executor.subprocess, "run", lambda cmd, **kw: removed.append(cmd))
    result = executor.Executor("docker").build("", tmp_path, None, lambda: False)
    assert not result.ok and result.error_type == "resource_limit"
    assert result.error_message == "Execution exceeded its hosted output-byte limit"
    assert worker.waited and removed[0][:3] == ["docker", "rm", "-f"]


def test_explicit_job_budgets_and_study_values_have_no_hidden_schema_ceiling() -> None:
    assert (
        WorkspaceRequest(prompt="Build", max_actions=10000, max_seconds=100000).max_actions == 10000
    )
    assert EditRequest(max_actions=10000, max_seconds=100000).max_seconds == 100000
    assert (
        len(Action(kind="study", explanation="Compare", study_values=list(range(100))).study_values)
        == 100
    )


def test_unlimited_real_worker_cancellation_terminates_descendants(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path('child.pid').write_text(str(child.pid)); time.sleep(60)"
    )
    started = time.monotonic()
    with pytest.raises(ValueError, match="cancelled"):
        processing.bounded_process(
            [sys.executable, "-c", script],
            tmp_path,
            None,
            lambda: child_pid.exists() or time.monotonic() - started > 5,
        )
    assert child_pid.exists(), "The test must actually launch a descendant before cancellation"
    pid = int(child_pid.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert time.monotonic() - started < 8
