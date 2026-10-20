from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.llm import codex_client
from forge_mind.workspace import budget, executor, processing
from forge_mind.workspace.models import Action


class FakeProcess:
    pid = 12345
    returncode = 0

    def __init__(self) -> None:
        self.stdin = io.BytesIO()
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return None if self.polls < 3 else 0

    def wait(self, **kwargs: Any) -> int:
        return 0


@pytest.mark.parametrize("timeout", [None, 500.0])
def test_worker_survives_old_cad_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        processing, "time", SimpleNamespace(sleep=lambda _: setattr(clock, "now", 190.0))
    )
    monkeypatch.setattr(processing.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1), children=lambda **kw: [])
    monkeypatch.setattr(processing.psutil, "Process", lambda _: process)
    killed: list[int] = []
    monkeypatch.setattr(processing, "_kill_tree", killed.append)
    processing.bounded_process(["fake"], tmp_path, timeout, lambda: False)
    assert killed == [FakeProcess.pid]
    assert clock.now == 190.0


@pytest.mark.parametrize("timeout", [None, 500.0])
def test_processing_passes_budget_without_local_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None
) -> None:
    observed = []

    def run(
        command: Any, directory: Path, seconds: float | None, cancelled: Any, **kwargs: Any
    ) -> None:
        observed.append(seconds)
        (directory / "processed.json").write_text("{}")

    monkeypatch.setattr(processing, "bounded_process", run)
    processing.process_revision(tmp_path, {"operation": "engineering"}, timeout, lambda: False)
    assert observed == [timeout]


@pytest.mark.parametrize("timeout", [None, 500.0])
def test_local_executor_passes_budget_without_local_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None
) -> None:
    observed = []
    monkeypatch.setattr(
        executor, "bounded_process", lambda cmd, path, seconds, cancel: observed.append(seconds)
    )
    executor.Executor("trusted_local").build("", tmp_path, timeout, lambda: False)
    assert observed == [timeout]


@pytest.mark.parametrize("timeout", [None, 500.0])
def test_model_survives_old_model_cap(
    monkeypatch: pytest.MonkeyPatch, timeout: float | None
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        codex_client, "time", SimpleNamespace(sleep=lambda _: setattr(clock, "now", 400.0))
    )
    monkeypatch.setattr(codex_client, "_command", lambda: ["fake"])

    def launch(command: list[str], **kwargs: Any) -> FakeProcess:
        answer = Path(command[command.index("--output-last-message") + 1])
        answer.write_text(json.dumps({"kind": "finish", "explanation": "Done"}))
        return FakeProcess()

    monkeypatch.setattr(codex_client.subprocess, "Popen", launch)
    result, _ = codex_client.CodexClient(timeout_s=timeout).generate_structured(
        system="", user="", output_model=Action, model="gpt-6-astra", max_tokens=10
    )
    assert result.kind == "finish"
    assert clock.now == 400.0


@pytest.mark.parametrize(
    "timeout,cancel,error", [(None, True, "cancelled"), (100.0, False, "wall-clock")]
)
def test_worker_still_stops_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, timeout: float | None, cancel: bool, error: str
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        processing, "time", SimpleNamespace(sleep=lambda _: setattr(clock, "now", 190.0))
    )
    monkeypatch.setattr(processing.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1), children=lambda **kw: [])
    monkeypatch.setattr(processing.psutil, "Process", lambda _: process)
    killed: list[int] = []
    monkeypatch.setattr(processing, "_kill_tree", killed.append)
    with pytest.raises(ValueError, match=error):
        processing.bounded_process(["fake"], tmp_path, timeout, lambda: cancel and clock.now > 0)
    assert killed == [FakeProcess.pid]


@pytest.mark.parametrize(
    "timeout,cancel,expected",
    [
        (None, False, None),
        (500.0, False, None),
        (100.0, False, "timeout"),
        (None, True, "cancelled"),
    ],
)
def test_docker_optional_budget_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timeout: float | None,
    cancel: bool,
    expected: str | None,
) -> None:
    from forge_mind.agentic.sandbox import SandboxResult

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(budget, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(
        executor,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda _: setattr(clock, "now", 190.0)),
    )
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *a, **kw: FakeProcess())
    removed = []
    monkeypatch.setattr(executor.subprocess, "run", lambda cmd, **kw: removed.append(cmd))
    monkeypatch.setattr(executor.Executor, "_result", lambda *a: SandboxResult(ok=True))
    result = executor.Executor("docker").build(
        "", tmp_path, timeout, lambda: cancel and clock.now > 0
    )
    assert result.ok == (expected is None)
    assert result.error_type == expected
    assert len(removed) == 1
    assert removed[0][:3] == ["docker", "rm", "-f"]
