from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.agentic.sandbox import SandboxResult
from forge_mind.workspace import executor, processing

THREAD_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@pytest.mark.parametrize("cpu_limit", [None, 1])
def test_actual_local_worker_receives_thread_policy_before_native_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_limit: int | None
) -> None:
    for name in THREAD_NAMES:
        monkeypatch.setenv(name, "64")
    monkeypatch.setenv("CADKITTY_THREAD_TEST_SECRET", "parent-only")
    parent = dict(os.environ)
    # Capture the environment before importing the native math library. This
    # exercises a real child, rather than only inspecting a mocked Popen call.
    script = (
        "import os, json, sys; "
        f"names = {THREAD_NAMES!r}; "
        "before = {name: os.environ.get(name) for name in names}; "
        "not_loaded = 'numpy' not in sys.modules; "
        "import numpy; "
        "from pathlib import Path; "
        "Path('environment.json').write_text(json.dumps({"
        "'before_import': before, 'numpy_not_loaded_before': not_loaded, "
        "'after_import': {name: os.environ.get(name) for name in names}, "
        "'secret': os.environ.get('CADKITTY_THREAD_TEST_SECRET')}))"
    )
    processing.bounded_process(
        [sys.executable, "-c", script], tmp_path, 20, lambda: False, cpu_limit=cpu_limit
    )
    data = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert data["numpy_not_loaded_before"]
    assert data["before_import"] == dict.fromkeys(THREAD_NAMES, "1")
    assert data["after_import"] == data["before_import"]
    assert data["secret"] is None
    assert dict(os.environ) == parent


@pytest.mark.parametrize("route", ["typed", "authored"])
def test_both_docker_routes_override_thread_defaults_without_changing_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    for name in THREAD_NAMES:
        monkeypatch.setenv(name, "64")
    parent = dict(os.environ)
    limits = dict(processing.HOSTED_LIMITS)
    commands: list[list[str]] = []
    removals: list[list[str]] = []

    class Process:
        returncode = 0
        waited = False

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> int:
            self.waited = True
            return 0

    process = Process()

    def launch(command: list[str], **kwargs: Any) -> Process:
        commands.append(command)
        return process

    def remove(command: list[str], **kwargs: Any) -> SimpleNamespace:
        removals.append(command)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    monkeypatch.setattr(executor.subprocess, "run", remove)
    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        executor.Executor, "_result", staticmethod(lambda *args: SandboxResult(ok=True))
    )
    if route == "typed":
        executor.Executor._run_container(
            "docker",
            tmp_path,
            "worker.module",
            "request.json",
            1,
            lambda: False,
            image="sha256:" + "a" * 64,
        )
    else:
        assert executor.Executor("docker").build("", tmp_path, 1, lambda: False).ok
    command = commands[0]
    env = [command[index + 1] for index, value in enumerate(command) if value == "--env"]
    for name in THREAD_NAMES:
        assert env.count(name + "=1") == 1
        assert all(not value.startswith(name + "=") or value == name + "=1" for value in env)
    for flag, value in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--user", "65534:65534"),
        ("--pids-limit", "64"),
        ("--memory", "2147483648"),
        ("--cpus", "1"),
        ("--ulimit", "fsize=67108864:67108864"),
        ("--tmpfs", "/tmp:rw,nosuid,size=268435456"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--read-only" in command
    assert command.count("--mount") == 1
    name = command[command.index("--name") + 1]
    assert removals == [["docker", "rm", "-f", name]]
    assert process.waited
    assert limits == processing.HOSTED_LIMITS
    assert dict(os.environ) == parent
