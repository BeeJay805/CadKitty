"""Warm sandbox workers for the parametric fit (engine research idea 1, 2026-07-03).

The (1+lambda)-ES polish re-runs the SAME authored script 24-48 times with different PARAMS
floats -- code that already survived the full one-shot sandbox. The measured cost of treating
each re-run as hostile was 7.3 s of subprocess spawn + kernel cold-import per evaluation (18%
of a 41.4 s eval; warm-process floor 26.3 s). This pool keeps a few `_sandbox_runner --loop`
children alive with the kernel imported ONCE and feeds them one job per stdin line.

What is kept from the sandbox's guarantees, by construction:
  * crash containment -- a worker that dies, hangs, or blows the memory cap is killed and
    respawned; only its own candidate fails (scored +inf upstream, same as today);
  * the import guard, env whitelist, and per-job fresh exec namespace run IN the child
    exactly as in one-shot mode (same runner code path);
  * fail-closed to the SLOW path -- any worker/harness failure makes the caller fall back to
    the cold one-shot sandbox for that evaluation (param_opt), so the pool can only ever cost
    time, never correctness.

What is deliberately NOT offered: running FIRST-CONTACT authored code. The one-shot sandbox
with its per-run cold isolation remains the only entry point for code that has never built.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from forge_mind.agentic.sandbox import (
    _ENV_PASSTHROUGH,
    SandboxLimits,
    SandboxResult,
    _kill_tree,
    _watch_memory,
    parse_result_file,
)

_RUNNER_MODULE = "forge_mind.agentic._sandbox_runner"
_SPAWN_TIMEOUT_S = 60.0  # kernel cold-import happens once, at spawn ("READY")


class WarmWorker:
    """One persistent runner child. Not thread-safe; the pool serializes access per worker."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[str] = queue.Queue()

    # -- lifecycle ---------------------------------------------------------------------------

    def _spawn(self) -> bool:
        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env["PYTHONHASHSEED"] = "0"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", "-m", _RUNNER_MODULE, "--loop"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        except Exception:
            self._proc = None
            return False
        # The reader thread gets ITS OWN queue object: a dying reader from a killed worker must
        # never deliver its EOF sentinel into a respawned worker's queue (the caught race: the
        # sentinel arrived during the new worker's READY handshake and failed the respawn).
        lines: queue.Queue[str] = queue.Queue()
        self._lines = lines
        threading.Thread(target=self._read_stdout, args=(self._proc, lines), daemon=True).start()
        return self._wait_line(_SPAWN_TIMEOUT_S) == "READY"

    def _read_stdout(self, proc: subprocess.Popen[bytes], lines: queue.Queue[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            lines.put(raw.decode("utf-8", "replace").strip())
        lines.put("")  # EOF sentinel: the child exited

    def _wait_line(self, timeout_s: float) -> str | None:
        try:
            return self._lines.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def kill(self) -> None:
        if self._proc is not None:
            _kill_tree(self._proc.pid)
            self._proc = None

    # -- one evaluation ----------------------------------------------------------------------

    def run(
        self, code: str, job_dir: Path, *, part_name: str, limits: SandboxLimits
    ) -> SandboxResult:
        """Execute one job in the warm child. Any worker-level failure returns a structured
        harness/timeout/memory result (never raises); the worker is respawned lazily."""
        if not self._alive() and not self._spawn():
            return SandboxResult(
                ok=False, error_type="harness", error_message="warm worker failed to spawn"
            )
        assert self._proc is not None and self._proc.stdin is not None
        job_dir = job_dir.resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        result_file = job_dir / "result.json"
        if result_file.exists():
            result_file.unlink()
        job_file = job_dir / "job.json"
        job_file.write_text(
            json.dumps(
                {
                    "code": code,
                    "part_name": part_name,
                    "out_dir": str(job_dir),
                    "memory_mb": limits.memory_mb,
                }
            ),
            encoding="utf-8",
        )
        flag: dict[str, bool] = {"memory": False}
        done = threading.Event()
        threading.Thread(
            target=_watch_memory,
            args=(self._proc.pid, limits.memory_mb * 1024 * 1024, flag, done),
            daemon=True,
        ).start()
        start = time.monotonic()
        try:
            self._proc.stdin.write(f"{job_file}\n".encode())
            self._proc.stdin.flush()
        except Exception:
            done.set()
            self.kill()
            return SandboxResult(
                ok=False, error_type="harness", error_message="warm worker pipe broke"
            )
        line = self._wait_line(limits.timeout_s)
        done.set()
        duration = time.monotonic() - start
        if flag["memory"]:
            self.kill()  # a worker past the cap is not reusable
            return SandboxResult(
                ok=False,
                error_type="memory",
                error_message=f"exceeded memory cap of {limits.memory_mb} MB",
                duration_s=duration,
            )
        if line is None or not line.startswith("DONE"):
            self.kill()  # hang or death mid-job: the worker is gone, the candidate failed
            return SandboxResult(
                ok=False,
                error_type="timeout" if line is None else "harness",
                error_message=(
                    f"exceeded time limit of {limits.timeout_s:.0f}s (warm)"
                    if line is None
                    else "warm worker exited mid-job"
                ),
                duration_s=duration,
            )
        return parse_result_file(result_file, duration_s=duration)


class WarmPool:
    """A small pool of warm workers with checkout semantics (one eval per worker at a time)."""

    def __init__(self, size: int) -> None:
        self._free: queue.Queue[WarmWorker] = queue.Queue()
        for _ in range(max(1, size)):
            self._free.put(WarmWorker())

    def run(
        self, code: str, job_dir: Path, *, part_name: str, limits: SandboxLimits
    ) -> SandboxResult:
        worker = self._free.get()
        try:
            return worker.run(code, job_dir, part_name=part_name, limits=limits)
        finally:
            self._free.put(worker)

    def close(self) -> None:
        while True:
            try:
                self._free.get_nowait().kill()
            except queue.Empty:
                return


__all__ = ["WarmPool", "WarmWorker"]
