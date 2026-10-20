"""Sandbox executor: run model-authored build123d code in an isolated subprocess.

THREAT MODEL: contain BUGGY, NON-ADVERSARIAL generated geometry code (the author is
Opus, not an attacker). What it guarantees:
  - a hard wall-clock timeout kills hangs / infinite loops
  - a memory watchdog kills runaway tessellation / boolean blow-ups (and descendants)
  - an isolated subprocess means a crash / segfault never takes down the host process
  - cwd is the per-job dir; ANTHROPIC_API_KEY + proxies are stripped from the child env
  - an import guard (in the child) blocks network + subprocess modules

What it does NOT guarantee (would need a container to close): a determined adversary
could still write files via absolute paths. For UNTRUSTED third-party code, wrap this
in a container. Adequate for Wave 1 because the only code author is our own model.

Determinism: PYTHONHASHSEED is pinned and the build123d / OCP versions are recorded on
every result, so a fixed program reproduces across runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

_RUNNER_MODULE = "forge_mind.agentic._sandbox_runner"

# Non-secret infra env vars passed to the sandbox child (a WHITELIST). Everything else --
# notably ANTHROPIC_API_KEY / SUPABASE_* / STRIPE_* -- is withheld so authored code cannot read
# a secret from os.environ or /proc/self/environ. build123d/OCP need PATH (+ Windows shell vars
# on dev machines); the per-job dir is handed to the child via job.json, not the env, so no
# secret is required. See SECURITY.md (code-execution sandbox).
_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    # Windows essentials (stdlib / subprocess creation need these on Windows dev machines):
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "PATHEXT",
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
)

# Error types the child / harness can report (mirrors _sandbox_runner + parent guards).
ErrorType = str  # timeout|memory|exec|no_result|export|kernel_import|harness


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings for one sandboxed execution."""

    # A fresh subprocess COLD-imports the build123d/OCP kernel (several seconds) before the author's
    # code runs, so the wall-clock budget must cover startup, not just execution. 30s starved a cold
    # kernel under heavy parallel load (xdist sandbox tests timing out); 60s leaves ample headroom and
    # still fast-fails a runaway. The agentic loop overrides this per part as needed.
    timeout_s: float = 60.0
    memory_mb: int = 2048


@dataclass(frozen=True)
class SandboxResult:
    """Structured outcome of one sandboxed execution. ``ok`` => STL + STEP on disk."""

    ok: bool
    stl_path: Path | None = None
    step_path: Path | None = None
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    extents: tuple[float, float, float] | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None
    traceback: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    build123d_version: str | None = None
    ocp_version: str | None = None
    # The self-check epilogue's measured deltas (FORGE_AUTHOR_SELFCHECK; None when no spec
    # rode the job). Feedback only -- the validator still gates.
    selfcheck: dict[str, object] | None = None
    # Result-plumbing advisory (ADR-0117): the script built a solid much larger than the one
    # bound to ``result`` (the measured bottle_cap shipped-the-base-plate class). Feedback only.
    plumbing_note: str | None = None
    # Per-op measurement transcript (ADR-0119, FORGE_OP_TRACE): one entry per builder op --
    # what that op did to volume / solid count / extents. Feedback only, never gates.
    optrace: list[dict[str, object]] | None = None
    # ADR-0133 fix 3 (FORGE_EDGE_FINISH): safe_fillet/safe_chamfer outcome log --
    # requested vs realized radius/length, or a recorded SKIP. Feedback only, never gates.
    edge_finish: list[dict[str, object]] | None = None


def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants (best-effort)."""
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    victims = parent.children(recursive=True)
    victims.append(parent)
    for proc in victims:
        with contextlib.suppress(psutil.Error):
            proc.kill()


def _watch_memory(pid: int, cap_bytes: int, flag: dict[str, bool], done: threading.Event) -> None:
    """Poll the child (+ descendants) RSS; kill the tree if it exceeds ``cap_bytes``."""
    try:
        proc = psutil.Process(pid)
    except psutil.Error:
        return
    while not done.is_set():
        try:
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    rss += child.memory_info().rss
        except psutil.Error:
            return  # process already gone
        if rss > cap_bytes:
            flag["memory"] = True
            _kill_tree(pid)
            return
        done.wait(0.1)


def run_build123d_code(
    code: str,
    job_dir: Path,
    *,
    part_name: str = "part",
    limits: SandboxLimits | None = None,
    selfcheck: dict[str, object] | None = None,
    runner_module: str = _RUNNER_MODULE,
) -> SandboxResult:
    """Execute ``code`` (which must bind the final solid to ``result``) in the sandbox.

    Writes ``<job_dir>/<part_name>.stl`` + ``.step`` on success. Never raises on a
    failure of the authored code: timeouts, memory blow-ups, exceptions, and missing
    results all come back as a ``SandboxResult`` with ``ok=False`` and an ``error_type``.

    ``selfcheck`` (optional, FORGE_AUTHOR_SELFCHECK): the brief-derived measurement targets;
    when given, the child measures the built part against them and the result carries the
    structured deltas (``SandboxResult.selfcheck``). Feedback only -- never gates.

    ``runner_module`` selects the child harness. The default runs build123d code; the
    CadQuery bridge (ADR-0112) passes ``_cadquery_runner`` so CAD-Coder's CadQuery output
    reuses this exact crash-isolation / timeout / memory-watch driver unchanged -- the
    child differs only in how it seeds the namespace and exports the bound ``result``.
    """
    limits = limits or SandboxLimits()
    # Resolve to an ABSOLUTE path first. The child runs with cwd=job_dir and is handed the job-file
    # path as an argument, so a RELATIVE job_dir double-resolves in the child (cwd/job_dir/job.json),
    # the child cannot find its job, and it dies before writing result.json -- surfacing only as the
    # opaque "child crashed before writing result.json". Resolving here makes the sandbox robust to a
    # relative work_dir (an absolute path is unchanged).
    job_dir = job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "job.json"
    result_file = job_dir / "result.json"
    if result_file.exists():
        result_file.unlink()
    job_payload: dict[str, object] = {
        "code": code,
        "part_name": part_name,
        "out_dir": str(job_dir),
        "memory_mb": limits.memory_mb,
    }
    if selfcheck is not None:
        job_payload["selfcheck"] = selfcheck
    # ADR-0119: per-op measurement transcript. The parent env decides; the child obeys the
    # job flag (the child env is whitelisted, so an env read there would never see it).
    if os.environ.get("FORGE_OP_TRACE") == "1":
        job_payload["optrace"] = True
    # ADR-0133 fix 3: same parent-decides pattern as optrace.
    if os.environ.get("FORGE_EDGE_FINISH") == "1":
        job_payload["edge_finish"] = True
    job_file.write_text(json.dumps(job_payload), encoding="utf-8")

    # WHITELIST the child env (see _ENV_PASSTHROUGH): authored code cannot read secrets
    # (ANTHROPIC / SUPABASE / STRIPE) via os.environ or /proc/self/environ. SECURITY.md finding #3.
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env["PYTHONHASHSEED"] = "0"

    flag: dict[str, bool] = {"memory": False}
    done = threading.Event()
    start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-m", runner_module, str(job_file)],
        cwd=str(job_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    watcher = threading.Thread(
        target=_watch_memory,
        args=(proc.pid, limits.memory_mb * 1024 * 1024, flag, done),
        daemon=True,
    )
    watcher.start()

    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc.pid)
        out_b, err_b = proc.communicate()
    finally:
        done.set()
    duration = time.monotonic() - start
    stdout = (out_b or b"").decode("utf-8", "replace")
    stderr = (err_b or b"").decode("utf-8", "replace")

    if flag["memory"]:
        return SandboxResult(
            ok=False,
            error_type="memory",
            error_message=f"exceeded memory cap of {limits.memory_mb} MB",
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )
    if timed_out:
        return SandboxResult(
            ok=False,
            error_type="timeout",
            error_message=f"exceeded time limit of {limits.timeout_s:.0f}s",
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )
    if not result_file.exists():
        # A NATIVE kernel crash (OCC segfault / access violation) kills the child before it can
        # write result.json OR a Python traceback -- so stderr is usually empty and the exit code
        # is a signal/fault code, not 0. Distinguish it from a generic harness miss and route it
        # through the ACTIONABLE ``kernel_crash`` hint: the author's revision pass reads this text,
        # and it can only avoid the crash if told the crash happened and why (measured 2026-07-13:
        # a forearm segfaulted identically on 3 attempts because the old opaque message taught the
        # author nothing). A file that merely failed to appear WITHOUT a crash stays "harness".
        rc = proc.returncode
        if rc is not None and rc != 0 and not stderr.strip():
            return SandboxResult(
                ok=False,
                error_type="kernel_crash",
                error_message=(
                    f"the geometry kernel crashed natively (exit {rc}) with no Python error -- "
                    "the authored code triggered an OCC fault, not a script bug"
                ),
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
            )
        return SandboxResult(
            ok=False,
            error_type="harness",
            error_message="sandbox produced no result.json (child crashed before writing)",
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )
    return parse_result_file(result_file, stdout=stdout, stderr=stderr, duration_s=duration)


def parse_result_file(
    result_file: Path, *, stdout: str = "", stderr: str = "", duration_s: float = 0.0
) -> SandboxResult:
    """Parse a child-written ``result.json`` into a SandboxResult (shared by the one-shot
    sandbox above and the warm pool, so both paths return byte-identical structures)."""
    duration = duration_s
    # Guard the read+parse: even though the child writes result.json atomically, a missing file
    # (raced/unlinked), an empty file, or corrupt JSON must surface as the SAME honest structured
    # harness failure -- never an unhandled OSError / JSONDecodeError that crashes the parent and
    # turns a containable bad part into a host-level exception.
    try:
        raw = result_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("result.json is not a JSON object")
    except (OSError, ValueError) as exc:
        return SandboxResult(
            ok=False,
            error_type="harness",
            error_message=f"sandbox result.json is missing/empty/corrupt: {type(exc).__name__}: {exc}",
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
        )
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    extents: tuple[float, float, float] | None = None
    if data.get("bbox"):
        mn, mx = data["bbox"]
        bbox = ((mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2]))
    if data.get("extents"):
        e = data["extents"]
        extents = (e[0], e[1], e[2])

    ok = bool(data.get("ok"))
    stl_path = Path(data["stl_path"]) if data.get("stl_path") else None
    step_path = Path(data["step_path"]) if data.get("step_path") else None
    error_type = data.get("error_type")
    error_message = data.get("error_message")
    # Defensive: a child that reports ok MUST have actually left the artifacts on disk for the
    # downstream loader. A missing file (an export that silently produced nothing) is a harness
    # failure, surfaced cleanly here rather than as a crash in validate/trimesh downstream.
    missing = (
        stl_path is None or not stl_path.exists() or step_path is None or not step_path.exists()
    )
    if ok and missing:
        ok = False
        error_type = "harness"
        error_message = "sandbox reported ok but the STL/STEP artifact is not on disk"

    sc = data.get("selfcheck")
    return SandboxResult(
        ok=ok,
        stl_path=stl_path,
        step_path=step_path,
        bbox=bbox,
        extents=extents,
        error_type=error_type,
        error_message=error_message,
        traceback=data.get("traceback"),
        stdout=stdout,
        stderr=stderr,
        duration_s=duration,
        build123d_version=data.get("build123d_version"),
        ocp_version=data.get("ocp_version"),
        selfcheck=sc if isinstance(sc, dict) else None,
        plumbing_note=data.get("plumbing_note") or None,
        optrace=data.get("optrace") if isinstance(data.get("optrace"), list) else None,
        edge_finish=(
            data.get("edge_finish") if isinstance(data.get("edge_finish"), list) else None
        ),
    )


__all__ = ["SandboxLimits", "SandboxResult", "parse_result_file", "run_build123d_code"]
