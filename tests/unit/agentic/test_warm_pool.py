"""Warm sandbox workers (engine research idea 1): the fit's re-runs skip the kernel
cold-import tax WITHOUT giving up crash containment, namespace freshness, or honest
structured failures. The leakage canary and warm-vs-cold identity here are load-bearing:
a warm pool that changes geometry or leaks state is worse than a slow one."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.agentic.warm_pool import WarmPool, WarmWorker

CODE = "result = Box(20, 10, 5)"
LIMITS = SandboxLimits(timeout_s=120)


@pytest.fixture(scope="module")
def worker() -> WarmWorker:  # type: ignore[misc]
    w = WarmWorker()
    yield w
    w.kill()


def _volume(stl: Path) -> float:
    return float(trimesh.load(str(stl), force="mesh").volume)


def test_warm_geometry_identical_to_cold(tmp_path: Path, worker: WarmWorker) -> None:
    """The identity requirement: the warm path must produce the SAME artifact as the cold
    sandbox for the same code -- byte-level volume agreement, not approximately."""
    cold = run_build123d_code(CODE, tmp_path / "cold", part_name="p", limits=LIMITS)
    warm = worker.run(CODE, tmp_path / "warm", part_name="p", limits=LIMITS)
    assert cold.ok and warm.ok
    assert cold.stl_path is not None and warm.stl_path is not None
    assert _volume(cold.stl_path) == _volume(warm.stl_path) == pytest.approx(1000.0)


def test_fresh_namespace_per_job(tmp_path: Path, worker: WarmWorker) -> None:
    """The leakage canary: a name bound by job 1 must NOT exist in job 2 -- each eval runs in
    a fresh exec namespace even inside one warm process."""
    first = worker.run(
        "LEAK = 7\nresult = Box(1, 1, 1)", tmp_path / "a", part_name="p", limits=LIMITS
    )
    assert first.ok
    second = worker.run("result = Box(LEAK, 1, 1)", tmp_path / "b", part_name="p", limits=LIMITS)
    assert not second.ok
    assert second.error_type == "exec"  # NameError: the leak did not survive


def test_worker_respawns_after_death(tmp_path: Path, worker: WarmWorker) -> None:
    """Crash containment: kill the child mid-life; the next run transparently respawns."""
    assert worker.run(CODE, tmp_path / "before", part_name="p", limits=LIMITS).ok
    worker.kill()
    after = worker.run(CODE, tmp_path / "after", part_name="p", limits=LIMITS)
    assert after.ok


def test_timeout_is_structured_and_worker_recovers(tmp_path: Path, worker: WarmWorker) -> None:
    """A hung candidate: honest timeout result, worker killed, next eval works again."""
    hung = worker.run(
        "x = 0\nwhile True:\n    x += 1\nresult = Box(1, 1, 1)",
        tmp_path / "hang",
        part_name="p",
        limits=SandboxLimits(timeout_s=4),
    )
    assert not hung.ok
    assert hung.error_type == "timeout"
    again = worker.run(CODE, tmp_path / "again", part_name="p", limits=LIMITS)
    assert again.ok


def test_pool_serializes_and_survives(tmp_path: Path) -> None:
    pool = WarmPool(1)
    try:
        r1 = pool.run(CODE, tmp_path / "p1", part_name="p", limits=LIMITS)
        r2 = pool.run(
            "result = Cylinder(radius=5, height=4)", tmp_path / "p2", part_name="p", limits=LIMITS
        )
        assert r1.ok and r2.ok
    finally:
        pool.close()
