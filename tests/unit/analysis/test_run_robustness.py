"""The analysis runner must survive a trial that blows up mid-pipeline.

One malformed LLM output (e.g. a disconnected body that breaks STL export) must
not abort a 30-trial run -- it should be recorded as a failed trial and the run
continues.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.analysis.run import run_failure_analysis

REPO = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO / "benchmarks" / "bench-001-nema17-mount-v1"


def test_one_exploding_trial_does_not_abort_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(prompt: str, output_dir: Path):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated pipeline explosion")

    # Patch the symbol used inside run.py.
    monkeypatch.setattr("forge_mind.analysis.run.run_pipeline", _boom)

    dataset = run_failure_analysis(
        BENCH_DIR, n_trials=3, output_root=tmp_path, run_id="robust", progress=False
    )

    # All three trials recorded (not aborted), all failed with an error.
    assert dataset.n_trials == 3
    assert dataset.n_passed == 0
    for rec in dataset.records:
        assert not rec.passed
        assert rec.first_failing_stage == "harness"
        assert rec.error is not None and "RuntimeError" in rec.error

    # The report still gets written.
    assert (tmp_path / "bench-001-nema17-mount-v1" / "robust" / "report.md").exists()
