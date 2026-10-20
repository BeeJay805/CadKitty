"""m1_probe.run_m1 must surface the REAL LoopResult cost (the subscription spend-leak tell), not a
hardwired 0.0 (regression: it read a nonexistent `cost` attr -> always 0.0). Loaded from its path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parents[3] / "scripts" / "arm_search" / "m1_probe.py"
_spec = importlib.util.spec_from_file_location("arm_m1_probe_test", _SRC)
assert _spec and _spec.loader
m1p = importlib.util.module_from_spec(_spec)
sys.modules["arm_m1_probe_test"] = m1p
_spec.loader.exec_module(m1p)


def test_run_m1_reports_real_cost_not_hardwired_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A no-build run carrying a NONZERO total_cost_usd: the report must surface it (leak detector).
    stub = SimpleNamespace(
        final_stl=None, total_cost_usd=0.42, stopped_reason="no_build", geometry_passed=False
    )
    monkeypatch.setattr(m1p, "run_agentic_pipeline", lambda *a, **k: stub)
    rep = m1p.run_m1(
        "p",
        target_mass_g=100.0,
        ref_mass_g=145.0,
        ref_deflection_mm=0.05,
        material_id="pla",
        out_dir=tmp_path,
        max_iterations=1,
    )
    assert rep["cost_usd"] == 0.42  # NOT 0.0 -- the spend-leak tell actually reflects real cost
    assert rep["verdict"] == "no_build"
    assert rep["built"] is False
