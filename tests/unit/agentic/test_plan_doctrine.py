"""Plan doctrine + iteration scaling (iteration 15): the L86 plan-level walls."""

from __future__ import annotations

import pytest

from forge_mind.agentic.functional import _PLAN_DOCTRINE, _scaled_iterations


def test_plan_doctrine_names_the_measured_classes() -> None:
    # The block must carry the two L86 classes verbatim enough to teach them.
    assert "SPLIT FEATURE-DENSE" in _PLAN_DOCTRINE
    assert "HOME-POSE CLEARANCE" in _PLAN_DOCTRINE
    assert "shin" in _PLAN_DOCTRINE and "hip" in _PLAN_DOCTRINE


def test_iteration_scaling_is_flag_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    dense = "a torso 120 x 80 x 40 with 4 bays 41.3 x 20.3, holes d4.5 on 49.5 x 10.0, wall 3.0"
    monkeypatch.delenv("FORGE_ITER_SCALE", raising=False)
    assert _scaled_iterations(dense, 3) == 3  # off = byte-identical budgets
    monkeypatch.setenv("FORGE_ITER_SCALE", "1")
    assert _scaled_iterations(dense, 3) == 5  # dense part earns +2
    assert _scaled_iterations("a plain spacer d10 x 5", 3) == 3  # simple part unchanged
