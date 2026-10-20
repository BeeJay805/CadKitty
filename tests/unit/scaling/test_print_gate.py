"""Step 4 (ADR-0016 Q3): the budget scale-run is BLOCKED until the oracle is
print-calibrated. The small-scale loop validation is NOT gated."""

from __future__ import annotations

import pytest

from forge_mind.scaling.engine import require_print_calibration


def test_scale_run_blocked_by_default() -> None:
    # The bottleneck is oracle trust: the budget run must not start without print
    # calibration of the validator.
    with pytest.raises(RuntimeError, match="print-calibrat"):
        require_print_calibration()


def test_scale_run_allowed_once_calibrated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forge_mind.config.SCALING_PRINT_CALIBRATED", True)
    require_print_calibration()  # no raise once the print-calibration flag is set
