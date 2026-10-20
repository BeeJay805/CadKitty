"""fea_is_validated() must FAIL CLOSED: a solver raise returns False, never propagates (audit).

The bool is the single switch callers check before reporting a decision-grade FEA number. The
contract is "False = not proven correct" -- so any failure to run the harness (scikit-fem absent OR
a solver raise) must read as not-validated, not crash the caller. This test monkeypatches the harness
to raise, so it needs no scikit-fem.
"""

from __future__ import annotations

import pytest

import forge_mind.physics.fea_validation as fv


def test_fea_is_validated_returns_false_when_harness_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> list[object]:
        raise RuntimeError("skfem solve blew up")

    monkeypatch.setattr(fv, "run_validation", _boom)
    fv.fea_is_validated.cache_clear()  # lru_cached: clear before (and after) so no stale/leaked result
    try:
        assert fv.fea_is_validated() is False
    finally:
        fv.fea_is_validated.cache_clear()
