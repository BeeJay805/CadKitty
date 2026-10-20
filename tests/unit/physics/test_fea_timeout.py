"""FORGE_FEA_TIMEOUT_S raises the subprocess FEA budget for offline batch callers (arm search).

Arm-scale parts (~35k tets) solve in 2-3 min, far past the live server's 30 s cap, and offline there
is no request worker to protect. The default stays 30 s so the live server is unchanged.
"""

from __future__ import annotations

import pytest

from forge_mind.physics import fea


def test_fea_timeout_defaults_to_30() -> None:
    assert fea._fea_timeout_s() == 30.0


def test_fea_timeout_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", "180")
    assert fea._fea_timeout_s() == 180.0
    monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", "45.5")
    assert fea._fea_timeout_s() == 45.5


def test_fea_timeout_fails_soft_on_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank or malformed value must NOT raise (it would crash the request worker through
    _fea_run) -- it falls back to the default, like max_param_evals / _build_workers."""
    for bad in ("", "   ", "400s", "abc", "4 00"):
        monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", bad)
        assert fea._fea_timeout_s() == 30.0
