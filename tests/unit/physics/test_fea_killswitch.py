"""The FEA kill switch (incident mitigation). The native tetrahedralizer (pytetwild/fTetWild) can OOM
or segfault on some geometry, which kills the worker process (uncatchable in-process) and 502s the
whole pre-print check. FORGE_FEA_STRESS=0 disables the ILLUSTRATIVE stress lens entirely so the honest
verdict can never be taken down by it.

This runs WITHOUT the FEA deps (the guard returns before importing them); in CI, where the deps ARE
present, it proves the flag overrides a working FEA (estimate_stress would otherwise return a field)."""

from __future__ import annotations

import pytest
import trimesh

from forge_mind.physics.fea import estimate_stress


def test_killswitch_returns_no_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FEA_STRESS", "0")
    box = trimesh.creation.box(extents=(60.0, 20.0, 20.0))
    box.apply_translation((0.0, 0.0, 10.0))
    assert estimate_stress(box) is None  # disabled -> NO-RESULT, no native tetrahedralize call


def test_explicit_on_does_not_killswitch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The flag is opt-OUT: only "0" disables. "1" (or unset) proceeds normally -- it must NOT raise.
    # (With FEA deps absent it still degrades to None downstream; with them present it solves.)
    monkeypatch.setenv("FORGE_FEA_STRESS", "1")
    box = trimesh.creation.box(extents=(60.0, 20.0, 20.0))
    box.apply_translation((0.0, 0.0, 10.0))
    estimate_stress(box)  # must not raise regardless of whether the deps are installed
