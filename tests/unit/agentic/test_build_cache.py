"""Content-hash build cache: the determinism handle. The key is prompt+settings (brief-independent),
and a hit restores the SAME geometry at zero cost -- so a repeat of the same prompt returns the same
part on the live path (evaluator #1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import build_cache
from forge_mind.agentic.loop import LoopResult
from forge_mind.fabrication import FabricationMode

_KW = {
    "material": None,
    "n_drafts": 1,
    "max_iterations": 4,
    "fabrication_mode": FabricationMode.PRINT_3D,
    "author_model": "opus",
    "critic_model": None,
    "brief_model": None,
}


def test_compute_key_is_stable_and_prompt_sensitive() -> None:
    k1 = build_cache.compute_key("a 40mm bracket", **_KW)
    k2 = build_cache.compute_key("a 40mm bracket", **_KW)
    k3 = build_cache.compute_key("a different part", **_KW)
    assert k1 == k2  # same prompt + settings -> same key (so the repeat hits the cache)
    assert k1 != k3


def test_round_trip_restores_geometry_for_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_BUILD_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "src"
    src.mkdir()
    stl = src / "final.stl"
    stl.write_text("solid x\nendsolid x\n")
    result = LoopResult(
        prompt="a 40mm bracket",
        brief=None,
        final_code="result = None",
        final_stl=stl,
        final_step=None,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=True,
        critic_approved=False,
        stopped_reason="passed",
        iterations=[],
        total_cost_usd=0.5,
        total_duration_s=1.0,
    )
    key = build_cache.compute_key("a 40mm bracket", **_KW)

    build_cache.save(key, result)
    loaded = build_cache.load(key, tmp_path / "work")

    assert loaded is not None
    assert loaded.geometry_passed is True
    assert loaded.stopped_reason == "cached"
    assert loaded.final_stl is not None and loaded.final_stl.exists()
    assert loaded.total_cost_usd == 0.0  # a hit is free


def test_disabled_cache_is_a_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_BUILD_CACHE", raising=False)
    assert build_cache.enabled() is False
    assert build_cache.load(build_cache.compute_key("x", **_KW), tmp_path) is None


def test_modified_cached_geometry_cannot_reuse_its_recorded_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_BUILD_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "source.stl"
    source.write_text("solid original\nendsolid original\n", encoding="utf-8")
    result = LoopResult(
        prompt="p",
        brief=None,
        final_code=None,
        final_stl=source,
        final_step=None,
        final_render=None,
        validation=None,
        critique=None,
        geometry_passed=True,
        critic_approved=True,
        stopped_reason="passed",
        iterations=[],
    )
    key = build_cache.compute_key("p", **_KW)
    build_cache.save(key, result)
    (tmp_path / "cache" / key / "final.stl").write_text("tampered", encoding="utf-8")
    assert build_cache.load(key, tmp_path / "work") is None
