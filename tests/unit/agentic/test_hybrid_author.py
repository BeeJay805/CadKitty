"""T4 hybrid physics-authoring hook: a declared structural link is physics-authored via
author_part instead of the LLM (byte-inert unless FORGE_PHYSICS_PARTS + FORGE_PHYSICS_CASE are
set; non-claimable -> falls through). The load case is OPERATOR-DECLARED, never invented; the
authored link is rotated so its +X length axis aligns with the port-to-port vector.

author_part (the T2 dispatch loop) is mocked at its module boundary -- these tests cover the
HOOK: gating, spec synthesis, claimable splice, refusal fall-through, and the orientation fix."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trimesh

import forge_mind.structures.dispatch as dispatch_mod
from forge_mind.agentic.hybrid_author import (
    _parse_case,
    _port_span,
    _synthesize_spec,
    try_hybrid_physics_author,
)
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.loadcase import G_M_S2


def _port(name: str, x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, origin=SimpleNamespace(x=x, y=y, z=z))


def _part(name: str = "upper_arm", ports: list[Any] | None = None) -> SimpleNamespace:
    # default: X-spanning bores (the M2-arm v4 layout: shoulder_bore x=0, elbow_bore x=120)
    ports = ports or [_port("shoulder_bore", 0, 0, 94), _port("elbow_bore", 120, 0, 94)]
    return SimpleNamespace(name=name, prompt=f"the {name} link, an open I-beam", ports=ports)


def _claimable_author(tmp: Path, *, long_axis: str = "x"):
    """A fake author_part whose analytic geometry is a bar 100 long on ``long_axis``."""

    def author(spec: Any, work_dir: Path, **kwargs: Any) -> dict[str, Any]:
        assert "agentic" in kwargs.get("authors", {})  # the LLM floor is a refusal
        work_dir.mkdir(parents=True, exist_ok=True)
        stl = work_dir / "analytic_link.stl"
        extents = {"x": (100, 10, 6), "y": (10, 100, 6)}[long_axis]
        trimesh.creation.box(extents=extents).export(str(stl))
        gate = {
            "checks": {"artifact": "pass", "deflection_bound": "pass", "full_hard_ok": "pass"},
            "measured": {"mass_g": 40.0},
            "claimable": True,
        }
        final = AuthorResult(provenance="analytic_isection", stl=stl, step=None, self_reported={})
        return {
            "chosen": "analytic",
            "attempts": [
                {
                    "rung": "analytic",
                    "provenance": "analytic_isection",
                    "gate": gate,
                    "self_reported": {},
                }
            ],
            "final": final,
        }

    return author


@pytest.fixture(autouse=True)
def _quiet_train_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_TRAIN_LOG", "0")  # never pollute the dataset from a test


def test_byte_inert_without_the_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FORGE_PHYSICS_PARTS", raising=False)
    monkeypatch.delenv("FORGE_PHYSICS_CASE", raising=False)

    def _boom(*a: object, **k: object) -> object:
        raise AssertionError("author_part must not run when the hook is disabled")

    monkeypatch.setattr(dispatch_mod, "author_part", _boom)
    assert (
        try_hybrid_physics_author(_part(), "upper_arm", tmp_path, SimpleNamespace(name="pla"))
        is None
    )
    # env present but part not in the target set -> also inert
    monkeypatch.setenv("FORGE_PHYSICS_PARTS", "forearm")
    monkeypatch.setenv("FORGE_PHYSICS_CASE", "payload_kg=0.5;bound_mm=0.046")
    assert (
        try_hybrid_physics_author(
            _part("upper_arm"), "upper_arm", tmp_path, SimpleNamespace(name="pla")
        )
        is None
    )


def test_claimable_link_is_physics_authored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_PARTS", "upper_arm,forearm")
    monkeypatch.setenv("FORGE_PHYSICS_CASE", "payload_kg=0.5;bound_mm=0.046;hy=24;wx=64")
    monkeypatch.setattr(dispatch_mod, "author_part", _claimable_author(tmp_path))
    lr = try_hybrid_physics_author(
        _part("upper_arm"), "upper_arm", tmp_path, SimpleNamespace(name="pla")
    )
    assert lr is not None
    assert lr.final_code is None and lr.stopped_reason == "accepted"  # resumed-style
    assert lr.geometry_passed is True and lr.resumed_hard_ok is True
    assert lr.final_stl is not None and lr.final_stl.exists()


def test_non_claimable_falls_through_to_the_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_PARTS", "upper_arm")
    monkeypatch.setenv("FORGE_PHYSICS_CASE", "payload_kg=50.0;bound_mm=0.001")  # infeasible

    def _refusing(spec: Any, work_dir: Path, **kwargs: Any) -> dict[str, Any]:
        gate = {"checks": {"deflection_bound": "fail"}, "measured": {}, "claimable": False}
        final = AuthorResult(provenance="analytic_isection", stl=None, step=None, self_reported={})
        return {
            "chosen": "analytic",
            "attempts": [
                {
                    "rung": "analytic",
                    "provenance": "analytic_isection",
                    "gate": gate,
                    "self_reported": {},
                }
            ],
            "final": final,
        }

    monkeypatch.setattr(dispatch_mod, "author_part", _refusing)
    assert (
        try_hybrid_physics_author(
            _part("upper_arm"), "upper_arm", tmp_path, SimpleNamespace(name="pla")
        )
        is None
    )


def test_spec_synthesis_declared_case_and_port_span() -> None:
    case = _parse_case("payload_kg=0.5;bound_mm=0.046;hy=24;wx=64")
    assert case == {"payload_kg": 0.5, "bound_mm": 0.046, "hy": 24.0, "wx": 64.0}
    part = _part("upper_arm")
    span = _port_span(part)
    assert span is not None and abs(span[0] - 120.0) < 1e-6  # x=0 -> x=120
    assert span[1] == (1.0, 0.0, 0.0)  # X-spanning (v4 layout) -> no rotation needed
    spec = _synthesize_spec(part, span[0], case, "pla")
    assert spec.envelope_mm == (120.0, 24.0, 64.0)  # (length, hy, wx)
    assert spec.load is not None and abs(spec.load.force_n - 0.5 * G_M_S2) < 1e-9  # DECLARED
    assert spec.deflection_bound_mm == 0.046
    assert spec.components == []  # bare span -> analytic


def test_orientation_rotates_the_length_axis_onto_the_port_vector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plan whose link ports span Y: the analytic geometry (built along +X) must be rotated so
    its long axis becomes Y, or the translation-only seat could never place it (the fix's point)."""
    monkeypatch.setenv("FORGE_PHYSICS_PARTS", "upper_arm")
    monkeypatch.setenv("FORGE_PHYSICS_CASE", "payload_kg=0.5;bound_mm=0.046")
    # author returns a bar long on X; ports span Y -> the hook must rotate X->Y
    monkeypatch.setattr(dispatch_mod, "author_part", _claimable_author(tmp_path, long_axis="x"))
    y_part = _part("upper_arm", ports=[_port("a_bore", 0, 0, 0), _port("b_bore", 0, 100, 0)])
    lr = try_hybrid_physics_author(y_part, "upper_arm", tmp_path, SimpleNamespace(name="pla"))
    assert lr is not None and lr.final_stl is not None
    mesh = trimesh.load(str(lr.final_stl), force="mesh")
    ex, ey, ez = mesh.extents
    assert ey > ex and ey > ez  # the longest extent is now Y (rotated), not X
