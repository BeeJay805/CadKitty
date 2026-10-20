"""Zero-LLM repairs (audit P1): the deterministic anchor seat and the motion shave.

The live i4 paid 4 x ~4 min LLM re-authors for pure-translation anchor misses, and its
one repair round re-authored a whole part for what a measured sub-mm overlap describes
exactly. Both fixes are self-gating (fail -> the LLM path stands), so these tests pin
the accept AND the refuse sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trimesh

from forge_mind.agentic.functional import (
    PartPlan,
    _deterministic_seat,
    _parse_sweep_collision,
    _port_anchor_callout,
    _try_deterministic_shave,
)
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3


def _make_box_part(tmp_path: Path, name: str = "part") -> tuple[Path, Path]:
    """A real 20 mm box at the LOCAL origin (STL + STEP), the misplaced-part shape."""
    from build123d import Box, export_step, export_stl

    solid = Box(20, 20, 20)
    stl, step = tmp_path / f"{name}.stl", tmp_path / f"{name}.step"
    export_stl(solid, str(stl))
    export_step(solid, str(step))
    return stl, step


@dataclass
class _LRStub:
    """Minimal dataclass stand-in for LoopResult (the seat/shave use dataclasses.replace,
    which a SimpleNamespace stub cannot satisfy)."""

    final_stl: Path | None
    final_step: Path | None
    brief: None = None


def _lr(stl: Path | None, step: Path | None) -> Any:
    return _LRStub(final_stl=stl, final_step=step)


def _port(name: str, x: float, y: float, z: float) -> Port:
    return Port(
        name=name,
        origin=Vec3(x=x, y=y, z=z),
        axis=Vec3(x=0, y=0, z=1),
        normal=Vec3(x=1, y=0, z=0),
    )


def test_seat_translates_a_local_frame_part_to_its_station(tmp_path: Path) -> None:
    stl, step = _make_box_part(tmp_path)
    part = PartPlan(name="rod", prompt="a rod", ports=[_port("big_end", 50.0, 0.0, 10.0)])
    lr = _lr(stl, step)
    assert _port_anchor_callout(part, lr) is not None  # misplaced as authored
    seated = _deterministic_seat(part, lr, "rod", tmp_path)
    assert seated is not None
    assert _port_anchor_callout(part, seated) is None  # the gate passes after the seat
    assert seated.final_stl.exists() and seated.final_step.exists()
    assert "anchor_seat" in str(seated.final_stl)
    m = trimesh.load(str(seated.final_stl), force="mesh")
    lo, hi = m.bounds
    assert lo[0] <= 50.0 <= hi[0]  # the port station is inside the seated bbox


def test_seat_refuses_wrong_multi_port_spacing(tmp_path: Path) -> None:
    """Two ports 100 mm apart on a 20 mm part: no single translation covers both -- the
    candidate search must fail closed to the LLM path."""
    stl, step = _make_box_part(tmp_path)
    part = PartPlan(
        name="beam",
        prompt="a beam",
        ports=[_port("a", 50.0, 0.0, 0.0), _port("b", -50.0, 0.0, 0.0)],
    )
    assert _deterministic_seat(part, _lr(stl, step), "beam", tmp_path) is None


def test_seat_requires_the_step(tmp_path: Path) -> None:
    stl, _step = _make_box_part(tmp_path)
    part = PartPlan(name="p", prompt="p", ports=[_port("a", 50.0, 0.0, 0.0)])
    assert _deterministic_seat(part, _lr(stl, None), "p", tmp_path) is None


def test_parse_sweep_collision_reads_the_frozen_grammar() -> None:
    got = _parse_sweep_collision(
        "rod3 hits block at 20 deg (depth 0.32 mm); +1 more colliding pair(s): a~b"
    )
    assert got == ("rod3", "block", 0.32)
    assert _parse_sweep_collision("rod3 hits block at 20 deg (depth unmeasurable)") is None
    assert _parse_sweep_collision("declared joints could not be validated") is None


def test_shave_thins_the_mover_and_reports_zero_llm(tmp_path: Path, capsys: Any) -> None:
    stl, step = _make_box_part(tmp_path)
    v0 = float(trimesh.load(str(stl), force="mesh").volume)
    motion = {"motion_status": "fail", "motion_detail": "part hits block at 40 deg (depth 0.30 mm)"}
    out = _try_deterministic_shave(motion, [("part", _lr(stl, step))], tmp_path)
    assert out is not None
    results, note = out
    assert "deterministic shave" in note and "0.20 mm" in note  # 0.30/2 + 0.05
    shaved = dict(results)["part"]
    m = trimesh.load(str(shaved.final_stl), force="mesh")
    assert m.is_watertight and 0 < float(m.volume) < v0  # thinner, still a solid
    assert "0 LLM calls" in capsys.readouterr().out


def test_shave_refuses_deep_collisions(tmp_path: Path) -> None:
    stl, step = _make_box_part(tmp_path)
    motion = {"motion_status": "fail", "motion_detail": "part hits block at 40 deg (depth 2.50 mm)"}
    assert _try_deterministic_shave(motion, [("part", _lr(stl, step))], tmp_path) is None


def test_shave_refuses_without_a_step(tmp_path: Path) -> None:
    stl, _step = _make_box_part(tmp_path)
    motion = {"motion_status": "fail", "motion_detail": "part hits block at 40 deg (depth 0.30 mm)"}
    assert _try_deterministic_shave(motion, [("part", _lr(stl, None))], tmp_path) is None


def test_seat_wins_before_the_llm_anchor_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring: a misplaced ported part must be seated deterministically -- the anchor-fix
    LLM re-author must NOT run (pipeline called exactly once, for the base authoring)."""
    from forge_mind.agentic import functional as fm

    stl, step = _make_box_part(tmp_path)
    plan = fm.FunctionalPlan(
        object_name="thing",
        parts=[
            PartPlan(name="rod", prompt="rod", ports=[_port("big_end", 50.0, 0.0, 10.0)]),
            PartPlan(name="base", prompt="base", ports=[_port("hub", 50.0, 0.0, 10.0)]),
        ],
        assembly_note="rod on base",
        joints=[Joint(type=JointType.FIXED, a="base.hub", b="rod.big_end")],
    )
    calls = {"n": 0}

    @dataclass
    class _FullLR:  # dataclass: the seat path calls dataclasses.replace on it
        prompt: str
        final_stl: Path | None
        final_step: Path | None
        final_render: None = None
        total_cost_usd: float = 0.0
        validation: None = None
        geometry_passed: bool = True
        critic_approved: bool = True
        stopped_reason: str = "accepted"
        iterations: tuple[object, ...] = (object(),)
        critique: None = None
        brief: None = None

    def _pipeline(prompt: str, out_dir: Path, **_k: object) -> _FullLR:
        calls["n"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        return _FullLR(prompt=prompt, final_stl=stl, final_step=step)

    monkeypatch.setattr(
        fm, "decompose_functional", lambda *a, **k: (plan, SimpleNamespace(cost_usd=0.0))
    )
    monkeypatch.setattr(fm, "run_agentic_pipeline", _pipeline)
    monkeypatch.setattr(
        fm, "acceptance_judge", lambda *a, **k: (None, SimpleNamespace(cost_usd=0.0))
    )
    brief = SimpleNamespace(object_name="thing", material=None, visual_description="x")
    fm.produce_functional_part("thing", tmp_path / "run", client=object(), brief=brief)
    # base part authors once each (2 parts); the rod's anchor fix is the SEAT, not a 3rd call
    assert calls["n"] == 2
    assert (tmp_path / "run" / "rod_anchor_seat").exists()
