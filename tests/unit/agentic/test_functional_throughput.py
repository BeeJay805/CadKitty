"""Throughput + resilience (audit P2): per-part checkpoint resume, parallel authoring,
structural early-exit, and finish telemetry.

A session boundary killing a 90-min run at part 8/10 used to cost all 10 parts; a failed
crankcase used to burn ~half a run authoring parts for an inevitable no_build; the finish
pass used to make 10 'chose nothing' decisions with zero trace.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.agentic import functional as fm
from forge_mind.agentic.functional import FunctionalPlan, PartPlan
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3


def _port(name: str) -> Port:
    return Port(
        name=name, origin=Vec3(x=0, y=0, z=0), axis=Vec3(x=0, y=0, z=1), normal=Vec3(x=1, y=0, z=0)
    )


def _plan(names: list[str], structural: str | None = None) -> FunctionalPlan:
    """A plan whose joint graph passes the topology gate; ``structural`` (if given) appears
    in 2 joints (a hub every other part attaches to)."""
    parts = [PartPlan(name=n, prompt=f"{n} card", ports=[_port("hub")]) for n in names]
    hub = structural or names[0]
    joints = [Joint(type=JointType.FIXED, a=f"{hub}.hub", b=f"{n}.hub") for n in names if n != hub]
    return FunctionalPlan(object_name="thing", parts=parts, assembly_note="joined", joints=joints)


@dataclass
class _LR:
    prompt: str
    final_stl: Path | None
    final_step: Path | None = None
    final_render: Path | None = None
    total_cost_usd: float = 0.0
    validation: Any = None
    critique: Any = None
    geometry_passed: bool = True
    critic_approved: bool = True
    stopped_reason: str = "accepted"
    iterations: list[Any] = field(default_factory=lambda: [object()])
    brief: Any = None
    resumed_hard_ok: bool = False


def _ok_pipeline_factory(calls: dict[str, int]) -> Any:
    def _pipeline(prompt: str, out_dir: Path, **_k: Any) -> _LR:
        calls[prompt] = calls.get(prompt, 0) + 1
        out_dir.mkdir(parents=True, exist_ok=True)
        stl = out_dir / "p.stl"
        stl.write_text("solid x")
        return _LR(prompt=prompt, final_stl=stl)

    return _pipeline


def _wire(monkeypatch: pytest.MonkeyPatch, plan: FunctionalPlan, pipeline: Any) -> None:
    monkeypatch.setattr(
        fm, "decompose_functional", lambda *a, **k: (plan, SimpleNamespace(cost_usd=0.0))
    )
    monkeypatch.setattr(fm, "run_agentic_pipeline", pipeline)
    monkeypatch.setattr(
        fm, "acceptance_judge", lambda *a, **k: (None, SimpleNamespace(cost_usd=0.0))
    )
    monkeypatch.delenv("FORGE_PARALLEL_AUTHORS", raising=False)
    monkeypatch.delenv("FORGE_RESUME_PARTS", raising=False)


_BRIEF = SimpleNamespace(object_name="thing", material=None, visual_description="x")


def test_checkpoint_resume_skips_finished_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(["a", "b"])
    calls: dict[str, int] = {}
    _wire(monkeypatch, plan, _ok_pipeline_factory(calls))
    run = tmp_path / "run"
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    assert calls == {"a card": 1, "b card": 1}
    assert (run / "a" / "part_checkpoint.json").exists()
    # relaunch into the SAME dir: both parts resume, ZERO pipeline calls
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    assert calls == {"a card": 1, "b card": 1}


def test_checkpoint_is_keyed_on_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed part card must NOT resurrect the stale body."""
    plan1 = _plan(["a", "b"])
    calls: dict[str, int] = {}
    _wire(monkeypatch, plan1, _ok_pipeline_factory(calls))
    run = tmp_path / "run"
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    plan2 = plan1.model_copy(
        update={
            "parts": [
                PartPlan(name="a", prompt="a card REVISED", ports=[_port("hub")]),
                plan1.parts[1],
            ]
        }
    )
    monkeypatch.setattr(
        fm, "decompose_functional", lambda *a, **k: (plan2, SimpleNamespace(cost_usd=0.0))
    )
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    assert calls["a card REVISED"] == 1  # re-authored under the new card
    assert calls["b card"] == 1  # untouched part resumed


def test_checkpoint_reauthors_a_tampered_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(["a", "b"])
    calls: dict[str, int] = {}
    _wire(monkeypatch, plan, _ok_pipeline_factory(calls))
    run = tmp_path / "run"
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    (run / "a" / "p.stl").write_text("tampered", encoding="utf-8")

    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)

    assert calls == {"a card": 2, "b card": 1}


def test_resume_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(["a", "b"])
    calls: dict[str, int] = {}
    _wire(monkeypatch, plan, _ok_pipeline_factory(calls))
    run = tmp_path / "run"
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    monkeypatch.setenv("FORGE_RESUME_PARTS", "0")
    fm.produce_functional_part("thing", run, client=object(), brief=_BRIEF)
    assert calls == {"a card": 2, "b card": 2}


def test_parallel_authoring_preserves_order_and_authors_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(["a", "b", "c", "d"])
    calls: dict[str, int] = {}
    seen_threads: set[int] = set()
    first_wave = threading.Barrier(3, timeout=10)
    base = _ok_pipeline_factory(calls)

    def _pipeline(prompt: str, out_dir: Path, **k: Any) -> _LR:
        seen_threads.add(threading.get_ident())
        if prompt in {"a card", "b card", "c card"}:
            # Require overlapping work instead of relying on fast fake I/O to yield.
            first_wave.wait()
        return base(prompt, out_dir, **k)

    _wire(monkeypatch, plan, _pipeline)
    monkeypatch.setenv("FORGE_PARALLEL_AUTHORS", "3")
    res = fm.produce_functional_part("thing", tmp_path / "run", client=object(), brief=_BRIEF)
    assert [n for n, _lr in res.parts] == ["a", "b", "c", "d"]  # plan order preserved
    assert calls == {f"{n} card": 1 for n in "abcd"}
    assert len(seen_threads) >= 2  # actually ran on worker threads


def test_structural_failure_early_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'hub' is referenced by 2 joints; when it fails, the remaining parts are never
    authored and the run is an honest no_build naming the early exit."""
    plan = _plan(["hub", "x", "y"], structural="hub")
    calls: dict[str, int] = {}

    def _pipeline(prompt: str, out_dir: Path, **_k: Any) -> _LR:
        calls[prompt] = calls.get(prompt, 0) + 1
        out_dir.mkdir(parents=True, exist_ok=True)
        if prompt == "hub card":
            return _LR(prompt=prompt, final_stl=None, stopped_reason="no_build")
        stl = out_dir / "p.stl"
        stl.write_text("solid x")
        return _LR(prompt=prompt, final_stl=stl)

    _wire(monkeypatch, plan, _pipeline)
    res = fm.produce_functional_part("thing", tmp_path / "run", client=object(), brief=_BRIEF)
    assert res.disposition == "no_build"
    assert res.failure is not None and "early exit" in res.failure and "hub" in res.failure
    assert "x card" not in calls and "y card" not in calls  # never authored


def test_finish_telemetry_records_every_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One part gets an op (applied), one gets restraint ([]) -- both rows recorded, and
    the report carries them (the i4-finish lesson: silence is unauditable)."""
    import forge_mind.agentic.finish_planner as fp

    plan = _plan(["knob", "shaft"])
    calls: dict[str, int] = {}
    _wire(monkeypatch, plan, _ok_pipeline_factory(calls))
    monkeypatch.setenv("FORGE_FINISH", "1")

    def _fake_plan(ctx: Any, **_k: Any) -> list[Any]:
        if ctx.name == "knob":
            return [SimpleNamespace(op="knurl", params={"kind": "straight"})]
        return []

    def _fake_finish_step(step: Path, ops: list[Any], out_stl: Path, out_step: Path) -> bool:
        out_stl.write_text("solid finished")
        out_step.write_text("STEP")
        return True

    monkeypatch.setattr(fp, "plan_part_finish", _fake_plan)
    monkeypatch.setattr(fp, "finish_step_file", _fake_finish_step)
    monkeypatch.setattr(fp, "fit_not_worse", lambda *_a, **_k: True)
    # 08e4553 revalidates the FINISHED geometry before accepting it -- the fake STEP text
    # here would fail that parse and fail-closed revert the row. Stub the seam (the
    # revalidation semantics own their tests in test_designed_finish.py); this test is
    # about the telemetry rows.
    monkeypatch.setattr(fm, "_finished_part_not_worse", lambda lr, s, p: (True, "not-worse", None))

    # the finish pass measures the part's REAL bbox before planning, so the stub must
    # export a genuine mesh (a text placeholder has no bounds -> 'skipped: unmeasurable')
    import trimesh as _tm

    def _pipeline_with_step(prompt: str, out_dir: Path, **_k: Any) -> _LR:
        calls[prompt] = calls.get(prompt, 0) + 1
        out_dir.mkdir(parents=True, exist_ok=True)
        stl, step = out_dir / "p.stl", out_dir / "p.step"
        _tm.creation.box(extents=[10, 10, 10]).export(str(stl))
        step.write_text("STEP")  # never parsed: finish_step_file is stubbed
        return _LR(prompt=prompt, final_stl=stl, final_step=step)

    monkeypatch.setattr(fm, "run_agentic_pipeline", _pipeline_with_step)
    res = fm.produce_functional_part("thing", tmp_path / "run", client=object(), brief=_BRIEF)
    assert res.finish is not None and len(res.finish) == 2
    by_part = {t["part"]: t for t in res.finish}
    assert by_part["knob"]["applied"] is True
    assert by_part["knob"]["ops"][0]["op"] == "knurl"
    assert by_part["shaft"]["applied"] is False
    assert "restraint" in by_part["shaft"]["note"]
    import json

    report = json.loads((tmp_path / "run" / "functional_report.json").read_text(encoding="utf-8"))
    assert report["finish"] and {t["part"] for t in report["finish"]} == {"knob", "shaft"}
