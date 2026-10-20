"""The bounded MOTION-REPAIR round: a failed sweep re-authors the MOVING part with the
measured collision callout, re-assembles, and re-validates -- exactly once. The repair
changes the PART, never a check; a still-failing sweep stays an honest FAIL. Also pins
the decompose-prompt mechanism rules (pin-is-a-part, clearance arithmetic) and the
judge's home-pose render note."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from build123d import Box, Pos, export_stl

from forge_mind.agentic import functional as functional_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.functional import (
    _DECOMPOSE_SYSTEM,
    FunctionalPlan,
    PartPlan,
    _motion_repair_prompt,
    _motion_repair_target,
    produce_functional_part,
)
from forge_mind.agentic.loop import LoopResult
from forge_mind.agentic.validate import AgenticValidation
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

pytestmark = pytest.mark.slow  # exports tiny STLs through the kernel


def _check(cid: str, status: str, message: str = "", actual: str | None = None):
    return SimpleNamespace(
        id=cid, status=SimpleNamespace(value=status), message=message, actual=actual
    )


def _pivot_port() -> Port:
    return Port(
        name="pivot",
        origin=Vec3(x=0.0, y=0.0, z=10.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )


def _plan() -> FunctionalPlan:
    return FunctionalPlan(
        object_name="test mechanism",
        parts=[
            PartPlan(name="base", prompt="the base body", ports=[_pivot_port()]),
            PartPlan(name="lever", prompt="the lever body", ports=[_pivot_port()]),
        ],
        assembly_note="lever pivots on base",
        joints=[
            Joint(type=JointType.REVOLUTE, a="base.pivot", b="lever.pivot", range_deg=(0.0, 90.0))
        ],
    )


# ---------------------------------------------------------------- target + prompt


def test_repair_target_resolves_the_moving_part() -> None:
    report = SimpleNamespace(
        checks=[
            _check("assembly.body.base.watertight", "pass"),
            _check("assembly.sweep.base__lever", "fail", "lever hits base at 0 deg.", "0 deg"),
        ]
    )
    lr = SimpleNamespace(final_stl=None)
    target = _motion_repair_target(report, _plan(), [("base", lr), ("lever", lr)])  # type: ignore[list-item]
    assert target is not None
    mover, plan_part, detail = target
    assert mover == "lever" and plan_part.name == "lever"
    assert "hits base at 0 deg" in detail


def test_repair_target_none_without_a_sweep_fail() -> None:
    report = SimpleNamespace(checks=[_check("assembly.sweep.base__lever", "pass")])
    lr = SimpleNamespace(final_stl=None)
    assert _motion_repair_target(report, _plan(), [("base", lr), ("lever", lr)]) is None  # type: ignore[list-item]
    assert _motion_repair_target(None, _plan(), []) is None


def test_repair_prompt_carries_spec_and_callout() -> None:
    p = _motion_repair_prompt(_plan().parts[1], "lever hits base at 0 deg.")
    assert "the lever body" in p  # the ORIGINAL spec rides along
    assert "MOTION-REPAIR ROUND" in p
    assert "lever hits base at 0 deg" in p
    assert "at least 1.0 mm of air" in p


# ---------------------------------------------------------------- decompose rules pinned


def test_decompose_system_teaches_pin_and_clearance_arithmetic() -> None:
    assert "THE PIN IS A PART" in _DECOMPOSE_SYSTEM
    assert "DO THE CLEARANCE ARITHMETIC" in _DECOMPOSE_SYSTEM
    assert "RUNNING fit" in _DECOMPOSE_SYSTEM


# ---------------------------------------------------------------- the bounded loop


def _real_part(tmp: Path, name: str, x: float) -> LoopResult:
    stl = tmp / f"{name}.stl"
    export_stl(Pos(x, 0, 0) * Box(20, 20, 20), str(stl))
    validation = AgenticValidation(
        passed=True, hard_ok=True, report=None, body_count=1, watertight=True, volume_mm3=8000.0
    )
    return LoopResult(
        prompt=name,
        brief=None,
        final_code="result = Box(20, 20, 20)",
        final_stl=stl,
        final_step=None,
        final_render=None,
        validation=validation,
        critique=None,
        geometry_passed=True,
        critic_approved=True,
        stopped_reason="accepted",
    )


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="test mechanism",
        summary="a lever on a base",
        visual_description="a lever on a base",
        overall_envelope_mm="60 x 20 x 20",
        features=[BriefFeature(name="body", purpose="bulk", approx_dims_mm="-", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def test_one_bounded_repair_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweep fails on attempt 1 naming base__lever -> ONE repair re-author of 'lever' ->
    sweep passes on attempt 2. A permanently-failing sweep gets exactly one repair too."""
    plan = _plan()
    monkeypatch.setattr(
        functional_mod,
        "decompose_functional",
        lambda *a, **k: (
            plan,
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )

    authored: list[str] = []

    def _fake_pipeline(prompt: str, out_dir: Path, **_k: object) -> LoopResult:
        authored.append(prompt)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _real_part(out_dir, "p", 30.0 * len(authored))

    monkeypatch.setattr(functional_mod, "run_agentic_pipeline", _fake_pipeline)
    # the port-anchor gate has its own tests; these fixtures scatter parts deliberately
    monkeypatch.setattr(functional_mod, "_port_anchor_callout", lambda *a, **k: None)

    motion_calls: list[int] = []
    fail_report = SimpleNamespace(
        checks=[_check("assembly.sweep.base__lever", "fail", "lever hits base at 0 deg.")]
    )

    def _fake_motion(part_results: object, _plan_arg: object):
        motion_calls.append(1)
        if len(motion_calls) == 1:
            return (
                {
                    "articulates": False,
                    "motion_status": "fail",
                    "joints_declared": 1,
                    "joints_validated": 1,
                    "checks": {},
                },
                SimpleNamespace(joints=[object()]),
                fail_report,
            )
        return (
            {
                "articulates": True,
                "motion_status": "pass",
                "joints_declared": 1,
                "joints_validated": 1,
                "checks": {},
            },
            SimpleNamespace(joints=[object()]),
            SimpleNamespace(checks=[_check("assembly.sweep.base__lever", "pass")]),
        )

    monkeypatch.setattr(functional_mod, "_validate_motion", _fake_motion)
    monkeypatch.setattr(functional_mod, "_articulated_glb", lambda *a, **k: None)
    monkeypatch.setattr(
        functional_mod,
        "acceptance_judge",
        lambda *a, **k: (
            SimpleNamespace(
                is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )

    res = produce_functional_part("a lever mechanism", tmp_path, client=object(), brief=_brief())
    # 2 part authors + exactly 1 repair author; the repair prompt names the collision.
    assert len(authored) == 3
    assert "MOTION-REPAIR ROUND" in authored[2]
    assert "lever hits base at 0 deg" in authored[2]
    assert len(motion_calls) == 2  # assemble->validate ran exactly twice (bounded)
    assert res.motion is not None and res.motion["articulates"] is True
    # These deliberately scattered fixtures do not seat; strict disposition now
    # outranks the repaired sweep and reports the physical assembly failure.
    assert res.disposition == "not_assemblable"


def test_still_failing_sweep_stays_an_honest_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        functional_mod,
        "decompose_functional",
        lambda *a, **k: (
            plan,
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )
    authored: list[str] = []

    def _fake_pipeline(prompt: str, out_dir: Path, **_k: object) -> LoopResult:
        authored.append(prompt)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _real_part(out_dir, "p", 30.0 * len(authored))

    monkeypatch.setattr(functional_mod, "run_agentic_pipeline", _fake_pipeline)
    # the port-anchor gate has its own tests; these fixtures scatter parts deliberately
    monkeypatch.setattr(functional_mod, "_port_anchor_callout", lambda *a, **k: None)
    fail_report = SimpleNamespace(
        checks=[_check("assembly.sweep.base__lever", "fail", "lever hits base at 0 deg.")]
    )
    motion_calls: list[int] = []

    def _always_fail(*_a: object):
        motion_calls.append(1)
        return (
            {
                "articulates": False,
                "motion_status": "fail",
                "joints_declared": 1,
                "joints_validated": 1,
                "checks": {},
            },
            SimpleNamespace(joints=[object()]),
            fail_report,
        )

    monkeypatch.setattr(functional_mod, "_validate_motion", _always_fail)
    monkeypatch.setattr(functional_mod, "_articulated_glb", lambda *a, **k: None)
    monkeypatch.setattr(
        functional_mod,
        "acceptance_judge",
        lambda *a, **k: (
            SimpleNamespace(
                is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )

    res = produce_functional_part("a lever mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 3  # exactly one repair, never more
    assert len(motion_calls) == 2  # bounded re-validate
    assert res.motion is not None and res.motion["articulates"] is False  # honest FAIL kept
    # Cannot-build outranks cannot-move under the default strict disposition.
    assert res.disposition == "not_assemblable"


# -------------------------------------------------- validate-vs-ship: checkpoint the FINAL set


def test_final_checkpoint_reflects_post_repair_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate-vs-ship regression: the motion-repair loop re-authors 'lever' at a NEW
    position (x=90 vs the pre-repair x=60). The persisted part checkpoint (what a resume /
    m2_score / the deliverable reads) MUST point to the post-repair geometry the report
    validated, not the stale pre-repair geometry checkpointed before the loop. Without the
    re-checkpoint after the loop the certificate and the deliverable diverge (the M2-arm bug)."""
    import json

    import trimesh

    plan = _plan()
    monkeypatch.setattr(
        functional_mod,
        "decompose_functional",
        lambda *a, **k: (
            plan,
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )
    authored: list[str] = []

    def _fake_pipeline(prompt: str, out_dir: Path, **_k: object) -> LoopResult:
        authored.append(prompt)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _real_part(out_dir, "p", 30.0 * len(authored))

    monkeypatch.setattr(functional_mod, "run_agentic_pipeline", _fake_pipeline)
    monkeypatch.setattr(functional_mod, "_port_anchor_callout", lambda *a, **k: None)
    motion_calls: list[int] = []

    def _fake_motion(part_results: object, _plan_arg: object):
        motion_calls.append(1)
        passed = len(motion_calls) > 1
        return (
            {
                "articulates": passed,
                "motion_status": "pass" if passed else "fail",
                "joints_declared": 1,
                "joints_validated": 1,
                "checks": {},
            },
            SimpleNamespace(joints=[object()]),
            SimpleNamespace(
                checks=[
                    _check(
                        "assembly.sweep.base__lever",
                        "pass" if passed else "fail",
                        "" if passed else "lever hits base at 0 deg.",
                    )
                ]
            ),
        )

    monkeypatch.setattr(functional_mod, "_validate_motion", _fake_motion)
    monkeypatch.setattr(functional_mod, "_articulated_glb", lambda *a, **k: None)
    monkeypatch.setattr(
        functional_mod,
        "acceptance_judge",
        lambda *a, **k: (
            SimpleNamespace(
                is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
            ),
            LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
        ),
    )

    res = produce_functional_part("a lever mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 3 and res.motion is not None and res.motion["articulates"] is True

    ck = json.loads((tmp_path / "lever" / "part_checkpoint.json").read_text(encoding="utf-8"))
    final_lever = next(lr for name, lr in res.parts if name == "lever")
    # the checkpoint points to the SAME geometry the report validated (the final part set)...
    assert ck["final_stl"] == str(final_lever.final_stl)
    # ...which is the POST-repair lever (x~90), not the stale pre-repair lever (x~60).
    mesh = trimesh.load(ck["final_stl"], force="mesh")
    assert abs(float(mesh.bounds.mean(axis=0)[0]) - 90.0) < 5.0


# ---------------------------------------------------------------- judge render note


def test_judge_render_note_appends_only_when_given() -> None:
    from forge_mind.agentic import judge as judge_mod

    captured: dict[str, str] = {}

    class _Client:
        def generate_structured_with_image(self, **kw: object):
            captured["user"] = str(kw["user"])
            return (
                SimpleNamespace(
                    is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
                ),
                LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0),
            )

    png = Path(__file__)  # any readable file's bytes work for the stub
    judge_mod.acceptance_judge(png, _brief(), client=_Client(), render_note="HOME pose note")
    assert "RENDER NOTE" in captured["user"] and "HOME pose note" in captured["user"]
    judge_mod.acceptance_judge(png, _brief(), client=_Client())
    assert "RENDER NOTE" not in captured["user"]
