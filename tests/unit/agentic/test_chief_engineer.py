"""CHIEF-ENGINEER mode (FORGE_CHIEF=1, founder architecture): one brain (the BRAIN_MODEL) writes
exact build cards and revises ITS OWN plan against measured assembly failures; cheap
drafters write the per-part code. Pins: the flag-off path is untouched (legacy prompt,
SONNET default, single-mover motion repair -- covered by test_motion_repair.py), the
chief overlay + BRAIN_MODEL default when on, failure lines name parts (the fit check reports
indices), and the bounded multi-part revision loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from build123d import Box, Pos, export_step, export_stl

from forge_mind.agentic import functional as functional_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.functional import (
    FunctionalPlan,
    PartPlan,
    PlanRevision,
    _assembly_failure_lines,
    _chief_enabled,
    decompose_functional,
    produce_functional_part,
)
from forge_mind.agentic.loop import LoopResult
from forge_mind.agentic.validate import AgenticValidation
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.config import BRAIN_MODEL, PLAN_MAX_TOKENS, SONNET
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

pytestmark = pytest.mark.slow  # exports tiny STLs through the kernel


def _usage() -> LLMUsage:
    return LLMUsage(model="stub", input_tokens=1, output_tokens=1, cost_usd=0.0)


class _CaptureClient:
    """Captures generate_structured kwargs; returns a canned plan."""

    def __init__(self, result: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result

    def generate_structured(self, **kw: object) -> tuple[object, LLMUsage]:
        self.calls.append(kw)
        return self._result, _usage()


def _tiny_plan() -> FunctionalPlan:
    pivot = Port(
        name="pivot",
        origin=Vec3(x=0.0, y=0.0, z=10.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    return FunctionalPlan(
        object_name="test mechanism",
        parts=[
            PartPlan(name="base", prompt="the base card", ports=[pivot]),
            PartPlan(name="lever", prompt="the lever card", ports=[pivot]),
        ],
        assembly_note="lever pivots on base",
        joints=[
            Joint(type=JointType.REVOLUTE, a="base.pivot", b="lever.pivot", range_deg=(0.0, 90.0))
        ],
    )


# ---------------------------------------------------------------- flag + decompose


def test_flag_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_CHIEF", raising=False)
    assert _chief_enabled() is False
    monkeypatch.setenv("FORGE_CHIEF", "1")
    assert _chief_enabled() is True


def test_decompose_flag_off_is_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_CHIEF", raising=False)
    client = _CaptureClient(_tiny_plan())
    decompose_functional("a thing", client=client)  # type: ignore[arg-type]
    call = client.calls[0]
    assert "CHIEF ENGINEER" not in str(call["system"])
    assert call["model"] == SONNET
    assert call["max_tokens"] == PLAN_MAX_TOKENS


def test_decompose_flag_on_adds_chief_overlay_and_brain_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_CHIEF", "1")
    client = _CaptureClient(_tiny_plan())
    decompose_functional("a thing", client=client)  # type: ignore[arg-type]
    call = client.calls[0]
    system = str(call["system"])
    assert "CHIEF ENGINEER" in system and "BUILD CARD" in system
    assert "every dimension a" in system.lower() or "NUMBER" in system
    assert call["model"] == BRAIN_MODEL
    assert call["max_tokens"] == PLAN_MAX_TOKENS


# ---------------------------------------------------------------- failure lines


def test_failure_lines_name_parts() -> None:
    fit = {
        "seated": False,
        "pairs": [
            {
                "pair": [2, 4],
                "touches": True,
                "seated": False,
                "note": "buried too deep",
                "interference_mm3": 1239.0,
                "overlap_ratio": 1.0,
            }
        ],
        "floating_parts": [1],
    }
    motion = {
        "articulates": False,
        "motion_detail": "turret hits plate at -180 deg",
        "checks": {"pin_coaxial": {"status": "fail", "message": "no pin found"}},
    }
    names = ["base", "turret", "base_pin", "arm", "elbow_pin"]
    lines = _assembly_failure_lines(fit, motion, names)
    joined = "\n".join(lines)
    assert "base_pin <-> elbow_pin" in joined  # indices resolved to NAMES
    assert "100% of the smaller part" in joined
    assert "turret: FLOATING" in joined
    assert "turret hits plate at -180 deg" in joined
    assert "pin_coaxial" in joined


def test_failure_lines_empty_when_all_green() -> None:
    assert _assembly_failure_lines({"seated": True, "pairs": []}, {"articulates": True}, []) == []


# ---------------------------------------------------------------- the chief loop


def _real_part(tmp: Path, name: str, x: float) -> LoopResult:
    stl = tmp / f"{name}.stl"
    step = tmp / f"{name}.step"
    solid = Pos(x, 0, 0) * Box(20, 20, 20)
    export_stl(solid, str(stl))
    export_step(solid, str(step))
    validation = AgenticValidation(
        passed=True, hard_ok=True, report=None, body_count=1, watertight=True, volume_mm3=8000.0
    )
    return LoopResult(
        prompt=name,
        brief=None,
        final_code="result = Box(20, 20, 20)",
        final_stl=stl,
        final_step=step,
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


def _wire_common(
    monkeypatch: pytest.MonkeyPatch, plan: FunctionalPlan, authored: list[str]
) -> None:
    monkeypatch.setenv("FORGE_CHIEF", "1")
    monkeypatch.setattr(functional_mod, "decompose_functional", lambda *a, **k: (plan, _usage()))

    def _fake_pipeline(prompt: str, out_dir: Path, **_k: object) -> LoopResult:
        authored.append(prompt)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _real_part(out_dir, "p", 30.0 * len(authored))

    monkeypatch.setattr(functional_mod, "run_agentic_pipeline", _fake_pipeline)
    # the port-anchor gate has its own tests; these fixtures scatter parts deliberately
    monkeypatch.setattr(functional_mod, "_port_anchor_callout", lambda *a, **k: None)
    monkeypatch.setattr(functional_mod, "_articulated_glb", lambda *a, **k: None)
    monkeypatch.setattr(
        functional_mod,
        "acceptance_judge",
        lambda *a, **k: (
            SimpleNamespace(
                is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
            ),
            _usage(),
        ),
    )


def test_chief_revises_named_parts_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit fails on attempt 1 -> the chief revises BOTH cards -> both parts re-drafted to
    *_chief_fix dirs, the plan carries the new cards, and the loop is bounded at one round."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)

    fit_calls: list[int] = []

    def _fake_fit(_meshes: object, **_kw: object) -> dict[str, object]:
        fit_calls.append(1)
        if len(fit_calls) == 1:
            return {
                "seated": False,
                "pairs": [
                    {
                        "pair": [0, 1],
                        "touches": True,
                        "seated": False,
                        "note": "buried",
                        "interference_mm3": 500.0,
                        "overlap_ratio": 0.9,
                        "min_gap_mm": 0.0,
                    }
                ],
                "floating_parts": [],
                "interference_mm3": 500.0,
                "overlap_ratio": 0.9,
                "min_gap_mm": 0.0,
                "note": "buried",
            }
        return {
            "seated": True,
            "pairs": [],
            "floating_parts": [],
            "interference_mm3": 0.0,
            "overlap_ratio": 0.0,
            "min_gap_mm": 0.5,
            "note": "ok",
        }

    monkeypatch.setattr(functional_mod, "_assembly_fit", _fake_fit)
    monkeypatch.setattr(
        functional_mod, "_validate_motion", lambda *a, **k: (None, None, None)
    )  # rigid: fit is the failing gate

    revision = PlanRevision(
        diagnosis="both cards placed the parts at the same spot",
        revised_parts=[
            # contract-compliant cards: joint-referenced ports carried (an omission or a
            # dropped referenced port is REFUSED -- covered by the port-contract tests)
            PartPlan(name="base", prompt="REVISED base card at x=0", ports=plan.parts[0].ports),
            PartPlan(name="lever", prompt="REVISED lever card at x=30", ports=plan.parts[1].ports),
        ],
    )
    revise_calls: list[list[str]] = []

    def _fake_revise(plan_arg: FunctionalPlan, failures: list[str], **_k: object) -> PlanRevision:
        revise_calls.append(failures)
        return revision

    monkeypatch.setattr(functional_mod, "_chief_revise_plan", _fake_revise)

    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(revise_calls) == 1
    assert "base <-> lever" in "\n".join(revise_calls[0])  # names, not indices
    # 2 originals + 2 chief re-drafts, from the REVISED cards
    assert len(authored) == 4
    assert authored[2:] == ["REVISED base card at x=0", "REVISED lever card at x=30"]
    assert len(fit_calls) == 2  # bounded: exactly one re-measure
    assert res.fit is not None and res.fit["seated"] is True
    assert (tmp_path / "base_chief_fix").exists() and (tmp_path / "lever_chief_fix").exists()


def test_chief_unknown_part_names_keep_honest_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    bad_fit = {
        "seated": False,
        "pairs": [],
        "floating_parts": [0],
        "interference_mm3": 0.0,
        "overlap_ratio": 0.0,
        "min_gap_mm": 9.0,
        "note": "floating",
    }
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: dict(bad_fit))
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        functional_mod,
        "_chief_revise_plan",
        lambda *a, **k: PlanRevision(
            diagnosis="x", revised_parts=[PartPlan(name="ghost", prompt="not a real part")]
        ),
    )
    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 2  # no re-draft happened
    assert res.fit is not None and res.fit["seated"] is False  # the honest FAIL stands


def test_chief_prose_names_match_slug_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row-56 no-fire suspect class: the chief answers in prose-space ('base plate') while
    cards are slug-space -- the revision must still land via slug-insensitive matching."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    bad_fit = {
        "seated": False,
        "pairs": [],
        "floating_parts": [0],
        "interference_mm3": 0.0,
        "overlap_ratio": 0.0,
        "min_gap_mm": 9.0,
        "note": "floating",
    }
    fits = [dict(bad_fit), {"seated": True, "pairs": [], "floating_parts": []}]
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: dict(fits.pop(0)))
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        functional_mod,
        "_chief_revise_plan",
        lambda *a, **k: PlanRevision(
            diagnosis="x",
            # prose-case name; ports carried (the port contract refuses an omission)
            revised_parts=[
                PartPlan(name="Base", prompt="REVISED base card", ports=plan.parts[0].ports)
            ],
        ),
    )
    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 3  # 2 originals + the slug-matched re-draft
    assert authored[2] == "REVISED base card"
    assert res.repair_note is not None and "base" in res.repair_note


def test_chief_call_failure_is_named_in_repair_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row-56's actual gap: the revise call died silently. The failure must surface."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    bad_fit = {
        "seated": False,
        "pairs": [],
        "floating_parts": [0],
        "interference_mm3": 0.0,
        "overlap_ratio": 0.0,
        "min_gap_mm": 9.0,
        "note": "floating",
    }
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: dict(bad_fit))
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))

    def _dying_revise(*_a: object, notes: list[str] | None = None, **_k: object) -> None:
        if notes is not None:
            notes.append("chief revision call failed: LLMError: claude -p timed out after 600s")
        return None

    monkeypatch.setattr(functional_mod, "_chief_revise_plan", _dying_revise)
    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 2  # no re-draft
    assert res.repair_note is not None and "timed out" in res.repair_note


# ---------------------------------------------------------------- revision port contract


def _buried_fit() -> dict[str, object]:
    return {
        "seated": False,
        "pairs": [
            {
                "pair": [0, 1],
                "touches": True,
                "seated": False,
                "note": "buried",
                "interference_mm3": 500.0,
                "overlap_ratio": 0.9,
                "min_gap_mm": 0.0,
            }
        ],
        "floating_parts": [],
        "interference_mm3": 500.0,
        "overlap_ratio": 0.9,
        "min_gap_mm": 0.0,
        "note": "buried",
    }


def _seated_fit() -> dict[str, object]:
    return {
        "seated": True,
        "pairs": [],
        "floating_parts": [],
        "interference_mm3": 0.0,
        "overlap_ratio": 0.0,
        "min_gap_mm": 0.5,
        "note": "ok",
    }


def test_chief_revision_losing_referenced_port_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The north-star-smoke class: a revised card that LOSES a joint-referenced port would
    silently unresolve those joints at re-validation (13/13 -> 9/13, zero trace). A card
    declaring a different non-empty port set is ambiguous intent -> refused, honest FAIL."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: _buried_fit())
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    wrong = Port(
        name="other",
        origin=Vec3(x=0.0, y=0.0, z=0.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    revision = PlanRevision(
        diagnosis="moved both parts",
        revised_parts=[
            PartPlan(name="base", prompt="REVISED base", ports=[wrong]),
            PartPlan(name="lever", prompt="REVISED lever", ports=[wrong]),
        ],
    )
    monkeypatch.setattr(functional_mod, "_chief_revise_plan", lambda *a, **k: revision)

    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 2  # both cards refused: no re-drafts
    assert res.repair_note is not None and "port contract refused" in res.repair_note
    assert "base" in res.repair_note and "lever" in res.repair_note


def test_chief_revision_with_omitted_ports_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole-card re-emit that OMITS ports is refused too (NO carry-forward): stale
    frames carried onto relocated geometry could sweep-pass about the wrong axis -- the
    adversarial review's false-PASS scenario. The original card stays; honest FAIL."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: _buried_fit())
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    revision = PlanRevision(
        diagnosis="lever card re-emitted",
        revised_parts=[PartPlan(name="lever", prompt="REVISED lever card")],  # ports omitted
    )
    monkeypatch.setattr(functional_mod, "_chief_revise_plan", lambda *a, **k: revision)

    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert len(authored) == 2  # refused: no re-draft
    assert res.repair_note is not None and "port contract refused ['lever']" in res.repair_note


def test_port_guard_uses_the_resolver_predicates() -> None:
    """The two adversarial-review breaks, pinned at the guard level: (1) a DOTTED part
    name ('link.a') must be attributed like the resolver's longest-prefix match -- a
    first-dot split saw no referenced ports and waved the card through; (2) separator
    variants ('main axis' vs 'main-axis') must MATCH -- a _slug comparison refused a
    legitimately-revised card."""
    from forge_mind.agentic.functional import _revision_port_guard

    def _p(name: str) -> Port:
        return Port(
            name=name,
            origin=Vec3(x=0.0, y=0.0, z=0.0),
            axis=Vec3(x=0.0, y=1.0, z=0.0),
            normal=Vec3(x=1.0, y=0.0, z=0.0),
        )

    # (1) dotted part name: the joint references link.a's pivot; a card without it refuses
    dotted = FunctionalPlan(
        object_name="t",
        parts=[
            PartPlan(name="base", prompt="b", ports=[_p("pivot")]),
            PartPlan(name="link.a", prompt="l", ports=[_p("pivot")]),
        ],
        assembly_note="n",
        joints=[
            Joint(type=JointType.REVOLUTE, a="base.pivot", b="link.a.pivot", range_deg=(0, 90))
        ],
    )
    bad = PartPlan(name="link.a", prompt="revised", ports=[_p("other")])
    refusal = _revision_port_guard(dotted, bad, "link.a")
    assert refusal is not None and "pivot" in refusal

    # (2) separator variants: joint says 'main-axis', the card declares 'main axis' -> OK
    sep = FunctionalPlan(
        object_name="t",
        parts=[
            PartPlan(name="base", prompt="b", ports=[_p("main-axis")]),
            PartPlan(name="lever", prompt="l", ports=[_p("main-axis")]),
        ],
        assembly_note="n",
        joints=[
            Joint(
                type=JointType.REVOLUTE, a="base.main-axis", b="lever.main-axis", range_deg=(0, 90)
            )
        ],
    )
    ok = PartPlan(name="lever", prompt="revised", ports=[_p("main axis")])
    assert _revision_port_guard(sep, ok, "lever") is None


# ---------------------------------------------------------------- gear checks gate the ship


def _gear_fail_motion() -> dict[str, object]:
    return {
        "articulates": True,
        "fully_validated": True,
        "motion_status": "pass",
        "motion_detail": "clear",
        "joints_declared": 1,
        "joints_validated": 1,
        "checks": {
            "gear_center_distance": {
                "status": "fail",
                "message": "mis-spaced: 40.00mm vs required 30.00mm",
            }
        },
        # the report-derived contract (deep-review): the COMPLETE fail list rides the
        # verdict and is what gates; the checks dict above is the older fallback shape.
        "assembly_fails": [
            {
                "id": "assembly.gear_center_distance",
                "message": "mis-spaced: 40.00mm vs required 30.00mm",
            }
        ],
    }


def test_gear_failure_caps_the_clean_ship(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep cannot see a TOO-FAR gear pair (gears that never touch sweep clean), so
    a measured gear_center_distance FAIL must cap the disposition on its own -- the
    adversarial review's false-PASS scenario for the gear-mesh plan gate."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: _seated_fit())
    monkeypatch.setattr(
        functional_mod, "_validate_motion", lambda *a, **k: (_gear_fail_motion(), None, None)
    )
    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert res.disposition == "shipped_with_caveats"

    # two-sided: the same run with the gear check PASSING ships clean
    ok_motion = _gear_fail_motion()
    checks = ok_motion["checks"]
    assert isinstance(checks, dict)
    checks["gear_center_distance"] = {"status": "pass", "message": "ok"}
    ok_motion["assembly_fails"] = []  # a clean report gates nothing
    monkeypatch.setattr(
        functional_mod, "_validate_motion", lambda *a, **k: (dict(ok_motion), None, None)
    )
    res2 = produce_functional_part("a mechanism", tmp_path / "ok", client=object(), brief=_brief())
    assert res2.disposition == "shipped"


def test_failing_relational_checks_reach_the_chief_even_when_articulating() -> None:
    """A gear/pin check FAIL on an ARTICULATING assembly must still become a failure line
    (the repair round cannot fix what it never hears about)."""
    lines = _assembly_failure_lines(None, _gear_fail_motion(), ["ring", "planet"])
    assert any("gear_center_distance" in ln and "mis-spaced" in ln for ln in lines)


def test_chief_fix_parts_get_the_deterministic_anchor_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """North-star iteration 4: the chief's replacement pins were authored at the ORIGIN
    and SKIPPED the anchor gate + zero-LLM seat that first-pass parts get -- three
    coincident pins buried in the sun; the 'repair' made the assembly worse. Revision
    results now get the same deterministic seat before being swapped in."""
    from forge_mind.agentic.functional import _port_anchor_callout as real_callout

    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    monkeypatch.setattr(functional_mod, "_port_anchor_callout", real_callout)  # the real gate
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: _buried_fit())
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    far_pivot = Port(
        name="pivot",
        origin=Vec3(x=200.0, y=0.0, z=0.0),  # the revised card's declared station
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    revision = PlanRevision(
        diagnosis="lever re-stationed",
        revised_parts=[PartPlan(name="lever", prompt="REVISED lever", ports=[far_pivot])],
    )
    monkeypatch.setattr(functional_mod, "_chief_revise_plan", lambda *a, **k: revision)

    res = produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    lever_lr = dict(res.parts)["lever"]
    # the re-drafted lever (authored far from x=200 by the fake pipeline) was SEATED
    assert lever_lr.final_stl is not None and "_chief_fix_anchor_seat" in str(lever_lr.final_stl)


def test_enclosed_parts_reach_the_judge_as_an_occlusion_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pump-in-a-tube false-reject killer: parts the fit tier measured as ENCLOSED
    are named to the judge (mapped through the mesh keys' stl identity), so it stops
    rejecting a correct assembly for 'missing' what it cannot see."""
    plan = _tiny_plan()
    authored: list[str] = []
    _wire_common(monkeypatch, plan, authored)
    enclosed_fit = _seated_fit()
    enclosed_fit["enclosed_parts"] = [1]  # the second loaded mesh = the lever
    monkeypatch.setattr(functional_mod, "_assembly_fit", lambda _m, **_kw: dict(enclosed_fit))
    monkeypatch.setattr(functional_mod, "_validate_motion", lambda *a, **k: (None, None, None))
    seen_notes: list[str | None] = []

    def _judge(render: object, brief: object, **kw: object):
        seen_notes.append(kw.get("render_note"))  # type: ignore[arg-type]
        return (
            SimpleNamespace(
                is_the_object=True, confidence="high", missing_or_wrong=[], reason="ok"
            ),
            _usage(),
        )

    monkeypatch.setattr(functional_mod, "acceptance_judge", _judge)
    produce_functional_part("a mechanism", tmp_path, client=object(), brief=_brief())
    assert seen_notes and seen_notes[0] is not None
    assert "ENCLOSED inside other parts" in seen_notes[0]
    assert "lever" in seen_notes[0]
