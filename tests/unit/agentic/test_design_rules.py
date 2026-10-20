"""The smartness stack (row-54 response): design-rules packs + the un-blinded chief.

Rules are deterministic keyword-selected reference text; the chief revision call now
carries the assembled render + as-built part sizes. These tests pin selection, the
decompose wiring, and the vision passthrough."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from forge_mind.agentic import functional as functional_mod
from forge_mind.agentic.design_rules import select_design_rules
from forge_mind.agentic.functional import PlanRevision, _chief_revise_plan


def test_engine_prompt_selects_mechanism_engine_and_fit_packs() -> None:
    rules = select_design_rules("a four cylinder piston engine with a rotating crankshaft")
    assert rules is not None
    assert "DIMENSION-CHAIN CLOSURE" in rules
    assert "INSERTABILITY" in rules
    assert "PISTON-ENGINE SPECIFICS" in rules
    assert "PRINTED-FIT TABLE" in rules


def test_mechanism_rules_stay_narrow_finish_is_broad() -> None:
    # a static bracket gets DESIGNED FINISH (chamfers etc. -- every real part is designed)
    # but NOT the mechanism/engine/fit packs (those must not over-fire on a non-moving part)
    rules = select_design_rules("a flat wall bracket with two countersunk screw holes") or ""
    assert "DESIGNED FINISH" in rules
    assert "DIMENSION-CHAIN CLOSURE" not in rules and "PISTON-ENGINE" not in rules


def test_featureless_prompt_selects_nothing() -> None:
    assert select_design_rules("a smooth calibration cube 20 by 20 by 20") is None


def test_engine_rules_carry_web_vs_rod_cheek_and_visual_promises() -> None:
    # row-55 residuals: coincident web/rod-cheek faces + unbuilt promised windows
    rules = select_design_rules("a piston engine mechanism with a rotating crankshaft")
    assert rules is not None
    assert "connecting-rod cheeks" in rules and "AXIAL gap" in rules
    assert "VISUAL PROMISES ARE BINDING" in rules


def test_decompose_user_carries_the_rules(monkeypatch) -> None:
    captured: dict = {}

    class _Client:
        def generate_structured(self, **kw):
            captured.update(kw)
            return (
                functional_mod.FunctionalPlan(
                    object_name="engine",
                    parts=[
                        functional_mod.PartPlan(name="a", prompt="a card"),
                        functional_mod.PartPlan(name="b", prompt="b card"),
                    ],
                    assembly_note="a on b",
                    joints=[],
                ),
                SimpleNamespace(cost_usd=0.0),
            )

    functional_mod.decompose_functional("a piston engine mechanism", client=_Client())
    assert "DIMENSION-CHAIN CLOSURE" in captured["user"]


def test_decompose_carries_the_briefs_visual_promise(monkeypatch) -> None:
    # row-55 drift: the brief promised side windows the plan never built -- the planner
    # must SEE the judge's reference
    captured: dict = {}

    class _Client:
        def generate_structured(self, **kw):
            captured.update(kw)
            return (
                functional_mod.FunctionalPlan(
                    object_name="engine",
                    parts=[
                        functional_mod.PartPlan(name="a", prompt="a card"),
                        functional_mod.PartPlan(name="b", prompt="b card"),
                    ],
                    assembly_note="a on b",
                    joints=[],
                ),
                SimpleNamespace(cost_usd=0.0),
            )

    fake_brief = SimpleNamespace(
        visual_description="large side windows reveal the crankshaft and rods"
    )
    functional_mod.decompose_functional("a piston engine", client=_Client(), brief=fake_brief)
    assert "THE JUDGE'S REFERENCE" in captured["user"]
    assert "large side windows reveal the crankshaft" in captured["user"]


def test_chief_revision_sees_render_and_extents(tmp_path: Path) -> None:
    png = tmp_path / "assembled.png"
    png.write_bytes(b"\x89PNG-fake")
    captured: dict = {}

    class _VisionClient:
        def generate_structured_with_image(self, **kw):
            captured.update(kw)
            return (
                PlanRevision(
                    diagnosis="d",
                    revised_parts=[functional_mod.PartPlan(name="block", prompt="fixed card")],
                ),
                SimpleNamespace(cost_usd=0.0),
            )

    plan = functional_mod.FunctionalPlan(
        object_name="piston engine",
        parts=[
            functional_mod.PartPlan(name="block", prompt="the block card"),
            functional_mod.PartPlan(name="crankshaft", prompt="the crank card"),
        ],
        assembly_note="crank in block",
        joints=[],
    )
    rev = _chief_revise_plan(
        plan,
        ["crankshaft hits engine_block at 0 deg"],
        client=_VisionClient(),
        render_png=png,
        extent_lines=["block: 120.0 x 80.0 x 145.0 mm, center (0.0, 0.0, 2.5), Z -70.0..75.0"],
    )
    assert rev is not None
    assert captured["image_png"] == b"\x89PNG-fake"
    assert "AS-BUILT PART SIZES" in captured["user"]
    assert "Z -70.0..75.0" in captured["user"]
    assert "DIMENSION-CHAIN CLOSURE" in captured["user"]  # rules keyed off the plan text


def test_chief_revision_without_render_stays_text_only(tmp_path: Path) -> None:
    captured: dict = {}

    class _TextClient:
        def generate_structured(self, **kw):
            captured.update(kw)
            return (
                PlanRevision(
                    diagnosis="d",
                    revised_parts=[functional_mod.PartPlan(name="block", prompt="fixed card")],
                ),
                SimpleNamespace(cost_usd=0.0),
            )

    plan = functional_mod.FunctionalPlan(
        object_name="thing",
        parts=[
            functional_mod.PartPlan(name="a", prompt="a card"),
            functional_mod.PartPlan(name="b", prompt="b card"),
        ],
        assembly_note="a on b",
        joints=[],
    )
    rev = _chief_revise_plan(plan, ["a <-> b: overlap"], client=_TextClient(), render_png=None)
    assert rev is not None and "MEASURED FAILURES" in captured["user"]


def test_mechanism_rules_carry_joint_type_guidance() -> None:
    # radial row-58 cause: chief typed all 13 joints coaxial_insert (zero DOF)
    rules = select_design_rules("a piston engine mechanism with a rotating crankshaft")
    assert rules is not None
    assert "JOINT TYPES ARE NOT ALL THE SAME" in rules
    assert "crankshaft turning in its case" in rules.lower() or "REVOLUTE joint" in rules
    assert "piston sliding in its cylinder is a PRISMATIC" in rules


def test_zero_moving_joint_becomes_an_actionable_repair_line() -> None:
    from forge_mind.agentic.functional import _assembly_failure_lines

    motion = {
        "articulates": False,
        "motion_detail": "no revolute or prismatic joint present",
        "checks": {
            "range_of_motion": {
                "status": "fail",
                "message": "A mechanism needs a revolute or prismatic joint, but none present.",
            }
        },
    }
    lines = _assembly_failure_lines(None, motion, ["crankshaft", "piston-1"])
    joined = "\n".join(lines)
    assert "JOINT-TYPE ERROR" in joined
    assert "REVOLUTE" in joined and "PRISMATIC" in joined


def test_designed_finish_rules_fire_and_protect_mating_surfaces() -> None:
    # user directive: parts must be DESIGNED (grip tread, knurl, chamfers), never bare
    # primitives -- but finish must NEVER touch a mating/bearing surface (changes the fit)
    rules = select_design_rules("a folding step-ladder with tread rungs and grip")
    assert rules is not None
    assert "DESIGNED FINISH" in rules
    assert "anti-slip texture" in rules and "grip holes" in rules
    assert "knurled or\n    ribbed" in rules or "knurled" in rules
    assert "NEVER to a mating" in rules  # the hard safety constraint
    assert "watertight" in rules  # printability guard
