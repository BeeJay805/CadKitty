"""Waves A-D + expert review (founder-directed 2026-07-12): plan-time dimension closure,
assembled-envelope budget, process-keyed worst-case material condition (PRINTED parts
only -- metals are a no-op, per founder), mobility note, expert panel (requirements
engine + deterministic enforcement, never a certifier), print pack, grasp-tab wiring,
insertability, and the transient stage retry.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trimesh

from forge_mind.agentic.functional import (
    FunctionalPlan,
    PartPlan,
    _assembled_envelope_issue,
    _call_with_transient_retry,
    _mobility_note,
    _plan_dimension_issues,
    _print_pack,
    _worst_case_issues,
    material_condition_margin_mm,
)
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.materials import resolve_material
from forge_mind.types import Vec3

_PLA = resolve_material(None)


def _port(name: str, x: float = 0, y: float = 0, z: float = 0, dia: float | None = None) -> Port:
    return Port(
        name=name,
        origin=Vec3(x=x, y=y, z=z),
        axis=Vec3(x=1, y=0, z=0),
        normal=Vec3(x=0, y=0, z=1),
        diameter_mm=dia,
    )


def _plan(ports_a: list[Port], ports_b: list[Port], joint: Joint) -> FunctionalPlan:
    return FunctionalPlan(
        object_name="toy",
        parts=[
            PartPlan(name="base", prompt="base", ports=ports_a),
            PartPlan(name="arm", prompt="arm", ports=ports_b),
        ],
        assembly_note="arm on base",
        joints=[joint],
    )


# ---------- A1: plan-time dimension closure ---------------------------------------------


def test_dimension_gate_flags_line_to_line_and_sloppy_fits() -> None:
    rev = Joint(type=JointType.REVOLUTE, a="base.hub", b="arm.bore", range_deg=(0, 90))
    line = _plan([_port("hub", dia=6.0)], [_port("bore", dia=6.0)], rev)
    assert any("line-to-line" in i for i in _plan_dimension_issues(line, _PLA))
    sloppy = _plan([_port("hub", dia=6.0)], [_port("bore", dia=9.0)], rev)
    assert any("beyond any running fit" in i for i in _plan_dimension_issues(sloppy, _PLA))
    good = _plan([_port("hub", dia=6.0)], [_port("bore", dia=6.4)], rev)
    assert _plan_dimension_issues(good, _PLA) == []


def test_dimension_gate_flags_mismatched_stations_and_skips_undeclared() -> None:
    rev = Joint(type=JointType.REVOLUTE, a="base.hub", b="arm.bore", range_deg=(0, 90))
    apart = _plan([_port("hub", x=0)], [_port("bore", x=50)], rev)
    assert any("SAME axis station" in i for i in _plan_dimension_issues(apart, _PLA))
    undeclared = _plan([_port("hub")], [_port("bore")], rev)  # no diameters: not checked
    assert all("line-to-line" not in i for i in _plan_dimension_issues(undeclared, _PLA))


def test_fit_floor_derives_from_the_material_slip_band() -> None:
    """L100 (the F2 postmortem): the plan floor IS the strict tier's bind threshold,
    2 * (clearance_mm[0] - 0.05) diametral -- derived, never hand-synced (law 2)."""
    rev = Joint(type=JointType.REVOLUTE, a="base.hub", b="arm.bore", range_deg=(0, 90))

    def gap_ok(gap: float, mat_name: str | None) -> bool:
        mat = resolve_material(mat_name)
        plan = _plan([_port("hub", dia=6.0)], [_port("bore", dia=6.0 + gap)], rev)
        return not any("printable running fit" in i for i in _plan_dimension_issues(plan, mat))

    # PLA: clearance (0.2, ...) -> floor 0.30
    assert not gap_ok(0.29, None) and gap_ok(0.30, None)
    # PETG: clearance (0.25, ...) -> floor 0.40 (the derivation, not a constant)
    assert not gap_ok(0.35, "PETG") and gap_ok(0.40, "PETG")
    # Steel: clearance (0.05, ...) derives 0.0 -> the line-to-line bound still holds
    assert not gap_ok(0.0, "steel") and gap_ok(0.06, "steel")


_F2_PLAN = Path(__file__).resolve().parents[2] / "fixtures" / "f2" / "plan.json"
_H4_PLAN = Path(__file__).resolve().parents[2] / "fixtures" / "h4" / "plan.json"


def test_f2_archived_plan_now_fails_its_measured_killers_at_plan_time() -> None:
    """The REAL F2 plan (verbatim copy; TEST_LOG L100) declared 0.15 mm piston-pin and
    0.20 mm dowel fits -- authored faithfully, they bound at the strict tier 14.9 h
    later (measured min clearance 0.075 mm radial). The derived floor catches exactly
    those eight pairs at plan time; the 0.35/0.40+ pairs the strict tier tolerates
    still pass."""
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints

    plan = _canonicalize_plan_endpoints(
        FunctionalPlan.model_validate_json(_F2_PLAN.read_text(encoding="utf-8"))
    )
    hits = [i for i in _plan_dimension_issues(plan, _PLA) if "printable running fit" in i]
    # 6 piston pinbores (0.15) + 2 crankcase dowels (0.20) + the hand-crank knob (0.20)
    assert len(hits) == 9
    assert sum("pinbore" in i for i in hits) == 6
    assert sum("dowel" in i for i in hits) == 2
    assert sum("knob_hand" in i for i in hits) == 1
    assert all("0.30 mm" in i for i in hits)  # the derived PLA floor, named for the replan


def test_h4_archived_plan_keeps_exactly_its_station_errors() -> None:
    """Floor regression pin: H4's 0.35 mm pairs sit ABOVE the strict tier's tolerance,
    so the floor must not add noise -- its dimension issues stay the 4 station errors."""
    from forge_mind.agentic.functional import _canonicalize_plan_endpoints

    plan = _canonicalize_plan_endpoints(
        FunctionalPlan.model_validate_json(_H4_PLAN.read_text(encoding="utf-8"))
    )
    issues = _plan_dimension_issues(plan, _PLA)
    assert len(issues) == 4 and all("SAME axis station" in i for i in issues)


# ---------- A2: assembled envelope -------------------------------------------------------


def test_dimension_gate_gear_mesh_wants_offset_parallel_axes() -> None:
    """Planetary north-star iteration 1: the chief declared six CORRECT gear meshes at
    exactly the designed 30.0 mm center distance and the same-station rule false-rejected
    the whole plan (no_build in 21.8 min). A mesh's stations differ BY DESIGN -- the plan
    gate wants a nonzero center distance between parallel axes instead; the assembly tier
    measures the real built center distance."""
    mesh = Joint(type=JointType.GEAR_MESH, a="base.hub", b="arm.bore")
    correct = _plan([_port("hub", x=0)], [_port("bore", x=30)], mesh)
    assert _plan_dimension_issues(correct, _PLA) == []
    degenerate = _plan([_port("hub", x=0)], [_port("bore", x=0)], mesh)
    assert any("nonzero center distance" in i for i in _plan_dimension_issues(degenerate, _PLA))
    skewed = _plan(
        [_port("hub", x=0)],
        [
            Port(
                name="bore",
                origin=Vec3(x=30, y=0, z=0),
                axis=Vec3(x=0, y=1, z=0),  # perpendicular to the hub axis: cannot mesh
                normal=Vec3(x=0, y=0, z=1),
            )
        ],
        mesh,
    )
    assert any("PARALLEL" in i for i in _plan_dimension_issues(skewed, _PLA))


def test_envelope_issue_fires_only_over_budget() -> None:
    big = trimesh.creation.box(extents=[300.0, 50.0, 50.0])
    small = trimesh.creation.box(extents=[100.0, 50.0, 50.0])
    prompt = "a thing. Overall assembled envelope under 220 x 110 x 150 mm."
    assert _assembled_envelope_issue([big], prompt, None) is not None
    assert _assembled_envelope_issue([small], prompt, None) is None
    assert _assembled_envelope_issue([big], "a thing with no stated budget", None) is None


# ---------- A3: mobility note ------------------------------------------------------------


def test_mobility_note_reports_gruebler() -> None:
    plan = _plan(
        [_port("hub")],
        [_port("bore")],
        Joint(type=JointType.REVOLUTE, a="base.hub", b="arm.bore", range_deg=(0, 90)),
    )
    note = _mobility_note(plan)
    assert "M=1" in note  # 3(2-1) - 2*1 = 1: a clean single-DOF hinge


# ---------- B1: worst-case material condition (process-keyed, founder rule) --------------


def test_worst_case_margin_is_printed_process_only() -> None:
    assert material_condition_margin_mm(resolve_material("PLA")) > 0.0
    assert material_condition_margin_mm(resolve_material("aluminum")) == 0.0  # metal: no-op


def test_worst_case_catches_the_row60_pair_on_pla_but_not_metal() -> None:
    fit = {
        "pairs": [
            {"pair": [0, 1], "touches": True, "interference_mm3": 0.0, "min_gap_mm": 0.01},
            {"pair": [0, 2], "touches": True, "interference_mm3": 0.0, "min_gap_mm": 0.40},
        ]
    }
    motion = {"checks": {"pin_clearance": {"actual": "0.400 mm"}}}
    pla = _worst_case_issues(fit, motion, resolve_material("PLA"), ["crank", "rod1", "rod2"])
    assert pla is not None and len(pla["issues"]) == 1
    assert "crank <-> rod1" in pla["issues"][0] and "bind as printed" in pla["issues"][0]
    metal = _worst_case_issues(fit, motion, resolve_material("steel"), ["crank", "rod1", "rod2"])
    assert metal is None  # founder: these parts are metal -> nominal certification stands


def test_worst_case_flags_tight_pins() -> None:
    motion = {"checks": {"pin_clearance": {"actual": "0.100 mm"}}}
    out = _worst_case_issues(None, motion, resolve_material("PLA"), [])
    assert out is not None and any("pins will bind" in i for i in out["issues"])


# ---------- expert review ----------------------------------------------------------------


def test_expert_review_fail_open_and_disabled() -> None:
    from forge_mind.agentic.expert_review import run_expert_review

    class _Boom:
        def generate_structured(self, **kw: Any) -> Any:
            raise RuntimeError("down")

    out = run_expert_review(
        stage="plan", prompt="x", summary="y", mat=resolve_material("PLA"), client=_Boom()
    )
    assert out is None  # fail-open: the deterministic validators still stand


def test_expert_review_deterministic_temp_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rocket-engine case: the PANEL extracts the environment; CODE compares the
    numbers. PLA at 300 C = deterministic blocker, no LLM opinion involved."""
    from forge_mind.agentic import expert_review as er

    review = er.ExpertReview(
        environment=er.ExpertEnvironment(operating_temp_c=300.0, duty="functional"),
        requirements=[
            er.ExpertRequirement(
                discipline="thermal",
                requirement="chamber must be regeneratively cooled",
                why="combustion gas exceeds any printable material's ceiling",
                quantity="wall < 350 C",
                severity="blocker",
                cad_verifiable=False,
            )
        ],
        material_concern="PLA cannot survive combustion temperatures",
    )

    class _Client:
        def generate_structured(self, **kw: Any) -> Any:
            return review, SimpleNamespace(cost_usd=0.0)

    out = er.run_expert_review(
        stage="plan",
        prompt="a working rocket engine",
        summary="chamber; nozzle",
        mat=resolve_material("PLA"),
        client=_Client(),
    )
    assert out is not None
    assert out["blockers"] == 2  # 1 LLM blocker + 1 deterministic temp finding
    det = out["deterministic_findings"]
    assert len(det) == 1 and det[0]["deterministic"] is True
    assert "service ceiling" in det[0]["finding"]


def test_expert_review_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_mind.agentic.expert_review import run_expert_review

    monkeypatch.setenv("FORGE_EXPERT_REVIEW", "0")
    assert run_expert_review(stage="plan", prompt="x", summary="y", mat=None) is None


# ---------- D2: print pack ---------------------------------------------------------------


def test_print_pack_recommends_an_orientation(tmp_path: Path) -> None:
    stl = tmp_path / "p.stl"
    trimesh.creation.box(extents=[20, 20, 10]).export(str(stl))
    lr = SimpleNamespace(final_stl=stl)
    pack = _print_pack([("plate", lr)])
    assert pack is not None and pack[0]["part"] == "plate"
    assert "down" in pack[0]["orientation"]


# ---------- D1: grasp tabs (wiring seam) --------------------------------------------------


def test_grasp_tabs_only_under_cell_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import dataclass

    from forge_mind.agentic import functional as fm

    @dataclass
    class _LR:
        final_stl: Path | None
        final_step: Path | None

    stl, step = tmp_path / "p.stl", tmp_path / "p.step"
    stl.write_text("x")
    step.write_text("y")
    parts = [("a", _LR(stl, step))]
    monkeypatch.delenv("FORGE_CELL_PROFILE", raising=False)
    assert fm._apply_grasp_tabs(parts, tmp_path, ["a"], resolve_material("PLA"), None) == parts

    calls = {"n": 0}

    def _fake_emit(step_path: Path, stl_path: Path, out_dir: Path, **_k: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(ok=True, stl_path=stl_path, step_path=step_path)

    import forge_mind.agentic.grasp_tab as gt

    monkeypatch.setattr(gt, "emit_grasp_tab", _fake_emit)
    monkeypatch.setenv("FORGE_CELL_PROFILE", "1")
    out = fm._apply_grasp_tabs(parts, tmp_path, ["a"], resolve_material("PLA"), None)
    assert calls["n"] == 1 and len(out) == 1


# ---------- C2: insertability ------------------------------------------------------------


def test_assembly_order_exists_for_separated_parts() -> None:
    from forge_mind.validator.insertion import assembly_order_issues

    a = trimesh.creation.box(extents=[10, 10, 10])
    b = trimesh.creation.box(extents=[10, 10, 10])
    b.apply_translation([30, 0, 0])
    c = trimesh.creation.box(extents=[10, 10, 10])
    c.apply_translation([60, 0, 0])
    out = assembly_order_issues([("a", a), ("b", b), ("c", c)])
    assert out is not None and out["stuck"] == []


def test_enclosed_part_is_reported_stuck() -> None:
    from forge_mind.validator.insertion import assembly_order_issues

    shell = trimesh.boolean.difference(
        [trimesh.creation.box(extents=[40, 40, 40]), trimesh.creation.box(extents=[30, 30, 30])]
    )
    ball = trimesh.creation.icosphere(radius=8)
    far = trimesh.creation.box(extents=[10, 10, 10])
    far.apply_translation([100, 0, 0])
    out = assembly_order_issues([("shell", shell), ("ball", ball), ("far", far)])
    assert out is not None
    assert "ball" in out["stuck"] and "shell" in out["stuck"]  # mutually locked
    assert "verify manually" in out["note"]


# ---------- transient stage retry --------------------------------------------------------


def test_transient_retry_retries_401_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_mind.agentic import functional as fm

    monkeypatch.setattr(fm.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("claude -p failed: API Error: 401 Invalid authentication")
        return "ok"

    assert _call_with_transient_retry(_flaky) == "ok"
    assert calls["n"] == 2

    def _real_error() -> str:
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError):
        _call_with_transient_retry(_real_error)
