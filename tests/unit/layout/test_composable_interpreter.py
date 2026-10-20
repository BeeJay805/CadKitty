"""ComposablePlateInterpreter (ADR-0014): the unified plate-family interpreter.

The migration gate is BYTE-EQUIVALENCE: the composable interpreter, on a config that
re-expresses a legacy pattern, must emit a PartProgram bit-for-bit equal to the legacy
interpreter's output (full ``model_dump_json``). Plus a grammar-budget guard: the
collapsed 2-member LLM wrapper must not be larger than the current 3-spec wrapper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from forge_mind.dsl.primitives import Box, Cylinder
from forge_mind.layout.adapter_interpreter import interpret_adapter_plate
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    BaseDimsSpec,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.fan_mount import interpret_fan_mount
from forge_mind.layout.interpreter import InfeasibleRequestError, LayoutError
from forge_mind.layout.interpreter import interpret as interpret_l_bracket
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.layout.placement import LayoutPattern
from forge_mind.layout.spec import LBracketMotorMountSpec
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"


# ---------------------------------------------------------------------------
# Step 0 — grammar budget: the collapsed wrapper must fit (<= current known-good).
# ---------------------------------------------------------------------------


class _CurrentWrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")
    l_bracket: LBracketMotorMountSpec | None = None
    pi4_case: ElectronicsTraySpec | None = None
    adapter_plate: AdapterPlateSpec | None = None


class _ComposableWrapper(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plate: ComposablePlateSpec | None = None
    pi4_case: ElectronicsTraySpec | None = None


def test_composable_wrapper_fits_grammar_budget() -> None:
    # Proxy for the Anthropic structured-output grammar limit: the current 3-spec
    # wrapper is known to work, so the collapsed 2-member wrapper being no larger
    # proves it is within budget (ADR-0014 Q2). Collapsing 3 specs -> 2 members
    # removes more than the per-item interface list adds.
    cur = len(json.dumps(_CurrentWrapper.model_json_schema()))
    new = len(json.dumps(_ComposableWrapper.model_json_schema()))
    assert new <= cur, f"composable wrapper schema ({new}) exceeds current ({cur}) -- over budget"


# ---------------------------------------------------------------------------
# Step 1 — adapter_plate byte-equivalence (the go/no-go).
# ---------------------------------------------------------------------------


def _adapter_legacy() -> AdapterPlateSpec:
    return AdapterPlateSpec.model_validate(
        json.loads((FIX / "bench003_adapter_plate.layoutspec.json").read_text())
    )


def test_adapter_plate_byte_equivalent() -> None:
    legacy = _adapter_legacy()
    config = ComposablePlateSpec(
        name=legacy.name,
        envelope_mm=legacy.envelope_mm,
        material=legacy.material,
        min_wall_mm=legacy.min_wall_mm,
        max_overhang_deg=legacy.max_overhang_deg,
        base=BaseArchetype.FLAT_PLATE,
        base_dims=BaseDimsSpec(width_mm=legacy.plate_width_mm, depth_mm=legacy.plate_depth_mm),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=legacy.motor_center_x_mm,
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=legacy.extrusion_center_x_mm,
                spacing_mm=20.0,
            ),
        ],
    )
    assert (
        interpret_composable_plate(config).model_dump_json()
        == interpret_adapter_plate(legacy).model_dump_json()
    )


# ---------------------------------------------------------------------------
# Step 2 — fan_mount byte-equivalence.
# ---------------------------------------------------------------------------


def test_fan_mount_byte_equivalent() -> None:
    legacy = AdapterPlateSpec.model_validate(
        json.loads((FIX / "bench006_fan_mount.layoutspec.json").read_text())
    )
    config = ComposablePlateSpec(
        name=legacy.name,
        envelope_mm=legacy.envelope_mm,
        material=legacy.material,
        min_wall_mm=legacy.min_wall_mm,
        max_overhang_deg=legacy.max_overhang_deg,
        base=BaseArchetype.FLAT_PLATE,
        base_dims=BaseDimsSpec(width_mm=legacy.plate_width_mm, depth_mm=legacy.plate_depth_mm),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE,
                face=Face.PLATE_TOP,
                offset_mm=legacy.motor_center_x_mm,
                component_id=legacy.fan_component_id,
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=legacy.extrusion_center_x_mm,
                spacing_mm=20.0,
            ),
        ],
    )
    assert (
        interpret_composable_plate(config).model_dump_json()
        == interpret_fan_mount(legacy).model_dump_json()
    )


# ---------------------------------------------------------------------------
# Step 3 — l_bracket byte-equivalence (the hard one: gusset + dual-face + op-ids).
# ---------------------------------------------------------------------------


def test_l_bracket_byte_equivalent() -> None:
    legacy = LBracketMotorMountSpec.model_validate(
        json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())
    )
    assert legacy.base is not None and legacy.upright is not None
    config = ComposablePlateSpec(
        name=legacy.name,
        envelope_mm=legacy.envelope_mm,
        material=legacy.material,
        min_wall_mm=legacy.min_wall_mm,
        max_overhang_deg=legacy.max_overhang_deg,
        base=BaseArchetype.L_BRACKET_BEND,
        base_dims=BaseDimsSpec(
            base_width_mm=legacy.base.width_mm,
            base_depth_mm=legacy.base.depth_mm,
            upright_width_mm=legacy.upright.width_mm,
            upright_height_mm=legacy.upright.height_mm,
        ),
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.BASE_TOP, name="motor_face"),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.UPRIGHT_FRONT,
                spacing_mm=20.0,
                name="extrusion_face",
            ),
        ],
    )
    assert (
        interpret_composable_plate(config).model_dump_json()
        == interpret_l_bracket(legacy).model_dump_json()
    )


def test_l_bracket_face_swap_is_canonicalized() -> None:
    # Reliability finding (Step 4): the LLM sometimes swaps the l_bracket faces (motor on
    # the upright, extrusion on the base). The archetype owns the face mapping, so a
    # swapped config must produce the SAME canonical part as the legacy l_bracket, not fail.
    legacy = LBracketMotorMountSpec.model_validate(
        json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())
    )
    assert legacy.base is not None and legacy.upright is not None
    bd = BaseDimsSpec(
        base_width_mm=legacy.base.width_mm,
        base_depth_mm=legacy.base.depth_mm,
        upright_width_mm=legacy.upright.width_mm,
        upright_height_mm=legacy.upright.height_mm,
    )
    swapped = ComposablePlateSpec(
        name=legacy.name,
        envelope_mm=legacy.envelope_mm,
        material=legacy.material,
        min_wall_mm=legacy.min_wall_mm,
        max_overhang_deg=legacy.max_overhang_deg,
        base=BaseArchetype.L_BRACKET_BEND,
        base_dims=bd,
        interfaces=[
            # Deliberately swapped faces (the LLM's failure mode).
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.UPRIGHT_FRONT, name="motor_face"
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.BASE_TOP,
                spacing_mm=20.0,
                name="extrusion_face",
            ),
        ],
    )
    assert (
        interpret_composable_plate(swapped).model_dump_json()
        == interpret_l_bracket(legacy).model_dump_json()
    )


# ---------------------------------------------------------------------------
# Validation guarantees (Codex review): the composable interpreter must not emit
# invalid geometry, matching the validate-before-geometry contract of the others.
# ---------------------------------------------------------------------------


def _spec(**kw: object) -> ComposablePlateSpec:
    fields: dict[str, object] = {
        "name": "t",
        "envelope_mm": Vec3(x=200.0, y=200.0, z=20.0),
        "material": "PLA",
        "min_wall_mm": 4.0,
        "base": BaseArchetype.FLAT_PLATE,
        **kw,
    }
    return ComposablePlateSpec(**fields)  # type: ignore[arg-type]


def test_nondefault_m5_spacing_uses_generic_path() -> None:
    # Non-default M5 spacing (40mm) has no LEGACY delegate, so it routes to the generic path.
    # ADR-0017: null base_dims now DERIVE (the LLM path); explicit base_dims still use the
    # given dims. Either way the 40mm spacing is honored (M5 holes land at +/-20 in Y).
    derived = [
        InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP),
        InterfacePlacement(
            ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, spacing_mm=40.0
        ),
    ]
    prog = interpret_composable_plate(_spec(interfaces=derived))  # null dims -> derive
    ys = sorted(o.position.y for o in prog.operations if o.op_id.startswith("ext_bolt"))
    assert ys == [-20.0, 20.0]
    # Explicit base_dims still accepted (config path); but a null offset with explicit dims
    # is rejected (supply every offset, or leave base_dims null to derive both).
    explicit = [
        InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=15.0),
        InterfacePlacement(
            ref=InterfaceRef.EXTRUSION_2020,
            face=Face.PLATE_BOTTOM,
            offset_mm=-15.0,
            spacing_mm=40.0,
        ),
    ]
    prog2 = interpret_composable_plate(
        _spec(base_dims=BaseDimsSpec(width_mm=80.0, depth_mm=70.0), interfaces=explicit)
    )
    ys2 = sorted(o.position.y for o in prog2.operations if o.op_id.startswith("ext_bolt"))
    assert ys2 == [-20.0, 20.0]


def test_new_families_are_config_only_and_validate(tmp_path: Path) -> None:
    # Scaling proof (ADR-0014 Q6): NEW plate families ship as pure config -- ZERO new
    # interpreter / spec / validator code -- and validate end-to-end. Two new combos on
    # flat_plate: a standalone fan grille and a standalone motor plate.
    grille = _spec(
        name="fan_grille",
        envelope_mm=Vec3(x=90.0, y=90.0, z=10.0),
        base_dims=BaseDimsSpec(width_mm=70.0, depth_mm=70.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE,
                face=Face.PLATE_TOP,
                offset_mm=0.0,
                component_id="fan_60mm",
            )
        ],
    )
    motor_plate = _spec(
        name="motor_plate",
        envelope_mm=Vec3(x=60.0, y=60.0, z=10.0),
        base_dims=BaseDimsSpec(width_mm=46.0, depth_mm=46.0),
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=0.0)
        ],
    )
    for spec in (grille, motor_plate):
        result = run_layout_pipeline(spec, tmp_path / spec.name)
        assert result.error is None, result.error
        assert result.report is not None and result.passed, (
            f"{spec.name} failed: {result.report.failed_check_ids() if result.report else 'no report'}"
        )


def test_multi_interface_families_validate_with_localization(tmp_path: Path) -> None:
    # ADR-0015 payoff: the two families ADR-0014 Step-5 reported as BLOCKED (two M3
    # interfaces on one plate -> 8 M3 holes counted globally) now validate end-to-end,
    # each interface INDEPENDENTLY certified, via checker localization + op-id
    # namespace-on-repeat. Deterministic ($0).
    motor_fan = _spec(
        name="motor_fan",
        envelope_mm=Vec3(x=150.0, y=100.0, z=10.0),
        base_dims=BaseDimsSpec(width_mm=130.0, depth_mm=70.0),
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=25.0),
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE,
                face=Face.PLATE_BOTTOM,
                offset_mm=-30.0,
                component_id="fan_60mm",
            ),
        ],
    )
    r = run_layout_pipeline(motor_fan, tmp_path / "motor_fan")
    assert r.error is None and r.report is not None and r.passed, r.error or (
        r.report and r.report.failed_check_ids()
    )
    mf = {c.id: c.status.value for c in r.report.checks}
    assert mf["interfaces.motor_face.bolt_holes_present"] == "pass"
    assert mf["interfaces.fan_face.bolt_holes_present"] == "pass"

    dual_motor = _spec(
        name="dual_motor",
        envelope_mm=Vec3(x=120.0, y=80.0, z=10.0),
        base_dims=BaseDimsSpec(width_mm=90.0, depth_mm=50.0),
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=21.0),
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=-21.0),
        ],
    )
    r2 = run_layout_pipeline(dual_motor, tmp_path / "dual_motor")
    assert r2.error is None and r2.report is not None and r2.passed, r2.error or (
        r2.report and r2.report.failed_check_ids()
    )
    dm = {c.id: c.status.value for c in r2.report.checks}
    # Both motors independently certified; the repeated ref auto-namespaced to motor_face_2.
    assert dm["interfaces.motor_face.bolt_holes_present"] == "pass"
    assert dm["interfaces.motor_face_2.bolt_holes_present"] == "pass"


def test_generic_path_keeps_fan_bore_to_bolt_land_check() -> None:
    # fan_40mm at min_wall=4 has a ~2.03mm bore-to-bolt land < 4 -> infeasible (ADR-0011).
    # With non-default spacing it routes to the generic path; the fan-specific land check
    # must still run there (Codex), so a big plate cannot mask the real infeasibility.
    cfg = _spec(
        min_wall_mm=4.0,
        base_dims=BaseDimsSpec(width_mm=120.0, depth_mm=120.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE,
                face=Face.PLATE_TOP,
                offset_mm=20.0,
                component_id="fan_40mm",
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=-30.0,
                spacing_mm=40.0,
            ),
        ],
    )
    with pytest.raises(LayoutError, match="corner bolts"):
        interpret_composable_plate(cfg)


def test_fallback_combo_validates_before_geometry() -> None:
    # A new combo on an explicit but too-small plate must be rejected up front (not emit
    # cutouts outside the body). Single nema17 face on a 20x20 plate: 17.1 + 4 > 10.
    cfg = _spec(
        base_dims=BaseDimsSpec(width_mm=20.0, depth_mm=20.0),
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, offset_mm=0.0)
        ],
    )
    with pytest.raises(LayoutError, match=r"land < min_wall|exceeds envelope"):
        interpret_composable_plate(cfg)


# ---------------------------------------------------------------------------
# ADR-0017 — registry motor/extrusion multiplier (RADIUS-DISTINCT combos only).
# New hardware validates end-to-end as PURE CONFIG: zero new interpreter/checker code,
# each interface independently certified against its OWN pattern (47.14mm for NEMA 23,
# 26mm for NEMA 14, M6 for 3030) -- proving the checker reads the registry, not 31mm/M5.
# ---------------------------------------------------------------------------


def _checks(spec: ComposablePlateSpec, tmp_path: Path):  # type: ignore[no-untyped-def]
    r = run_layout_pipeline(spec, tmp_path / spec.name)
    assert r.error is None and r.report is not None and r.passed, r.error or (
        r.report and r.report.failed_check_ids()
    )
    return {c.id: c.status.value for c in r.report.checks}


def test_nema14_on_3030_is_config_only(tmp_path: Path) -> None:
    # NEMA 14 (M3, 26mm) motor on a 3030 (M6) extrusion -- BOTH new hardware, radius-distinct
    # (M3 vs M6). Pure config; the checker certifies the 26mm square + the M6 T-nut pair.
    cfg = _spec(
        name="nema14_3030",
        base_dims=BaseDimsSpec(width_mm=100.0, depth_mm=55.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=25.0,
                component_id="nema14",
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=-25.0,
                component_id="ext_3030",
            ),
        ],
    )
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.motor_face.bolt_pattern_26mm_square"] == "pass"  # NEMA 14, not 31mm
    assert ck["interfaces.extrusion_face.two_m5_holes"] == "pass"  # M6 pair certified via registry


def test_nema23_plus_fan_is_config_only(tmp_path: Path) -> None:
    # NEMA 23 (M5) + 60mm fan (M3): radius-distinct, so the per-placement region separates
    # them with no new checker code. Certifies the 47.14mm square (NEMA 23) + the fan square.
    cfg = _spec(
        name="nema23_fan",
        base_dims=BaseDimsSpec(width_mm=150.0, depth_mm=70.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=40.0,
                component_id="nema23",
            ),
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE,
                face=Face.PLATE_BOTTOM,
                offset_mm=-40.0,
                component_id="fan_60mm",
            ),
        ],
    )
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.motor_face.bolt_pattern_47mm_square"] == "pass"  # NEMA 23, not 31mm
    assert ck["interfaces.fan_face.bolt_holes_present"] == "pass"


def test_nema23_on_3030_is_config_only(tmp_path: Path) -> None:
    # The headline combo, made radius-distinct: NEMA 23 (M5) + 3030 (M6). M5 != M6, so the
    # current radius-filtered oracle certifies it with ZERO new code (the M5/M5 case stays
    # deferred -- see test_same_m5_motor_extrusion_is_deferred).
    cfg = _spec(
        name="nema23_3030",
        base_dims=BaseDimsSpec(width_mm=140.0, depth_mm=65.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=35.0,
                component_id="nema23",
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=-35.0,
                component_id="ext_3030",
            ),
        ],
    )
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.motor_face.bolt_pattern_47mm_square"] == "pass"
    assert ck["interfaces.extrusion_face.two_m5_holes"] == "pass"  # M6 pair


def test_standalone_nema23_and_nema14_validate(tmp_path: Path) -> None:
    # Single new-motor plates validate as config; the pilot bore window tracks the registry
    # (NEMA 23's 38.1mm register sits in the shifted window, not the NEMA 17 [9,14]).
    n23 = _spec(
        name="n23",
        base_dims=BaseDimsSpec(width_mm=65.0, depth_mm=65.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=0.0,
                component_id="nema23",
            )
        ],
    )
    n14 = _spec(
        name="n14",
        base_dims=BaseDimsSpec(width_mm=45.0, depth_mm=45.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=0.0,
                component_id="nema14",
            )
        ],
    )
    assert _checks(n23, tmp_path)["interfaces.motor_face.shaft_bore_present"] == "pass"
    assert _checks(n14, tmp_path)["interfaces.motor_face.bolt_pattern_26mm_square"] == "pass"


def test_nema23_plus_2020_validates_with_localization(tmp_path: Path) -> None:
    # ADR-0017 payoff (was deferred): the same-M5 combo -- NEMA 23 (M5 motor) + 2020 (M5
    # extrusion) -- now VALIDATES end-to-end. The localized extrusion checker counts only its
    # own 2 M5 holes (the NEMA 23's 4 M5 bolts are out-of-region) and the localized motor
    # checker counts only its 4. Each interface is independently certified.
    cfg = _spec(
        name="nema23_2020",
        base_dims=BaseDimsSpec(width_mm=150.0, depth_mm=70.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE,
                face=Face.PLATE_TOP,
                offset_mm=40.0,
                component_id="nema23",
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020,
                face=Face.PLATE_BOTTOM,
                offset_mm=-40.0,
                component_id="ext_2020",
            ),
        ],
    )
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.motor_face.bolt_pattern_47mm_square"] == "pass"  # NEMA 23 certified
    assert ck["interfaces.extrusion_face.two_m5_holes"] == "pass"  # exactly its own 2 M5 holes
    assert ck["interfaces.extrusion_face.m5_spacing"] == "pass"


def test_dual_extrusion_validates_with_localization(tmp_path: Path) -> None:
    # ADR-0017 payoff (was deferred): two same-radius extrusions (two 3030 M6 faces) now
    # VALIDATE -- the localized checker counts each pair in its own region, and namespace-on-
    # repeat gives the second its own ids (extrusion_face_2). Null dims -> derived.
    cfg = _spec(
        name="dual_3030",
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_TOP, component_id="ext_3030"
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, component_id="ext_3030"
            ),
        ],
    )
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.extrusion_face.two_m5_holes"] == "pass"
    assert ck["interfaces.extrusion_face_2.two_m5_holes"] == "pass"  # second pair, own region


def test_partial_explicit_geometry_is_rejected() -> None:
    # Explicit geometry must be COMPLETE or fully null -- a partial base_dims must NOT be
    # silently discarded into a full derive (Codex). width-only base_dims -> clear LayoutError.
    cfg = _spec(
        name="partial",
        base_dims=BaseDimsSpec(width_mm=120.0),  # depth_mm left null
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, component_id="nema23"
            )
        ],
    )
    with pytest.raises(LayoutError, match="partial explicit geometry"):
        interpret_composable_plate(cfg)


def test_explicit_legacy_ids_are_byte_equivalent_to_null() -> None:
    # Naming the legacy default ids explicitly (nema17 / ext_2020) must behave identically to
    # leaving component_id null: same legacy derive delegate, no extra params -> byte-identical
    # PartProgram (ADR-0017 Codex round 2 -- a regression guard).
    null_cfg = _spec(
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP),
            InterfacePlacement(ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM),
        ]
    )
    explicit_cfg = _spec(
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, component_id="nema17"
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, component_id="ext_2020"
            ),
        ]
    )
    assert (
        interpret_composable_plate(explicit_cfg).model_dump_json()
        == interpret_composable_plate(null_cfg).model_dump_json()
    )


def test_l_bracket_mispick_rebases_to_flat_plate() -> None:
    # A hardware feature on an l_bracket base is the LLM mis-picking the base (l_bracket is only the
    # NEMA 17 + 2020 right-angle mount) -> RE-BASE to the flat plate it really is (correct, don't
    # reject -> fallback). The face is remapped (base_top -> plate_top) and the part carries its
    # hardware -- never mis-built as the NEMA 17 mount.
    cfg = ComposablePlateSpec(
        name="lb_mispick",
        envelope_mm=Vec3(x=200.0, y=200.0, z=80.0),
        material="PLA",
        min_wall_mm=4.0,
        base=BaseArchetype.L_BRACKET_BEND,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.BASE_TOP, component_id="608", fit="press"
            ),
        ],
    )
    program = interpret_composable_plate(cfg)
    assert any(
        op.op_id == "plate" for op in program.operations
    )  # a flat slab, not motor_flange + ext_flange
    assert any(f.category == "bearing_seat" and f.key == "608" for f in program.hardware_features)


def test_l_bracket_nema23_adapter_rebases_grows_and_validates(tmp_path: Path) -> None:
    # The Step-0 miss: a NEMA 23 -> 3030 adapter the LLM put on an l_bracket base, in an under-sized
    # 80mm envelope. Re-base to flat_plate (built with the REAL NEMA 23 pattern, not NEMA 17) + grow
    # the envelope -> it validates every tier, no fallback (base-selection + grow together).
    cfg = ComposablePlateSpec(
        name="lb_n23_3030",
        envelope_mm=Vec3(x=80.0, y=80.0, z=40.0),
        material="PLA",
        min_wall_mm=4.0,
        base=BaseArchetype.L_BRACKET_BEND,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.BASE_TOP, component_id="nema23"
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.UPRIGHT_FRONT, component_id="ext_3030"
            ),
        ],
    )
    _checks(cfg, tmp_path)  # re-bases + grows + validates (run_layout_pipeline asserts passed)


# ---------------------------------------------------------------------------
# ADR-0017 (derive generalization) — new hardware derives geometry from NULL dims (the LLM
# path: leave base_dims + offsets null, code derives per-component). Completes the multiplier
# end-to-end without explicit dims. Deterministic ($0); the paid probe only adds the LLM pick.
# ---------------------------------------------------------------------------


def test_nema23_flat_plate_derives_with_null_dims(tmp_path: Path) -> None:
    cfg = _spec(
        name="n23_derived",
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, component_id="nema23"
            )
        ],
    )
    assert cfg.base_dims is None  # null dims -> derive path (no explicit geometry)
    ck = _checks(cfg, tmp_path)
    assert (
        ck["interfaces.motor_face.bolt_pattern_47mm_square"] == "pass"
    )  # 47.14mm derived + certified
    assert ck["interfaces.motor_face.shaft_bore_present"] == "pass"


def test_nema23_on_3030_derives_with_null_dims(tmp_path: Path) -> None:
    cfg = _spec(
        name="n23_3030_derived",
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP, component_id="nema23"
            ),
            InterfacePlacement(
                ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM, component_id="ext_3030"
            ),
        ],
    )
    assert cfg.base_dims is None and all(p.offset_mm is None for p in cfg.interfaces)
    ck = _checks(cfg, tmp_path)
    assert ck["interfaces.motor_face.bolt_pattern_47mm_square"] == "pass"
    assert ck["interfaces.extrusion_face.two_m5_holes"] == "pass"  # M6 pair derived + certified


# ---------------------------------------------------------------------------
# Grow-don't-reject — an under-sized (LLM-estimated) envelope is GROWN to the derived footprint
# instead of being rejected -> direct-DSL fallback (the #1 askability fix); a user hard max is
# honoured (honest refusal, never grown past).
# ---------------------------------------------------------------------------


def test_flat_plate_grows_undersized_envelope_for_hardware(tmp_path: Path) -> None:
    # A 608 bearing seat (centered) + four M3 clearance holes (corner pattern) derive a compact
    # ~41mm square; an under-sized 30mm envelope is GROWN, not rejected, so the part still validates
    # -- grow-don't-reject end to end (the corner layout is more compact than the old row, but a
    # 30mm box still can't hold the 608 boss + corner holes).
    cfg = _spec(
        name="grow_hw",
        envelope_mm=Vec3(x=30.0, y=30.0, z=40.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[
                InterfacePlacement(
                    ref=InterfaceRef.CLEARANCE_HOLE,
                    face=Face.PLATE_TOP,
                    component_id="M3",
                    fit="normal",
                )
                for _ in range(4)
            ],
        ],
    )
    program = interpret_composable_plate(cfg)
    assert (
        program.constraints.max_bbox is not None and program.constraints.max_bbox.x > 30.0
    )  # grew
    _checks(
        cfg, tmp_path
    )  # the GROWN part compiles + validates every tier (run_layout_pipeline asserts passed)


def test_fan_40mm_is_infeasible_refusal() -> None:
    # A 40mm fan can't keep a min_wall land between its airflow bore and its corner bolts (ADR-0011)
    # -- a GENUINE physical infeasibility -> InfeasibleRequestError (a clean refusal the pipeline
    # surfaces), NOT a fallback that builds a wrong part. Raised before any geometry, so it is fast.
    cfg = _spec(
        name="fan40",
        envelope_mm=Vec3(x=200.0, y=200.0, z=20.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.FAN_FACE, face=Face.PLATE_TOP, component_id="fan_40mm"
            ),
            InterfacePlacement(ref=InterfaceRef.EXTRUSION_2020, face=Face.PLATE_BOTTOM),
        ],
    )
    with pytest.raises(InfeasibleRequestError, match="physically too small"):
        interpret_composable_plate(cfg)


def test_flat_plate_hard_max_refuses_when_unfittable() -> None:
    # The same part with a user HARD max it cannot meet -> honest refusal, never grown past the max.
    cfg = _spec(
        name="refuse_hw",
        envelope_mm=Vec3(x=30.0, y=30.0, z=40.0),
        max_envelope_mm=Vec3(x=30.0, y=30.0, z=40.0),
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[
                InterfacePlacement(
                    ref=InterfaceRef.CLEARANCE_HOLE,
                    face=Face.PLATE_TOP,
                    component_id="M3",
                    fit="normal",
                )
                for _ in range(4)
            ],
        ],
    )
    with pytest.raises(LayoutError, match="cannot fit within the requested max envelope"):
        interpret_composable_plate(cfg)


# ---------------------------------------------------------------------------
# Placement patterns — sensible layout by construction (the LLM picks a pattern, code derives
# every coordinate; the existing edge-land/overlap check still gates validity).
# ---------------------------------------------------------------------------


def _hole(component_id: str = "M3") -> InterfacePlacement:
    return InterfacePlacement(
        ref=InterfaceRef.CLEARANCE_HOLE,
        face=Face.PLATE_TOP,
        component_id=component_id,
        fit="normal",
    )


def _hole_centers(program: object) -> list[tuple[float, float]]:
    return [
        (round(op.position.x, 3), round(op.position.y, 3))
        for op in program.operations  # type: ignore[attr-defined]
        if op.op_id.startswith("clr_")
    ]


def test_four_holes_default_to_a_corner_pattern_not_a_row() -> None:
    # The headline fix: four mounting holes land a 2x2 bolt pattern by default, not a centerline row.
    cfg = _spec(name="corner4", interfaces=[_hole() for _ in range(4)])
    centers = _hole_centers(interpret_composable_plate(cfg))
    assert len(centers) == 4
    xs = sorted({x for x, _ in centers})
    ys = sorted({y for _, y in centers})
    assert len(xs) == 2 and len(ys) == 2  # a rectangle: two distinct x AND two distinct y
    assert any(y != 0.0 for _, y in centers)  # NOT all on the y=0 centerline (the old row bug)


def test_explicit_row_pattern_is_collinear() -> None:
    # When the user asks for a row, honour it: all holes share one y.
    cfg = _spec(name="row4", layout=LayoutPattern.ROW, interfaces=[_hole() for _ in range(4)])
    centers = _hole_centers(interpret_composable_plate(cfg))
    assert len(centers) == 4
    assert len({y for _, y in centers}) == 1  # collinear


def test_single_bearing_is_centered() -> None:
    cfg = _spec(
        name="brg1",
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            )
        ],
    )
    program = interpret_composable_plate(cfg)
    boss = next(op for op in program.operations if op.op_id.startswith("brgboss_"))
    assert (round(boss.position.x, 6), round(boss.position.y, 6)) == (0.0, 0.0)


def test_bearing_centered_with_holes_around_it() -> None:
    # A primary seat stays centered; the mounting holes arrange AROUND it (off the origin).
    cfg = _spec(
        name="brg_holes",
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[_hole() for _ in range(4)],
        ],
    )
    program = interpret_composable_plate(cfg)
    boss = next(op for op in program.operations if op.op_id.startswith("brgboss_"))
    assert (round(boss.position.x, 6), round(boss.position.y, 6)) == (0.0, 0.0)  # bearing centered
    for x, y in _hole_centers(program):  # every hole clears the centre
        assert (x, y) != (0.0, 0.0)


def test_face_mount_with_holes_clears_corner_bolts(tmp_path: Path) -> None:
    # Regression: a NEMA face's bolt holes sit at the CORNERS of its square; corner-placed mounting
    # holes must clear that DIAGONAL reach (not just the axis half-extent) or they merge with the
    # bolt holes and `motor_face.bolt_holes_present` fails. The part must validate end to end.
    cfg = _spec(
        name="nema_holes",
        interfaces=[
            InterfacePlacement(ref=InterfaceRef.NEMA17_FACE, face=Face.PLATE_TOP),
            *[_hole() for _ in range(4)],
        ],
    )
    _checks(cfg, tmp_path)  # compiles + validates every tier (bolt holes present, min_wall, ...)


# ---------------------------------------------------------------------------
# open_box base body — a flat_plate FLOOR + a perimeter wall ring (additive).
# ---------------------------------------------------------------------------


def test_open_box_is_flat_plate_floor_plus_walls() -> None:
    # The open_box must be PURELY additive over the flat plate: same features (ops, declarations,
    # hardware) byte-identical, only the bare slab replaced by an outer box minus a top cavity.
    ifaces = [_hole() for _ in range(4)]
    flat = interpret_composable_plate(_spec(name="x", interfaces=ifaces))  # FLAT_PLATE default
    box = interpret_composable_plate(
        _spec(name="x", base=BaseArchetype.OPEN_BOX, interfaces=ifaces)
    )
    assert [o.op_id for o in box.operations[:2]] == ["box_outer", "box_cavity"]
    # the floor's feature ops (after the 2 box ops) are byte-identical to the flat plate's (after 1 slab op)
    assert box.operations[2:] == flat.operations[1:]
    assert box.hardware_features == flat.hardware_features
    assert box.interfaces == flat.interfaces
    # walls sit OUTSIDE the floor footprint: outer box is the slab + 2*min_wall on each in-plane axis
    flat_slab = flat.operations[0]
    box_outer = box.operations[0]
    assert isinstance(flat_slab, Box) and isinstance(box_outer, Box)
    assert box_outer.width == flat_slab.width + 2 * 4.0
    assert box_outer.depth == flat_slab.depth + 2 * 4.0


def test_open_box_validates_with_hardware(tmp_path: Path) -> None:
    spec = _spec(
        name="box_val",
        base=BaseArchetype.OPEN_BOX,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[_hole() for _ in range(4)],
        ],
    )
    _checks(spec, tmp_path)  # floor + walls + bearing + 4 holes compile + validate every tier


# ---------------------------------------------------------------------------
# angle_bracket base body — a flat_plate BASE + a rear upright flange (additive).
# ---------------------------------------------------------------------------


def test_angle_bracket_is_flat_plate_base_plus_upright() -> None:
    # Additive over the flat plate: the base slab + features are byte-identical; an upright is added.
    ifaces = [_hole() for _ in range(4)]
    flat = interpret_composable_plate(_spec(name="x", interfaces=ifaces))
    brk = interpret_composable_plate(
        _spec(name="x", base=BaseArchetype.ANGLE_BRACKET, interfaces=ifaces)
    )
    assert brk.operations[0] == flat.operations[0]  # same base slab
    assert brk.operations[1].op_id == "bracket_upright"
    # the base features (after slab + upright) are byte-identical to the flat plate's (after slab)
    flange = [o for o in brk.operations if o.op_id.startswith("flange_bolt_")]
    base_feats = [o for o in brk.operations[2:] if o not in flange]
    assert base_feats == flat.operations[1:]
    assert brk.hardware_features == flat.hardware_features
    assert brk.interfaces == flat.interfaces


def test_angle_bracket_validates_and_flanges_a_wide_base(tmp_path: Path) -> None:
    # A wide base (a 608 bearing) gets the default M5 mounting pair on the upright; validates.
    spec = _spec(
        name="brk_val",
        base=BaseArchetype.ANGLE_BRACKET,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            )
        ],
    )
    program = interpret_composable_plate(spec)
    assert any(o.op_id == "flange_bolt_left" for o in program.operations)  # wide -> flange
    _checks(spec, tmp_path)


# ---------------------------------------------------------------------------
# walled-base grammar — new bodies are pure config (no new interpreter code).
# ---------------------------------------------------------------------------


def test_round_plate_is_a_circular_floor_with_bolt_circle(tmp_path: Path) -> None:
    # A round plate: a CIRCULAR floor (Cylinder) hosting a centered bearing + a bolt-circle of holes.
    spec = _spec(
        name="round",
        base=BaseArchetype.ROUND_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[_hole() for _ in range(4)],
        ],
    )
    program = interpret_composable_plate(spec)
    floor = program.operations[0]
    assert isinstance(floor, Cylinder)  # circular floor, not a Box
    boss = next(o for o in program.operations if o.op_id.startswith("brgboss_"))
    assert (round(boss.position.x, 6), round(boss.position.y, 6)) == (0.0, 0.0)  # bearing centered
    holes = [(o.position.x, o.position.y) for o in program.operations if o.op_id.startswith("clr_")]
    assert len(holes) == 4
    radii = {round(math.hypot(x, y), 3) for x, y in holes}
    assert len(radii) == 1 and radii.pop() > 0.0  # all 4 holes on ONE pitch circle (a bolt circle)
    _checks(spec, tmp_path)  # circular edge-land + conformance validate end to end


def test_round_plate_rim_crowding_is_rejected() -> None:
    # No-false-PASS: the circular edge-land must REJECT a feature crowded against the rim. Force it
    # with an explicit base_dims-free spec is derive-only (always grows to fit), so drive the check
    # directly with a too-small plate radius.
    from forge_mind.layout.composable_interpreter import _check_round_feasible

    spec = _spec(
        name="crowd",
        base=BaseArchetype.ROUND_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            )
        ],
    )
    with pytest.raises(LayoutError, match="rim land"):
        _check_round_feasible(spec, 4.0, plate_radius=5.0, offsets=[(0.0, 0.0)])  # 608 boss >> 5mm


def test_round_captive_nut_disc_clears_the_box_corner(tmp_path: Path) -> None:
    # Fix 3: a captive-nut pocket is a rectangular BOX boss; on a round floor its true reach is the
    # CORNER DIAGONAL, not the axis footprint. The disc must be sized to clear the corner so the part
    # validates -- the previously-rejected M8-captive-nut-on-round case now passes.
    from forge_mind.dsl.standards import captive_nut_pocket
    from forge_mind.layout.composable_interpreter import _footprint_reach, _round_reach

    nut = InterfacePlacement(
        ref=InterfaceRef.CAPTIVE_NUT_POCKET, face=Face.PLATE_TOP, component_id="M8"
    )
    spec = _spec(name="round_nut", base=BaseArchetype.ROUND_PLATE, interfaces=[nut])
    program = interpret_composable_plate(spec)
    floor = program.operations[0]
    assert isinstance(floor, Cylinder)

    w = spec.min_wall_mm
    af, ac, _depth = captive_nut_pocket("M8")
    corner = math.hypot(af / 2.0 + w, ac / 2.0 + w)  # the box's farthest corner from its centre
    assert _round_reach(nut, w) == pytest.approx(corner)
    assert _round_reach(nut, w) > _footprint_reach(
        nut, w
    )  # diagonal strictly exceeds the axis reach
    # a lone feature is centred -> disc radius = corner reach + a min_wall rim land.
    assert floor.radius == pytest.approx(corner + w)
    _checks(spec, tmp_path)  # builds + validates end to end: the box corner clears the rim


def test_round_circular_feature_reach_is_unchanged() -> None:
    # A circular feature (bearing/hole/magnet) reaches only its footprint radius, so _round_reach
    # equals the flat footprint reach -> non-captive-nut round parts are byte-identical (no growth).
    from forge_mind.layout.composable_interpreter import _footprint_reach, _round_reach

    bearing = InterfacePlacement(
        ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
    )
    assert _round_reach(bearing, 4.0) == _footprint_reach(bearing, 4.0)


def test_interface_cap_raised_to_ten(tmp_path: Path) -> None:
    # Fix 4: the spec now accepts up to 10 interfaces, so a bearing + a 6-hole bolt circle (7 total,
    # over the old cap of 6) is buildable + valid; 11 is still rejected at the schema.
    seven = _spec(
        name="bearing_6bolt",
        base=BaseArchetype.ROUND_PLATE,
        interfaces=[
            InterfacePlacement(
                ref=InterfaceRef.BEARING_SEAT, face=Face.PLATE_TOP, component_id="608", fit="press"
            ),
            *[_hole() for _ in range(6)],
        ],
    )
    program = interpret_composable_plate(seven)
    assert len(program.hardware_features) == 7  # the previously-over-cap part builds
    _checks(seven, tmp_path)  # and validates end to end

    with pytest.raises(ValidationError):
        _spec(name="eleven", interfaces=[_hole() for _ in range(11)])  # 11 > 10 -> rejected


@pytest.mark.parametrize(
    "base",
    [BaseArchetype.U_CHANNEL, BaseArchetype.TRAY, BaseArchetype.PARTIAL_ENCLOSURE],
)
def test_new_walled_bases_validate(base: BaseArchetype, tmp_path: Path) -> None:
    # The 3 config-only bases build + validate end to end. partial_enclosure (adjacent walls) must be
    # watertight -- the corner-overlap rule fuses adjacent walls instead of leaving a corner gap.
    spec = _spec(name=base.value, base=base, interfaces=[_hole() for _ in range(4)])
    checks = _checks(spec, tmp_path)
    assert checks.get("mesh.watertight") == "pass"
    # the floor features are byte-identical to the flat plate's (the grammar is additive).
    flat = interpret_composable_plate(
        _spec(name=base.value, interfaces=[_hole() for _ in range(4)])
    )
    built = interpret_composable_plate(spec)
    assert built.hardware_features == flat.hardware_features
    assert built.interfaces == flat.interfaces
