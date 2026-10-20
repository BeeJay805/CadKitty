"""Static-load spike acceptance harness (ADR-0039): the council's NO-RESULT floor.

The doctrine under test: numbers are deterministic and auditable, ranges not points,
refuse-not-guess outside the model's claimed envelope, strength structurally absent,
and the sizing inversion re-derives through the REAL interpreter (never interpolates).
All money-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.layout.interpreter import interpret as interpret_l_bracket
from forge_mind.layout.spec import LBracketMotorMountSpec
from forge_mind.physics.static_load import (
    StaticLoadCase,
    StaticLoadEstimate,
    StaticLoadRefusal,
    estimate_static_load,
    extract_bracket_dims,
    size_wall_for_deflection,
)
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "layout"

LOAD = StaticLoadCase(payload_kg=2.0)


def _bench_spec(min_wall: float = 4.0) -> LBracketMotorMountSpec:
    data = json.loads((FIX / "bench001_l_bracket.layoutspec.json").read_text())
    data["min_wall_mm"] = min_wall
    return LBracketMotorMountSpec.model_validate(data)


def _bench_program(min_wall: float = 4.0) -> PartProgram:
    return interpret_l_bracket(_bench_spec(min_wall))


def _synthetic_bracket(
    wall: float, base_depth: float, base_width: float, bolt_z: float
) -> PartProgram:
    """A hand-built motor_flange/ext_flange program for closed-form limit checks."""
    return PartProgram(
        metadata=Metadata(name="synthetic_bracket"),
        operations=[
            Box(
                op_id="motor_flange",
                width=base_width,
                depth=base_depth,
                height=wall,
                position=Vec3(z=wall / 2.0),
            ),
            Box(
                op_id="ext_flange",
                width=base_width,
                depth=wall,
                height=bolt_z * 2.0,
                position=Vec3(y=base_depth / 2.0 - wall / 2.0, z=bolt_z),
            ),
            Cylinder(
                op_id="ext_bolt_left",
                mode=OpMode.SUBTRACT,
                radius=2.75,
                height=wall,
                position=Vec3(x=-10.0, y=base_depth / 2.0 - wall / 2.0, z=bolt_z),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# The closed form is right: slender-plate limit matches Euler-Bernoulli.
# ---------------------------------------------------------------------------


def test_plate_term_is_euler_bernoulli() -> None:
    """Typo guard (the analytic 'patch test'): the plate-bending term equals F*L^3/(3*E*I)
    computed independently here, to numerical noise."""
    wall, depth, width = 4.0, 208.0, 40.0  # L = depth/2 - wall = 100mm, L/t = 25 (slender)
    program = _synthetic_bracket(wall, depth, width, bolt_z=wall / 2.0 + 0.1)
    est = estimate_static_load(program, LOAD, "PETG")
    assert isinstance(est, StaticLoadEstimate)
    f_n = (LOAD.payload_kg + LOAD.motor_kg) * 9.81
    l_b = depth / 2.0 - wall
    i_b = width * wall**3 / 12.0
    expected = f_n * l_b**3 / (3.0 * est.e_nominal_mpa * i_b)
    assert est.plate_bending_mm == pytest.approx(expected, rel=0.02)  # 2-sig-fig rounding


def test_slender_limit_is_bounded_above_euler_bernoulli() -> None:
    """In the slender-plate limit the total must sit ABOVE pure Euler-Bernoulli (the
    stubby-upright floor + shear always add compliance -- conservative by design) but
    bounded (< 1.3x): the extra terms are corrections, not the answer."""
    wall, depth, width = 4.0, 208.0, 40.0
    program = _synthetic_bracket(wall, depth, width, bolt_z=wall / 2.0 + 0.1)
    est = estimate_static_load(program, LOAD, "PETG")
    assert isinstance(est, StaticLoadEstimate)
    f_n = (LOAD.payload_kg + LOAD.motor_kg) * 9.81
    l_b = depth / 2.0 - wall
    i_b = width * wall**3 / 12.0
    eb = f_n * l_b**3 / (3.0 * est.e_nominal_mpa * i_b)
    nominal_total = (
        est.plate_bending_mm
        + est.plate_shear_mm
        + est.upright_bending_mm
        + est.upright_shear_mm
        + est.root_rotation_mm
    )
    assert eb < nominal_total < 1.3 * eb


def test_two_segment_exceeds_plate_only() -> None:
    """The council's core correction: root rotation + upright compliance ADD deflection.
    The full model must predict MORE than the plate-bending term alone (the old
    upright-only shape flattered the design)."""
    est = estimate_static_load(_bench_program(), LOAD, "PLA")
    assert isinstance(est, StaticLoadEstimate)
    total = (
        est.plate_bending_mm
        + est.plate_shear_mm
        + est.upright_bending_mm
        + est.upright_shear_mm
        + est.root_rotation_mm
    )
    assert total > est.plate_bending_mm
    assert est.root_rotation_mm > 0.0


# ---------------------------------------------------------------------------
# Honesty invariants: ranges, determinism, monotonicity, machine-honest payload.
# ---------------------------------------------------------------------------


def test_range_ordering_and_payload_honesty() -> None:
    est = estimate_static_load(_bench_program(), LOAD, "PLA")
    assert isinstance(est, StaticLoadEstimate)
    assert 0.0 < est.deflection_mm_lo <= est.deflection_mm_hi
    assert est.estimate is True and est.gates is False
    assert est.assumptions and est.disclaimer
    assert any("PLA softens" in w for w in est.warnings)  # the named thermal refusal-case
    # Strength is STRUCTURALLY absent -- no field can carry a stress/FoS claim.
    assert not any(
        "stress" in name or "yield" in name or "safety" in name for name in type(est).model_fields
    )


def test_deterministic() -> None:
    a = estimate_static_load(_bench_program(), LOAD, "PETG")
    b = estimate_static_load(_bench_program(), LOAD, "PETG")
    assert isinstance(a, StaticLoadEstimate) and isinstance(b, StaticLoadEstimate)
    assert a.model_dump() == b.model_dump()


def test_thicker_wall_deflects_less() -> None:
    """Pure model monotonicity on FIXED dims (the synthetic bracket): only the wall varies.
    (On a real spec the interpreter re-derives dependent dims, so this is the clean check.)"""
    thin = estimate_static_load(_synthetic_bracket(4.0, 120.0, 43.0, 15.0), LOAD, "PETG")
    thick = estimate_static_load(_synthetic_bracket(5.0, 120.0, 43.0, 15.0), LOAD, "PETG")
    assert isinstance(thin, StaticLoadEstimate) and isinstance(thick, StaticLoadEstimate)
    assert thick.deflection_mm_hi < thin.deflection_mm_hi
    assert thick.deflection_mm_lo < thin.deflection_mm_lo


# ---------------------------------------------------------------------------
# Refuse-not-guess: unknown material, wrong geometry.
# ---------------------------------------------------------------------------


def test_unknown_material_refuses() -> None:
    out = estimate_static_load(_bench_program(), LOAD, "unobtainium")
    assert isinstance(out, StaticLoadRefusal)
    assert any("refusing rather than guessing" in r for r in out.reasons)
    assert out.estimate is False and out.gates is False


def test_non_bracket_geometry_refuses() -> None:
    plate = PartProgram(
        metadata=Metadata(name="plate"),
        operations=[Box(op_id="plate", width=40.0, depth=20.0, height=4.0, position=Vec3(z=2.0))],
    )
    out = estimate_static_load(plate, LOAD, "PLA")
    assert isinstance(out, StaticLoadRefusal)
    assert any("L-bracket shape" in r for r in out.reasons)
    assert extract_bracket_dims(plate) is None


# ---------------------------------------------------------------------------
# Sizing inversion: re-derive through the REAL interpreter, snap up, verify forward.
# ---------------------------------------------------------------------------


def test_sizing_meets_target_via_reinterpretation() -> None:
    spec = _bench_spec(4.0)
    program = interpret_l_bracket(spec)
    target = 0.05
    out = size_wall_for_deflection(
        program, LOAD, "PETG", target, interpret=interpret_l_bracket, spec=spec
    )
    assert isinstance(out, StaticLoadEstimate), getattr(out, "reasons", None)
    assert out.sized_field == "min_wall_mm"
    assert out.sized_value_mm is not None and out.sized_value_mm >= 4.0
    assert (out.sized_value_mm * 2) == int(out.sized_value_mm * 2)  # 0.5mm increments
    nominal_total = (
        out.plate_bending_mm
        + out.plate_shear_mm
        + out.upright_bending_mm
        + out.upright_shear_mm
        + out.root_rotation_mm
    )
    assert nominal_total <= target * 1.05  # forward-verified at nominal E (2-sig-fig slack)


def test_sizing_refuses_an_impossible_target() -> None:
    spec = _bench_spec(4.0)
    program = interpret_l_bracket(spec)
    out = size_wall_for_deflection(
        program, LOAD, "PETG", 1e-9, interpret=interpret_l_bracket, spec=spec
    )
    assert isinstance(out, StaticLoadRefusal)
    assert any("different design" in r or "infeasible" in r for r in out.reasons)
