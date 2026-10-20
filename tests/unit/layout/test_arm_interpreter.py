"""First-arm interpreter: a computed_body strut fused with a NEMA-17 motor mount validates
end-to-end as ONE part (fusion integrity + strut printability + mount certified + mass)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from forge_mind.blueprint.arm_sizing import REACH_SPAN_MM, derive_nema17_arm_envelope
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import ComputedSolid
from forge_mind.layout.arm_interpreter import ArmSpec, interpret_arm
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.types import Vec3
from forge_mind.validator.metrics import density_for
from forge_mind.validator.run import validate

_ENVELOPE = Vec3(x=180.0, y=60.0, z=30.0)


def _spec(
    cross_section: str = "hexagonal", envelope: Vec3 = _ENVELOPE, min_wall: float = 4.0
) -> ArmSpec:
    return ArmSpec(
        name="arm", envelope_mm=envelope, min_wall_mm=min_wall, cross_section=cross_section
    )


def _validate(spec: ArmSpec, tmp_path: Path):  # type: ignore[no-untyped-def]
    program = interpret_arm(spec)
    model = compile_program(program, tmp_path / spec.name)
    report = validate(model, program, material=spec.material)
    return program, report, {c.id: c.status.value for c in report.checks}


def test_arm_fuses_and_validates_all_checks(tmp_path: Path) -> None:
    program, report, ck = _validate(_spec("hexagonal"), tmp_path)
    assert report.passed, report.failed_check_ids()
    # Fusion integrity: the strut + mount are ONE watertight body.
    assert ck["mesh.single_body"] == "pass"
    assert ck["mesh.watertight"] == "pass"
    # Strut printability.
    assert ck["features.min_wall_thickness"] == "pass"
    assert ck["printability.overhang_max_angle"] == "pass"
    # Motor mount certified.
    assert ck["interfaces.motor_face.bolt_holes_present"] == "pass"
    assert ck["interfaces.motor_face.bolt_pattern_31mm_square"] == "pass"
    assert ck["interfaces.motor_face.shaft_bore_present"] == "pass"
    # The body carries a computed strut op (the computed shell).
    assert any(isinstance(op, ComputedSolid) for op in program.operations)
    # Mass reported.
    assert report.metrics is not None and report.metrics.mass_g > 0.0


def test_arm_rect_section_validates(tmp_path: Path) -> None:
    _program, report, _ck = _validate(_spec("rectangular"), tmp_path)
    assert report.passed, report.failed_check_ids()


def test_interpret_layout_routes_arm(tmp_path: Path) -> None:
    program = interpret_layout(_spec("hexagonal"))
    assert any(isinstance(op, ComputedSolid) for op in program.operations)
    assert any(d.name == "motor_face" for d in program.interfaces)
    model = compile_program(program, tmp_path)
    assert validate(model, program).passed


def test_arm_infeasible_envelope_raises() -> None:
    # envelope.y too small to host the NEMA-17 mount pad -> rejected up front.
    with pytest.raises(LayoutError, match="mount needs"):
        interpret_arm(_spec(envelope=Vec3(x=180.0, y=30.0, z=30.0)))


# --- Part A: deterministic span-sizing (the LLM picks reach; code derives the envelope) ------


@pytest.mark.parametrize(
    ("cross_section", "expected_volume"),
    [
        ("hexagonal", 30621.765955),
        ("rectangular", 32217.192449),
    ],
)
def test_explicit_envelope_arm_is_byte_identical(
    cross_section: str, expected_volume: float, tmp_path: Path
) -> None:
    # BYTE-EQUIVALENCE: an explicit envelope is honored verbatim, so the existing arm geometry is
    # unchanged by the span-sizing feature. Pinned volume snapshot (the current arm).
    program = interpret_arm(_spec(cross_section))
    model = compile_program(program, tmp_path / cross_section)
    assert math.isclose(model.part.volume, expected_volume, abs_tol=1e-3)


@pytest.mark.parametrize("reach", ["compact", "standard", "long"])
@pytest.mark.parametrize("cross_section", ["hexagonal", "rectangular"])
@pytest.mark.parametrize("min_wall", [3.0, 4.0, 5.0])
def test_derived_envelope_always_feasible(reach: str, cross_section: str, min_wall: float) -> None:
    # DRIFT GUARD: a derived envelope (no LLM numbers) must NEVER trip the interpreter's own
    # feasibility guards, across every reach x cross_section x min_wall. If arm_sizing and the
    # interpreter's pad/strut math drift apart, this raises LayoutError and fails here.
    spec = ArmSpec(
        name="arm", envelope_mm=None, min_wall_mm=min_wall, cross_section=cross_section, reach=reach
    )
    program = interpret_arm(spec)  # must not raise
    assert any(isinstance(op, ComputedSolid) for op in program.operations)


def test_derived_arm_validates_and_certifies(tmp_path: Path) -> None:
    # The fully-derived arm (envelope null) validates clean + certifies the motor face.
    spec = ArmSpec(
        name="arm", envelope_mm=None, min_wall_mm=4.0, cross_section="hexagonal", reach="standard"
    )
    program = interpret_arm(spec)
    model = compile_program(program, tmp_path)
    report = validate(model, program, material=spec.material)
    ck = {c.id: c.status.value for c in report.checks}
    assert report.passed, report.failed_check_ids()
    assert ck["interfaces.motor_face.bolt_pattern_31mm_square"] == "pass"


def test_reach_sets_span_monotonically(tmp_path: Path) -> None:
    assert REACH_SPAN_MM["compact"] < REACH_SPAN_MM["standard"] < REACH_SPAN_MM["long"]
    spans = []
    for reach in ("compact", "standard", "long"):
        program = interpret_arm(ArmSpec(name="arm", min_wall_mm=4.0, reach=reach))
        model = compile_program(program, tmp_path / reach)
        spans.append(model.bounding_box.max.x - model.bounding_box.min.x)
    assert spans[0] < spans[1] < spans[2]  # derived span grows with the reach class
    assert math.isclose(
        spans[1], REACH_SPAN_MM["standard"], abs_tol=1.0
    )  # standard arm fills its span


# --- Part B: lattice-bodied arm (a latticed beam fused with the mount) -----------------------


@pytest.mark.parametrize("density", ["light", "medium"])
def test_lattice_arm_validates_certifies_and_is_lighter(density: str, tmp_path: Path) -> None:
    spec = ArmSpec(name="larm", min_wall_mm=4.0, body="lattice", density=density, reach="standard")
    program = interpret_arm(spec)
    # The body is a grid of straight wall members (the LatticeOp Box-union), not a single strut.
    assert sum(1 for op in program.operations if op.op_id.startswith("lat_")) >= 4
    assert not any(isinstance(op, ComputedSolid) for op in program.operations)  # no faceted strut
    model = compile_program(program, tmp_path)
    report = validate(model, program, material=spec.material)
    ck = {c.id: c.status.value for c in report.checks}
    assert report.passed, report.failed_check_ids()
    assert ck["mesh.single_body"] == "pass"  # the lattice walls + the mount fuse into ONE body
    assert ck["mesh.watertight"] == "pass"
    assert ck["printability.overhang_max_angle"] == "pass"  # all walls vertical
    assert ck["interfaces.motor_face.bolt_pattern_31mm_square"] == "pass"  # motor mount certified
    # Lighter than a solid of equal envelope (the lattice payoff). The de-risk + Part-B validation
    # prove the tighter beam-vs-beam comparison end-to-end; this is the in-suite regression guard.
    env = derive_nema17_arm_envelope("standard", 4.0)
    solid_envelope_mass = env.x * env.y * env.z / 1000.0 * density_for("PLA")
    assert report.metrics is not None
    assert report.metrics.mass_g < 0.7 * solid_envelope_mass


def test_default_arm_body_is_solid() -> None:
    # Byte-equivalence safety: the body defaults to solid, so an unspecified arm is the original
    # faceted-strut arm (no lattice), keeping every existing arm unchanged.
    assert ArmSpec(name="a", min_wall_mm=4.0).body == "solid"
    program = interpret_arm(ArmSpec(name="a", min_wall_mm=4.0))
    assert any(isinstance(op, ComputedSolid) for op in program.operations)
