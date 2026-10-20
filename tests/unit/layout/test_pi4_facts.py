"""Pi 4 mount facts: one source feeding the spec defaults + the checker (ADR-0007 Stage B).

Pi4 differs from NEMA17/M5: its mount geometry is spec-parameterized
(``StandoffLayout`` exposes the hole rectangle + screw clearance), so the
interpreter reads the SPEC (which DEFAULTS to these facts) rather than a
``place_*()`` helper. The facts module is still the single source the spec defaults
and the validator checker both resolve to -- these tests prove it, and that the
fact-derived defaults build geometry the checker accepts.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces import pi4 as facts
from forge_mind.layout.pi4_interpreter import interpret_pi4
from forge_mind.layout.pi4_spec import Pi4CaseSpec, StandoffLayout
from forge_mind.types import Vec3
from forge_mind.validator.interfaces import pi4_mount as checker
from forge_mind.validator.run import validate


def test_checker_rectangle_comes_from_the_shared_fact() -> None:
    # The 58 x 49 rectangle the checker validates IS the shared fact (same objects).
    assert checker.RECT_DX is facts.HOLE_RECT_DX_MM
    assert checker.RECT_DY is facts.HOLE_RECT_DY_MM


def test_spec_defaults_come_from_the_shared_fact() -> None:
    so = StandoffLayout()
    assert so.hole_dx_mm == facts.HOLE_RECT_DX_MM
    assert so.hole_dy_mm == facts.HOLE_RECT_DY_MM


def test_m25_clearance_has_one_canonical_radius_source() -> None:
    # The checker validates against the canonical radius; the spec field is the
    # diameter derived from it -- one literal (the radius), no dia/radius drift.
    assert checker.M25_CLEARANCE_RADIUS is facts.M25_CLEARANCE_RADIUS_MM
    assert facts.M25_CLEARANCE_DIAMETER_MM == 2.0 * facts.M25_CLEARANCE_RADIUS_MM
    assert StandoffLayout().screw_clearance_mm == facts.M25_CLEARANCE_DIAMETER_MM


def test_pi4_mount_placement_check_parity(tmp_path: Path) -> None:
    # A spec using the fact-derived standoff DEFAULTS builds geometry the pi4_mount
    # checker accepts: facts -> placement -> check round-trips.
    spec = Pi4CaseSpec(
        name="pi4_parity",
        envelope_mm=Vec3(x=95.0, y=70.0, z=30.0),
        material="PLA",
        min_wall_mm=2.0,
        clearance_mm=2.0,
    )
    program = interpret_pi4(spec)
    report = validate(compile_program(program, tmp_path), program)
    for cid in (
        "interfaces.pi4_mount.mount_holes_present",
        "interfaces.pi4_mount.hole_rectangle_58x49",
        "interfaces.pi4_mount.standoff_land",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
