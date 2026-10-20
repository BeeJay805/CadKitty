"""Pi4CaseInterpreter: deterministic Pi4CaseSpec -> valid PartProgram (no LLM).

The headline test compiles + validates the hand-authored fixture against the real
validator (pi4_mount + pi4_ports). The rest prove infeasible specs are rejected
before any geometry, that a closed (no-ports) tray omits the pi4_ports claim and
still validates, and that the pi4_ports checker still flags a wall it is told to
have an opening but does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.layout.interpreter import InfeasibleRequestError
from forge_mind.layout.pi4_interpreter import (
    _derived_standoff_od,
    _effective_clearance,
    _resolve_board,
    interpret_pi4,
)
from forge_mind.layout.pi4_spec import Pi4CaseSpec, PortPolicy, StandoffLayout
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.types import Vec3
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench002_pi4_case.layoutspec.json"

REQUIRED_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.pi4_mount.mount_holes_present",
    "interfaces.pi4_mount.hole_rectangle_58x49",
    "interfaces.pi4_mount.standoff_land",
    "interfaces.pi4_ports.long_edge_opening",
    "interfaces.pi4_ports.short_edge_opening",
]


def _spec(**overrides: object) -> Pi4CaseSpec:
    spec = Pi4CaseSpec.model_validate(json.loads(FIXTURE.read_text()))
    return spec.model_copy(update=overrides) if overrides else spec


def _validate(spec: Pi4CaseSpec, work: Path) -> ValidationReport:
    program = interpret_pi4(spec)
    model = compile_program(program, work)
    return validate(model, program)


def test_pi4_fixture_compiles_and_validates(tmp_path: Path) -> None:
    report = _validate(_spec(), tmp_path)
    failed = set(report.failed_check_ids())
    assert not failed, f"unexpected failures: {failed}"
    for cid in REQUIRED_CHECKS:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    assert report.passed


def test_interpreter_reproduces_reference_geometry(tmp_path: Path) -> None:
    # Same bbox as the hand-authored reference (93 x 64 x 16).
    model = compile_program(interpret_pi4(_spec()), tmp_path)
    size = tuple(round(v, 1) for v in model.bounding_box.size.as_tuple())
    assert size == (93.0, 64.0, 16.0)


def test_thin_standoff_od_silently_grown_to_derived_minimum(tmp_path: Path) -> None:
    # Pre-fix: OD 6 (radius 3, land 1.65 mm < 2 mm min_wall) was rejected.
    # Post-fix (bench-002 perturbation work): the interpreter treats the LLM /
    # legacy field as a FLOOR and grows it to the derived minimum (2*(1.35+2) =
    # 6.7 mm). The spec validates rather than rejecting.
    thin = StandoffLayout(outer_diameter_mm=6.0)
    spec = _spec(standoffs=thin)
    rb = _resolve_board(spec)
    assert rb.standoff_od_mm > 6.0  # grown past the LLM's pick
    assert abs(rb.standoff_od_mm - 6.7) < 1e-3  # to the derived min, +eps
    # And the part now interprets + validates cleanly.
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


def test_undersized_envelope_grows_instead_of_rejecting(tmp_path: Path) -> None:
    """Grow-don't-reject (ADR-0035 A2): the outer footprint derives from board facts, so an
    LLM-under-sized envelope GROWS to fit instead of killing the build. Pre-fix this exact spec
    raised LayoutError 'exceeds envelope'."""
    spec = _spec(envelope_mm=Vec3(x=80.0, y=70.0, z=30.0))
    program = interpret_pi4(spec)
    # The grown constraint covers the real footprint (the canonical case is 93 x 64 x 16).
    assert program.constraints.max_bbox is not None
    assert program.constraints.max_bbox.x >= 93.0
    report = validate(compile_program(program, tmp_path), program)
    assert report.passed, report.failed_check_ids()


def test_user_hard_max_still_refuses_honestly() -> None:
    """A USER-stated hard maximum is a real limit: growth past it is an InfeasibleRequestError
    (the clean refusal path), never a silent build outside stated intent."""
    spec = _spec(
        envelope_mm=Vec3(x=80.0, y=70.0, z=30.0),
        max_envelope_mm=Vec3(x=80.0, y=70.0, z=30.0),
    )
    with pytest.raises(InfeasibleRequestError, match="cannot fit within the requested max"):
        interpret_pi4(spec)


def test_pi4_case_at_production_defaults_builds(tmp_path: Path) -> None:
    """THE LIVE KILL, un-killed (ADR-0035 A2): at the production blueprint default min_wall=4 the
    derived standoff OD forces a ~105x76mm tray vs the LLM's ~100x70 estimate. Pre-fix:
    LayoutError 'case length 105mm exceeds envelope X' -> every Raspberry Pi prompt burned the
    standard lane and fell to the ~25-50% agentic path. Now the envelope grows and the case
    builds + validates at production defaults."""
    spec = Pi4CaseSpec(
        name="pi4_case_production_defaults",
        envelope_mm=Vec3(x=100.0, y=70.0, z=40.0),
        material="PLA",
        min_wall_mm=4.0,
        clearance_mm=4.0,
        board_component_id="raspberry_pi_4",
    )
    program = interpret_pi4(spec)
    model = compile_program(program, tmp_path)
    size = model.bounding_box.size
    assert size.x > 100.0  # grown past the estimate to the derived footprint (~104.7)
    assert size.y > 70.0  # ~75.7
    report = validate(model, program)
    assert report.passed, report.failed_check_ids()


def test_small_clearance_silently_grown_to_derived_minimum(tmp_path: Path) -> None:
    # Pre-fix: a 0.5 mm clearance was rejected because the corner standoff would
    # sit 0.5 mm from the wall (< min_wall = 2 mm). Post-fix the interpreter
    # treats spec.clearance_mm as a FLOOR and grows it to the derived minimum
    # that keeps gap >= min_wall. The part validates rather than rejecting.
    spec = _spec(clearance_mm=0.5)
    rb = _resolve_board(spec)
    eff = _effective_clearance(spec, rb)
    assert eff > 0.5  # grown past the LLM's pick
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


def test_derived_standoff_od_for_canonical_min_wall() -> None:
    # min_wall = 2, screw_r = 1.35 -> derived OD ~ 6.7 mm (+ FP epsilon).
    assert abs(_derived_standoff_od(1.35, 2.0) - 6.7) < 1e-3


def test_resolved_standoff_od_grows_under_thicker_wall() -> None:
    # Canonical bench-002 fixture but at min_wall=4. Registry OD-7 fails the
    # land rule; the resolver grows it to >= 10.7 so the structural failure
    # mode the bench-002 perturbation surfaced becomes impossible.
    spec = _spec(min_wall_mm=4.0)
    rb = _resolve_board(spec)
    assert rb.standoff_od_mm >= 10.7
    # Effective clearance grows in turn so the standoff-to-wall gap stays >= w.
    eff = _effective_clearance(spec, rb)
    standoff_r = rb.standoff_od_mm / 2.0
    gap_y = (rb.width_mm / 2.0 + eff) - (rb.hole_dy_mm / 2.0 + standoff_r)
    assert gap_y >= 4.0 - 1e-6


def test_ports_none_omits_pi4_ports_and_validates(tmp_path: Path) -> None:
    # ADR-0007 Stage D: a no-ports tray does NOT claim the pi4_ports interface, so it
    # is a valid closed tray rather than a Pi 4 case missing its openings.
    spec = _spec(ports=PortPolicy.NONE)
    program = interpret_pi4(spec)
    assert not any(d.name == "pi4_ports" for d in program.interfaces)
    report = validate(compile_program(program, tmp_path), program)
    assert report.passed
    assert not any(cid.startswith("interfaces.pi4_ports") for cid in report.failed_check_ids())


def test_pi4_ports_checker_flags_a_closed_wall_it_is_told_to_open(tmp_path: Path) -> None:
    # The pi4_ports checker still catches missing openings: take the closed (no-ports)
    # tray and attach a pi4_ports claim it does not satisfy -> the port checks fail.
    closed = interpret_pi4(_spec(ports=PortPolicy.NONE))
    claimed = closed.model_copy(
        update={
            "interfaces": [
                *closed.interfaces,
                InterfaceDeclaration(
                    name="pi4_ports",
                    interface_type=InterfaceType.PI4_PORTS,
                    location=Vec3(x=0.0, y=0.0, z=8.0),
                    normal=Vec3(x=0.0, y=1.0, z=0.0),
                ),
            ]
        }
    )
    report = validate(compile_program(claimed, tmp_path), claimed)
    for cid in (
        "interfaces.pi4_ports.long_edge_opening",
        "interfaces.pi4_ports.short_edge_opening",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "fail", cid
