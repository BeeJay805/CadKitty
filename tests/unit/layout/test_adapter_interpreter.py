"""AdapterPlateInterpreter (ADR-0008): deterministic AdapterPlateSpec ->
valid PartProgram, with the same validator passing it that the hand-authored
bench-003 reference passes. No LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.layout.adapter_interpreter import (
    NEMA17_HALF_PITCH,
    derive_adapter_offsets,
    derive_plate_dims,
    interpret_adapter_plate,
)
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench003_adapter_plate.layoutspec.json"

# Same required-check list as bench-001 (ADR-0008: bench-003 reuses these
# checkers verbatim, no new validator code).
REQUIRED_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
    "interfaces.extrusion_face.two_m5_holes",
    "interfaces.extrusion_face.m5_spacing",
]


def _spec(**overrides: object) -> AdapterPlateSpec:
    spec = AdapterPlateSpec.model_validate(json.loads(FIXTURE.read_text()))
    return spec.model_copy(update=overrides) if overrides else spec


# ---------------------------------------------------------------------------
# Geometry contract
# ---------------------------------------------------------------------------


def test_interpret_emits_expected_operations() -> None:
    program = interpret_adapter_plate(_spec())
    op_ids = {op.op_id for op in program.operations}
    # One plate + 4 motor bolts + 1 shaft bore + 2 T-nut holes = 8 ops.
    assert "plate" in op_ids
    assert sum(1 for o in program.operations if o.op_id.startswith("motor_bolt")) == 4
    assert "shaft_clearance" in op_ids
    assert {"ext_bolt_back", "ext_bolt_front"} <= op_ids
    assert len(program.operations) == 8
    # First op is ADD (compiler seeds the body with it).
    assert program.operations[0].mode.value == "add"
    # Two interfaces declared, both axis-along-Z.
    assert len(program.interfaces) == 2
    motor = next(i for i in program.interfaces if i.name == "motor_face")
    extrusion = next(i for i in program.interfaces if i.name == "extrusion_face")
    assert motor.normal.as_tuple() == (0.0, 0.0, 1.0)
    assert extrusion.normal.as_tuple() == (0.0, 0.0, -1.0)


def test_geometry_matches_handwritten_reference() -> None:
    # ADR-0008 contract: the interpreter, on the canonical spec, produces a
    # part equivalent to the hand-authored bench-003 reference (same op ids +
    # positions + sizes). This is the "deterministic reference" gate.
    from examples.handwritten_nema17_2020_adapter import build_adapter_plate_reference

    interpreted = interpret_adapter_plate(_spec())
    reference = build_adapter_plate_reference()

    assert [o.op_id for o in interpreted.operations] == [o.op_id for o in reference.operations]
    by_ref = {o.op_id: o for o in reference.operations}
    for op in interpreted.operations:
        ref = by_ref[op.op_id]
        assert op.mode == ref.mode
        assert op.position.as_tuple() == ref.position.as_tuple()
        assert op.rotation.x == ref.rotation.x
        assert op.rotation.y == ref.rotation.y
        assert op.rotation.z == ref.rotation.z


def test_adapter_layoutspec_passes_validator(tmp_path: Path) -> None:
    # THE DELIVERABLE: hand-authored spec -> valid bench-003 part, all
    # checks pass via the existing checkers with no new validator code.
    result = run_layout_pipeline(_spec(), tmp_path)
    assert result.error is None, result.error
    assert result.report is not None
    failed = set(result.report.failed_check_ids())
    assert not failed, f"unexpected failures: {failed}"
    for cid in REQUIRED_CHECKS:
        check = result.report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    assert result.passed


# ---------------------------------------------------------------------------
# Relational invariants enforced before geometry exists
# ---------------------------------------------------------------------------


def test_hole_to_edge_land_is_at_least_min_wall() -> None:
    spec = _spec()
    program = interpret_adapter_plate(spec)
    w = spec.min_wall_mm
    # The canonical fixture has explicit dims + offsets; ADR-0009 Task 1 made
    # them optional in the spec but the fixture pins them so the values are
    # non-None at runtime.
    assert spec.plate_width_mm is not None and spec.plate_depth_mm is not None
    assert spec.motor_center_x_mm is not None
    half_w = spec.plate_width_mm / 2.0
    half_d = spec.plate_depth_mm / 2.0
    # M3 bolt holes around the motor face: every outer edge clears the plate
    # boundary by >= min_wall in both X and Y.
    motor_bolts = [o for o in program.operations if o.op_id.startswith("motor_bolt")]
    for op in motor_bolts:
        land_x = half_w - (abs(op.position.x) + 1.6)  # M3 clearance r
        land_y = half_d - (abs(op.position.y) + 1.6)
        assert land_x >= w - 1e-9
        assert land_y >= w - 1e-9
    # And the bolt holes sit exactly on the 31mm square around the motor centre.
    centred = [(b.position.x - spec.motor_center_x_mm, b.position.y) for b in motor_bolts]
    assert all(abs(abs(x) - NEMA17_HALF_PITCH) < 1e-9 for x, _ in centred)


def test_too_small_envelope_rejected() -> None:
    # Plate width 65mm cannot fit in a 60mm envelope -- caught BEFORE compile.
    with pytest.raises(LayoutError, match="exceeds envelope"):
        interpret_adapter_plate(_spec(envelope_mm=Vec3(x=60.0, y=50.0, z=8.0)))


def test_too_small_plate_rejects_with_land_problem() -> None:
    # Shrink the plate to where the M3 hole cluster cannot keep a min_wall
    # land to the plate edge. The interpreter must reject up front.
    with pytest.raises(LayoutError, match="too small"):
        interpret_adapter_plate(_spec(plate_width_mm=40.0))


def test_motor_and_tnut_overlap_rejected() -> None:
    # Slide both centres so the motor M3 cluster and the T-nut pair overlap in
    # plan view. Caught by the in-plane-clearance check.
    with pytest.raises(LayoutError, match=r"overlap|touch"):
        interpret_adapter_plate(_spec(motor_center_x_mm=0.0, extrusion_center_x_mm=-5.0))


# ---------------------------------------------------------------------------
# Individual interface checks pass via the existing checkers (no new code).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plate dims are derivable: the LLM-side fix for the perturbation finding.
# ---------------------------------------------------------------------------


def test_derive_plate_dims_for_canonical_offsets() -> None:
    # M=+11, E=-22, m5_spacing=20, min_wall=4:
    #   half_w = max(11 + 17.1, 22 + 2.6, 11 + 11.0) + 4 = max(28.1, 24.6, 22.0) + 4 = 32.1
    #   half_d = max(17.1, 10 + 2.6) + 4 = 21.1
    #   -> width 64.2, depth 42.2
    spec = _spec(plate_width_mm=None, plate_depth_mm=None)
    width, depth = derive_plate_dims(spec)
    assert abs(width - 64.2) < 1e-9
    assert abs(depth - 42.2) < 1e-9


def test_interpret_uses_derived_dims_when_plate_fields_omitted(tmp_path: Path) -> None:
    # Omit both plate fields; the interpreter must derive a passing part.
    spec = _spec(plate_width_mm=None, plate_depth_mm=None)
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed
    # The plate op carries the derived dims (not the fixture's explicit 65x45).
    plate_op = next(o for o in result.program.operations if o.op_id == "plate")  # type: ignore[union-attr]
    assert abs(plate_op.width - 64.2) < 1e-9  # type: ignore[attr-defined]
    assert abs(plate_op.depth - 42.2) < 1e-9  # type: ignore[attr-defined]


def test_explicit_plate_dims_still_win_for_backward_compat(tmp_path: Path) -> None:
    # The hand-authored fixture has explicit 65x45 plate dims (with extra
    # headroom over the 64.2x42.2 derived minimum). Those values must be
    # used as-is so the byte-equivalent reference comparison stays valid.
    spec = _spec()  # explicit values from the fixture
    result = run_layout_pipeline(spec, tmp_path)
    assert result.passed
    plate_op = next(o for o in result.program.operations if o.op_id == "plate")  # type: ignore[union-attr]
    assert plate_op.width == 65.0  # type: ignore[attr-defined]
    assert plate_op.depth == 45.0  # type: ignore[attr-defined]


def test_tight_x_envelope_with_adapted_offsets_now_passes(tmp_path: Path) -> None:
    # The tight-envelope failure mode from the perturbation experiment was a
    # too-narrow plate (model picked 55 wide when the canonical motor offset
    # +11 demands >= 64.2). Fix: model picks smaller offsets, interpreter
    # derives the matching plate. Tighten X only (the Y direction is fixed
    # at 42.2 mm by the NEMA17 M3 cluster + 4 mm wall, regardless of offsets).
    spec = _spec(
        envelope_mm=Vec3(x=60.0, y=50.0, z=8.0),
        plate_width_mm=None,
        plate_depth_mm=None,
        motor_center_x_mm=8.0,
        extrusion_center_x_mm=-16.0,
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed
    plate_op = next(o for o in result.program.operations if o.op_id == "plate")  # type: ignore[union-attr]
    # Derived plate: half_w = max(8+17.1, -8+11, 16+2.6) + 4 = 29.1; half_d = 21.1.
    assert abs(plate_op.width - 58.2) < 1e-9  # type: ignore[attr-defined]
    assert abs(plate_op.depth - 42.2) < 1e-9  # type: ignore[attr-defined]


def test_envelope_y_below_m3_minimum_still_rejected(tmp_path: Path) -> None:
    # Honest negative case: the NEMA17 M3 cluster needs depth >= 2*(17.1+wall)
    # regardless of offsets. An envelope with Y below that minimum is
    # GEOMETRICALLY infeasible, derivation cannot help, interpreter must
    # reject (this is correct behaviour, not a bug).
    spec = _spec(
        envelope_mm=Vec3(x=60.0, y=40.0, z=8.0),
        plate_width_mm=None,
        plate_depth_mm=None,
        motor_center_x_mm=8.0,
        extrusion_center_x_mm=-16.0,
    )
    import pytest

    from forge_mind.layout.adapter_interpreter import interpret_adapter_plate
    from forge_mind.layout.interpreter import LayoutError

    with pytest.raises(LayoutError, match="exceeds envelope Y"):
        interpret_adapter_plate(spec)


def test_thicker_wall_with_adapted_offsets_passes_with_derived_dims(tmp_path: Path) -> None:
    # The thicker-wall variant from the perturbation experiment, now run via
    # the derived-dims path. min_wall=6, motor at +12 (legal land), T-nut at
    # -22; envelope 80x60x10. The interpreter derives plate 70.2 x 46.2.
    spec = _spec(
        envelope_mm=Vec3(x=80.0, y=60.0, z=10.0),
        plate_width_mm=None,
        plate_depth_mm=None,
        min_wall_mm=6.0,
        motor_center_x_mm=12.0,
        extrusion_center_x_mm=-22.0,
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


# ---------------------------------------------------------------------------
# Offsets are derivable too: ADR-0009 Task 1 (recover bench-003 decontaminated)
# ---------------------------------------------------------------------------


def test_derive_adapter_offsets_for_canonical_min_wall() -> None:
    # min_wall = 4, envelope X = 70 (canonical bench-003).
    # Balanced solution from derive_adapter_offsets:
    #   no_overlap_floor = 17.1 + 2.6 + 4 = 23.7
    #   motor_abs = (23.7 - 14.5) / 2 = 4.6 (+ 1e-6 eps)
    #   ext_abs = 4.6 + 14.5 = 19.1 (+ 1e-6 eps)
    # Result: motor at +4.6, T-nut at -19.1.
    motor_x, ext_x = derive_adapter_offsets(envelope_x_mm=70.0, min_wall_mm=4.0)
    tol = 1e-4
    assert abs(motor_x - 4.6) < tol
    assert abs(ext_x - (-19.1)) < tol
    # The pair satisfies the no-overlap rule by min_wall.
    assert abs(motor_x - ext_x) >= 17.1 + 2.6 + 4.0 - 1e-6


def test_derive_adapter_offsets_grows_with_min_wall() -> None:
    # min_wall = 6: no_overlap_floor = 25.7, motor_abs = (25.7 - 14.5)/2 = 5.6.
    motor_x, ext_x = derive_adapter_offsets(envelope_x_mm=80.0, min_wall_mm=6.0)
    tol = 1e-4
    assert abs(motor_x - 5.6) < tol
    assert abs(ext_x - (-20.1)) < tol


def test_derive_adapter_offsets_rejects_envelope_too_small() -> None:
    # min_wall=4 needs balanced plate width = 2 * (4.6 + 17.1 + 4) = 51.4 mm.
    # An envelope X under 51.4 cannot host a balanced layout regardless of
    # other choices; the helper rejects up-front rather than returning
    # values that would only get rejected one check downstream.
    with pytest.raises(LayoutError, match="envelope X"):
        derive_adapter_offsets(envelope_x_mm=50.0, min_wall_mm=4.0)


def test_interpret_uses_derived_offsets_when_both_omitted(tmp_path: Path) -> None:
    # ADR-0009 Task 1 deliverable: a spec with no motor/T-nut offsets at all
    # produces a passing part via the derived offsets + derived plate dims.
    # This is exactly the shape an LLM emits under the decontaminated prompt.
    spec = _spec(
        plate_width_mm=None,
        plate_depth_mm=None,
        motor_center_x_mm=None,
        extrusion_center_x_mm=None,
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed
    # Derived offsets: motor at +4.6, T-nut at -19.1; derived plate at
    # 2 * (4.6 + 17.1 + 4) = 51.4 wide, 2 * (17.1 + 4) = 42.2 deep.
    plate_op = next(o for o in result.program.operations if o.op_id == "plate")  # type: ignore[union-attr]
    tol = 1e-3
    assert abs(plate_op.width - 51.4) < tol  # type: ignore[attr-defined]
    assert abs(plate_op.depth - 42.2) < tol  # type: ignore[attr-defined]


def test_explicit_offsets_still_win_for_backward_compat(tmp_path: Path) -> None:
    # The canonical fixture has explicit motor=+11, T-nut=-22; those values
    # must be used as-is so the byte-equivalent reference comparison stays
    # valid. The derived offsets (+4.6, -19.1) are NOT what the fixture
    # produces.
    spec = _spec()  # explicit values from the fixture
    result = run_layout_pipeline(spec, tmp_path)
    assert result.passed
    motor_bolt = next(
        o
        for o in result.program.operations
        if o.op_id == "motor_bolt_ne"  # type: ignore[union-attr]
    )
    # motor_bolt_ne sits at (motor_x + 15.5, +15.5); for the fixture motor_x = 11
    # the bolt is at x = 26.5.
    assert abs(motor_bolt.position.x - 26.5) < 1e-9  # type: ignore[attr-defined]


def test_geometry_matches_handwritten_reference_with_derived_offsets() -> None:
    # When offsets are omitted, the interpreter's emitted geometry is NOT
    # byte-equivalent to the hand-authored reference (which uses explicit
    # canonical M=+11/E=-22). Document that this is the intended difference,
    # and verify the existing byte-equivalence test still holds for the
    # explicit-fixture path.
    from examples.handwritten_nema17_2020_adapter import build_adapter_plate_reference

    explicit_program = interpret_adapter_plate(_spec())  # explicit fixture
    reference = build_adapter_plate_reference()
    # Explicit-fixture path: byte-equivalent (unchanged from pre-ADR-0009).
    assert [o.op_id for o in explicit_program.operations] == [o.op_id for o in reference.operations]
    by_ref = {o.op_id: o for o in reference.operations}
    for op in explicit_program.operations:
        ref = by_ref[op.op_id]
        assert op.position.as_tuple() == ref.position.as_tuple()


def test_nema17_interface_checks_pass(tmp_path: Path) -> None:
    result = run_layout_pipeline(_spec(), tmp_path)
    assert result.report is not None
    for cid in (
        "interfaces.motor_face.bolt_holes_present",
        "interfaces.motor_face.bolt_pattern_31mm_square",
        "interfaces.motor_face.shaft_bore_present",
    ):
        check = result.report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_extrusion_interface_checks_pass(tmp_path: Path) -> None:
    result = run_layout_pipeline(_spec(), tmp_path)
    assert result.report is not None
    for cid in (
        "interfaces.extrusion_face.two_m5_holes",
        "interfaces.extrusion_face.m5_spacing",
    ):
        check = result.report.check(cid)
        assert check is not None and check.status.value == "pass", cid
