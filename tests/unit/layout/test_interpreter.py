"""LayoutInterpreter: deterministic LayoutSpec -> valid PartProgram (no LLM).

The headline test (`test_bench001_layoutspec_passes_validator`) is the ADR-0004
deliverable: a hand-authored l_bracket_motor_mount spec compiles into a part the
EXISTING validator passes outright -- the geometry the direct DSL path produced
only ~10-23% of the time and the reviser could not repair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.layout.interpreter import (
    M3_CLEARANCE_RADIUS,
    NEMA17_HALF_PITCH,
    LayoutError,
    derive_l_bracket_dims,
    interpret,
)
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.layout.spec import BaseDims, LBracketMotorMountSpec, UprightDims
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench001_l_bracket.layoutspec.json"

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


def _spec(**overrides: object) -> LBracketMotorMountSpec:
    spec = LBracketMotorMountSpec.model_validate(json.loads(FIXTURE.read_text()))
    return spec.model_copy(update=overrides) if overrides else spec


def test_interpret_emits_expected_operations() -> None:
    program = interpret(_spec())
    op_ids = {op.op_id for op in program.operations}
    assert {"motor_flange", "ext_flange", "shaft_clearance"} <= op_ids
    assert sum(1 for o in program.operations if o.op_id.startswith("motor_bolt")) == 4
    assert sum(1 for o in program.operations if o.op_id.startswith("ext_bolt")) == 2
    # First op must be an ADD (the compiler seeds the body with it).
    assert program.operations[0].mode.value == "add"
    assert len(program.interfaces) == 2


def test_bench001_layoutspec_passes_validator(tmp_path: Path) -> None:
    # THE DELIVERABLE: hand-authored spec -> valid bench-001 part, all checks pass.
    result = run_layout_pipeline(_spec(), tmp_path)
    assert result.error is None
    assert result.report is not None
    failed = set(result.report.failed_check_ids())
    assert not failed, f"unexpected failures: {failed}"
    for cid in REQUIRED_CHECKS:
        check = result.report.check(cid)
        assert check is not None and check.status.value == "pass", cid
    assert result.passed


def test_hole_to_edge_land_is_at_least_min_wall() -> None:
    # The relational invariant the LLM kept violating, now guaranteed by code:
    # every M3 hole's outer edge clears the base edge by >= min_wall.
    spec = _spec()
    program = interpret(spec)
    w = spec.min_wall_mm
    half_w = spec.base.width_mm / 2.0
    half_d = spec.base.depth_mm / 2.0
    for op in program.operations:
        if op.op_id.startswith("motor_bolt"):
            land_x = half_w - (abs(op.position.x) + M3_CLEARANCE_RADIUS)
            land_y = half_d - (abs(op.position.y) + M3_CLEARANCE_RADIUS)
            assert land_x >= w - 1e-9
            assert land_y >= w - 1e-9
    # And the bolt holes sit exactly on the 31mm square.
    bolts = [o for o in program.operations if o.op_id.startswith("motor_bolt")]
    assert all(abs(abs(b.position.x) - NEMA17_HALF_PITCH) < 1e-9 for b in bolts)


def test_canonical_fixture_includes_gusset_when_envelope_has_room(tmp_path: Path) -> None:
    # ADR-0009 Task 2: gusset is now interpreter-controlled (always AUTO
    # semantics). On the canonical fixture (base 43x50, envelope 60x60x40)
    # there IS room for the rib: max_depth = 50/2 - 4 - (11+4) = 6 mm >= 4
    # mm min_wall. The interpreter adds it; the part still validates.
    result = run_layout_pipeline(_spec(), tmp_path)
    assert result.program is not None
    assert any(o.op_id == "corner_gusset" for o in result.program.operations)
    assert result.passed


def test_too_tight_base_silently_skips_gusset_no_layout_error(tmp_path: Path) -> None:
    # ADR-0009 Task 2 deliverable: on a base depth where the rib cannot fit
    # past the shaft bore + min_wall land, the interpreter SILENTLY skips
    # the rib rather than raising LayoutError. Pre-fix this exact shape (LLM
    # picking gusset=required with the derived tightest legal base depth)
    # was the bench-001 decontaminated failure mode (23/30 trials).
    # base_depth = 44 mm: M3 land = 22 - 15.5 - 1.6 = 4.9 mm (>= 4 mm wall),
    # but gusset max_depth = 22 - 4 - (11+4) = 3 mm (< 4 mm wall) -> rib skipped.
    spec = _spec(
        base=BaseDims(width_mm=43.0, depth_mm=44.0),
        upright=UprightDims(width_mm=43.0, height_mm=30.0),
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.program is not None
    assert not any(o.op_id == "corner_gusset" for o in result.program.operations)
    assert result.passed


def test_legacy_gusset_field_no_longer_loadable() -> None:
    # ADR-0009 Task 2: `gusset` was a real field; old fixtures that set it
    # must now be rejected by extra="forbid" rather than silently coerced.
    import pytest as _pytest
    from pydantic import ValidationError

    payload = json.loads(FIXTURE.read_text())
    payload["gusset"] = "required"  # legacy value
    with _pytest.raises(ValidationError):
        LBracketMotorMountSpec.model_validate(payload)


def test_derived_dims_path_silently_skips_gusset_no_layout_error(tmp_path: Path) -> None:
    # The recovery scenario that drives ADR-0009 Task 2 specifically:
    # derived base 42.2 x 42.2 (the 096f997 tightest legal fix) leaves no
    # room for a rib. The interpreter must skip rather than raise.
    spec = _spec(base=None, upright=None)
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.program is not None
    assert not any(o.op_id == "corner_gusset" for o in result.program.operations)
    assert result.passed


def test_rejects_envelope_too_small() -> None:
    small = _spec().envelope_mm.model_copy(update={"x": 30.0})
    with pytest.raises(LayoutError, match="exceeds envelope"):
        interpret(_spec(envelope_mm=small))


def test_rejects_base_too_narrow_for_holes_and_wall() -> None:
    with pytest.raises(LayoutError, match="too small"):
        interpret(_spec(base=BaseDims(width_mm=30.0, depth_mm=50.0)))


def test_rejects_min_wall_that_cannot_fit_lands() -> None:
    with pytest.raises(LayoutError, match="too small"):
        interpret(_spec(min_wall_mm=12.0))


def test_rejects_upright_too_short_for_m5_row() -> None:
    with pytest.raises(LayoutError, match="upright height"):
        interpret(_spec(upright=UprightDims(width_mm=43.0, height_mm=10.0)))


# ---------------------------------------------------------------------------
# Derived dims: the LLM-side fix for the bench-001 perturbation finding.
# (Mirrors the bench-003 `derive_plate_dims` pattern; ADR-0008.)
# ---------------------------------------------------------------------------


def test_derive_l_bracket_dims_for_canonical_min_wall() -> None:
    # min_wall=4, m5_spacing=20 (with a 1e-6 FP epsilon in the derive helper):
    #   base.width = base.depth ~ 2*(17.1 + 4) = 42.2 (+ 2e-6)
    #   upright.width ~ 2*(10 + 2.6 + 4) = 33.2 (+ 2e-6)
    #   upright.height ~ 2*(2.6 + 4) = 13.2 (+ 2e-6)
    spec = _spec(base=None, upright=None)
    base, upright = derive_l_bracket_dims(spec)
    tol = 1e-4  # well above the 2e-6 epsilon, well below any mech tolerance
    assert abs(base.width_mm - 42.2) < tol
    assert abs(base.depth_mm - 42.2) < tol
    assert abs(upright.width_mm - 33.2) < tol
    assert abs(upright.height_mm - 13.2) < tol


def test_derive_l_bracket_dims_grows_with_min_wall() -> None:
    # min_wall=6 -> base.width ~ 2*(17.1 + 6) = 46.2; upright.height ~ 17.2.
    spec = _spec(base=None, upright=None, min_wall_mm=6.0)
    base, upright = derive_l_bracket_dims(spec)
    tol = 1e-4
    assert abs(base.width_mm - 46.2) < tol
    assert abs(upright.height_mm - 17.2) < tol


def test_interpret_uses_derived_dims_when_base_and_upright_omitted(tmp_path: Path) -> None:
    # Omit both -- the interpreter must derive a passing part on the
    # canonical envelope.
    spec = _spec(base=None, upright=None)
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed
    # The motor_flange Box carries the derived width/depth (~42.2 each),
    # not the fixture's explicit 43x50.
    plate = next(o for o in result.program.operations if o.op_id == "motor_flange")  # type: ignore[union-attr]
    tol = 1e-4
    assert abs(plate.width - 42.2) < tol  # type: ignore[attr-defined]
    assert abs(plate.depth - 42.2) < tol  # type: ignore[attr-defined]


def test_explicit_base_and_upright_still_win_for_backward_compat(tmp_path: Path) -> None:
    # The hand-authored l_bracket fixture has explicit 43x50 base + 43x30
    # upright. Those values must be used as-is so the bench-001 reference and
    # any saved LLM output keep producing byte-equivalent geometry.
    spec = _spec()  # explicit values from the fixture
    result = run_layout_pipeline(spec, tmp_path)
    assert result.passed
    plate = next(o for o in result.program.operations if o.op_id == "motor_flange")  # type: ignore[union-attr]
    assert plate.width == 43.0  # type: ignore[attr-defined]
    assert plate.depth == 50.0  # type: ignore[attr-defined]


def test_thicker_wall_with_derived_dims_passes_where_pre_fix_failed(tmp_path: Path) -> None:
    # The bench-001 perturbation thicker-wall case (60x60x40, min_wall=6) had
    # 2/5 LLM trials emit base width 46.0 (off-by-0.2 from the required 46.2)
    # and the interpreter rightly rejected them. With derived dims the
    # interpreter computes 46.2 directly -- failure mode structurally
    # impossible.
    spec = _spec(base=None, upright=None, min_wall_mm=6.0)
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


def test_thicker_wall_with_derived_dims_rejected_when_envelope_too_small() -> None:
    # Honest negative case: min_wall=6 requires base width 46.2; an envelope
    # X under 46.2 is geometrically infeasible regardless of derivation.
    # Reject up-front with the same exceeds-envelope message bench-003 uses.
    spec = _spec(
        base=None,
        upright=None,
        min_wall_mm=6.0,
        envelope_mm=Vec3(x=44.0, y=60.0, z=40.0),
    )
    with pytest.raises(LayoutError, match="exceeds envelope"):
        interpret(spec)
