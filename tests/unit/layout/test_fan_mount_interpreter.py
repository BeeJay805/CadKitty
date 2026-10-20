"""Fan-mount interpreter (bench-006 Stage A): an AdapterPlateSpec with a
fan_component_id builds through the EXISTING adapter_plate pattern, placing a
fan_face on +Z and reusing the -Z M5 mount. No new Pattern, no LLM.

The deliverable gate: on the explicit 60mm fixture the interpreter is byte-
equivalent to the hand-authored Stage-1 reference; the motor (NEMA17) path is
unchanged; an infeasible spec (the 40mm fan at a 4mm wall) is rejected up front.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.layout.adapter_interpreter import interpret_adapter_plate
from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.fan_mount import (
    derive_fan_offsets,
    derive_fan_plate_dims,
    interpret_fan_mount,
    resolve_fan_facts,
)
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.pipeline import run_layout_pipeline
from forge_mind.types import Vec3

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench006_fan_mount.layoutspec.json"

REQUIRED_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.fan_face.bolt_holes_present",
    "interfaces.fan_face.bolt_pattern_square",
    "interfaces.fan_face.airflow_bore_present",
    "interfaces.extrusion_face.two_m5_holes",
    "interfaces.extrusion_face.m5_spacing",
]


def _spec(**overrides: object) -> AdapterPlateSpec:
    spec = AdapterPlateSpec.model_validate(json.loads(FIXTURE.read_text()))
    return spec.model_copy(update=overrides) if overrides else spec


# ---------------------------------------------------------------------------
# Dispatch + byte-equivalence to the Stage-1 hand-authored reference.
# ---------------------------------------------------------------------------


def test_layout_dispatch_routes_fan_to_fan_path() -> None:
    # interpret_layout (the union dispatch) routes a fan spec through the
    # adapter_plate pattern; the resulting +Z interface is a fan_face.
    program = interpret_layout(_spec())
    fan = next(i for i in program.interfaces if i.name == "fan_face")
    assert fan.interface_type.value == "fan_face"
    assert fan.normal.as_tuple() == (0.0, 0.0, 1.0)


def test_geometry_matches_handwritten_60mm_reference() -> None:
    # THE STAGE-A GATE: on the explicit 60mm fixture the interpreter produces a
    # part equivalent to the hand-authored bench-006 60mm reference.
    from examples.handwritten_fan_2020_adapter import build_fan_mount_reference

    interpreted = interpret_adapter_plate(_spec())
    reference = build_fan_mount_reference(size="60mm")

    assert [o.op_id for o in interpreted.operations] == [o.op_id for o in reference.operations]
    by_ref = {o.op_id: o for o in reference.operations}
    for op in interpreted.operations:
        ref = by_ref[op.op_id]
        assert op.mode == ref.mode
        assert op.position.as_tuple() == ref.position.as_tuple()
        assert op.rotation.x == ref.rotation.x
        assert op.rotation.y == ref.rotation.y
        assert op.rotation.z == ref.rotation.z


def test_fan_layoutspec_passes_validator(tmp_path: Path) -> None:
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
# Derive-not-pick: omitting plate dims + offsets still yields a passing part.
# ---------------------------------------------------------------------------


def test_interpret_uses_derived_dims_and_offsets_when_omitted(tmp_path: Path) -> None:
    # The shape an LLM emits under a decontaminated prompt: fan id + envelope +
    # min_wall, no geometry. The interpreter derives a passing part.
    spec = _spec(
        plate_width_mm=None, plate_depth_mm=None, motor_center_x_mm=None, extrusion_center_x_mm=None
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


def test_resolve_fan_facts_known_and_unknown() -> None:
    assert resolve_fan_facts("fan_60mm") == (50.0, 28.0)
    assert resolve_fan_facts("fan_40mm") == (32.0, 19.0)
    with pytest.raises(LayoutError, match="unknown fan component"):
        resolve_fan_facts("fan_999mm")


def test_derive_fan_dims_grow_with_min_wall() -> None:
    # Deriving for the 60mm fan at min_wall 4: feasible, plate covers the cluster.
    spec = _spec(
        plate_width_mm=None, plate_depth_mm=None, motor_center_x_mm=None, extrusion_center_x_mm=None
    )
    f_x, e_x = derive_fan_offsets(spec.envelope_mm.x, spec.min_wall_mm, 50.0, 28.0)
    assert f_x > 0 and e_x < 0
    w, d = derive_fan_plate_dims(spec, 50.0, 28.0)
    assert w > 0 and d > 0


# ---------------------------------------------------------------------------
# Feasibility: the 40mm fan at a 4mm wall is the ADR-0011 known limitation.
# ---------------------------------------------------------------------------


def test_40mm_fan_at_4mm_wall_rejected_bore_eats_bolt_land() -> None:
    # KNOWN LIMITATION (ADR-0011): a flat 40mm fan mount cannot keep the bore-to-
    # bolt land at a 4mm wall (land ~2.03mm). The interpreter must reject it up
    # front with the bore-to-bolt diagnostic, not emit a failing part.
    spec = _spec(
        name="fan_40mm_tight",
        fan_component_id="fan_40mm",
        min_wall_mm=4.0,
        plate_width_mm=None,
        plate_depth_mm=None,
        motor_center_x_mm=None,
        extrusion_center_x_mm=None,
        envelope_mm=Vec3(x=90.0, y=60.0, z=8.0),
    )
    with pytest.raises(LayoutError, match="bore"):
        interpret_fan_mount(spec)


def test_40mm_fan_at_2mm_wall_is_feasible(tmp_path: Path) -> None:
    # The same 40mm fan at a 2mm wall IS feasible (land ~2.03mm >= 2mm) -- proving
    # the rejection above is a real geometric limit, not a blanket "no 40mm".
    spec = _spec(
        name="fan_40mm_2mm",
        fan_component_id="fan_40mm",
        min_wall_mm=2.0,
        plate_width_mm=None,
        plate_depth_mm=None,
        motor_center_x_mm=None,
        extrusion_center_x_mm=None,
        envelope_mm=Vec3(x=90.0, y=60.0, z=8.0),
    )
    result = run_layout_pipeline(spec, tmp_path)
    assert result.error is None, result.error
    assert result.report is not None and result.passed


def test_too_small_envelope_rejected() -> None:
    with pytest.raises(LayoutError, match=r"exceeds envelope|too small to host"):
        interpret_fan_mount(
            _spec(
                plate_width_mm=None,
                plate_depth_mm=None,
                motor_center_x_mm=None,
                extrusion_center_x_mm=None,
                envelope_mm=Vec3(x=50.0, y=68.0, z=8.0),
            )
        )
