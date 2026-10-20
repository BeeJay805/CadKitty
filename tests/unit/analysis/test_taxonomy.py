"""Unit tests for the failure-taxonomy classifier (no LLM).

We construct synthetic PipelineResults to drive each classification path.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.analysis.taxonomy import (
    Expectations,
    FailureLabel,
    classify,
    label_for_check,
    remap_validate_label,
)
from forge_mind.blueprint.schema import Blueprint, BlueprintVec3
from forge_mind.intent.schema import IntentConstraints, IntentSpec, MatingInterfaceRequest
from forge_mind.pipeline import PipelineResult
from forge_mind.validator.report import Check, CheckStatus, Tier, ValidationReport, build_report

REPO = Path(__file__).resolve().parents[3]
BLUEPRINT_FIXTURE = REPO / "tests" / "fixtures" / "blueprints" / "nema17_mount.blueprint.json"

EXP = Expectations(
    required_interface_types={"nema17_face", "extrusion_2020_slot"},
    envelope_cap_mm=(60.0, 60.0, 40.0),
    min_wall_mm=4.0,
)


def _intent(interfaces: list[str]) -> IntentSpec:
    return IntentSpec(
        part_family="motor_mount",
        summary="x",
        mating_interfaces=[
            MatingInterfaceRequest(interface_type=t, target="t")
            for t in interfaces  # type: ignore[arg-type]
        ],
        constraints=IntentConstraints(
            min_wall_mm=4.0, material="PLA", max_envelope_mm=[60.0, 60.0, 40.0]
        ),
    )


def _good_intent() -> IntentSpec:
    return _intent(["nema17_face", "extrusion_2020_slot"])


def _good_blueprint() -> Blueprint:
    return Blueprint.model_validate(json.loads(BLUEPRINT_FIXTURE.read_text()))


def _failing_report(check_id: str, tier: Tier) -> ValidationReport:
    return build_report(
        [Check(id=check_id, tier=tier, status=CheckStatus.FAIL, message="boom")], []
    )


def _passing_report() -> ValidationReport:
    return build_report(
        [Check(id="mesh.watertight", tier=Tier.MESH, status=CheckStatus.PASS, message="ok")], []
    )


def test_passed() -> None:
    r = PipelineResult(
        prompt="x", intent=_good_intent(), blueprint=_good_blueprint(), report=_passing_report()
    )
    assert classify(r, EXP) == FailureLabel.PASSED


def test_intent_missing_requirement_is_root_even_when_validate_fails() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_intent(["nema17_face"]),  # dropped the extrusion interface
        blueprint=_good_blueprint(),
        report=_failing_report("interfaces.extrusion_face.two_m5_holes", Tier.INTERFACES),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.INTENT_MISSING_REQUIREMENT


def test_blueprint_missing_requirement() -> None:
    bp = _good_blueprint()
    bp = bp.model_copy(update={"required_interfaces": [bp.required_interfaces[0]]})
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=bp,
        report=_failing_report("interfaces.extrusion_face.two_m5_holes", Tier.INTERFACES),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.BLUEPRINT_MISSING_REQUIREMENT


def test_blueprint_bad_geometry_constraint() -> None:
    bp = _good_blueprint().model_copy(
        update={"envelope_mm": BlueprintVec3(x=200.0, y=60.0, z=40.0)}
    )
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=bp,
        report=_failing_report("mesh.bbox_within_max", Tier.MESH),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.BLUEPRINT_BAD_GEOMETRY_CONSTRAINT


def test_dsl_schema_invalid() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        failed_stage="dsl",
        error="parse",
    )
    assert classify(r, EXP) == FailureLabel.DSL_SCHEMA_INVALID


def test_dsl_compile_failed() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        failed_stage="compile",
        error="boom",
    )
    assert classify(r, EXP) == FailureLabel.DSL_COMPILE_FAILED


def test_validate_topology() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=_failing_report("mesh.single_body", Tier.MESH),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.TOPOLOGY_WRONG


def test_validate_interface() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=_failing_report("interfaces.motor_face.bolt_pattern_31mm_square", Tier.INTERFACES),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.INTERFACE_FAILED


def test_validate_envelope() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=_failing_report("mesh.bbox_within_max", Tier.MESH),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.ENVELOPE_FAILED


def test_validate_wall_too_thin() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=_failing_report("features.min_wall_thickness", Tier.FEATURES),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.WALL_TOO_THIN


def test_validate_overhang_too_steep() -> None:
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=_failing_report("printability.overhang_max_angle", Tier.PRINTABILITY),
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.OVERHANG_TOO_STEEP


def test_wall_outranks_overhang_when_both_fail() -> None:
    # A thin wall is more fundamental than an overhang; it wins the primary label.
    report = build_report(
        [
            Check(
                id="features.min_wall_thickness",
                tier=Tier.FEATURES,
                status=CheckStatus.FAIL,
                message="thin",
            ),
            Check(
                id="printability.overhang_max_angle",
                tier=Tier.PRINTABILITY,
                status=CheckStatus.FAIL,
                message="steep",
            ),
        ],
        [],
    )
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=report,
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.WALL_TOO_THIN


def test_primary_check_priority_topology_over_envelope() -> None:
    # Both topology and envelope fail; topology is the more fundamental root.
    report = build_report(
        [
            Check(
                id="mesh.bbox_within_max", tier=Tier.MESH, status=CheckStatus.FAIL, message="big"
            ),
            Check(id="mesh.single_body", tier=Tier.MESH, status=CheckStatus.FAIL, message="split"),
        ],
        [],
    )
    r = PipelineResult(
        prompt="x",
        intent=_good_intent(),
        blueprint=_good_blueprint(),
        report=report,
        failed_stage="validate",
    )
    assert classify(r, EXP) == FailureLabel.TOPOLOGY_WRONG


def test_label_for_check_mapping() -> None:
    assert label_for_check("features.min_wall_thickness") == FailureLabel.WALL_TOO_THIN
    assert label_for_check("printability.overhang_max_angle") == FailureLabel.OVERHANG_TOO_STEEP
    assert label_for_check("interfaces.motor_face.x") == FailureLabel.INTERFACE_FAILED
    assert label_for_check("mesh.bbox_within_max") == FailureLabel.ENVELOPE_FAILED
    assert label_for_check("mesh.watertight") == FailureLabel.TOPOLOGY_WRONG
    # Genuinely disconnected/non-manifold solids stay topology_wrong.
    assert label_for_check("mesh.single_body") == FailureLabel.TOPOLOGY_WRONG
    # Other printability checks (not overhang) keep the generic label.
    assert label_for_check("printability.bridge_span") == FailureLabel.PRINTABILITY_FAILED


def test_remap_validate_label_splits_old_topology_wrong() -> None:
    # An old topology_wrong whose primary check was a thin wall -> wall_too_thin.
    assert (
        remap_validate_label("topology_wrong", "features.min_wall_thickness")
        == FailureLabel.WALL_TOO_THIN
    )
    # An old topology_wrong from a genuine disconnect stays topology_wrong.
    assert remap_validate_label("topology_wrong", "mesh.single_body") == FailureLabel.TOPOLOGY_WRONG
    # An old printability_failed whose check was overhang -> overhang_too_steep.
    assert (
        remap_validate_label("printability_failed", "printability.overhang_max_angle")
        == FailureLabel.OVERHANG_TOO_STEEP
    )


def test_remap_preserves_upstream_root_labels() -> None:
    # Upstream root causes never came from a validator check, so they are kept.
    for label in (
        "intent_missing_requirement",
        "blueprint_bad_geometry_constraint",
        "dsl_compile_failed",
        "passed",
        "unknown",
    ):
        assert remap_validate_label(label, "features.min_wall_thickness") == FailureLabel(label)
