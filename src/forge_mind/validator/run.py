"""Validator orchestrator — runs all four tiers and assembles a ValidationReport.

    validate(model, program) -> ValidationReport

Tiers are independent and all run (we want the full picture, not first-failure).
Fix hints from every tier are collected for the Reviser.
"""

from __future__ import annotations

import contextlib

from forge_mind.compiler.compile import CADModel
from forge_mind.dsl.program import PartProgram
from forge_mind.fabrication import FabricationMode
from forge_mind.validator.conformance import check_conformance_all, check_measured_holes
from forge_mind.validator.feature_checks import run_feature_checks
from forge_mind.validator.interface_checks import run_interface_checks
from forge_mind.validator.mesh_checks import run_mesh_checks
from forge_mind.validator.metrics import DEFAULT_MATERIAL, compute_physics
from forge_mind.validator.printability import run_printability_checks
from forge_mind.validator.report import Check, Fix, PhysicsMetrics, ValidationReport, build_report
from forge_mind.validator.step_proof import verify_step_roundtrip


def validate(
    model: CADModel,
    program: PartProgram,
    material: str = DEFAULT_MATERIAL,
    *,
    fabrication_mode: FabricationMode = FabricationMode.PRINT_3D,
) -> ValidationReport:
    checks: list[Check] = []
    fixes: list[Fix] = []

    # Tier 1: mesh (from STL).
    checks.extend(run_mesh_checks(model.stl_path, program, fabrication_mode=fabrication_mode))

    # Tier 2: features (wall thickness, from STL via ray casting).
    f_checks, f_fixes = run_feature_checks(model.stl_path, program)
    checks.extend(f_checks)
    fixes.extend(f_fixes)

    # Tier 3: printability (overhangs, mesh per-triangle from STL) -- FDM-specific, so it RUNS only
    # when the fabrication mode enforces printability (ADR-0031). A GENERAL (not-3D-printed) part is
    # never checked for overhangs at all; PRINT_3D runs the tier. Default PRINT_3D keeps the
    # DSL/template path byte-identical (it passes no mode). This wires `enforces_printability`.
    if fabrication_mode.enforces_printability:
        p_checks, p_fixes = run_printability_checks(model.stl_path, program)
        checks.extend(p_checks)
        fixes.extend(p_fixes)

    # Tier 4: interfaces (from B-rep; stl_path lets non-cylinder checks run).
    i_checks, i_fixes = run_interface_checks(model.part, program, model.stl_path)
    checks.extend(i_checks)
    fixes.extend(i_fixes)

    # Tier 4b: hardware conformance (A4). Verifies each declared registry-backed feature (bearing
    # seat, clearance hole, heat-set boss, ...) against the SAME standards registry that built it --
    # the no-false-PASS gate for composed hardware (a non-conformant or unrecognized feature makes
    # the report non-PASS). Empty hardware_features (every non-composed part) adds zero checks, so
    # existing parts are byte/behaviour identical.
    checks.extend(check_conformance_all(program.hardware_features))

    # Tier 4c: declared-vs-MEASURED join (ADR-0035 B1, advisory-first). 4b verdicts the
    # DECLARATION against the registry; this measures the compiled B-rep -- a declared round
    # feature must exist as a CONCAVE cylinder of the declared diameter. INFO status only:
    # gates nothing this pass (promotion criterion pre-registered in ADR-0035); best-effort,
    # a measurement crash must never break validation.
    with contextlib.suppress(Exception):
        checks.extend(check_measured_holes(model.part, program.hardware_features))

    # Physics metric (ADR-0019): measured mass/COM, REPORTED not gated. Best-effort —
    # a metric failure must never break validation. `material` is threaded from the source
    # blueprint/spec by the generation pipelines; callers that hold only a bare PartProgram
    # (grader / reviser / candidate-eval / CLI) default to PLA, because material is not on
    # the PartProgram (putting it there would break byte-equivalence). Those paths consume
    # the report's CHECKS, not the metric, so the default is harmless there.
    metrics: PhysicsMetrics | None = None
    try:
        metrics = compute_physics(model.part, material)
    except Exception:
        metrics = None

    # STEP export integrity proof (ISO 10303-49 spirit): re-import the exported STEP and confirm a
    # downstream CAD tool reads back the SAME solid (volume/area/centroid). REPORTED, never gates;
    # cheap (~40ms re-import) and best-effort. Mirrors the physics-metric pattern above.
    step_proof = None
    try:
        step_proof = verify_step_roundtrip(model.part, model.step_path)
    except Exception:
        step_proof = None

    return build_report(checks, fixes, metrics, step_proof)


__all__ = ["validate"]
