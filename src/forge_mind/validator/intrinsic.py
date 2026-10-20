"""Intrinsic pre-print check (ADR-0056, Flavor A) — the validator WITHOUT declared intent.

`validate(model, program)` joins declared intent against geometry; its high-value tiers
(interfaces, hardware conformance, declared-vs-measured holes) all NEED that intent. On a model we
know nothing about — an uploaded STL/STEP — only the *intrinsic* checks survive: manifold/watertight,
a generic printable-thickness floor, overhang angle, mass/CoM, stand-or-tip stability, STEP round-trip.

This is that intrinsic check, as a SEPARATE fail-closed entry. It is deliberately NOT `validate()` with
an empty program — that would silently skip the intent tiers and hand back a hollow PASS (the council's
#1 guardrail, ADR-0056). The verdict here is structurally honest: ``fit_function_verified`` is ALWAYS
False and the prose says so. No verdict ever uses the word "safe". The validator itself is untouched —
we drive its existing tier functions with a minimal printability-constraints carrier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh
from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.primitives import Box as DslBox
from forge_mind.dsl.primitives import OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.fabrication import FabricationMode
from forge_mind.physics.stability import compute_stability
from forge_mind.types import Vec3
from forge_mind.validator.feature_checks import run_feature_checks
from forge_mind.validator.mesh_checks import run_mesh_checks
from forge_mind.validator.metrics import DEFAULT_MATERIAL, density_for
from forge_mind.validator.printability import run_printability_checks
from forge_mind.validator.report import (
    Check,
    CheckStatus,
    PhysicsMetrics,
    StepProof,
    Tier,
    aggregate_status,
)
from forge_mind.validator.step_proof import verify_step_roundtrip

# A generic printability floor (~2 perimeters on a 0.4 mm nozzle) used when there is NO declared intent.
# This is a printing-geometry minimum, NOT a user's min_wall — a foreign part thinner than this is
# flagged as likely-not-printable. Conservative on purpose: a false-FAIL is acceptable here, a false-PASS
# is not (ADR-0056 bedrock). Tunable.
INTRINSIC_MIN_WALL_FLOOR_MM = 0.8
# Steeper-than-this downward faces need support; the printability default that needs no declared intent.
INTRINSIC_MAX_OVERHANG_DEG = 45.0


class StabilityView(BaseModel):
    """Stand-or-tip projection (Z-up mm), matching the /generate ``metrics.stability`` shape so the
    viewer can reuse the same balance lens. None when there is no clear flat base."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stable: bool
    margin_mm: float
    tip_angle_deg: float
    com_mm: Vec3
    com_height_mm: float
    base_z_mm: float
    footprint_mm: list[list[float]]


class IntrinsicReport(BaseModel):
    """The honest pre-print verdict for a model with no declared intent. ``printable`` aggregates the
    intrinsic checks; ``fit_function_verified`` is ALWAYS False (we cannot certify fit/function without
    intent) and the prose ``verdict`` says so. Never a bare PASS."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    printable: CheckStatus
    fit_function_verified: bool = False
    verdict: str
    checks: list[Check]
    metrics: PhysicsMetrics | None = None
    stability: StabilityView | None = None
    step_proof: StepProof | None = Field(default=None)


# The placeholder operation exists ONLY so PartProgram (operations: min_length=1) validates; it is never
# compiled — `validate_intrinsic` measures the uploaded geometry directly. The tier functions read only
# `program.constraints`, so this carries the intrinsic printability floors and nothing else.
_PLACEHOLDER_OP = DslBox(
    op_id="upload",
    mode=OpMode.ADD,
    position=Vec3(x=0.0, y=0.0, z=0.0),
    width=1.0,
    depth=1.0,
    height=1.0,
)


def _intrinsic_program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="upload", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[_PLACEHOLDER_OP],
        constraints=Constraints(
            min_wall_thickness=INTRINSIC_MIN_WALL_FLOOR_MM,
            max_overhang_deg=INTRINSIC_MAX_OVERHANG_DEG,
            max_bbox=None,
        ),
    )


def _mesh_physics(mesh: trimesh.Trimesh, material: str) -> PhysicsMetrics | None:
    """Mass/CoM from the mesh (foreign-geometry friendly — no B-rep needed). REPORTED, never a gate,
    so an approximate value on a non-watertight mesh is harmless (the watertight check is the verdict)."""
    try:
        volume_mm3 = abs(float(mesh.volume))
        com = mesh.center_mass
        density = density_for(material)
        return PhysicsMetrics(
            volume_mm3=volume_mm3,
            mass_g=(volume_mm3 / 1000.0) * density,
            center_of_mass=Vec3(x=float(com[0]), y=float(com[1]), z=float(com[2])),
            material=material,
            density_g_cm3=density,
        )
    except Exception:
        return None


def _stability_view(mesh: trimesh.Trimesh) -> StabilityView | None:
    try:
        result = compute_stability(mesh)
    except Exception:
        return None
    if result is None:
        return None
    return StabilityView(
        stable=result.stable,
        margin_mm=result.margin_mm,
        tip_angle_deg=result.tip_angle_deg,
        com_mm=Vec3(x=result.com[0], y=result.com[1], z=result.com[2]),
        com_height_mm=result.com_height_mm,
        base_z_mm=round(result.com[2] - result.com_height_mm, 2),
        footprint_mm=[[x, y] for x, y in result.footprint],
    )


def _verdict(printable: CheckStatus, failed: list[str]) -> str:
    if printable == CheckStatus.PASS:
        head = "Printable: yes."
    elif printable == CheckStatus.WARN:
        head = "Printable: with cautions."
    else:
        detail = f" ({', '.join(failed)})" if failed else ""
        head = f"Printable: NO{detail}."
    # The honest ceiling: no declared intent means we cannot say it fits or works.
    return f"{head} Fit/function: NOT verified — you didn't tell us what it's for."


def validate_intrinsic(
    stl_path: Path,
    *,
    part: Any = None,
    step_path: Path | None = None,
    material: str = DEFAULT_MATERIAL,
) -> IntrinsicReport:
    """Run the intent-free pre-print checks on a mesh (+ optional B-rep for the STEP round-trip).

    Fail-closed: if the geometry can't even be read, the report is FAIL — never a silent PASS. The
    intent tiers (interfaces / hardware conformance) are NOT run, and ``fit_function_verified`` stays
    False, so this can never be mistaken for a fit/function certificate (ADR-0056 Flavor A)."""
    program = _intrinsic_program()
    try:
        checks: list[Check] = list(
            run_mesh_checks(stl_path, program, fabrication_mode=FabricationMode.PRINT_3D)
        )
        f_checks, _ = run_feature_checks(stl_path, program)
        checks.extend(f_checks)
        p_checks, _ = run_printability_checks(stl_path, program)
        checks.extend(p_checks)
    except Exception as exc:  # fail CLOSED — an unreadable / un-measurable model is not a PASS
        checks = [
            Check(
                id="intrinsic.unreadable",
                tier=Tier.MESH,
                status=CheckStatus.FAIL,
                message=f"Could not read the model for an intrinsic check ({type(exc).__name__}).",
            )
        ]

    # REPORTED metrics (mass/CoM, stand-or-tip) — best-effort, never gate. Loaded once.
    metrics: PhysicsMetrics | None = None
    stability: StabilityView | None = None
    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
        if isinstance(mesh, trimesh.Trimesh):
            metrics = _mesh_physics(mesh, material)
            stability = _stability_view(mesh)
    except Exception:
        metrics = None
        stability = None

    step_proof: StepProof | None = None
    if part is not None and step_path is not None:
        try:
            step_proof = verify_step_roundtrip(part, step_path)
        except Exception:
            step_proof = None

    printable = aggregate_status(checks)
    failed = [c.id for c in checks if c.status == CheckStatus.FAIL]
    return IntrinsicReport(
        printable=printable,
        verdict=_verdict(printable, failed),
        checks=checks,
        metrics=metrics,
        stability=stability,
        step_proof=step_proof,
    )


__all__ = [
    "INTRINSIC_MAX_OVERHANG_DEG",
    "INTRINSIC_MIN_WALL_FLOOR_MM",
    "IntrinsicReport",
    "StabilityView",
    "validate_intrinsic",
]
