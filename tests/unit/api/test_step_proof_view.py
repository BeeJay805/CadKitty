"""The /generate ValidationView surfaces the STEP roundtrip proof (ADR-0024) when a STEP was exported
and re-read, and null otherwise -- so the frontend can show whether a downstream CAD tool reads back
the SAME solid. The proof is REPORTED, never a gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from forge_mind.api import core
from forge_mind.types import Vec3
from forge_mind.validator.report import StepProof, build_report


def _result_with_report(report: Any) -> Any:
    """A minimal stand-in for the PartResult shell _validation_view reads (loop.validation.report)."""
    val = SimpleNamespace(passed=True, report=report)
    return SimpleNamespace(
        loop=SimpleNamespace(validation=val, geometry_passed=True), disposition="shipped"
    )


def test_nonshipping_disposition_overrides_green_geometry() -> None:
    result = _result_with_report(build_report([], []))
    result.disposition = "not_assemblable"
    view = core._validation_view(result)
    assert view.passed is False and view.overall == "fail"
    assert any(check.id == "delivery.disposition" for check in view.failed_checks)


def test_validation_view_surfaces_step_proof() -> None:
    proof = StepProof(
        matches=True,
        volume_mm3=1000.0,
        area_mm2=600.0,
        centroid=Vec3(x=1.0, y=2.0, z=3.0),
        volume_rel_err=0.001,
        area_rel_err=0.002,
        centroid_dist_mm=0.01,
    )
    view = core._validation_view(_result_with_report(build_report([], [], step_proof=proof)))
    assert view.step_proof is not None
    assert view.step_proof.matches is True
    assert view.step_proof.volume_mm3 == 1000.0
    assert view.step_proof.centroid.z == 3.0  # the centroid is carried through, axis-faithful


def test_validation_view_step_proof_absent_is_none() -> None:
    view = core._validation_view(_result_with_report(build_report([], [])))
    assert view.step_proof is None
