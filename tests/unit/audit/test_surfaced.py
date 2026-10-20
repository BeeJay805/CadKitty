"""Tests for the ADR-0083 v2 'surfaced' classifier (audit_validator._surfaced_by).

v1 measured only `validation.passed`, which is advisory-rich, so it under-counted the moat: a
defect the validator SURFACES as an advisory issue (but doesn't gate) read as a full escape. v2
adds `_surfaced_by` = "did the reward signal SEE this defect anywhere (gate OR advisory)?", the
measure that matters for using the validator as an RL reward. These pin the mapping per class.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import audit_validator as av  # noqa: E402


def _v(issues: list[str] | None = None, failed: list[str] | None = None, fm: bool = False) -> dict:
    return {"issues": issues or [], "failed_checks": failed or [], "features_missing": fm}


def test_stated_dim_surfaced_via_existing_findings() -> None:
    # v2.2: stated-dim drift has partial CLEAN coverage -- a bore drifted out of its ID window
    # surfaces as SIZE MISMATCH, envelope drift via ENVELOPE. (A bolt-on advisory was too noisy on
    # good parts to add, so this measures the existing clean coverage, ~9% -- honest, not inflated.)
    assert av._surfaced_by("off_brief_stated_dim", _v(issues=["SIZE MISMATCH: ..."])) is True
    assert av._surfaced_by("off_brief_stated_dim", _v(issues=["ENVELOPE: X is off"])) is True
    # a linear stated dim (arm length) with no bore/envelope owning check does NOT surface
    assert av._surfaced_by("off_brief_stated_dim", _v(issues=["all fine"])) is False


def test_envelope_surfaced_via_issue_or_check() -> None:
    assert av._surfaced_by("envelope_violation", _v(issues=["ENVELOPE: X is 83 mm..."])) is True
    assert av._surfaced_by("envelope_violation", _v(failed=["mesh.bbox_scale"])) is True
    assert av._surfaced_by("envelope_violation", _v(issues=["something else"])) is False


def test_subfloor_wall_surfaced_even_when_advisory() -> None:
    # min_wall is ADVISORY on the agentic path (doesn't gate `passed`) but IS a failed check id ->
    # the reward signal still SEES it, which is the whole point of the surfaced measure
    assert av._surfaced_by("sub_floor_wall", _v(failed=["features.min_wall_thickness"])) is True
    assert av._surfaced_by("sub_floor_wall", _v(issues=["the wall is too THIN here"])) is True
    assert av._surfaced_by("sub_floor_wall", _v()) is False


def test_hole_classes_surfaced_via_measured_bores_or_features_missing() -> None:
    assert av._surfaced_by("blind_where_through", _v(fm=True)) is True
    assert (
        av._surfaced_by("blind_where_through", _v(issues=["MEASURED BORES vs PLAN: MISSING"]))
        is True
    )
    assert av._surfaced_by("dropped_named_feature", _v(issues=["MISSING FEATURES: ..."])) is True
    assert av._surfaced_by("dropped_named_feature", _v()) is False


def test_off_footprint_surfaced_via_bore_findings() -> None:
    assert av._surfaced_by("off_footprint_bore", _v(issues=["SIZE MISMATCH: ..."])) is True
    assert av._surfaced_by("off_footprint_bore", _v(fm=True)) is True
    assert av._surfaced_by("off_footprint_bore", _v(issues=["nothing relevant"])) is False


def test_unknown_class_returns_none() -> None:
    assert av._surfaced_by("some_new_class", _v(issues=["MISSING"])) is None
