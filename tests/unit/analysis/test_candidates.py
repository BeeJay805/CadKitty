"""Unit tests for the deterministic best-of-k scorer + selection (no LLM)."""

from __future__ import annotations

from forge_mind.candidates import (
    CandidateResult,
    candidate_label,
    pick_best,
    score_candidate,
)
from forge_mind.types import BBox, Vec3
from forge_mind.validator.report import Check, CheckStatus, Tier, build_report

CAP = (60.0, 60.0, 40.0)
FIT_BBOX = BBox(min=Vec3(x=0, y=0, z=0), max=Vec3(x=43, y=50, z=30))  # within envelope


def _report(*fails: tuple[str, Tier]):  # type: ignore[no-untyped-def]
    checks = [Check(id="mesh.watertight", tier=Tier.MESH, status=CheckStatus.PASS, message="ok")]
    checks += [
        Check(id=cid, tier=tier, status=CheckStatus.FAIL, message="x") for cid, tier in fails
    ]
    return build_report(checks, [])


def _passing_report():  # type: ignore[no-untyped-def]
    return build_report(
        [Check(id="mesh.watertight", tier=Tier.MESH, status=CheckStatus.PASS, message="ok")], []
    )


def _score(report, *, compiled=True, op_count=9, bbox=FIT_BBOX):  # type: ignore[no-untyped-def]
    return score_candidate(
        compiled=compiled, report=report, op_count=op_count, bbox=bbox, envelope_cap_mm=CAP
    )


def test_passing_beats_any_failing() -> None:
    passing = _score(_passing_report())
    # Worst plausible failing candidate: thin wall + overhang + bbox, compiled.
    worst_fail = _score(
        _report(
            ("features.min_wall_thickness", Tier.FEATURES),
            ("printability.overhang_max_angle", Tier.PRINTABILITY),
            ("mesh.bbox_within_max", Tier.MESH),
        )
    )
    assert passing > worst_fail


def test_thin_wall_is_penalized() -> None:
    clean = _score(_report(("interfaces.x.y", Tier.INTERFACES)))  # interface fail only
    thin = _score(_report(("features.min_wall_thickness", Tier.FEATURES)))
    # A thin wall (heavy penalty) but interfaces OK still beats an interface failure
    # (hard priority), yet thinness must reduce the score vs a clean interface-OK part.
    clean_iface_ok = _score(_report(("mesh.winding_consistent", Tier.MESH)))
    assert thin < clean_iface_ok  # thin-wall penalty applied
    assert thin > clean  # interface pass beats interface failure (hard priority)


def test_overhang_is_penalized() -> None:
    base = _score(_report(("mesh.winding_consistent", Tier.MESH)))  # some non-penalized fail
    overhang = _score(_report(("printability.overhang_max_angle", Tier.PRINTABILITY)))
    assert overhang < base


def test_compile_success_beats_compile_failure() -> None:
    compiled = _score(_report(("features.min_wall_thickness", Tier.FEATURES)))
    not_compiled = score_candidate(
        compiled=False, report=None, op_count=9, bbox=None, envelope_cap_mm=CAP
    )
    assert compiled > not_compiled


def test_interface_pass_beats_interface_failure() -> None:
    iface_ok = _score(_report(("features.min_wall_thickness", Tier.FEATURES)))  # iface OK
    iface_fail = _score(
        _report(("interfaces.motor_face.bolt_pattern_31mm_square", Tier.INTERFACES))
    )
    assert iface_ok > iface_fail


def test_bbox_exceedance_penalized() -> None:
    fit = _score(_passing_report(), bbox=FIT_BBOX)
    oversized = _score(
        _passing_report(), bbox=BBox(min=Vec3(x=0, y=0, z=0), max=Vec3(x=80, y=60, z=40))
    )
    assert oversized < fit  # drift penalty for exceeding the envelope


def test_excessive_op_count_penalized() -> None:
    few = _score(_passing_report(), op_count=9)
    many = _score(_passing_report(), op_count=30)
    assert many < few


def test_pick_best_selects_passing_candidate() -> None:
    cands = [
        CandidateResult(
            0,
            passed=False,
            compiled=True,
            score=_score(_report(("features.min_wall_thickness", Tier.FEATURES))),
            primary_label="wall_too_thin",
            failed_check_ids=["features.min_wall_thickness"],
            cost_usd=0.01,
            duration_ms=8000,
        ),
        CandidateResult(
            1,
            passed=True,
            compiled=True,
            score=_score(_passing_report()),
            primary_label="passed",
            failed_check_ids=[],
            cost_usd=0.01,
            duration_ms=8000,
        ),
        CandidateResult(
            2,
            passed=False,
            compiled=False,
            score=score_candidate(
                compiled=False, report=None, op_count=0, bbox=None, envelope_cap_mm=CAP
            ),
            primary_label="dsl_compile_failed",
            failed_check_ids=[],
            cost_usd=0.01,
            duration_ms=8000,
        ),
    ]
    assert pick_best(cands) == 1


def test_pick_best_tie_breaks_lowest_id() -> None:
    s = _score(_passing_report())
    cands = [
        CandidateResult(
            0,
            passed=True,
            compiled=True,
            score=s,
            primary_label="passed",
            failed_check_ids=[],
            cost_usd=0.0,
            duration_ms=0.0,
        ),
        CandidateResult(
            1,
            passed=True,
            compiled=True,
            score=s,
            primary_label="passed",
            failed_check_ids=[],
            cost_usd=0.0,
            duration_ms=0.0,
        ),
    ]
    assert pick_best(cands) == 0


def test_candidate_label() -> None:
    assert candidate_label(True, True, _passing_report()) == "passed"
    assert candidate_label(False, False, None) == "dsl_compile_failed"
    assert (
        candidate_label(False, True, _report(("features.min_wall_thickness", Tier.FEATURES)))
        == "wall_too_thin"
    )
