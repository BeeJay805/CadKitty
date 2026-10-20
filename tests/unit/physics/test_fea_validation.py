"""FEA validation harness (slice 6) — the gate that lets the solver be called decision-grade.

These tests ARE the gate in CI: if the linear-elastic solver regresses, the analytic axial cases stop
matching and this goes red. Skipped cleanly when scikit-fem is absent (we can't validate -> not a false
failure, and fea_is_validated() returns False, never an unearned 'validated')."""

from __future__ import annotations

import pytest

pytest.importorskip("skfem")

from forge_mind.physics.fea_validation import fea_is_validated, run_validation


def test_axial_cases_match_analytic() -> None:
    outcomes = run_validation()
    axial = [o for o in outcomes if o.case.startswith("axial")]
    assert axial, "scikit-fem is present but the harness produced no axial outcomes"
    for o in axial:
        assert o.passed, (
            f"{o.case}/{o.quantity}: FEM {o.fem} vs analytic {o.analytical} "
            f"({o.rel_error:.1%} > tol {o.tolerance:.0%})"
        )
    # uniaxial stress should be essentially exact (P1 reproduces constant strain)
    stress = [o for o in axial if o.quantity == "stress"]
    assert stress and all(o.rel_error < 0.03 for o in stress)


def test_fea_is_validated_gate_is_green() -> None:
    assert fea_is_validated() is True


def test_cantilever_bending_is_decision_grade_with_p2() -> None:
    outcomes = run_validation()
    cant = [o for o in outcomes if o.case == "cantilever_bending"]
    assert cant, "expected a cantilever outcome"
    # P2 (quadratic) tets remove the shear-locking that made P1 ~47% off, so bending deflection now
    # matches the Euler-Bernoulli closed form within a few percent and is a GATING (decision-grade) case.
    assert cant[0].gating is True
    assert cant[0].passed is True
    assert cant[0].rel_error < 0.06


def test_gate_requires_gating_cases_to_pass() -> None:
    # The gate is exactly: every GATING outcome passed (and at least one exists).
    outcomes = run_validation()
    gating = [o for o in outcomes if o.gating]
    assert gating and fea_is_validated() == all(o.passed for o in gating)
